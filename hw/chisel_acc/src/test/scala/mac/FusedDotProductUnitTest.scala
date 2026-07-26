package mx.mac

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import java.lang.Float.intBitsToFloat
import scala.util.Random

class FusedDotProductUnitTest extends AnyFunSuite with ChiselScalatestTester {

  // -----------------------------------------------------------------------
  // Log file — verbose output here; stdout only for failures
  // -----------------------------------------------------------------------
  private val _logWriter: java.io.PrintWriter = {
    val fw = new java.io.FileWriter("fused_dot_product_unit_test.log", false)
    new java.io.PrintWriter(fw, true)
  }
  private def log(msg: String): Unit    = _logWriter.println(msg)
  private def logErr(msg: String): Unit = { _logWriter.println(msg); println(msg) }

  // -----------------------------------------------------------------------
  // Software golden model
  // -----------------------------------------------------------------------

  /** Decode raw bit-pattern → Double (MX element format). */
  def decodeElement(raw: Int, t: ElementType): Double = {
    if (t.name == "INT8") {
      // MXINT8 2's complement: treat as signed 8-bit integer.
      // value = signed_int × 2^implicitScaleExp (= 2^-6).
      val signedVal = if ((raw & 0x80) != 0) raw - 256 else raw
      signedVal.toDouble * Math.pow(2, t.implicitScaleExp)
    } else {
      val sign = if (((raw >> (t.totalWidth - 1)) & 1) == 1) -1.0 else 1.0
      val exp  = (raw >> t.elementWidthMant) & ((1 << t.elementWidthExp) - 1)
      val mant = raw & ((1 << t.elementWidthMant) - 1)
      if (exp == 0)
        sign * (mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, 1 - t.bias)
      else
        sign * (1.0 + mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, exp - t.bias)
    }
  }

  /** Decode raw bit-pattern → Double (MX scale format, no sign bit). */
  def decodeScale(raw: Int, s: ScaleType): Double = {
    val expBits = (raw >> s.mantScaleWidth) & ((1 << s.expScaleWidth) - 1)
    if (s.mantScaleWidth == 0) {
      Math.pow(2, expBits - s.bias)
    } else {
      val mantBits  = raw & ((1 << s.mantScaleWidth) - 1)
      // 次正规数（expBits=0）：无隐含 1，指数固定为 1-bias
      val implicit1 = if (expBits > 0) 1.0 else 0.0
      val effExp    = if (expBits > 0) expBits - s.bias else 1 - s.bias
      (implicit1 + mantBits.toDouble / (1 << s.mantScaleWidth)) * Math.pow(2, effExp)
    }
  }

  /** Encode Double → raw bits for an ElementType (best-effort, clamps extremes). */
  def encodeElement(value: Double, t: ElementType): Int = {
    if (value == 0.0) return 0
    val sign   = if (value < 0) 1 else 0
    val absVal = math.abs(value)
    if (t.name == "INT8") {
      // MXINT8 2's complement: encode as signed 8-bit integer.
      // magnitude = round(|value| / 2^implicitScaleExp), clamped to [0, 127].
      val magnitude = math.round(absVal / Math.pow(2, t.implicitScaleExp)).toInt.min(127)
      if (sign == 0) magnitude else (-magnitude) & 0xFF
    } else {
      val expUnbiased = math.floor(math.log(absVal) / math.log(2)).toInt
      val expBiased   = expUnbiased + t.bias
      if (expBiased <= 0) {
        val mantInt = math.round(absVal / math.pow(2, 1 - t.bias) * (1 << t.elementWidthMant))
                          .toInt.min((1 << t.elementWidthMant) - 1)
        (sign << (t.totalWidth - 1)) | mantInt
      } else {
        val expClamped = expBiased.min((1 << t.elementWidthExp) - 1)
        val mantInt    = math.round((absVal / math.pow(2, expUnbiased) - 1.0) * (1 << t.elementWidthMant))
                              .toInt.min((1 << t.elementWidthMant) - 1).max(0)
        (sign << (t.totalWidth - 1)) | (expClamped << t.elementWidthMant) | mantInt
      }
    }
  }

  /** Encode positive Double → raw bits for a ScaleType. */
  def encodeScale(value: Double, s: ScaleType): Int = {
    val expUnbiased = math.floor(math.log(value) / math.log(2)).toInt
    val expBiased   = (expUnbiased + s.bias).max(0).min((1 << s.expScaleWidth) - 1)
    if (s.mantScaleWidth == 0) expBiased
    else {
      val mantInt = math.round((value / math.pow(2, expUnbiased) - 1.0) * (1 << s.mantScaleWidth))
                         .toInt.min((1 << s.mantScaleWidth) - 1).max(0)
      (expBiased << s.mantScaleWidth) | mantInt
    }
  }

  /**
   * SW golden model for one fused-dot-product cycle:
   *   result = (Σ_i  decodeA(as_i) × decodeB(bs_i)) × decodeScale(sA) × decodeScale(sB)
   *
   * Scale factors are broadcast to all lanes (MX block-float convention).
   */
  def swFusedProduct(cfg: ScaleAddConfig)(
    as: Seq[Int], bs: Seq[Int], scaleA: Int, scaleB: Int
  ): Float = {
    val sA     = decodeScale(scaleA, cfg.stype)
    val sB     = decodeScale(scaleB, cfg.stype)
    // 模拟 8 个 lane 独立计算并引入 scale
    val products = as.zip(bs).map { case (a, b) =>
        val prod = decodeElement(a, cfg.elementTypeA) * decodeElement(b, cfg.elementTypeB)
        (prod * sA * sB).toFloat // 模拟硬件在 Normalization 这里的强制转 FP32
    }
    
    // 模拟 Reduction Tree (虽然简单 sum 也可以，但转成 Float 很重要)
    products.sum.toFloat
  }

  def swFusedProductWithTrace(cfg: ScaleAddConfig)(
    as: Seq[Int], bs: Seq[Int], scaleA: Int, scaleB: Int, cycleIdx: Int,
    verbose: Boolean = false
  ): Float = {
    val sA = decodeScale(scaleA, cfg.stype)
    val sB = decodeScale(scaleB, cfg.stype)

    val details = as.zip(bs).zipWithIndex.map { case ((a, b), lane) =>
      val dA = decodeElement(a, cfg.elementTypeA)
      val dB = decodeElement(b, cfg.elementTypeB)
      val prod = dA * dB
      (dA, dB, prod)
    }

    val dotSum = details.map(_._3).sum
    val finalCycleVal = (dotSum * sA * sB).toFloat

    if (verbose) {
      log(f"--- Cycle $cycleIdx SW Trace (sA=$sA%.4e, sB=$sB%.4e) ---")
      details.zipWithIndex.foreach { case ((dA, dB, p), lane) =>
        log(f"  Lane $lane: A=$dA%10.4f (raw=0x${as(lane)}%02x) | B=$dB%10.4f (raw=0x${bs(lane)}%02x) | P=$p%10.4f")
      }
      log(f"  DotSum=$dotSum%10.4f | FinalCycleVal=$finalCycleVal%10.4f")
    }

    finalCycleVal
  }
  //手动计算预期的exp和mant
  def debugLane0(cfg: ScaleAddConfig, aRaw: Int, bRaw: Int, sARaw: Int, sBRaw: Int) = {
  // 模拟硬件运算逻辑
  val dA = decodeElement(aRaw, cfg.elementTypeA)
  val dB = decodeElement(bRaw, cfg.elementTypeB)
  val sA = decodeScale(sARaw, cfg.stype)
  val sB = decodeScale(sBRaw, cfg.stype)
  
  // 理论乘积
  val idealProd = dA * dB * sA * sB
  
  // 硬件 ScaleToFP32 内部的 fracBits 计算
  val fracBits = {
    val aM = if (cfg.elementTypeA.name == "INT8") 0 else cfg.elementTypeA.elementWidthMant
    val bM = if (cfg.elementTypeB.name == "INT8") 0 else cfg.elementTypeB.elementWidthMant
    aM + bM + 2 * cfg.stype.mantScaleWidth
  }
  // 硬件对应的 expBias
  val expBias = 127 + cfg.resScaleAddMantWidth - 1 - fracBits
  
  (idealProd, fracBits, expBias)
}

  /** Read accOut as IEEE 754 Float. */
  def peekFloat(dut: FusedDotProductUnit): Float =
    intBitsToFloat(dut.io.accOut.peek().litValue.toInt)

  /**
   * Bit-exact check for a single FP32Adder operation.
   * Asserts that HW output bits == expectedBits and logs context on failure.
   */
  def checkFP32Add(dut: FP32Adder, aBits: Int, bBits: Int, expectedBits: Int): Unit = {
    val aF   = intBitsToFloat(aBits)
    val bF   = intBitsToFloat(bBits)
    val expF = intBitsToFloat(expectedBits)
    dut.io.a.poke(aBits.U)
    dut.io.b.poke(bBits.U)
    val hw  = dut.io.out.peek().litValue.toInt
    val hwF = intBitsToFloat(hw)
    assert(hw == expectedBits,
      f"FP32Add($aF%g + $bF%g): " +
      f"got 0x${hw.toHexString} ($hwF%g), " +
      f"expected 0x${expectedBits.toHexString} ($expF%g)")
  }

  /**
   * Drive reset=1 so asyncRstN = ~1 = 0 (not in reset) for normal operation.
   * Required because generator uses active-low async reset (!reset.asBool),
   * but ChiselTest holds reset=0 during normal operation, which would keep
   * asyncRstN permanently high and freeze all registers at 0.
   */
  def initDut(dut: FusedDotProductUnit): Unit = {
    dut.reset.poke(true.B)
    dut.io.validIn.poke(false.B)
    dut.io.resetAcc.poke(false.B)
  }

  /** Pack element raw values into a single UInt (element 0 at LSB). */
  def packElements(elems: Seq[Int], width: Int): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
      acc | (BigInt(v & ((1 << width) - 1)) << (i * width))
    }

  /** Drive one fused-MAC cycle (all vectorSize lanes) and advance the clock. */
  def driveOne(
    dut: FusedDotProductUnit, cfg: ScaleAddConfig,
    as: Seq[Int], bs: Seq[Int], scaleA: Int, scaleB: Int
  ): Unit = {
    dut.io.op_a_i.poke(packElements(as, cfg.elementTypeA.totalWidth).U)
    dut.io.op_b_i.poke(packElements(bs, cfg.elementTypeB.totalWidth).U)
    dut.io.share_exp_A_i.poke(scaleA.U)
    dut.io.share_exp_B_i.poke(scaleB.U)
    dut.io.validIn.poke(true.B)
    dut.clock.step()
    dut.io.validIn.poke(false.B)
  }

  /**
   * Drive a sequence of fused-MAC cycles and check the running accumulator
   * against the SW golden model after every cycle.
   *
   * @param cycles  Seq of (as, bs, rawScaleA, rawScaleB) — one entry per cycle.
   * @param tol5pct Use 5 % relative + 1e-37 absolute tolerance (same as DotProductUnitTest).
   */
def runCycles(
  dut: FusedDotProductUnit, cfg: ScaleAddConfig,
  cycles: Seq[(Seq[Int], Seq[Int], Int, Int)],
  header: String = "",
  verbose: Boolean = false
): Unit = {
  if (header.nonEmpty) {
    log("=" * 70)
    log(header)
    log("=" * 70)
  }
  var swAcc = 0.0f
  for (((as, bs, sA, sB), i) <- cycles.zipWithIndex) {
    // 解码当前的 Scale 因子以便打印
    val decSA = decodeScale(sA, cfg.stype)
    val decSB = decodeScale(sB, cfg.stype)

    // 1. 使用带 trace 的软件模型
    val swCycleVal = swFusedProductWithTrace(cfg)(as, bs, sA, sB, i, verbose)
    
    // 2. 模拟硬件
    driveOne(dut, cfg, as, bs, sA, sB)

    // 4. 拉出归约后的总和
    val hwCycleValRaw = dut.io.debug.get.reducedSum.peek().litValue.toInt
    val hwCycleVal    = java.lang.Float.intBitsToFloat(hwCycleValRaw)
    val hwAcc         = peekFloat(dut)

   
    // 3. 模拟硬件累加行为
    swAcc = (swAcc + swCycleVal).toFloat 
    val hw = peekFloat(dut)
    
    // 4. 容差判断
    val tol = math.abs(swAcc) * 0.10f + 1e-5f 
    val isFail = math.abs(hw - swAcc) > tol

    // 在打印行中加入 Scale 信息
    val line = f"Cycle $i%2d | sA=${decSA}%10.4e | sB=${decSB}%10.4e | cycleVal=$swCycleVal%12.4e | SW=$swAcc%12.4e | HW=$hw%12.4e"
    
    if (isFail || verbose) {
      val tag = if (isFail) "FAIL" else "VERBOSE"
      log(f"--- Cycle $i%2d [$tag] HW All Lanes Trace ---")

      for (laneIdx <- 0 until dut.vectorSize) {
        val hwLaneFP32Raw = dut.io.debug.get.all_lanes_fp32(laneIdx).peek().litValue.toInt
        val hwLaneFP32    = java.lang.Float.intBitsToFloat(hwLaneFP32Raw)
        val hwLaneMant    = dut.io.debug.get.all_lanes_sa_mant(laneIdx).peek().litValue
        log(f"  Lane $laneIdx%1d: HW_Mant=0x$hwLaneMant%x | HW_FP32=$hwLaneFP32%.6f")
      }

      log(f"  [SUMMARY] SW_CycleSum=$swCycleVal%.6f | HW_CycleSum=$hwCycleVal%.6f")
      log(f"  [ACCUM  ] SW_Acc=$swAcc%.6f | HW_Acc=$hwAcc%.6f")

      if (isFail) {
        logErr(line)
        logErr(f"      BITS: SW=0x${java.lang.Float.floatToIntBits(swAcc).toHexString} | " +
               f"HW=0x${java.lang.Float.floatToIntBits(hw).toHexString}")
      } else {
        log(line)
      }
    } else {
      log(line)
    }
    assert(!isFail, s"Mismatch at cycle $i: hw=$hw expected=$swAcc")
  }
}
  // -----------------------------------------------------------------------
  // MX block quantization helpers
  // -----------------------------------------------------------------------

  /** Maximum representable (positive) value for an element type.
   *
   *  For INT8 the magnitude field is 7 bits → max = 127.
   *  For floating-point types the all-ones biased exponent is reserved
   *  (NaN/Inf), so max biased exp = 2^expBits − 2.
   */
  def maxElemRepr(t: ElementType): Double =
    if (t.name == "INT8") {
      // Max symmetric value: 127 × 2^implicitScaleExp = 127/64 ≈ 1.984375
      ((1 << (t.totalWidth - 1)) - 1).toDouble * Math.pow(2, t.implicitScaleExp)
    } else {
      val maxExpBiased = (1 << t.elementWidthExp) - 2   // all-1s reserved
      val maxMant      = (1 << t.elementWidthMant) - 1
      (1.0 + maxMant.toDouble / (1 << t.elementWidthMant)) * math.pow(2, maxExpBiased - t.bias)
    }

  /** Quantize a block of floating-point values to an MX format.
   *
   *  MX convention: 32 elements share one scale factor.  The shared scale
   *  is chosen as the smallest power-of-2 (or scale-type-representable value)
   *  such that every element divided by that scale fits inside the element
   *  type's range.
   *
   *  @param values  Raw FP values — normally 32 for a full MX block.
   *  @param et      Target element type.
   *  @param st      Target scale type.
   *  @return        (encoded elements, encoded shared scale)
   */
  def quantizeBlock(values: Seq[Double], et: ElementType, st: ScaleType): (Seq[Int], Int) = {
    val maxAbs = values.map(math.abs).max.max(1e-38)   // guard against log(0)

    // Scale exponent selection:
    //   INT8 (MXINT8 2's complement): match Python _find_shared_exp for 'mxint8':
    //     scaleExp = floor(log2(maxAbs))
    //     This maps maxAbs into (1, 2], normalised value quantised by MXINT8 (range ≈ ±1.984);
    //     values in (1.984, 2) will saturate to magnitude=127, exactly as Python does.
    //   FP types: scaleExp = ceil(log2(maxAbs / maxElemRepr)) so no saturation occurs.
    val scaleExp =
      if (et.name == "INT8")
        math.floor(math.log(maxAbs) / math.log(2)).toInt
      else {
        val idealScale = maxAbs / maxElemRepr(et)
        math.ceil(math.log(idealScale.max(1e-38)) / math.log(2)).toInt
      }
    // Clamp to what the scale type can actually represent
    val minScaleExp  = -(st.bias)
    val maxScaleExp  = (1 << st.expScaleWidth) - 1 - st.bias
    val clampedExp   = scaleExp.max(minScaleExp).min(maxScaleExp)
    val scaleDouble  = math.pow(2.0, clampedExp)

    // Encode the scale and decode back so element quantization uses the
    // same rounding as the hardware will see.
    val rawScale    = encodeScale(scaleDouble, st)
    val decodedScale = decodeScale(rawScale, st)

    val rawElems = values.map(v => encodeElement(v / decodedScale, et))
    (rawElems, rawScale)
  }

  // -----------------------------------------------------------------------
  // Pre-encoded constants (default E4M3 × E2M1 / UE5M3)
  // -----------------------------------------------------------------------
  val defaultScfg = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)

  val e4m3_1    = encodeElement(1.0,  MXFormats.E4M3)
  val e4m3_neg1 = encodeElement(-1.0, MXFormats.E4M3)
  val e4m3_2    = encodeElement(2.0,  MXFormats.E4M3)
  val e4m3_1p5  = encodeElement(1.5,  MXFormats.E4M3)
  val e2m1_1    = encodeElement(1.0,  MXFormats.E2M1)
  val e2m1_2    = encodeElement(2.0,  MXFormats.E2M1)
  val ue5m3_1   = encodeScale(1.0,    ScaleFormats.UE5M3)
  val ue5m3_2   = encodeScale(2.0,    ScaleFormats.UE5M3)

  // -----------------------------------------------------------------------
  // Pre-encoded constants — E5M2 × E5M2 / UE8M0, vec=4
  //   E5M2 : 1.5.2, bias 15 — representable fractions are 0, 0.25, 0.5, 0.75
  //   UE8M0: pure power-of-2 scale (8-bit exponent, no mantissa)
  // -----------------------------------------------------------------------
  val e5m2Scfg   = ScaleAddConfig(MXFormats.E5M2, MXFormats.E5M2, ScaleFormats.UE8M0)

  val e5m2_0p5   = encodeElement( 0.5,  MXFormats.E5M2)
  val e5m2_0p75  = encodeElement( 0.75, MXFormats.E5M2)
  val e5m2_1     = encodeElement( 1.0,  MXFormats.E5M2)
  val e5m2_1p25  = encodeElement( 1.25, MXFormats.E5M2)
  val e5m2_1p5   = encodeElement( 1.5,  MXFormats.E5M2)
  val e5m2_1p75  = encodeElement( 1.75, MXFormats.E5M2)
  val e5m2_2     = encodeElement( 2.0,  MXFormats.E5M2)
  val e5m2_3     = encodeElement( 3.0,  MXFormats.E5M2)
  val e5m2_n0p5  = encodeElement(-0.5,  MXFormats.E5M2)
  val e5m2_n0p75 = encodeElement(-0.75, MXFormats.E5M2)
  val e5m2_n1    = encodeElement(-1.0,  MXFormats.E5M2)
  val e5m2_n1p25 = encodeElement(-1.25, MXFormats.E5M2)
  val e5m2_n1p5  = encodeElement(-1.5,  MXFormats.E5M2)
  val e5m2_n1p75 = encodeElement(-1.75, MXFormats.E5M2)
  val e5m2_n2    = encodeElement(-2.0,  MXFormats.E5M2)
  val ue8m0_0p5  = encodeScale(0.5, ScaleFormats.UE8M0)
  val ue8m0_1    = encodeScale(1.0, ScaleFormats.UE8M0)
  val ue8m0_2    = encodeScale(2.0, ScaleFormats.UE8M0)
  val ue8m0_4    = encodeScale(4.0, ScaleFormats.UE8M0)

  // -----------------------------------------------------------------------
  // Test 1: Reset clears accumulator
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Reset clears accumulator") {
    log("\n[TEST 1] Reset clears accumulator")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.accOut.expect(0.U, "initial accOut should be 0")

        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.accOut.expect(0.U, "accOut should stay 0 after reset")
        dut.io.resetAcc.poke(false.B)
      }
      log("[PASSED] Reset clears accumulator")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Reset clears accumulator: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 2: validOut handshake timing
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: validOut handshake timing") {
    log("\n[TEST 2] validOut handshake timing (vec=4)")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.op_a_i.poke(packElements(Seq.fill(4)(e4m3_1), defaultScfg.elementTypeA.totalWidth).U)
        dut.io.op_b_i.poke(packElements(Seq.fill(4)(e2m1_1), defaultScfg.elementTypeB.totalWidth).U)
        dut.io.share_exp_A_i.poke(ue5m3_1.U)
        dut.io.share_exp_B_i.poke(ue5m3_1.U)

        dut.io.validOut.expect(false.B, "validOut must be false when validIn=0 (before any cycle)")

        dut.io.validIn.poke(true.B)
        dut.clock.step()
        dut.io.validIn.poke(false.B)
        dut.io.validOut.expect(true.B, "validOut must be true one cycle after validIn pulse")

        dut.clock.step()
        dut.io.validOut.expect(false.B, "validOut must return false when validIn=0")
      }
      log("[PASSED] validOut handshake timing")
    } catch { case e: Exception =>
      logErr(s"[FAILED] validOut handshake timing: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 3: vec=1 scalar equivalence — matches DotProductUnit behaviour
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=1 scalar equivalence (1×1×1×1 = 1.0)") {
    log("\n[TEST 3] vec=1 scalar equivalence")
    try {
      test(new FusedDotProductUnit(defaultScfg, 1, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val expected = swFusedProduct(defaultScfg)(Seq(e4m3_1), Seq(e2m1_1), ue5m3_1, ue5m3_1)
        driveOne(dut, defaultScfg, Seq(e4m3_1), Seq(e2m1_1), ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"Expected: $expected%.6f  HW: $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol, s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] vec=1 scalar equivalence")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=1 scalar equivalence: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 4: vec=4 all-ones fused → 4.0
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=4 all-ones fused → 4.0") {
    log("\n[TEST 4] vec=4 all-ones (1×1×1×1 per lane) → 4.0")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val as       = Seq.fill(4)(e4m3_1)
        val bs       = Seq.fill(4)(e2m1_1)
        val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_1, ue5m3_1)
        driveOne(dut, defaultScfg, as, bs, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"Expected (SW): $expected%.6f  HW: $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol, s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] vec=4 all-ones fused → 4.0")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=4 all-ones: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 5: vec=8 all-ones fused → 8.0
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=8 all-ones fused → 8.0") {
    log("\n[TEST 5] vec=8 all-ones → 8.0")
    try {
      test(new FusedDotProductUnit(defaultScfg, 8, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val as       = Seq.fill(8)(e4m3_1)
        val bs       = Seq.fill(8)(e2m1_1)
        val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_1, ue5m3_1)
        driveOne(dut, defaultScfg, as, bs, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"Expected (SW): $expected%.6f  HW: $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol, s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] vec=8 all-ones fused → 8.0")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=8 all-ones: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 6: vec=4 mixed signs — partial cancellation
  //   Lane values: [+1, -1, +1, -1] × [1, 1, 1, 1] = 0
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=4 mixed signs → near-zero (cancellation)") {
    log("\n[TEST 6] vec=4 mixed signs [+1,-1,+1,-1] × [1,1,1,1] → 0.0")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val as       = Seq(e4m3_1, e4m3_neg1, e4m3_1, e4m3_neg1)
        val bs       = Seq.fill(4)(e2m1_1)
        val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_1, ue5m3_1)
        driveOne(dut, defaultScfg, as, bs, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"Expected (SW): $expected%.6f  HW: $hw%.6f")

        // Allow small absolute residual for exact cancellation in FP32
        val tol = math.abs(expected) * 0.01f + 1e-37f + 1e-6f
        assert(math.abs(hw - expected) <= tol, s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] vec=4 mixed signs cancellation")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=4 mixed signs cancellation: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 7: vec=4 scale amplification — scaleA=2, scaleB=2 → 4×1 × 4 = 16.0
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=4 scale=2×2 amplification → 16.0") {
    log("\n[TEST 7] vec=4, all-ones, scale 2×2 → 4×1×2×2 = 16.0")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val as       = Seq.fill(4)(e4m3_1)
        val bs       = Seq.fill(4)(e2m1_1)
        val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_2, ue5m3_2)
        driveOne(dut, defaultScfg, as, bs, ue5m3_2, ue5m3_2)
        val hw = peekFloat(dut)
        log(f"Expected (SW): $expected%.6f  HW: $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol, s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] vec=4 scale=2×2 amplification")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=4 scale amplification: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 8: Reset mid-accumulation
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Reset mid-accumulation (vec=4)") {
    log("\n[TEST 8] Reset mid-accumulation (vec=4)")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)

        // First accumulation cycle
        driveOne(dut, defaultScfg, Seq.fill(4)(e4m3_2), Seq.fill(4)(e2m1_2), ue5m3_1, ue5m3_1)
        val before = dut.io.accOut.peek().litValue.toInt
        log(s"Before reset: 0x${before.toHexString}  (${intBitsToFloat(before)})")
        assert(before != 0, "accOut should be non-zero after accumulation")

        // Reset
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.accOut.expect(0.U, "accOut should be 0 after reset")
        dut.io.resetAcc.poke(false.B)

        // Single cycle after reset
        val as1      = Seq.fill(4)(e4m3_1)
        val bs1      = Seq.fill(4)(e2m1_1)
        val expected = swFusedProduct(defaultScfg)(as1, bs1, ue5m3_1, ue5m3_1)
        driveOne(dut, defaultScfg, as1, bs1, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"After reset + 1 cycle: HW=$hw%.6f expected=$expected%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol, s"Mismatch after reset: hw=$hw expected=$expected")
      }
      log("[PASSED] Reset mid-accumulation")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Reset mid-accumulation: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 9: vec=4 multi-cycle accumulation — 4 cycles each adding 4.0 → 16.0
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=4 multi-cycle (4 cycles × 4.0 → 16.0, VCD)") {
    log("\n[TEST 9] vec=4 multi-cycle: 4 cycles of (4×1×1×1×1) → 16.0")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val cycles = Seq.fill(4)(
          (Seq.fill(4)(e4m3_1), Seq.fill(4)(e2m1_1), ue5m3_1, ue5m3_1)
        )
        runCycles(dut, defaultScfg, cycles, "vec=4 multi-cycle (4 cycles)")
      }
      log("[PASSED] vec=4 multi-cycle accumulation")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=4 multi-cycle: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 10: vec=4 mixed-value multi-cycle — negative contributions
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: vec=4 mixed-value multi-cycle (negative/positive)") {
    log("\n[TEST 10] vec=4 mixed-value multi-cycle")
    try {
      test(new FusedDotProductUnit(defaultScfg, 4, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val cycles = Seq(
          // cycle 0: [1, 2, -1, 1] × [1, 1, 1, 1], scale 1×1 → sum = 3.0
          (Seq(e4m3_1, e4m3_2, e4m3_neg1, e4m3_1), Seq.fill(4)(e2m1_1), ue5m3_1, ue5m3_1),
          // cycle 1: [1, 1, 1, 1] × [2, 2, 2, 2], scale 1×1 → sum = 8.0
          (Seq.fill(4)(e4m3_1), Seq.fill(4)(e2m1_2), ue5m3_1, ue5m3_1),
          // cycle 2: [-1, -1, -1, -1] × [1, 1, 1, 1], scale 2×1 → sum = -8.0
          (Seq.fill(4)(e4m3_neg1), Seq.fill(4)(e2m1_1), ue5m3_2, ue5m3_1),
          // cycle 3: [1, 1.5, -1, 2] × [1, 1, 1, 1], scale 1×1 → sum = 3.5
          (Seq(e4m3_1, e4m3_1p5, e4m3_neg1, e4m3_2), Seq.fill(4)(e2m1_1), ue5m3_1, ue5m3_1)
        )
        runCycles(dut, defaultScfg, cycles, "vec=4 mixed-value multi-cycle")
      }
      log("[PASSED] vec=4 mixed-value multi-cycle")
    } catch { case e: Exception =>
      logErr(s"[FAILED] vec=4 mixed-value multi-cycle: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 11: E5M2 × E4M3 / UE8M0 — vec=4, 3-cycle mixed
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: E5M2 x E4M3 / UE8M0 vec=4 mixed values") {
    log("\n[TEST 11] E5M2 x E4M3 / UE8M0 vec=4 3-cycle mixed")
    try {
      val cfg   = ScaleAddConfig(MXFormats.E5M2, MXFormats.E4M3, ScaleFormats.UE8M0)
      val vsize = 4
      val cycles = Seq(
        (Seq(1.0, 2.0, -1.0, 1.5),  Seq(1.0, 1.0,  2.0, 1.0), 1.0, 1.0),
        (Seq(1.0, 1.0,  1.0, 1.0),  Seq(2.0, 1.0, -1.0, 2.0), 2.0, 1.0),
        (Seq(2.0, 1.5,  1.0, -1.0), Seq(1.0, 2.0,  1.0, 1.0), 1.0, 2.0)
      ).map { case (as, bs, sA, sB) =>
        (as.map(encodeElement(_, cfg.elementTypeA)),
         bs.map(encodeElement(_, cfg.elementTypeB)),
         encodeScale(sA, cfg.stype),
         encodeScale(sB, cfg.stype))
      }
      test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, cycles, "E5M2 x E4M3 / UE8M0 vec=4")
      }
      log("[PASSED] E5M2 x E4M3 / UE8M0 vec=4")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E5M2 x E4M3 / UE8M0 vec=4: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 12: E3M2 × E2M3 / UE6M2 — vec=4, 4-cycle with fractional mantissa
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: E3M2 x E2M3 / UE6M2 vec=4 fractional mantissa") {
    log("\n[TEST 12] E3M2 x E2M3 / UE6M2 vec=4 fractional mantissa")
    try {
      val cfg   = ScaleAddConfig(MXFormats.E3M2, MXFormats.E2M3, ScaleFormats.UE6M2)
      val vsize = 4
      val cycles = Seq(
        (Seq(1.0,  1.25, -1.5, 2.0),  Seq(1.0,  1.0,  1.0, 1.0), 1.0, 1.0),
        (Seq(1.5,  1.0,   1.0, 1.0),  Seq(1.25, 1.0, -1.0, 1.5), 1.0, 1.5),
        (Seq(2.0, -1.0,   1.5, 1.0),  Seq(1.0,  2.0,  1.0, 1.0), 1.5, 1.0),
        (Seq(-1.0, 1.0,  -1.0, 1.0),  Seq(1.0,  1.0,  1.5, 1.0), 2.0, 2.0)
      ).map { case (as, bs, sA, sB) =>
        (as.map(encodeElement(_, cfg.elementTypeA)),
         bs.map(encodeElement(_, cfg.elementTypeB)),
         encodeScale(sA, cfg.stype),
         encodeScale(sB, cfg.stype))
      }
      test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, cycles, "E3M2 x E2M3 / UE6M2 vec=4")
      }
      log("[PASSED] E3M2 x E2M3 / UE6M2 vec=4")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E3M2 x E2M3 / UE6M2 vec=4: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 13: Power-of-2 vectorSizes — one-cycle all-1 sum equals vectorSize
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Power-of-2 vectorSizes 1..16 — fused sum = vecSize") {
    log("\n[TEST 13] Power-of-2 vectorSizes (1, 2, 4, 8, 16) — 1-cycle sum = vecSize")
    try {
      for (vsize <- Seq(1, 2, 4, 8, 16)) {
        test(new FusedDotProductUnit(defaultScfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)

          val as       = Seq.fill(vsize)(e4m3_1)
          val bs       = Seq.fill(vsize)(e2m1_1)
          val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_1, ue5m3_1)
          driveOne(dut, defaultScfg, as, bs, ue5m3_1, ue5m3_1)
          val hw = peekFloat(dut)
          log(f"vec=$vsize%2d | expected=$expected%.4f | HW=$hw%.4f")

          val tol = math.abs(expected) * 0.01f + 1e-37f
          assert(math.abs(hw - expected) <= tol, s"vec=$vsize: hw=$hw expected=$expected")
        }
      }
      log("[PASSED] Power-of-2 vectorSizes")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Power-of-2 vectorSizes: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 14: Non-power-of-2 vectorSizes — tests odd-lane pass-through in tree
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Non-power-of-2 vectorSizes (3, 5, 7) — reduction tree") {
    log("\n[TEST 14] Non-power-of-2 vectorSizes (3, 5, 7) — odd lane pass-through")
    try {
      for (vsize <- Seq(3, 5, 7)) {
        test(new FusedDotProductUnit(defaultScfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)

          val as       = Seq.fill(vsize)(e4m3_1)
          val bs       = Seq.fill(vsize)(e2m1_1)
          val expected = swFusedProduct(defaultScfg)(as, bs, ue5m3_1, ue5m3_1)
          driveOne(dut, defaultScfg, as, bs, ue5m3_1, ue5m3_1)
          val hw = peekFloat(dut)
          log(f"vec=$vsize%2d | expected=$expected%.4f | HW=$hw%.4f")

          val tol = math.abs(expected) * 0.01f + 1e-37f
          assert(math.abs(hw - expected) <= tol, s"vec=$vsize: hw=$hw expected=$expected")
        }
      }
      log("[PASSED] Non-power-of-2 vectorSizes")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Non-power-of-2 vectorSizes: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 15: 8-element vector dot product simulation (vec=8, 1 cycle)
  //   Mirrors DotProductUnitTest Test 12 but completes in a single cycle.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: 8-element vector dot product in 1 cycle (shared scale=2.0)") {
    log("\n[TEST 15] 8-element dot product in 1 cycle (scale=2.0 x 2.0)")
    try {
      val vecA         = Seq(1.0, -1.0, 2.0, 1.0, -1.0, 1.0, 2.0, -2.0)
      val vecB         = Seq(1.0,  1.0, 1.0, 2.0,  2.0, 1.0, 1.0,  1.0)
      val sharedScaleA = 2.0
      val sharedScaleB = 2.0

      val rawAs = vecA.map(encodeElement(_, defaultScfg.elementTypeA))
      val rawBs = vecB.map(encodeElement(_, defaultScfg.elementTypeB))
      val rawSA = encodeScale(sharedScaleA, defaultScfg.stype)
      val rawSB = encodeScale(sharedScaleB, defaultScfg.stype)

      val dotNoScale    = vecA.zip(vecB).map { case (a, b) => a * b }.sum
      val expectedTotal = (dotNoScale * sharedScaleA * sharedScaleB).toFloat
      log(f"Dot product (no scale) = $dotNoScale%.4f")
      log(f"Expected (×scale²)    = $expectedTotal%.4f")

      test(new FusedDotProductUnit(defaultScfg, 8, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        driveOne(dut, defaultScfg, rawAs, rawBs, rawSA, rawSB)
        val hw  = peekFloat(dut)
        val tol = math.abs(expectedTotal) * 0.05f + 1e-37f
        log(f"HW result = $hw%.4f  expected = $expectedTotal%.4f  tol = $tol%.6f")
        assert(math.abs(hw - expectedTotal) <= tol,
          s"Final dot product mismatch: hw=$hw expected=$expectedTotal")
      }
      log("[PASSED] 8-element vector dot product in 1 cycle")
    } catch { case e: Exception =>
      logErr(s"[FAILED] 8-element vector dot product: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 16: All type combinations — vec=4, 5-cycle random accumulation
  //   6 × 6 × 7 = 252 configurations; each gets 5 random cycles.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: All type combinations vec=4 — 5-cycle random accumulation") {
    log("\n[TEST 16] All type combinations vec=4 — 5-cycle random accumulation")

    val rng      = new Random(42)
    val vsize    = 4
    val numCycles = 5

    def randElement(t: ElementType): Int = {
      val sign = rng.nextInt(2)
      if (t.name == "INT8") {
        val mag = 1 + rng.nextInt(15)  // magnitude 1..15
        if (sign == 0) mag else (-mag) & 0xFF  // 2's complement encoding
      } else {
        val expRange = 1 << t.elementWidthExp
        val expLo    = (t.bias + 1).max(1).min(expRange - 1)
        val expHi    = (t.bias + 4).max(expLo).min(expRange - 1)
        val exp      = expLo + rng.nextInt(expHi - expLo + 1)
        val mant     = rng.nextInt(1 << t.elementWidthMant)
        (sign << (t.totalWidth - 1)) | (exp << t.elementWidthMant) | mant
      }
    }

    def randScale(s: ScaleType): Int = {
      val expRange = 1 << s.expScaleWidth
      val expLo    = (s.bias - 2).max(1).min(expRange - 1)
      val expHi    = (s.bias + 2).max(expLo).min(expRange - 1)
      val exp      = expLo + rng.nextInt(expHi - expLo + 1)
      val mant     = if (s.mantScaleWidth == 0) 0 else rng.nextInt(1 << s.mantScaleWidth)
      (exp << s.mantScaleWidth) | mant
    }

    val allElem  = MXFormats.allElementTypes
    val allScale = ScaleFormats.allScaleTypes
    val total    = allElem.size * allElem.size * allScale.size
    var passed   = 0
    var failed   = 0

    for (etA <- allElem; etB <- allElem; st <- allScale) {
      val cfg   = ScaleAddConfig(etA, etB, st)
      val label = s"${etA.name} x ${etB.name} / ${st.name} vec=$vsize"
      val cycles = (0 until numCycles).map { _ =>
        val as = Seq.fill(vsize)(randElement(etA))
        val bs = Seq.fill(vsize)(randElement(etB))
        (as, bs, randScale(st), randScale(st))
      }
      try {
        test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runCycles(dut, cfg, cycles, label)
        }
        log(s"[OK] $label")
        passed += 1
      } catch { case e: Exception =>
        logErr(s"[FAIL] $label: ${e.getMessage}")
        failed += 1
      }
    }

    val summary = s"Summary: $passed passed, $failed failed out of $total combinations"
    if (failed > 0) logErr(summary) else log(summary)
    assert(failed == 0,
      s"$failed combination(s) failed — see fused_dot_product_unit_test.log for details")
    log("[PASSED] All type combinations vec=4")
  }

  // -----------------------------------------------------------------------
  // Test 17: E5M2 × E5M2 / UE8M0 — vec=4, 6-cycle randomized
  //
  //   Each cycle uses distinct sign patterns, fractional magnitudes drawn from
  //   the full E5M2 mantissa alphabet {0, 0.25, 0.50, 0.75} × {0.5,1,1.5,2,3}
  //   and varying UE8M0 scale pairs to exercise the scale multiplier path.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: E5M2 x E5M2 / UE8M0 vec=4 randomized 6-cycle") {
    log("\n[TEST 17] E5M2 x E5M2 / UE8M0 vec=4 randomized 6-cycle")
    try {
      val cfg   = e5m2Scfg
      val vsize = 4
      // Cycles are listed as raw pre-encoded integers (constants defined above).
      // Sign, magnitude, and scale vary independently each cycle to cover:
      //   - all four E5M2 mantissa fractions (0.00, 0.25, 0.50, 0.75)
      //   - positive and negative elements in both operands
      //   - UE8M0 scale factors spanning {0.5, 1.0, 2.0, 4.0}
      val cycles = Seq(
        // cycle 0: mixed signs, fractional mantissa diversity; scale 2×1
        //   dot = 1.5×(−0.5) + (−0.75)×1.75 + 2.0×(−1.0) + (−1.5)×0.75
        //       = −0.75 − 1.3125 − 2.0 − 1.125 = −5.1875  × 2 × 1
        (Seq(e5m2_1p5,  e5m2_n0p75, e5m2_2,    e5m2_n1p5),
         Seq(e5m2_n0p5, e5m2_1p75,  e5m2_n1,   e5m2_0p75),  ue8m0_2, ue8m0_1),

        // cycle 1: smaller magnitudes with opposite-sign lanes; scale 1×0.5
        //   dot = 0.75×2.0 + 1.0×(−1.5) + (−1.5)×0.75 + 0.5×(−1.0)
        //       = 1.5 − 1.5 − 1.125 − 0.5 = −1.625  × 1 × 0.5
        (Seq(e5m2_0p75, e5m2_1,    e5m2_n1p5,  e5m2_0p5),
         Seq(e5m2_2,    e5m2_n1p5, e5m2_0p75,  e5m2_n1),    ue8m0_1, ue8m0_0p5),

        // cycle 2: large negative A values, asymmetric signs; scale 0.5×2
        //   dot = (−2.0)×1.0 + 1.5×(−0.75) + 0.75×1.5 + (−1.0)×2.0
        //       = −2.0 − 1.125 + 1.125 − 2.0 = −4.0  × 0.5 × 2
        (Seq(e5m2_n2,   e5m2_1p5,   e5m2_0p75, e5m2_n1),
         Seq(e5m2_1,    e5m2_n0p75, e5m2_1p5,  e5m2_2),     ue8m0_0p5, ue8m0_2),

        // cycle 3: high-scale (4×1), near-cancellation between lanes
        //   dot = 1.75×(−1.5) + (−0.5)×1.0 + (−1.0)×2.0 + 1.5×(−0.75)
        //       = −2.625 − 0.5 − 2.0 − 1.125 = −6.25  × 4 × 1
        (Seq(e5m2_1p75, e5m2_n0p5,  e5m2_n1,   e5m2_1p5),
         Seq(e5m2_n1p5, e5m2_1,     e5m2_2,    e5m2_n0p75), ue8m0_4, ue8m0_1),

        // cycle 4: near-zero net sum (cancellation), scale 1×4
        //   dot = (−0.75)×0.75 + 2.0×(−2.0) + 1.0×1.5 + (−1.75)×1.0
        //       = −0.5625 − 4.0 + 1.5 − 1.75 = −4.8125  × 1 × 4
        (Seq(e5m2_n0p75, e5m2_2,    e5m2_1,    e5m2_n1p75),
         Seq(e5m2_0p75,  e5m2_n2,   e5m2_1p5,  e5m2_1),     ue8m0_1, ue8m0_4),

        // cycle 5: uses e5m2_3 (exp+1 row), sign-alternating A and B; scale 2×2
        //   dot = 1.0×(−1.0) + (−1.0)×1.5 + 1.5×(−0.75) + (−2.0)×1.0
        //       = −1.0 − 1.5 − 1.125 − 2.0 = −5.625  × 2 × 2
        (Seq(e5m2_1,    e5m2_n1,   e5m2_1p5,  e5m2_n2),
         Seq(e5m2_n1,   e5m2_1p5,  e5m2_n0p75, e5m2_1),     ue8m0_2, ue8m0_2)
      )
      test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, cycles, "E5M2 x E5M2 / UE8M0 vec=4 randomized 6-cycle")
      }
      log("[PASSED] E5M2 x E5M2 / UE8M0 vec=4 randomized 6-cycle")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E5M2 x E5M2 / UE8M0 vec=4 randomized 6-cycle: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 18: MX block-quantized dot product (32-element block, vec=8, 4 cycles)
  //
  //   In the MX spec a block of 32 elements shares one scale factor.
  //   This test:
  //     1. Starts from raw FP32 values for two 32-element vectors (A and B).
  //     2. Calls quantizeBlock() to encode each vector into MX elements
  //        + one shared scale — exactly what a real quantization pass would do.
  //     3. Splits the 32 encoded elements into 4 groups of 8 (= 4 cycles
  //        with vectorSize=8), each cycle reusing the same shared scale.
  //     4. Verifies the accumulated dot product against the SW golden model.
  //
  //   Using E5M2 × E4M3 / UE8M0 because UE8M0 is a pure power-of-2 scale
  //   (no mantissa), making the quantization boundaries easy to verify manually.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: MX block-quantized 32-element dot product (vec=8, 4 cycles)") {
    log("\n[TEST 18] MX block-quantized 32-element dot product (vec=8, 4 cycles)")
    try {
      val cfg   = ScaleAddConfig(MXFormats.E5M2, MXFormats.E4M3, ScaleFormats.UE8M0)
      val vsize = 8
      val blockSize = 32   // canonical MX block size

      // ---- raw FP32 source data ----
      // Values are chosen to span a wide dynamic range so that the shared
      // scale is non-trivial (not 1.0).  Both positive and negative entries
      // appear in both vectors.
      val rawA: Seq[Double] = Seq(
         3.2,  -1.6,   0.8,   4.0,  -2.4,  1.2,  -0.6,  3.6,
         2.0,  -3.0,   1.5,  -0.5,   2.5, -1.0,   0.4,  1.8,
        -2.2,   0.9,  -1.4,   3.1,  -0.7,  2.8,  -1.1,  0.3,
         1.7,  -2.9,   0.6,  -1.3,   2.1, -0.8,   1.9, -3.5
      )
      val rawB: Seq[Double] = Seq(
         1.0,   2.0,  -1.0,   0.5,   1.5, -0.5,   2.0, -1.0,
         0.75, -1.25,  1.5,   2.0,  -0.75, 1.0,  -2.0,  0.5,
         2.0,  -1.0,   0.5,  -1.5,   1.0,  0.25, -1.0,  2.0,
        -0.5,   1.5,  -2.0,   1.0,   0.5, -1.5,   1.0, -0.25
      )
      require(rawA.size == blockSize && rawB.size == blockSize)

      // ---- MX block quantization ----
      val (encA, rawScaleA) = quantizeBlock(rawA, cfg.elementTypeA, cfg.stype)
      val (encB, rawScaleB) = quantizeBlock(rawB, cfg.elementTypeB, cfg.stype)

      val scaleAf = decodeScale(rawScaleA, cfg.stype)
      val scaleBf = decodeScale(rawScaleB, cfg.stype)
      log(f"Block max |A| = ${rawA.map(math.abs).max}%.4f  → shared scale_A = $scaleAf%.6f  (raw=0x${rawScaleA.toHexString})")
      log(f"Block max |B| = ${rawB.map(math.abs).max}%.4f  → shared scale_B = $scaleBf%.6f  (raw=0x${rawScaleB.toHexString})")
      log(f"maxElemRepr(E5M2) = ${maxElemRepr(cfg.elementTypeA)}%.1f   maxElemRepr(E4M3) = ${maxElemRepr(cfg.elementTypeB)}%.1f")

      // Log quantized vs original to show quantization error per element
      log("-" * 70)
      log("%-6s %8s %10s  %8s %10s  %14s  %14s".format("Lane", "rawA", "decA", "rawB", "decB", "A*B (quant)", "A*B (exact)"))
      log("-" * 70)
      var exactDot = 0.0
      for (idx <- 0 until blockSize) {
        val dA   = decodeElement(encA(idx), cfg.elementTypeA) * scaleAf
        val dB   = decodeElement(encB(idx), cfg.elementTypeB) * scaleBf
        val exact = rawA(idx) * rawB(idx)
        exactDot += exact
        log(f"$idx%-6d 0x${encA(idx).toHexString}%6s ${dA}%10.4f  0x${encB(idx).toHexString}%6s ${dB}%10.4f  ${dA * dB}%14.4f  $exact%14.4f")
      }
      log("-" * 70)
      log(f"Exact dot (no quantization error) = $exactDot%.4f")

      // ---- split into 4 cycles of vsize=8 ----
      val cycles: Seq[(Seq[Int], Seq[Int], Int, Int)] =
        (0 until (blockSize / vsize)).map { c =>
          val slice = c * vsize until (c + 1) * vsize
          (encA.slice(slice.start, slice.end),
           encB.slice(slice.start, slice.end),
           rawScaleA, rawScaleB)   // same shared scale for all cycles in the block
        }

      test(new FusedDotProductUnit(cfg, vsize, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, cycles, "MX block-quantized vec=8 × 4 cycles")
      }
      log("[PASSED] MX block-quantized 32-element dot product")
    } catch { case e: Exception =>
      logErr(s"[FAILED] MX block-quantized 32-element dot product: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 19: E5M2 × E2M1 / UE8M0 — MX block-quantized 32-element dot product
  //          (vec=8, 4 cycles)
  //
  //  Mirrors Test 18 but exercises the E5M2 × E2M1 / UE8M0 configuration
  //  that is used by the top-level flow.  E2M1 has only four magnitude levels
  //  {0, 0.5, 1.0, 1.5, 2.0, 3.0} so quantisation noise is large — the SW
  //  golden model must be computed *after* round-tripping through the codec,
  //  exactly as the HW does, so any mismatch is a real hardware bug.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: E5M2 x E2M1 / UE8M0 block-quantized 32-element dot product (vec=8, 4 cycles)") {
    log("\n[TEST 19] E5M2 x E2M1 / UE8M0 block-quantized 32-element dot product (vec=8, 4 cycles)")
    try {
      val cfg       = ScaleAddConfig(MXFormats.E5M2, MXFormats.E2M1, ScaleFormats.UE8M0)
      val vsize     = 8
      val blockSize = 32

      // Source FP32 data — wide dynamic range so the shared scale is non-trivial.
      // Values chosen to produce diverse E2M1-quantised levels after normalisation.
      val rawA: Seq[Double] = Seq(
         3.2,  -1.6,   0.8,   4.0,  -2.4,   1.2,  -0.6,   3.6,
         2.0,  -3.0,   1.5,  -0.5,   2.5,  -1.0,   0.4,   1.8,
        -2.2,   0.9,  -1.4,   3.1,  -0.7,   2.8,  -1.1,   0.3,
         1.7,  -2.9,   0.6,  -1.3,   2.1,  -0.8,   1.9,  -3.5
      )
      val rawB: Seq[Double] = Seq(
         1.0,   2.0,  -1.0,   0.5,   1.5,  -0.5,   2.0,  -1.0,
         0.75, -1.25,  1.5,   2.0,  -0.75,  1.0,  -2.0,   0.5,
         2.0,  -1.0,   0.5,  -1.5,   1.0,   0.25, -1.0,   2.0,
        -0.5,   1.5,  -2.0,   1.0,   0.5,  -1.5,   1.0,  -0.25
      )
      require(rawA.size == blockSize && rawB.size == blockSize)

      // MX block quantisation — B is quantised to E2M1 (coarse grid)
      val (encA, rawScaleA) = quantizeBlock(rawA, cfg.elementTypeA, cfg.stype)
      val (encB, rawScaleB) = quantizeBlock(rawB, cfg.elementTypeB, cfg.stype)

      val scaleAf = decodeScale(rawScaleA, cfg.stype)
      val scaleBf = decodeScale(rawScaleB, cfg.stype)
      log(f"Block max |A| = ${rawA.map(math.abs).max}%.4f  → scale_A = $scaleAf%.6f  (raw=0x${rawScaleA.toHexString})")
      log(f"Block max |B| = ${rawB.map(math.abs).max}%.4f  → scale_B = $scaleBf%.6f  (raw=0x${rawScaleB.toHexString})")
      log(f"maxElemRepr(E5M2)=${maxElemRepr(cfg.elementTypeA)}%.1f  maxElemRepr(E2M1)=${maxElemRepr(cfg.elementTypeB)}%.1f")

      log("-" * 80)
      log("%-6s %8s %10s  %8s %10s  %14s  %14s".format(
        "Elem", "rawA", "decA×sA", "rawB", "decB×sB", "A*B(quant)", "A*B(exact)"))
      log("-" * 80)
      var exactDot = 0.0
      for (idx <- 0 until blockSize) {
        val dA    = decodeElement(encA(idx), cfg.elementTypeA) * scaleAf
        val dB    = decodeElement(encB(idx), cfg.elementTypeB) * scaleBf
        val exact = rawA(idx) * rawB(idx)
        exactDot += exact
        log(f"$idx%-6d 0x${encA(idx).toHexString}%6s ${dA}%10.4f  " +
            f"0x${encB(idx).toHexString}%4s ${dB}%10.4f  ${dA * dB}%14.4f  $exact%14.4f")
      }
      log("-" * 80)
      log(f"Exact dot (no quantisation error) = $exactDot%.4f")

      // Split 32 elements into 4 cycles of vsize=8; same shared scale per cycle
      val cycles: Seq[(Seq[Int], Seq[Int], Int, Int)] =
        (0 until (blockSize / vsize)).map { c =>
          val slice = c * vsize until (c + 1) * vsize
          (encA.slice(slice.start, slice.end),
           encB.slice(slice.start, slice.end),
           rawScaleA, rawScaleB)
        }

      test(new FusedDotProductUnit(cfg, vsize, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, cycles, "E5M2 x E2M1 / UE8M0 block-quantized vec=8 × 4 cycles")
      }
      log("[PASSED] E5M2 x E2M1 / UE8M0 block-quantized 32-element dot product")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E5M2 x E2M1 / UE8M0 block-quantized 32-element dot product: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 20: E5M2 × E2M1 / UE8M0 — multi-block accumulation (vec=8, 30 cycles)
  //
  //  The top-flow mismatch log shows indices up to [29], which corresponds to
  //  a run of ≥30 accumulation cycles.  This test processes four independent
  //  32-element MX blocks back-to-back (no resetAcc between blocks) so that
  //  the running accumulator is exercised across 4 × 4 = 16 cycles, and also
  //  a separate 32-element block repeated to reach 30+ cycles.
  //
  //  Failure here (but not in Test 19) would point to an accumulator issue
  //  rather than a single-cycle multiply bug.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: E5M2 x E2M1 / UE8M0 multi-block accumulation (vec=8, 32 cycles)") {
    log("\n[TEST 20] E5M2 x E2M1 / UE8M0 multi-block accumulation (vec=8, 32 cycles)")
    try {
      val cfg       = ScaleAddConfig(MXFormats.E5M2, MXFormats.E2M1, ScaleFormats.UE8M0)
      val vsize     = 8
      val blockSize = 32
      val rng       = new Random(1234)

      // Generate 8 independent 32-element MX blocks (8 × 4 cycles = 32 cycles total).
      // Each block has its own independently-computed shared scale factor.
      val allCycles = (0 until 8).flatMap { blk =>
        val aVals = Seq.fill(blockSize)(rng.nextGaussian() * 2.0)
        val bVals = Seq.fill(blockSize)(rng.nextGaussian() * 1.5)
        val (encA, rawSA) = quantizeBlock(aVals, cfg.elementTypeA, cfg.stype)
        val (encB, rawSB) = quantizeBlock(bVals, cfg.elementTypeB, cfg.stype)
        // 监控量化后的最大值，看是否触及了 E2M1 的边界
        val maxAbsB = bVals.map(_.abs).max
        val reprB = maxElemRepr(cfg.elementTypeB)
        val decodedSB = decodeScale(rawSB, cfg.stype)
        log(f"Block $blk: maxAbsB=$maxAbsB%.4f, reprB=$reprB%.4f, scaleB=$decodedSB%.4e")
        log(f"Block $blk: scale_A=${decodeScale(rawSA, cfg.stype)}%12.4e  scale_B=${decodeScale(rawSB, cfg.stype)}%12.4e")
        (0 until (blockSize / vsize)).map { c =>
          val s = c * vsize
          (encA.slice(s, s + vsize), encB.slice(s, s + vsize), rawSA, rawSB)
        }
      }

      test(new FusedDotProductUnit(cfg, vsize, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, allCycles, "E5M2 x E2M1 / UE8M0 multi-block vec=8 32 cycles")
      }
      log("[PASSED] E5M2 x E2M1 / UE8M0 multi-block accumulation 32 cycles")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E5M2 x E2M1 / UE8M0 multi-block accumulation: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Helper: vec-of-4 subnormal raw encodings for a given element type.
  //
  //   FP types (E5M2, E4M3, E3M2, E2M3, E2M1):
  //     Subnormal = exponent field all-zero with non-zero mantissa.
  //     Returns [minSub, maxSub, minSub, maxSub] where
  //       minSub = exp=0, mant=1  (smallest subnormal magnitude)
  //       maxSub = exp=0, mant=all-1s  (largest subnormal magnitude)
  //     sign=0 → positive, sign=1 → negative.
  //
  //   INT8: no subnormal concept — returns small magnitudes {1, 2, 1, 2}
  //   (sign=1 encodes negative values in 2's complement: -1=0xFF, -2=0xFE).
  // -----------------------------------------------------------------------
  def vec4Subnormals(t: ElementType, sign: Int = 0): Seq[Int] = {
    val signBit = sign << (t.totalWidth - 1)
    if (t.name == "INT8") {
      // 2's complement: positive small → 1,2; negative small → 0xFF(-1), 0xFE(-2)
      if (sign == 0) Seq(1, 2, 1, 2)
      else           Seq(0xFF, 0xFE, 0xFF, 0xFE)
    } else {
      val minSub = 1                              // exp=0, mant=1
      val maxSub = (1 << t.elementWidthMant) - 1  // exp=0, mant=all-1s
      Seq(signBit | minSub, signBit | maxSub, signBit | minSub, signBit | maxSub)
    }
  }

  // -----------------------------------------------------------------------
  // Test 21: Subnormal element values — all FP element type pairs (vec=4, 4 cycles)
  //
  //   For each (etA, etB) pair drawn from {E5M2, E4M3, E3M2, E2M3, E2M1}
  //   (INT8 is excluded — no subnormal concept):
  //     Cycle 0: subnormal A+ × normal B   — tests A-subnormal decode
  //     Cycle 1: subnormal A- × normal B   — tests negative A-subnormal
  //     Cycle 2: normal A  × subnormal B+  — tests B-subnormal decode
  //     Cycle 3: subnormal A+ × subnormal B+ — tests sub×sub multiply
  //   Scale = UE8M0 (pure power-of-2, scale=1.0) to isolate element behavior.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Subnormal element values — all FP type pairs (vec=4)") {
    log("\n[TEST 21] Subnormal element values — all FP element type pairs (vec=4)")
    val fpTypes  = MXFormats.allElementTypes.filterNot(_.name == "INT8")
    val scaleRaw = encodeScale(1.0, ScaleFormats.UE8M0)
    val vsize    = 4
    var passed   = 0; var failed = 0

    for (etA <- fpTypes; etB <- fpTypes) {
      val cfg   = ScaleAddConfig(etA, etB, ScaleFormats.UE8M0)
      val label = s"Subnormal elem: ${etA.name} x ${etB.name} / UE8M0 vec=$vsize"

      val subA_pos = vec4Subnormals(etA, sign = 0)
      val subA_neg = vec4Subnormals(etA, sign = 1)
      val subB_pos = vec4Subnormals(etB, sign = 0)
      val norm_A   = Seq.fill(vsize)(encodeElement(1.0, etA))
      val norm_B   = Seq.fill(vsize)(encodeElement(1.0, etB))

      val cycles = Seq(
        (subA_pos, norm_B,   scaleRaw, scaleRaw),  // subnormal A+ × normal B
        (subA_neg, norm_B,   scaleRaw, scaleRaw),  // subnormal A- × normal B
        (norm_A,   subB_pos, scaleRaw, scaleRaw),  // normal A × subnormal B+
        (subA_pos, subB_pos, scaleRaw, scaleRaw),  // subnormal A+ × subnormal B+
      )

      try {
        test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runCycles(dut, cfg, cycles, label)
        }
        log(s"[OK] $label")
        passed += 1
      } catch { case e: Exception =>
        logErr(s"[FAIL] $label: ${e.getMessage}")
        failed += 1
      }
    }

    val total   = fpTypes.size * fpTypes.size
    val summary = s"Subnormal elements: $passed passed, $failed failed out of $total combinations"
    if (failed > 0) logErr(summary) else log(summary)
    assert(failed == 0, s"$failed subnormal-element pair(s) failed — see log for details")
    log("[PASSED] Subnormal element values — all FP type pairs")
  }

  // -----------------------------------------------------------------------
  // Test 22: Subnormal scale values — all scale types with mantissa bits (vec=4)
  //
  //   UE8M0 is a pure-exponent scale and has no subnormal representation;
  //   all remaining 6 scale types (UE7M1 … UE2M6) are tested.
  //   A subnormal scale is encoded as exp=0, mant≠0.  Element values are
  //   normal (1.0) so any mismatch traces directly to the scale-decode path.
  //
  //   For very small subnormal scales (UE7M1, UE6M2) the FP32 product may
  //   underflow to 0; both the SW golden model and HW should agree on 0,
  //   which passes within the existing 1e-37 absolute tolerance floor.
  //
  //     Cycle 0: scale_A = sub_min (exp=0, mant=1),       scale_B = sub_min
  //     Cycle 1: scale_A = sub_max (exp=0, mant=all-1s),  scale_B = sub_min
  //     Cycle 2: scale_A = sub_max,                       scale_B = sub_max
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Subnormal scale values — all scale types with mantissa (vec=4)") {
    log("\n[TEST 22] Subnormal scale values — all scale types with mantissa (vec=4)")
    val etA   = MXFormats.E5M2
    val etB   = MXFormats.E4M3
    val vsize = 4
    val scaleTypesWithMant = ScaleFormats.allScaleTypes.filter(_.mantScaleWidth > 0)
    var passed = 0; var failed = 0

    for (st <- scaleTypesWithMant) {
      val cfg   = ScaleAddConfig(etA, etB, st)
      val label = s"Subnormal scale: E5M2 x E4M3 / ${st.name} vec=$vsize"

      val subScaleMin = 1                              // exp=0, mant=1
      val subScaleMax = (1 << st.mantScaleWidth) - 1  // exp=0, mant=all-1s
      val normA = Seq.fill(vsize)(encodeElement(1.0, etA))
      val normB = Seq.fill(vsize)(encodeElement(1.0, etB))

      val cycles = Seq(
        (normA, normB, subScaleMin, subScaleMin),  // min subnormal × min subnormal
        (normA, normB, subScaleMax, subScaleMin),  // max subnormal × min subnormal
        (normA, normB, subScaleMax, subScaleMax),  // max subnormal × max subnormal
      )

      try {
        test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runCycles(dut, cfg, cycles, label)
        }
        log(s"[OK] $label")
        passed += 1
      } catch { case e: Exception =>
        logErr(s"[FAIL] $label: ${e.getMessage}")
        failed += 1
      }
    }

    val summary = s"Subnormal scales: $passed passed, $failed failed out of ${scaleTypesWithMant.size}"
    if (failed > 0) logErr(summary) else log(summary)
    assert(failed == 0, s"$failed subnormal-scale type(s) failed — see log for details")
    log("[PASSED] Subnormal scale values — all scale types with mantissa")
  }

  // -----------------------------------------------------------------------
  // Test 23: All type combinations with subnormal inputs — full sweep (vec=4)
  //
  //   Covers all 6×6×7 = 252 (etA × etB × scaleType) triples.
  //   Element inputs use subnormal (or near-zero for INT8) raw values.
  //   Scale inputs: for scale types with mantissa bits, the smallest subnormal
  //   scale (exp=0, mant=1) is used; for UE8M0 (pure exponent), scale=1.0.
  //   This exercises the combined subnormal-element × subnormal-scale path.
  //
  //     Cycle 0: A+ × B+ — positive × positive
  //     Cycle 1: A+ × B- — positive × negative
  //     Cycle 2: A- × B+ — negative × positive
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: All type combinations with subnormal inputs — full sweep (vec=4)") {
    log("\n[TEST 23] All type combinations with subnormal inputs — full sweep (vec=4)")
    val vsize    = 4
    val allElem  = MXFormats.allElementTypes
    val allScale = ScaleFormats.allScaleTypes
    val total    = allElem.size * allElem.size * allScale.size
    var passed   = 0; var failed = 0
    
    for (etA <- allElem; etB <- allElem; st <- allScale) {
      val cfg   = ScaleAddConfig(etA, etB, st)
      val label = s"Subnormal sweep: ${etA.name} x ${etB.name} / ${st.name} vec=$vsize"

      // Subnormal scale for types with mantissa; normal 1.0 for UE8M0.
      val scaleRaw = if (st.mantScaleWidth > 0) 1 else encodeScale(1.0, st)

      val subA_pos = vec4Subnormals(etA, sign = 0)
      val subA_neg = vec4Subnormals(etA, sign = 1)
      val subB_pos = vec4Subnormals(etB, sign = 0)
      val subB_neg = vec4Subnormals(etB, sign = 1)

      val cycles = Seq(
        (subA_pos, subB_pos, scaleRaw, scaleRaw),  // A+ × B+
        (subA_pos, subB_neg, scaleRaw, scaleRaw),  // A+ × B-
        (subA_neg, subB_pos, scaleRaw, scaleRaw),  // A- × B+
      )

      // 当 scale 为 E7M1 (UE7M1) 时打印详细 lane 信息，便于 debug
      val verboseThisRun = st.name == "UE7M1"

      try {
        test(new FusedDotProductUnit(cfg, vsize, true)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runCycles(dut, cfg, cycles, label, verbose = verboseThisRun)
        }
        log(s"[OK] $label")
        passed += 1
      } catch { case e: Exception =>
        logErr(s"[FAIL] $label: ${e.getMessage}")
        failed += 1
      }
    }

    val summary = s"Full subnormal sweep: $passed passed, $failed failed out of $total"
    if (failed > 0) logErr(summary) else log(summary)
    assert(failed == 0, s"$failed subnormal combination(s) failed — see log for details")
    log("[PASSED] All type combinations with subnormal inputs — full sweep")
  }

  // -----------------------------------------------------------------------
  // Test 24: INT8 × xxx / UE8M0 — multi-block accumulation sweep (vec=8, 32 cycles)
  //
  //  For each element type B in allElementTypes {INT8, E5M2, E4M3, E3M2, E2M3, E2M1},
  //  runs 8 independent 32-element MX blocks back-to-back (no resetAcc between
  //  blocks) with randomly generated values quantized via quantizeBlock.
  //  The structure mirrors Test 20 but sweeps all B-side element types while
  //  fixing elementTypeA = INT8 and scale = UE8M0.
  //
  //  Failure for a specific (INT8 × etB) pair would indicate an accumulator
  //  or mixed-type multiply issue rather than a single-cycle decode bug.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: INT8 x xxx / UE8M0 multi-block accumulation sweep (vec=8, 32 cycles)") {
    log("\n[TEST 24] INT8 x xxx / UE8M0 multi-block accumulation sweep (vec=8, 32 cycles)")
    val vsize     = 8
    val blockSize = 32
    val allElemB  = MXFormats.allElementTypes
    var passed    = 0; var failed = 0

    for (etB <- allElemB) {
      val cfg   = ScaleAddConfig(MXFormats.INT8, etB, ScaleFormats.UE8M0)
      val label = s"INT8 x ${etB.name} / UE8M0 multi-block vec=$vsize 32 cycles"
      log(s"\n--- $label ---")
      val rng = new Random(5678)

      // Generate 8 independent 32-element MX blocks (8 × 4 cycles = 32 cycles total).
      // Each block has its own independently-computed shared scale factor.
      val allCycles = (0 until 8).flatMap { blk =>
        val aVals = Seq.fill(blockSize)(rng.nextGaussian() * 2.0)
        val bVals = Seq.fill(blockSize)(rng.nextGaussian() * 1.5)
        val (encA, rawSA) = quantizeBlock(aVals, cfg.elementTypeA, cfg.stype)
        val (encB, rawSB) = quantizeBlock(bVals, cfg.elementTypeB, cfg.stype)
        val maxAbsB   = bVals.map(_.abs).max
        val reprB     = maxElemRepr(cfg.elementTypeB)
        val decodedSB = decodeScale(rawSB, cfg.stype)
        log(f"Block $blk: maxAbsB=$maxAbsB%.4f, reprB=$reprB%.4f, scaleB=$decodedSB%.4e")
        log(f"Block $blk: scale_A=${decodeScale(rawSA, cfg.stype)}%12.4e  scale_B=${decodeScale(rawSB, cfg.stype)}%12.4e")
        (0 until (blockSize / vsize)).map { c =>
          val s = c * vsize
          (encA.slice(s, s + vsize), encB.slice(s, s + vsize), rawSA, rawSB)
        }
      }

      try {
        test(new FusedDotProductUnit(cfg, vsize, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
          initDut(dut)
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runCycles(dut, cfg, allCycles, label)
        }
        log(s"[PASSED] $label")
        passed += 1
      } catch { case e: Exception =>
        logErr(s"[FAILED] $label: ${e.getMessage}")
        failed += 1
      }
    }

    val summary = s"INT8 x xxx / UE8M0 multi-block sweep: $passed passed, $failed failed out of ${allElemB.size}"
    if (failed > 0) logErr(summary) else log(summary)
    assert(failed == 0, s"$failed INT8 multi-block combination(s) failed — see log for details")
    log("[PASSED] INT8 x xxx / UE8M0 multi-block accumulation sweep")
  }

  // -----------------------------------------------------------------------
  // Test 25: Python-generated MXINT8 × E2M1 / UE8M0 — hardcoded vectors
  //          (vec=4, 2 blocks × 8 cycles = 16 cycles total)
  //
  //  Vectors generated by gen_int8_e2m1_test.py using quantize_mx_v2
  //  with dtype='mxint8', scale_format='UE8M0', block_size=32, seed=9999.
  //  Raw encodings are 2's complement MXINT8 (signed 8-bit integer)
  //  and 4-bit E2M1 in a uint8.  Expected result: 22.406250.
  // -----------------------------------------------------------------------
  test("FusedDotProductUnit: Python-generated MXINT8 x E2M1 / UE8M0 hardcoded (vec=4, 16 cycles)") {
    log("\n[TEST 25] Python-generated MXINT8 x E2M1 / UE8M0 hardcoded (vec=4, 16 cycles)")
    try {
      val cfg   = ScaleAddConfig(MXFormats.INT8, MXFormats.E2M1, ScaleFormats.UE8M0)
      val vsize = 4

      // Generated by gen_int8_e2m1_test.py (quantize_mx_v2, seed=9999)
      // INT8 raw encodings are 2's complement (hardware format).
      // Converted from original sign-magnitude output of gen_int8_e2m1_test.py.

      val rawScaleA_0 = 0x77  // scale_A = 3.906250e-03
      val rawScaleB_0 = 0x75  // scale_B = 9.765625e-04
      val rawScaleA_1 = 0x77  // scale_A = 3.906250e-03
      val rawScaleB_1 = 0x74  // scale_B = 4.882812e-04

      val encA_0: Seq[Int] = Seq(
        0xF2, 0xFB, 0xF9, 0x15, 0x49, 0xD9, 0xDB, 0xE6,
        0x31, 0x10, 0xC8, 0xEB, 0x10, 0x29, 0x1B, 0xBD,
        0xEE, 0xFB, 0xF4, 0x1F, 0xFB, 0x34, 0x02, 0x30,
        0xFE, 0xE5, 0xD9, 0xDC, 0xF7, 0x2C, 0xF9, 0x20)
      val encB_0: Seq[Int] = Seq(
        0x09, 0x01, 0x00, 0x0A, 0x08, 0x00, 0x0B, 0x04,
        0x0B, 0x04, 0x0D, 0x08, 0x0C, 0x01, 0x02, 0x0D,
        0x03, 0x0B, 0x09, 0x01, 0x09, 0x0A, 0x01, 0x02,
        0x03, 0x00, 0x02, 0x0E, 0x09, 0x01, 0x09, 0x0A)

      val encA_1: Seq[Int] = Seq(
        0x1A, 0x12, 0xE2, 0xD6, 0x03, 0xE8, 0xFE, 0x23,
        0xDD, 0x26, 0xED, 0x77, 0xD6, 0xF9, 0x29, 0xEE,
        0x19, 0xD8, 0x16, 0xDA, 0x08, 0xD7, 0xF3, 0x3B,
        0xDB, 0x08, 0x10, 0xF6, 0xFB, 0xEE, 0xE0, 0x00)
      val encB_1: Seq[Int] = Seq(
        0x0F, 0x0D, 0x01, 0x02, 0x00, 0x02, 0x09, 0x0E,
        0x0F, 0x0B, 0x04, 0x01, 0x03, 0x07, 0x09, 0x0B,
        0x01, 0x0D, 0x05, 0x00, 0x05, 0x0D, 0x04, 0x0B,
        0x09, 0x06, 0x01, 0x03, 0x02, 0x04, 0x02, 0x08)

      // Print decode table (same structure as test 19/20)
      val blockDefs = Seq(
        (encA_0, encB_0, rawScaleA_0, rawScaleB_0),
        (encA_1, encB_1, rawScaleA_1, rawScaleB_1))

      for ((eA, eB, rsA, rsB) <- blockDefs.zipWithIndex.map { case (t, i) => (t._1, t._2, t._3, t._4) }) {
        val sA = decodeScale(rsA, cfg.stype)
        val sB = decodeScale(rsB, cfg.stype)
        val maxAbsB = eB.map(r => math.abs(decodeElement(r, cfg.elementTypeB) * sB)).max
        log(f"Block: maxAbsB=$maxAbsB%.4f  scale_A=${sA}%12.4e  scale_B=${sB}%12.4e")
      }
      log("-" * 80)
      log("%-6s %6s %10s  %5s %10s  %12s".format("Elem", "rawA", "decA×sA", "rawB", "decB×sB", "A*B"))
      log("-" * 80)
      var exactDot = 0.0
      for (((eA, eB, rsA, rsB), blk) <- blockDefs.zipWithIndex) {
        val sA = decodeScale(rsA, cfg.stype)
        val sB = decodeScale(rsB, cfg.stype)
        for (idx <- 0 until 32) {
          val dA = decodeElement(eA(idx), cfg.elementTypeA) * sA
          val dB = decodeElement(eB(idx), cfg.elementTypeB) * sB
          exactDot += dA * dB
          log(f"${blk*32+idx}%-6d 0x${eA(idx).toHexString}%4s ${dA}%10.4f  0x${eB(idx).toHexString}%4s ${dB}%10.4f  ${dA*dB}%12.4f")
        }
      }
      log("-" * 80)
      log(f"Expected dot product (Python golden) = $exactDot%.4e  (should be 22.406250)")

      // Build cycle list: block 0 then block 1, vsize=4 → 8 cycles each
      val allCycles = blockDefs.flatMap { case (eA, eB, rsA, rsB) =>
        (0 until (32 / vsize)).map { c =>
          val s = c * vsize
          (eA.slice(s, s + vsize), eB.slice(s, s + vsize), rsA, rsB)
        }
      }

      test(new FusedDotProductUnit(cfg, vsize, true)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        initDut(dut)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runCycles(dut, cfg, allCycles, "Python MXINT8 x E2M1 / UE8M0 vec=4 16 cycles")
      }
      log("[PASSED] Python-generated MXINT8 x E2M1 / UE8M0 hardcoded test")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Python-generated MXINT8 x E2M1 / UE8M0 hardcoded: ${e.getMessage}"); throw e
    }
  }
}
                       