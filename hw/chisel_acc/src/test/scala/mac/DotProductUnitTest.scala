package mx.mac

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import java.lang.Float.intBitsToFloat

class DotProductUnitTest extends AnyFunSuite with ChiselScalatestTester {

  val scfg = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)

  // -----------------------------------------------------------------------
  // Log file — all verbose output goes here; stdout only gets errors
  // -----------------------------------------------------------------------
  private val _logWriter: java.io.PrintWriter = {
    val fw = new java.io.FileWriter("dot_product_unit_test.log", false) // overwrite each run
    new java.io.PrintWriter(fw, true)
  }

  /** Write to the log file only. */
  private def log(msg: String): Unit = _logWriter.println(msg)

  /** Write to both the log file and stdout (used for errors/failures). */
  private def logErr(msg: String): Unit = { _logWriter.println(msg); println(msg) }

  // -----------------------------------------------------------------------
  // Software golden model — correct IEEE 754 arithmetic
  // -----------------------------------------------------------------------

  /** Decode a raw bit-pattern to Double using the MX element format rules. */
  def decodeElement(raw: Int, t: ElementType): Double = {
    val sign = if (((raw >> (t.totalWidth - 1)) & 1) == 1) -1.0 else 1.0
    if (t.name == "INT8") {
      val mag = raw & ((1 << (t.totalWidth - 1)) - 1)
      sign * mag.toDouble
    } else {
      val exp  = (raw >> t.elementWidthMant) & ((1 << t.elementWidthExp) - 1)
      val mant = raw & ((1 << t.elementWidthMant) - 1)
      if (exp == 0)
        sign * (mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, 1 - t.bias)
      else
        sign * (1.0 + mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, exp - t.bias)
    }
  }

  /** Decode a raw bit-pattern to Double using the MX scale format (no sign bit). */
  def decodeScale(raw: Int, s: ScaleType): Double = {
    val expBits = (raw >> s.mantScaleWidth) & ((1 << s.expScaleWidth) - 1)
    if (s.mantScaleWidth == 0) {
      Math.pow(2, expBits - s.bias)
    } else {
      val mantBits  = raw & ((1 << s.mantScaleWidth) - 1)
      val implicit1 = if (expBits > 0) 1.0 else 0.0
      (implicit1 + mantBits.toDouble / (1 << s.mantScaleWidth)) * Math.pow(2, expBits - s.bias)
    }
  }

  /** Encode a Double value into raw bits for a given ElementType. */
  def encodeElement(value: Double, t: ElementType): Int = {
    if (value == 0.0) return 0
    val sign   = if (value < 0) 1 else 0
    val absVal = math.abs(value)
    if (t.name == "INT8") {
      (sign << 7) | math.round(absVal).toInt.min(127)
    } else {
      val expUnbiased = math.floor(math.log(absVal) / math.log(2)).toInt
      val expBiased   = expUnbiased + t.bias
      if (expBiased <= 0) {
        val mantInt = math.round(absVal / math.pow(2, 1 - t.bias) * (1 << t.elementWidthMant)).toInt
                             .min((1 << t.elementWidthMant) - 1)
        (sign << (t.totalWidth - 1)) | mantInt
      } else {
        val expClamped = expBiased.min((1 << t.elementWidthExp) - 1)
        val mantInt    = math.round((absVal / math.pow(2, expUnbiased) - 1.0) * (1 << t.elementWidthMant)).toInt
                               .min((1 << t.elementWidthMant) - 1).max(0)
        (sign << (t.totalWidth - 1)) | (expClamped << t.elementWidthMant) | mantInt
      }
    }
  }

  /** Encode a positive scale value into raw bits for a given ScaleType. */
  def encodeScale(value: Double, s: ScaleType): Int = {
    val expUnbiased = math.floor(math.log(value) / math.log(2)).toInt
    val expBiased   = (expUnbiased + s.bias).max(0).min((1 << s.expScaleWidth) - 1)
    if (s.mantScaleWidth == 0) expBiased
    else {
      val mantInt = math.round((value / math.pow(2, expUnbiased) - 1.0) * (1 << s.mantScaleWidth)).toInt
                           .min((1 << s.mantScaleWidth) - 1).max(0)
      (expBiased << s.mantScaleWidth) | mantInt
    }
  }

  /** Correct product for one step using the default scfg. */
  def swProduct(inA: Int, inB: Int, scaleA: Int, scaleB: Int): Float =
    (decodeElement(inA, scfg.elementTypeA) *
     decodeElement(inB, scfg.elementTypeB) *
     decodeScale(scaleA, scfg.stype) *
     decodeScale(scaleB, scfg.stype)).toFloat

  /** Correct product for one step using an explicit config. */
  def swProductFor(cfg: ScaleAddConfig)(inA: Int, inB: Int, scaleA: Int, scaleB: Int): Float =
    (decodeElement(inA, cfg.elementTypeA) *
     decodeElement(inB, cfg.elementTypeB) *
     decodeScale(scaleA, cfg.stype) *
     decodeScale(scaleB, cfg.stype)).toFloat

  // Read accOut as IEEE 754 Float
  def peekFloat(dut: DotProductUnit): Float =
    intBitsToFloat(dut.io.accOut.peek().litValue.toInt)

  /** Drive one MAC step (1 cycle: assert validIn, step, de-assert). */
  def driveOne(dut: DotProductUnit, inA: Int, inB: Int, scaleA: Int, scaleB: Int): Unit = {
    dut.io.op_a_i.poke(inA.U)
    dut.io.op_b_i.poke(inB.U)
    dut.io.share_exp_A_i.poke(scaleA.U)
    dut.io.share_exp_B_i.poke(scaleB.U)
    dut.io.validIn.poke(true.B)
    dut.clock.step()            // accReg updated at posedge; validOut will be true after this
    dut.io.validIn.poke(false.B)
  }

  /**
   * Run a sequence of (rawA, rawB, rawScaleA, rawScaleB) steps and verify each against the SW
   * golden model.  All output goes to the log file; on mismatch the failing line is also printed
   * to stdout before the assertion is thrown.
   */
  def runSteps(dut: DotProductUnit, steps: Seq[(Int, Int, Int, Int)],
               cfg: ScaleAddConfig, header: String = ""): Unit = {
    if (header.nonEmpty) {
      log("=" * 70)
      log(header)
      log("=" * 70)
    }
    var swAcc = 0.0f
    for (((a, b, sA, sB), i) <- steps.zipWithIndex) {
      val stepVal = swProductFor(cfg)(a, b, sA, sB)
      driveOne(dut, a, b, sA, sB)
      swAcc += stepVal
      val hw  = peekFloat(dut)
      val aD  = decodeElement(a,  cfg.elementTypeA)
      val bD  = decodeElement(b,  cfg.elementTypeB)
      val sAD = decodeScale(sA, cfg.stype)
      val sBD = decodeScale(sB, cfg.stype)
      val tol     = math.abs(swAcc) * 0.05f + 1e-37f
      val isFail  = math.abs(hw - swAcc) > tol
      val line    = f"Step $i%2d | rawA=0x${a}%X($aD%7.4f) rawB=0x${b}%X($bD%7.4f) rawSA=0x${sA}%X($sAD%6.4f) rawSB=0x${sB}%X($sBD%6.4f) | step=$stepVal%9.4f | SW=$swAcc%10.4f | HW=$hw%10.4f${if (isFail) " *** MISMATCH ***" else ""}"
      if (isFail) logErr(line) else log(line)
      assert(!isFail, s"Mismatch at step $i: hw=$hw expected=$swAcc (tol=$tol)")
    }
  }

  // -----------------------------------------------------------------------
  // Pre-encoded constants for the default config (E4M3 × E2M1 / UE5M3)
  // -----------------------------------------------------------------------
  val e4m3_1    = encodeElement(1.0,  MXFormats.E4M3)   // 56
  val e4m3_neg1 = encodeElement(-1.0, MXFormats.E4M3)   // 184
  val e4m3_2    = encodeElement(2.0,  MXFormats.E4M3)   // 64
  val e2m1_1    = encodeElement(1.0,  MXFormats.E2M1)   // 2
  val e2m1_2    = encodeElement(2.0,  MXFormats.E2M1)   // 4
  val ue5m3_1   = encodeScale(1.0,    ScaleFormats.UE5M3) // 120
  val ue5m3_2   = encodeScale(2.0,    ScaleFormats.UE5M3) // 128

  // -----------------------------------------------------------------------
  // Test 1: Reset clears accumulator
  // -----------------------------------------------------------------------
  test("DotProductUnit: Reset clears accumulator") {
    log("\n[TEST 1] Reset clears accumulator")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)
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
  // Test 2: valid handshake timing — validOut is validIn delayed by 1 cycle
  // -----------------------------------------------------------------------
  test("DotProductUnit: valid handshake timing") {
    log("\n[TEST 2] valid handshake timing")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)
        dut.io.op_a_i.poke(e4m3_1.U); dut.io.op_b_i.poke(e2m1_1.U)
        dut.io.share_exp_A_i.poke(ue5m3_1.U); dut.io.share_exp_B_i.poke(ue5m3_1.U)

        dut.io.validOut.expect(false.B, "validOut must be false when validIn=0")

        dut.io.validIn.poke(true.B)
        dut.clock.step()                     // accReg updated at posedge
        dut.io.validIn.poke(false.B)
        dut.io.validOut.expect(true.B, "validOut must be true one cycle after validIn")

        dut.clock.step()
        dut.io.validOut.expect(false.B, "validOut must return false when validIn=0")
      }
      log("[PASSED] valid handshake timing")
    } catch { case e: Exception =>
      logErr(s"[FAILED] valid handshake timing: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 3: 1.0 × 1.0 × 1.0 × 1.0 = 1.0
  // -----------------------------------------------------------------------
  test("DotProductUnit: Single accumulation 1x1x1x1=1") {
    log("\n[TEST 3] Single accumulation 1x1x1x1=1")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val expected = swProduct(e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
        driveOne(dut, e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)

        log(f"Expected (SW): $expected%.6f")
        log(f"Got      (HW): $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol,
          s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] Single accumulation 1x1x1x1=1")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Single accumulation 1x1x1x1=1: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 4: Different scale factors — scaleA=2.0, scaleB=1.0 → 2.0
  // -----------------------------------------------------------------------
  test("DotProductUnit: Different scale factors A=2 B=1") {
    log("\n[TEST 4] Different scale factors (scaleA=2.0, scaleB=1.0)")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val expected = swProduct(e4m3_1, e2m1_1, ue5m3_2, ue5m3_1)
        driveOne(dut, e4m3_1, e2m1_1, ue5m3_2, ue5m3_1)
        val hw = peekFloat(dut)

        log(f"Expected (SW): $expected%.6f  [1.0 x 1.0 x 2.0 x 1.0 = 2.0]")
        log(f"Got      (HW): $hw%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol,
          s"Mismatch: hw=$hw expected=$expected")
      }
      log("[PASSED] Different scale factors A=2 B=1")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Different scale factors A=2 B=1: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 5: Multiple accumulations (4 steps of 1x1x1x1) → 4.0
  // -----------------------------------------------------------------------
  test("DotProductUnit: Multiple accumulations (4 steps)") {
    log("\n[TEST 5] Multiple accumulations (4 steps of 1x1x1x1)")
    try {
      test(new DotProductUnit(scfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        var swAcc = 0.0f
        for (i <- 0 until 4) {
          val step = swProduct(e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
          driveOne(dut, e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
          swAcc += step
          val hw = peekFloat(dut)
          log(f"Step $i: SW acc = $swAcc%.4f  HW acc = $hw%.4f")

          val tol = math.abs(swAcc) * 0.01f + 1e-37f
          assert(math.abs(hw - swAcc) <= tol,
            s"Mismatch at step $i: hw=$hw expected=$swAcc")
        }
      }
      log("[PASSED] Multiple accumulations (4 steps)")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Multiple accumulations (4 steps): ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 6: Negative element input — two steps of (-1×1×1×1) → -2.0
  // -----------------------------------------------------------------------
  test("DotProductUnit: Negative element input") {
    log("\n[TEST 6] Negative element input (-1.0 x 1.0 x 1.0 x 1.0)")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        var swAcc = 0.0f
        for (i <- 0 until 2) {
          val step = swProduct(e4m3_neg1, e2m1_1, ue5m3_1, ue5m3_1)
          driveOne(dut, e4m3_neg1, e2m1_1, ue5m3_1, ue5m3_1)
          swAcc += step
          val hw = peekFloat(dut)
          log(f"Step $i: SW acc = $swAcc%.4f  HW acc = $hw%.4f  [step value = $step%.4f]")
        }

        val hw  = peekFloat(dut)
        val tol = math.abs(swAcc) * 0.01f + 1e-37f
        assert(math.abs(hw - swAcc) <= tol,
          s"Mismatch: hw=$hw expected=$swAcc")
      }
      log("[PASSED] Negative element input")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Negative element input: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 7: Reset mid-accumulation
  // -----------------------------------------------------------------------
  test("DotProductUnit: Reset mid-accumulation") {
    log("\n[TEST 7] Reset mid-accumulation")
    try {
      test(new DotProductUnit(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)

        // Accumulate once
        driveOne(dut, e4m3_2, e2m1_2, ue5m3_1, ue5m3_1)
        val before = dut.io.accOut.peek().litValue.toInt
        log(s"Before reset: accOut = 0x${before.toHexString}  (${intBitsToFloat(before)})")
        assert(before != 0, "accOut should be non-zero after accumulation")

        // Reset
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.accOut.expect(0.U, "accOut should be 0 after reset")
        dut.io.resetAcc.poke(false.B)

        // Accumulate once more
        val expected = swProduct(e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
        driveOne(dut, e4m3_1, e2m1_1, ue5m3_1, ue5m3_1)
        val hw = peekFloat(dut)
        log(f"After reset + 1 step: HW = $hw%.6f  expected = $expected%.6f")

        val tol = math.abs(expected) * 0.01f + 1e-37f
        assert(math.abs(hw - expected) <= tol,
          s"Mismatch after reset: hw=$hw expected=$expected")
      }
      log("[PASSED] Reset mid-accumulation")
    } catch { case e: Exception =>
      logErr(s"[FAILED] Reset mid-accumulation: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 8: E5M2 × E4M3 / UE8M0 — 10-step mixed accumulation
  // -----------------------------------------------------------------------
  test("DotProductUnit: E5M2 x E4M3 / UE8M0 — 10-step mixed") {
    log("\n[TEST 8] E5M2 x E4M3 / UE8M0 — 10 steps")
    try {
      val cfg8 = ScaleAddConfig(MXFormats.E5M2, MXFormats.E4M3, ScaleFormats.UE8M0)
      val rawSteps = Seq(
        ( 1.0,  1.0,  1.0, 1.0),
        ( 2.0,  1.0,  1.0, 1.0),
        ( 1.0,  2.0,  1.0, 1.0),
        (-1.0,  2.0,  1.0, 1.0),
        ( 1.5,  1.0,  1.0, 1.0),
        ( 1.0,  1.5,  2.0, 1.0),
        ( 2.0,  2.0,  1.0, 1.0),
        ( 1.0,  1.0,  4.0, 1.0),
        (-2.0,  1.5,  1.0, 1.0),
        ( 1.0,  1.0,  0.5, 1.0)
      ).map { case (a, b, sA, sB) =>
        (encodeElement(a,  cfg8.elementTypeA),
         encodeElement(b,  cfg8.elementTypeB),
         encodeScale(sA,  cfg8.stype),
         encodeScale(sB,  cfg8.stype))
      }
      test(new DotProductUnit(cfg8)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runSteps(dut, rawSteps, cfg8, "E5M2 x E4M3 / UE8M0 — 10 steps")
      }
      log("[PASSED] E5M2 x E4M3 / UE8M0 — 10-step mixed")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E5M2 x E4M3 / UE8M0 — 10-step mixed: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 9: E3M2 × E2M3 / UE6M2 — 8-step with non-trivial mantissa values
  // -----------------------------------------------------------------------
  test("DotProductUnit: E3M2 x E2M3 / UE6M2 — 8-step mixed sign") {
    log("\n[TEST 9] E3M2 x E2M3 / UE6M2 — 8 steps")
    try {
      val cfg9 = ScaleAddConfig(MXFormats.E3M2, MXFormats.E2M3, ScaleFormats.UE6M2)
      val rawSteps = Seq(
        ( 1.0,   1.0,  1.0,  1.0),
        ( 2.0,   1.0,  1.0,  1.0),
        ( 1.0,  -1.0,  1.0,  1.0),
        ( 1.5,   1.25, 1.0,  1.0),
        (-1.5,   1.5,  1.0,  1.0),
        ( 2.0,   2.0,  2.0,  1.0),
        ( 1.0,   1.0,  1.5,  1.5),
        (-1.0,  -1.0,  1.0,  1.0)
      ).map { case (a, b, sA, sB) =>
        (encodeElement(a,  cfg9.elementTypeA),
         encodeElement(b,  cfg9.elementTypeB),
         encodeScale(sA,  cfg9.stype),
         encodeScale(sB,  cfg9.stype))
      }
      test(new DotProductUnit(cfg9)) { dut =>
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runSteps(dut, rawSteps, cfg9, "E3M2 x E2M3 / UE6M2 — 8 steps")
      }
      log("[PASSED] E3M2 x E2M3 / UE6M2 — 8-step mixed sign")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E3M2 x E2M3 / UE6M2 — 8-step mixed sign: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 10: E4M3 × E2M1 / UE4M4 — 8-step with fractional scale mantissa
  // -----------------------------------------------------------------------
  test("DotProductUnit: E4M3 x E2M1 / UE4M4 — 8-step fractional scale") {
    log("\n[TEST 10] E4M3 x E2M1 / UE4M4 — 8 steps with fractional scale")
    try {
      val cfg10 = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE4M4)
      val rawSteps = Seq(
        ( 1.0,  1.0,  1.0,   1.0),
        ( 1.5,  1.0,  1.0,   1.0),
        ( 1.0,  2.0,  1.5,   1.0),
        (-1.5,  1.0,  2.0,   1.0),
        ( 2.0,  1.0,  1.0,   1.5),
        ( 1.25, 1.0,  1.0,   1.0),
        (-1.0, -1.0,  1.0,   1.0),
        ( 1.0,  1.0,  1.0,   1.25)
      ).map { case (a, b, sA, sB) =>
        (encodeElement(a,  cfg10.elementTypeA),
         encodeElement(b,  cfg10.elementTypeB),
         encodeScale(sA,  cfg10.stype),
         encodeScale(sB,  cfg10.stype))
      }
      test(new DotProductUnit(cfg10)) { dut =>
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runSteps(dut, rawSteps, cfg10, "E4M3 x E2M1 / UE4M4 — 8 steps")
      }
      log("[PASSED] E4M3 x E2M1 / UE4M4 — 8-step fractional scale")
    } catch { case e: Exception =>
      logErr(s"[FAILED] E4M3 x E2M1 / UE4M4 — 8-step fractional scale: ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 11: Default config — 16-step mixed accumulation with VCD
  // -----------------------------------------------------------------------
  test("DotProductUnit: 16-step mixed accumulation (default config, VCD)") {
    log("\n[TEST 11] 16-step mixed accumulation (E4M3 x E2M1 / UE5M3)")
    try {
      val e4m3_1p5  = encodeElement(1.5,  MXFormats.E4M3)
      val e4m3_1p25 = encodeElement(1.25, MXFormats.E4M3)
      val e2m1_1p5  = encodeElement(1.5,  MXFormats.E2M1)
      val ue5m3_1p5 = encodeScale(1.5,    ScaleFormats.UE5M3)

      val rawSteps = Seq(
        (e4m3_1,     e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_1,     e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_neg1,  e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_2,     e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_1,     e2m1_2,   ue5m3_1,   ue5m3_1),
        (e4m3_neg1,  e2m1_2,   ue5m3_2,   ue5m3_1),
        (e4m3_1p5,   e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_1,     e2m1_1,   ue5m3_2,   ue5m3_2),
        (e4m3_neg1,  e2m1_1,   ue5m3_1,   ue5m3_2),
        (e4m3_1,     e2m1_1p5, ue5m3_1,   ue5m3_1),
        (e4m3_2,     e2m1_1,   ue5m3_2,   ue5m3_1),
        (e4m3_neg1,  e2m1_1,   ue5m3_2,   ue5m3_2),
        (e4m3_1p25,  e2m1_1,   ue5m3_1,   ue5m3_1),
        (e4m3_1,     e2m1_1,   ue5m3_1p5, ue5m3_1),
        (e4m3_2,     e2m1_2,   ue5m3_1,   ue5m3_1),
        (e4m3_neg1,  e2m1_1,   ue5m3_1,   ue5m3_1)
      )
      test(new DotProductUnit(scfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runSteps(dut, rawSteps, scfg, "16-step mixed (E4M3 x E2M1 / UE5M3)")
      }
      log("[PASSED] 16-step mixed accumulation (default config, VCD)")
    } catch { case e: Exception =>
      logErr(s"[FAILED] 16-step mixed accumulation (default config, VCD): ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 12: 8-element vector dot product simulation with shared scale=2.0
  // -----------------------------------------------------------------------
  test("DotProductUnit: 8-element vector dot product (shared scale=2.0, VCD)") {
    log("\n[TEST 12] 8-element vector dot product (scale=2.0, scale=2.0)")
    try {
      val vecA = Seq(1.0, -1.0, 2.0, 1.0, -1.0, 1.0, 2.0, -2.0)
      val vecB = Seq(1.0,  1.0, 1.0, 2.0,  2.0, 1.0, 1.0,  1.0)
      val sharedScaleA = 2.0
      val sharedScaleB = 2.0

      val rawSteps = vecA.zip(vecB).map { case (a, b) =>
        (encodeElement(a,            scfg.elementTypeA),
         encodeElement(b,            scfg.elementTypeB),
         encodeScale(sharedScaleA,  scfg.stype),
         encodeScale(sharedScaleB,  scfg.stype))
      }

      val dotNoScale    = vecA.zip(vecB).map { case (a, b) => a * b }.sum
      val expectedTotal = (dotNoScale * sharedScaleA * sharedScaleB).toFloat
      log(f"Dot product (no scale) = $dotNoScale%.4f")
      log(f"Expected total (xscale^2) = $expectedTotal%.4f")

      test(new DotProductUnit(scfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.resetAcc.poke(true.B); dut.clock.step()
        dut.io.resetAcc.poke(false.B)
        runSteps(dut, rawSteps, scfg, "8-element vector dot product (scale=2.0 x 2.0)")
        val hw  = peekFloat(dut)
        val tol = math.abs(expectedTotal) * 0.05f + 1e-37f
        assert(math.abs(hw - expectedTotal) <= tol,
          s"Final dot product mismatch: hw=$hw expected=$expectedTotal")
        log(f"Final HW result = $hw%.4f  expected = $expectedTotal%.4f")
      }
      log("[PASSED] 8-element vector dot product (shared scale=2.0, VCD)")
    } catch { case e: Exception =>
      logErr(s"[FAILED] 8-element vector dot product (shared scale=2.0, VCD): ${e.getMessage}"); throw e
    }
  }

  // -----------------------------------------------------------------------
  // Test 13: All type combinations — 10-step random accumulation
  //   Iterates over every (elementTypeA × elementTypeB × scaleType) pair
  //   (6 × 6 × 7 = 252 configurations) and drives 10 random raw-bit inputs
  //   per configuration, verifying the HW accumulator against the SW golden
  //   model at each step.  Element exponents are clamped near their bias
  //   (adjExp in [1,4]) and scale exponents near their bias (±2) to keep
  //   accumulated sums well within FP32 range.
  // -----------------------------------------------------------------------
  test("DotProductUnit: All type combinations — 10-step random accumulation") {
    log("\n[TEST 13] All type combinations — 10-step random accumulation")

    val rng = new scala.util.Random(42)

    /** Random element raw bits with adjExp in [1, 4] to stay in [2, ~32]. */
    def randElement(t: ElementType, r: scala.util.Random): Int = {
      val sign = r.nextInt(2)
      if (t.name == "INT8") {
        (sign << 7) | (1 + r.nextInt(15))          // magnitude 1..15
      } else {
        val expRange = 1 << t.elementWidthExp
        val expLo    = (t.bias + 1).max(1).min(expRange - 1)
        val expHi    = (t.bias + 4).max(expLo).min(expRange - 1)
        val exp      = expLo + r.nextInt(expHi - expLo + 1)
        val mant     = r.nextInt(1 << t.elementWidthMant)
        (sign << (t.totalWidth - 1)) | (exp << t.elementWidthMant) | mant
      }
    }

    /** Random scale raw bits with exp near bias ± 2 → scale in [0.25, 4.0]. */
    def randScale(s: ScaleType, r: scala.util.Random): Int = {
      val expRange = 1 << s.expScaleWidth
      val expLo    = (s.bias - 2).max(1).min(expRange - 1)
      val expHi    = (s.bias + 2).max(expLo).min(expRange - 1)
      val exp      = expLo + r.nextInt(expHi - expLo + 1)
      val mant     = if (s.mantScaleWidth == 0) 0 else r.nextInt(1 << s.mantScaleWidth)
      (exp << s.mantScaleWidth) | mant
    }

    val allElem  = MXFormats.allElementTypes
    val allScale = ScaleFormats.allScaleTypes
    val total    = allElem.size * allElem.size * allScale.size
    var passed   = 0
    var failed   = 0

    for (etA <- allElem; etB <- allElem; st <- allScale) {
      val cfg   = ScaleAddConfig(etA, etB, st)
      val label = s"${etA.name} x ${etB.name} / ${st.name}"
      val steps = (0 until 10).map { _ =>
        (randElement(etA, rng), randElement(etB, rng),
         randScale(st, rng),    randScale(st, rng))
      }
      try {
        test(new DotProductUnit(cfg)) { dut =>
          dut.io.resetAcc.poke(true.B); dut.clock.step()
          dut.io.resetAcc.poke(false.B)
          runSteps(dut, steps, cfg, label)
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
    assert(failed == 0, s"$failed combination(s) failed — see dot_product_unit_test.log for details")
    log("[PASSED] All type combinations — 10-step random accumulation")
  }
}
