package mx.requant

import chisel3._
import chisel3.util._
import mx.mac.ElementType

// ============================================================
// NVFP4 max-normal limits per element format (OCP-MX semantics)
// ============================================================
/**
 * Real OCP-MX max-normal (biased_exp, mant) for the ExMy / NVFP4 path.
 *
 * Differs from Chisel's "(1<<exp)-2" convention for E4M3/E3M2/E2M3 because
 * OCP-MX FP6 has no Inf/NaN and E4M3 reserves only (s, 1111, 111).  The
 * UE8M0 path keeps the legacy "all-ones-minus-one" convention for backward
 * compatibility (OCP only defines scale as a pure power of 2 for E8M0).
 */
private object NVFP4Limits {
  def apply(t: ElementType): (Int, Int) = t.name match {
    case "E5M2" => (30, (1 << t.elementWidthMant) - 1)  // IEEE-like; all-ones exp = Inf/NaN
    case "E4M3" => (15, (1 << t.elementWidthMant) - 2)  // OCP-MX: only (s,1111,111) is NaN
    case "E3M2" => (7,  (1 << t.elementWidthMant) - 1)  // FP6: no Inf/NaN
    case "E2M3" => (3,  (1 << t.elementWidthMant) - 1)  // FP6: no Inf/NaN
    case _      => ((1 << t.elementWidthExp) - 2, (1 << t.elementWidthMant) - 1)
  }
}

// ============================================================
// FP32 → MXFP8 single-element converter
// ============================================================
/**
 * Converts one IEEE-754 FP32 value to MXFP8 (E5M2 or E4M3) given the
 * block's shared scale.
 *
 * Scale formats (selected by cfg.scaleType at elaboration time, OCP MX semantics):
 *   UE8M0 — shared_scale is a pure power-of-2 encoding of the block scale X,
 *            where 2^X · max_normal(element_format) ≈ V_max
 *            (i.e. raw max biased FP32 exp minus the element format's emax).
 *   ExMy  — shared_scale = {biased_exp[E-1:0], mant[M-1:0]} encoding the
 *            floor-to-ExMy of V_max / 2^emax_element.
 * Both encodings place V_max at the element format's max-normal value, so
 * the consumer formula  out_biased_exp = fp32_biased_exp − shared_exp + outBias
 * (UE8M0) lands the max element at outMaxNormalExp.
 *
 * Rounding : round-to-nearest-even (RNE).
 * Saturation: clamp to ±max-normal when re-biased exponent > fp8MaxNormalExp.
 * Underflow : flush to ±0 when re-biased exponent ≤ 0.
 */
class FP32ToMXFP8(val cfg: RequantConfig) extends Module {
  override def desiredName =
    s"FP32ToMXFP_${cfg.outputType.name}_${cfg.scaleType.name}"

  private val outExpBits      = cfg.outputType.elementWidthExp
  private val outMantBits     = cfg.outputType.elementWidthMant
  private val outBias         = cfg.outputType.bias
  // UE8M0 path: legacy "all-ones-minus-one" max-normal (OCP power-of-2 scale).
  private val outMaxNormalExp = (1 << outExpBits) - 2
  // ExMy path: real OCP-MX max-normal (NVFP4-style); differs for E4M3/E3M2/E2M3.
  private val (exMyMaxNormalExp, exMyMaxNormalMant) = NVFP4Limits(cfg.outputType)

  val io = IO(new Bundle {
    val fp32_in      = Input(UInt(32.W))
    val shared_scale = Input(UInt(8.W))   // always 8 bits; format depends on cfg.scaleType
    val elem_out     = Output(UInt(cfg.outputType.totalWidth.W))
  })

  // ── Unpack FP32 ──────────────────────────────────────────
  val sign     = io.fp32_in(31)
  val fp32_exp = io.fp32_in(30, 23)   // biased exponent, 0-255
  val fp32_man = io.fp32_in(22, 0)    // 23-bit fractional mantissa

  val isZeroOrSubnormal = fp32_exp === 0.U
  val fp32_exp_s        = fp32_exp.zext   // SInt(9)

  val maxNormalMant = ((1 << outMantBits) - 1).U(outMantBits.W)
  val maxNormalExpU = outMaxNormalExp.U(outExpBits.W)

  if (cfg.scaleType.mantScaleWidth == 0) {
    // ── UE8M0 path ────────────────────────────────────────
    //
    // shared_scale = (max biased FP32 exponent in the block) − emax_element.
    // out_biased_exp = fp32_biased_exp − shared_scale + outBias
    // For the block's max element this evaluates to outMaxNormalExp
    // (= outBias + emax_element), using the full element-format exp range.
    val shared_exp_s = io.shared_scale.zext     // SInt(9)
    val out_exp_full = fp32_exp_s - shared_exp_s + outBias.S

    val underflow = isZeroOrSubnormal || out_exp_full <= 0.S
    val overflow  = out_exp_full > outMaxNormalExp.S

    val out_mant_raw = fp32_man(22, 23 - outMantBits)
    val guardBit     = fp32_man(22 - outMantBits)
    val stickyBits   =
      if (22 - outMantBits > 0) fp32_man(21 - outMantBits, 0).orR
      else false.B
    val roundUp  = guardBit && (out_mant_raw(0) || stickyBits)
    val out_mant = Mux(roundUp, out_mant_raw + 1.U, out_mant_raw)

    val out_exp_clamped = out_exp_full.asUInt(outExpBits - 1, 0)
    val normalResult    = Cat(sign, out_exp_clamped, out_mant)
    val maxResult       = Cat(sign, maxNormalExpU, maxNormalMant)

    io.elem_out := Mux(underflow, 0.U, Mux(overflow, maxResult, normalResult))

  } else {
    // ── ExMy path (NVFP4-style) ────────────────────────────
    //
    // shared_scale = {biased_scale_exp[E-1:0], scale_mant[M-1:0]}
    // where E = expScaleWidth, M = mantScaleWidth, E+M = 8.
    //
    // Scale value  = (1.scale_mant) × 2^(scale_biased_exp − scale_bias)   (normal)
    //              = (0.scale_mant) × 2^(1 − scale_bias)                  (subnormal)
    // FP32 value   = (1.fp32_mant)  × 2^(fp32_biased_exp  − 127)
    // Element      = FP32 / scale
    //
    // Saturation:  element format's real OCP-MX max-normal (exMyMaxNormalExp /
    //              exMyMaxNormalMant); for E4M3 the borderline (15, 7) is
    //              clamped down to (15, 6) to avoid the single NaN pattern.
    //
    // Subnormal scale: the encoded {0, mant} is *renormalized* to an
    //              (eff_biased_exp, normalized_mant) pair so the division
    //              path can reuse the normal "(1.mant) divisor" math.

    val M          = cfg.scaleType.mantScaleWidth   // 1..6
    val scaleBias  = cfg.scaleType.bias             // (1 << (E-1)) − 1
    val correction = scaleBias - 127 + outBias       // design-time Scala Int

    val scale_biased_exp = io.shared_scale(7, M)          // E bits
    val scale_mant_raw   = io.shared_scale(M - 1, 0)      // M bits
    val isZeroScale      = !scale_biased_exp.orR && !scale_mant_raw.orR
    val isSubnormScale   = !scale_biased_exp.orR &&  scale_mant_raw.orR

    // Subnormal renormalization: shift scale_mant_raw left so its leading
    // 1 sits at the implicit-1 position.  Log2(x) returns the position of
    // the highest set bit; the Mux gates Log2(0) when the mantissa is 0.
    val msbPos    = Log2(Mux(scale_mant_raw === 0.U, 1.U(M.W), scale_mant_raw))
    val leftShift = M.U - msbPos                                          // [1, M]
    val mantSubNorm = (scale_mant_raw << leftShift)(M - 1, 0)             // M-bit
    // Effective biased exp for subnormal: 1 − leftShift ∈ [1−M, 0].
    val effBiasedExpSubn = 1.S - leftShift.zext

    val effBiasedExp_s = Mux(isSubnormScale, effBiasedExpSubn, scale_biased_exp.zext.asSInt)
    val scaleFullMant  = Mux(isSubnormScale,
                              Cat(1.U(1.W), mantSubNorm),
                              Cat(scale_biased_exp.orR, scale_mant_raw))  // (M+1)-bit

    val out_exp_raw = fp32_exp_s - effBiasedExp_s + correction.S          // SInt

    // ── Mantissa division q = (1.fp32_mant) / (1.scale_mant) ──
    val EXTRA = outMantBits + 3
    val IMPL  = 23 - M + EXTRA      // e.g. E4M3+UE7M1: 23-1+6 = 28

    val fp32FullMant = Cat(fp32_exp.orR, fp32_man)    // 24 bits (0 for zero/subnormal)
    val qNum         = Cat(fp32FullMant, 0.U(EXTRA.W))  // (24+EXTRA) bits

    val safeDenom = Mux(scaleFullMant === 0.U, 1.U((M + 1).W), scaleFullMant)
    val q_int     = qNum / safeDenom
    val q_rem     = qNum % safeDenom

    val qGeq1 = q_int(IMPL)   // 1 iff q_frac ≥ 1.0

    val out_mant_raw = Mux(
      qGeq1,
      q_int(IMPL - 1,         IMPL - outMantBits),
      q_int(IMPL - 2,         IMPL - 1 - outMantBits)
    )
    val guardBit = Mux(qGeq1,
      q_int(IMPL - outMantBits - 1),
      q_int(IMPL - outMantBits - 2))
    val roundBit = Mux(qGeq1,
      q_int(IMPL - outMantBits - 2),
      q_int(IMPL - outMantBits - 3))

    val stickyQ_geq1 = q_int(IMPL - outMantBits - 3, 0).orR
    val stickyQ_lt1  = q_int(IMPL - outMantBits - 4, 0).orR
    val stickyBit    = (q_rem =/= 0.U) || Mux(qGeq1, stickyQ_geq1, stickyQ_lt1)

    val roundUp        = guardBit && (out_mant_raw(0) || roundBit || stickyBit)
    val out_mant_carry = out_mant_raw +& roundUp        // (outMantBits+1) bits
    val mantOverflow   = out_mant_carry(outMantBits)
    val out_mant       = out_mant_carry(outMantBits - 1, 0)

    val normAdj      = Mux(qGeq1, 0.S(2.W), (-1).S(2.W))
    val out_exp_full = out_exp_raw + normAdj + mantOverflow.zext

    val overflow  = out_exp_full > exMyMaxNormalExp.S
    // E4M3 borderline: (exp=15, mant=7) is NaN → clamp to (15, 6).
    val borderNaN = (out_exp_full === exMyMaxNormalExp.S) && (out_mant > exMyMaxNormalMant.U)
    val sat       = overflow || borderNaN

    // Subnormal element output: when out_exp_full ≤ 0, encode as
    //   subn_mant × 2^(1 − outBias − outMantBits)  via biased_exp = 0.
    // We approximate by right-shifting (1.out_mant) by (1 − out_exp_full)
    // bits and rounding the shifted-out portion (guard + sticky).
    val subnMaxShift = outMantBits + 2     // larger shifts fully underflow
    val subnShiftU   = Mux(out_exp_full > 0.S, 0.U, (1.S - out_exp_full).asUInt)
    val subnShift    = subnShiftU.min(subnMaxShift.U)
    val sigInt       = Cat(1.U(1.W), out_mant)                          // (outMantBits+1)-bit
    val padded       = sigInt ## 0.U(subnMaxShift.W)                    // pad with zeros for guard/sticky
    val shiftedPad   = (padded >> subnShift).asUInt
    val subnTrunc    = shiftedPad(subnMaxShift + outMantBits - 1, subnMaxShift)
    val subnGuard    = shiftedPad(subnMaxShift - 1)
    val subnSticky   = shiftedPad(subnMaxShift - 2, 0).orR
    val subnRoundUp  = subnGuard.asBool && (subnTrunc(0) || subnSticky)
    val subnSum      = subnTrunc +& subnRoundUp.asUInt
    val subnCarryToNormal = subnSum(outMantBits)                        // rounded up into smallest normal
    val subnMant     = subnSum(outMantBits - 1, 0)

    val isSubnormalRange = (out_exp_full <= 0.S) && (subnCarryToNormal || subnMant =/= 0.U)
    val isTrueZero = isZeroOrSubnormal || isZeroScale ||
                     ((out_exp_full <= 0.S) && !subnCarryToNormal && subnMant === 0.U)

    val subnExpField  = Mux(subnCarryToNormal, 1.U(outExpBits.W), 0.U(outExpBits.W))
    val subnMantField = Mux(subnCarryToNormal, 0.U(outMantBits.W), subnMant)
    val subnResult    = Cat(sign, subnExpField, subnMantField)

    val out_exp_clamped = out_exp_full.asUInt(outExpBits - 1, 0)
    val normalResult    = Cat(sign, out_exp_clamped, out_mant)
    val maxResult       = Cat(sign,
                              exMyMaxNormalExp.U(outExpBits.W),
                              exMyMaxNormalMant.U(outMantBits.W))

    val resultPick = Mux(isSubnormalRange, subnResult, normalResult)
    io.elem_out := Mux(isTrueZero, 0.U, Mux(sat, maxResult, resultPick))
  }
}

// ============================================================
// Block shared-scale finder (combinational reduction tree)
// ============================================================
/**
 * Finds the shared scale for one block of blockSize FP32 elements.
 *
 * Two different conventions per scale type:
 *
 * UE8M0 (OCP-MX, pure power-of-2):
 *   X = clamp(max_fp32_biased_exp − emaxElem, 0, 255),
 *   with emaxElem = (1 << outExpBits) − 2 − outBias.
 *   The max element lands at biased_exp = (1 << outExpBits) − 2 in the
 *   element format (Chisel's "all-ones-minus-one" max-normal convention).
 *
 * ExMy (NVFP4-style ceil):
 *   ideal_scale = max_abs / elem_max,
 *   elem_max    = (1.elemMaxMant) × 2^E_elem (real OCP-MX max-normal value),
 *   and the result is ceil-encoded into UExMy (guarantees encoded_scale ≥
 *   ideal_scale so no element saturates).  Subnormal range clamps to the
 *   smallest non-zero subnormal {0, 1}; FP32_zero block ⇒ {0, 0}.
 */
class MaxScaleFinder(val cfg: RequantConfig) extends Module {
  override def desiredName =
    s"MaxScaleFinder_${cfg.scaleType.name}_${cfg.outputType.name}_blk${cfg.blockSize}"

  
  private val outExpBits      = cfg.outputType.elementWidthExp
  private val outBias         = cfg.outputType.bias
  private val outMaxNormalExp = (1 << outExpBits) - 2
  private val emaxElem        = outMaxNormalExp - outBias
  // Element-format emax (max representable real exponent).
  // E5M2: 30 − 15 = 15;  E4M3: 14 − 7 = 7;  E3M2: 6 − 3 = 3;  E2M3: 2 − 1 = 1.

  val io = IO(new Bundle {
    val fp32_in   = Input(Vec(cfg.blockSize, UInt(32.W)))
    val max_scale = Output(UInt(8.W))
  })

  // Generic balanced max-tree for any UInt width
  def maxTree(vals: Seq[UInt]): UInt =
    if (vals.length == 1) vals.head
    else {
      val next = vals.grouped(2).map {
        case Seq(a, b) => Mux(a >= b, a, b)
        case Seq(a)    => a
        case g         => g.head  // unreachable
      }.toSeq
      maxTree(next)
    }

  if (cfg.scaleType.mantScaleWidth == 0) {
    // ── UE8M0: (max biased FP32 exponent) − emax_element ──
    val maxBiasedExp = maxTree(io.fp32_in.map(_(30, 23)))
    val raw          = maxBiasedExp.zext - emaxElem.S
    io.max_scale := Mux(raw <= 0.S,   0.U(8.W),
                    Mux(raw >= 255.S, 255.U(8.W),
                        raw.asUInt(7, 0)))

  } else {
    // ── ExMy: NVFP4-style ceil-encode max_abs / elem_max ────
    //
    // ideal_scale = max_abs / elem_max,   elem_max = (1.maxNormalMant) × 2^E_elem
    // Result is round-toward-+∞ encoded into UExMy (matches Python
    // quantize_round_up; guarantees encoded_scale ≥ ideal_scale so no
    // element saturates after division).
    val M            = cfg.scaleType.mantScaleWidth
    val E            = cfg.scaleType.expScaleWidth
    val scaleBias    = cfg.scaleType.bias
    val maxScaleExpV = (1 << E) - 1 //15

    // NVFP4 max-normal of the element format (OCP-MX semantics).
    val outMantBits     = cfg.outputType.elementWidthMant 
    val (elemMaxExp, elemMaxMant) = NVFP4Limits(cfg.outputType)
    val E_elem          = elemMaxExp - outBias                       // E5M2=15, E4M3=8, E3M2=4, E2M3=2
    val maxNormSignifInt = (1 << outMantBits) | elemMaxMant          // (1.maxMant) as integer

    // Find FP32 with largest absolute value (sign-stripped 31-bit comparison)
    val maxMag31     = maxTree(io.fp32_in.map(_(30, 0)))
    val maxBiasedExp = maxMag31(30, 23)
    val maxMant23    = maxMag31(22, 0)
    val maxIsZero    = !maxBiasedExp.orR        // FP32 zero or subnormal block-max

    // q_int = (1.maxMant23) × 2^EXTRA / maxNormSignifInt.
    // Implicit-1 of q (when q ≥ 1) lands at IMPL = 23 − outMantBits + EXTRA.
    val EXTRA = M + 3
    val IMPL  = 23 - outMantBits + EXTRA
    val fp32FullMant = Cat(maxBiasedExp.orR, maxMant23)              // 24-bit (0 for FP32 zero/subn)
    val qNum         = Cat(fp32FullMant, 0.U(EXTRA.W))               // (24+EXTRA)-bit
    val divisorU     = maxNormSignifInt.U((outMantBits + 1).W)
    val q_int        = qNum / divisorU
    val q_rem        = qNum % divisorU

    val qGeq1 = q_int(IMPL)                                          // 1 iff q ≥ 1

    // Extract top M mant bits + ceil-round increment (any bit below cut OR
    // remainder non-zero ⇒ round up).
    val mant_raw = Mux(qGeq1,
                       q_int(IMPL - 1, IMPL - M),
                       q_int(IMPL - 2, IMPL - 1 - M))
    val resid_geq1 = q_int(IMPL - M - 1, 0).orR
    val resid_lt1  = q_int(IMPL - M - 2, 0).orR
    val ceilInc    = (q_rem =/= 0.U) || Mux(qGeq1, resid_geq1, resid_lt1)

    val mant_sum     = mant_raw +& ceilInc                           // (M+1)-bit
    val mantOverflow = mant_sum(M)
    val mant_normal  = mant_sum(M - 1, 0)

    // Scale biased exp (signed: can underflow into subnormal range).
    val scaleBiasedExp = maxBiasedExp.zext - 127.S - E_elem.S +
                         Mux(qGeq1, 0.S(2.W), (-1).S(2.W)) +
                         mantOverflow.zext +
                         scaleBias.S

    val isSat = scaleBiasedExp >= maxScaleExpV.S
    val isSub = scaleBiasedExp <= 0.S

    // Output construction:
    //   FP32 max zero/subnormal → {0, 0}
    //   Saturated               → {maxScaleExpV, all-ones-mant}
    //   Subnormal scale range   → {0, 1}  (smallest subnormal; see design doc)
    //   Otherwise normal         → {expClamped, mant_normal}
    val expOut = Mux(maxIsZero, 0.U(E.W),
                 Mux(isSat,    maxScaleExpV.U(E.W),
                 Mux(isSub,    0.U(E.W),
                               scaleBiasedExp.asUInt(E - 1, 0))))
    val mantOut = Mux(maxIsZero, 0.U(M.W),
                  Mux(isSat,    ((1 << M) - 1).U(M.W),
                  Mux(isSub,    1.U(M.W),
                                mant_normal)))

    io.max_scale := Cat(expOut, mantOut)
  }
}

// ============================================================
// One-row block requantizer (combinational)
// ============================================================
/**
 * Requantizes blockSize FP32 values for one tile row.
 * Purely combinational: MaxScaleFinder → blockSize × FP32ToMXFP8.
 */
class RequantBlock(val cfg: RequantConfig) extends Module {
  override def desiredName =
    s"RequantBlock_${cfg.outputType.name}_${cfg.scaleType.name}_blk${cfg.blockSize}"

  val io = IO(new Bundle {
    val fp32_in      = Input(Vec(cfg.blockSize, UInt(32.W)))
    val shared_scale = Output(UInt(8.W))
    val elem_out     = Output(Vec(cfg.blockSize, UInt(cfg.outputType.totalWidth.W)))
  })

  val scaleFinder = Module(new MaxScaleFinder(cfg))
  scaleFinder.io.fp32_in := io.fp32_in
  io.shared_scale        := scaleFinder.io.max_scale

  for (i <- 0 until cfg.blockSize) {
    val conv = Module(new FP32ToMXFP8(cfg))
    conv.io.fp32_in      := io.fp32_in(i)
    conv.io.shared_scale := scaleFinder.io.max_scale
    io.elem_out(i)       := conv.io.elem_out
  }
}

// ============================================================
// Top-level: buffer + controller + tileRows parallel blocks
// ============================================================
/**
 * RequantFP8: receives the full TileRows × TileCols FP32 output of
 * a PE array each valid cycle, buffers it, and fires MXFP8
 * requantization once blockSize elements per row have arrived.
 *
 * The shared-scale format is selected at elaboration time via
 * cfg.scaleType (UE8M0, UE7M1, UE6M2, …).
 *
 * After batchesPerBlock = blockSize / tileCols valid pulses, valid_out
 * fires for one cycle:
 *   shared_scale_out[i]  — 8-bit scale for row i (format per cfg.scaleType)
 *   elem_out[i][k]       — k-th MX element of row i
 */
class RequantFP8(val cfg: RequantConfig) extends Module {
  override def desiredName =
    s"requant"

  private val B     = cfg.batchesPerBlock
  private val elemW = cfg.outputType.totalWidth
  private val nIn   = cfg.tileRows * cfg.tileCols

  val io = IO(new Bundle {
    val fp32_in          = Input(UInt((nIn * 32).W))
    val valid_in         = Input(Bool())
    val shared_scale_out = Output(UInt((cfg.tileRows * 8).W))
    val elem_out         = Output(UInt((cfg.tileRows * cfg.blockSize * elemW).W))
    val valid_out        = Output(Bool())
  })

  def extractFP32(row: Int, col: Int): UInt = {
    val k = row * cfg.tileCols + col
    io.fp32_in(nIn * 32 - k * 32 - 1, nIn * 32 - (k + 1) * 32)
  }

  // Active-low async reset — matches FusedDotProductUnit register convention.
  // Registers reset when reset=false (rst_ni=0 in SV), not when reset=true.
  val asyncRstN = (!reset.asBool).asAsyncReset

  val buffer   = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(32.W)))))
  val batchCnt = withReset(asyncRstN)(RegInit(0.U(6.W)))
  val blockDone = io.valid_in && (batchCnt === (B - 1).U)

  when(io.valid_in) {
    for (batch <- 0 until B) {
      when(batchCnt === batch.U) {
        for (row <- 0 until cfg.tileRows; col <- 0 until cfg.tileCols)
          buffer(row)(batch * cfg.tileCols + col) := extractFP32(row, col)
      }
    }
    batchCnt := Mux(blockDone, 0.U, batchCnt + 1.U)
  }

  val sharedScaleWire = Wire(Vec(cfg.tileRows, UInt(8.W)))
  val elemWire        = Wire(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(elemW.W))))

  for (row <- 0 until cfg.tileRows) {
    val rq = Module(new RequantBlock(cfg))
    for (col <- 0 until cfg.blockSize) {
      val batchIdx   = col / cfg.tileCols
      val colInBatch = col % cfg.tileCols
      rq.io.fp32_in(col) := (
        if (batchIdx == B - 1)
          Mux(blockDone, extractFP32(row, colInBatch), buffer(row)(col))
        else
          buffer(row)(col)
      )
    }
    sharedScaleWire(row) := rq.io.shared_scale
    elemWire(row)        := rq.io.elem_out
  }

  val validOutReg    = withReset(asyncRstN)(RegNext(blockDone, init = false.B))
  val sharedScaleReg = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, UInt(8.W))))
  val elemOutReg     = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(elemW.W)))))

  when(blockDone) {
    sharedScaleReg := sharedScaleWire
    elemOutReg     := elemWire
  }

  io.valid_out        := validOutReg
  // Pack row 0 in LSB so little-endian memory stores [row0, row1, …, rowN-1]
  // in ascending byte address (matches software golden).
  io.shared_scale_out := Cat(sharedScaleReg.reverse)
  io.elem_out := Cat(
     for (row <- 0 until cfg.tileRows; col <- 0 until cfg.blockSize)
      yield elemOutReg(row)(col)
  )
}

// ============================================================
// Emission helpers
// ============================================================

/** Emit the three baseline UE8M0 configs. */
object RequantFP8Main extends App {
  import DefaultRequantConfigs._

  Seq(e5m2_block32_4x4, e4m3_block16_8x8, e4m3_block64_4x4).foreach { cfg =>
    println(s"Generating RequantFP8: ${cfg.outputType.name} ${cfg.scaleType.name} " +
            s"blk${cfg.blockSize} ${cfg.tileRows}x${cfg.tileCols}")
    emitVerilog(
      new RequantFP8(cfg),
      Array("--target-dir",
            s"generated/requant/${cfg.outputType.name}_${cfg.scaleType.name}" +
            s"_blk${cfg.blockSize}_${cfg.tileRows}x${cfg.tileCols}")
    )
  }
}

/** Emit the baseline UE8M0 FP6 configs (E3M2 and E2M3). */
object RequantFP6Main extends App {
  import DefaultRequantConfigs._

  Seq(e3m2_block32_4x4, e3m2_block32_8x8, e2m3_block32_4x4, e2m3_block32_8x8).foreach { cfg =>
    println(s"Generating RequantFP6: ${cfg.outputType.name} ${cfg.scaleType.name} " +
            s"blk${cfg.blockSize} ${cfg.tileRows}x${cfg.tileCols}")
    emitVerilog(
      new RequantFP8(cfg),
      Array("--target-dir",
            s"generated/requant/${cfg.outputType.name}_${cfg.scaleType.name}" +
            s"_blk${cfg.blockSize}_${cfg.tileRows}x${cfg.tileCols}")
    )
  }
}

/** Emit Verilog for a matrix of output-type × scale-type × geometry combinations. */
object AllRequantFP8Main extends App {
  import mx.mac.{MXFormats, ScaleFormats}

  // (elemType, scaleType, blockSize, tileRows, tileCols)
  val configs = Seq(
    // FP8 variants
    (MXFormats.E5M2, ScaleFormats.UE8M0, 32, 4, 4),
    (MXFormats.E4M3, ScaleFormats.UE8M0, 16, 8, 8),
    (MXFormats.E4M3, ScaleFormats.UE8M0, 64, 4, 4),
    (MXFormats.E5M2, ScaleFormats.UE7M1, 32, 4, 4),
    (MXFormats.E4M3, ScaleFormats.UE7M1, 64, 4, 4),
    (MXFormats.E5M2, ScaleFormats.UE6M2, 32, 4, 4),
    (MXFormats.E4M3, ScaleFormats.UE6M2, 64, 4, 4),
    (MXFormats.E5M2, ScaleFormats.UE4M4, 32, 4, 4),
    (MXFormats.E4M3, ScaleFormats.UE4M4, 64, 4, 4),
    // FP6 variants
    (MXFormats.E3M2, ScaleFormats.UE8M0, 32, 4, 4),
    (MXFormats.E3M2, ScaleFormats.UE8M0, 32, 8, 8),
    (MXFormats.E2M3, ScaleFormats.UE8M0, 32, 4, 4),
    (MXFormats.E2M3, ScaleFormats.UE8M0, 32, 8, 8),
    (MXFormats.E3M2, ScaleFormats.UE7M1, 32, 4, 4),
    (MXFormats.E2M3, ScaleFormats.UE7M1, 32, 4, 4),
  )

  for ((elemType, scaleType, blockSize, tileRows, tileCols) <- configs) {
    val cfg = RequantConfig(blockSize, tileRows, tileCols, elemType, scaleType)
    println(s"Generating: ${elemType.name} ${scaleType.name} blk${blockSize} ${tileRows}x${tileCols}")
    emitVerilog(
      new RequantFP8(cfg),
      Array("--target-dir",
            s"generated/requant/${elemType.name}_${scaleType.name}" +
            s"_blk${blockSize}_${tileRows}x${tileCols}")
    )
  }
}
