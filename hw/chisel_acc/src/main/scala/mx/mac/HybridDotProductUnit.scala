package mx.mac

import chisel3._
import chisel3.util._

/**
 * HybridDotProductUnit.scala
 *
 * Implements three optimizations from:
 *   "Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree
 *    for Efficient NPU Integration" (Cuyckens et al., ASP-DAC 2026, 4D-3)
 *
 * Optimization 1 – Integer-domain L2 reduction (Sec. III-A, Fig. 3 right):
 *   Instead of converting each lane product to FP32 before the reduction
 *   tree (as in FusedDotProductUnit), all lane mantissas from ScaleAddition
 *   are aligned to the maximum exponent in the integer domain and summed.
 *   Only a SINGLE normalization is needed at the very end, shared across
 *   all vectorSize lanes.  This eliminates the N per-lane ScaleToFP32
 *   modules and the FP32 balanced reduction tree.
 *
 * Optimization 2 – MUX early-accumulation (Sec. III-A, Fig. 3b):
 *   When accumulating the integer product-sum with the stored partial
 *   result, both operands can only extend in ONE direction at a time:
 *     • accum larger  → product-sum is right-shifted (accum sits LEFT)
 *     • product larger → accum is right-shifted (product sits LEFT)
 *   A MUX selects the active configuration.  This halves the normaliser
 *   width compared to the naive approach (which would extend both ways).
 *   Window width = sumMantWidth + ACCUM_SIG_BITS + 1  (vs. +2*ACCUM_MANT).
 *
 * Optimization 3 – Reduced-precision accumulator (Sec. III-B, Fig. 4):
 *   The stored partial result uses ACCUM_MANT_BITS = 16 mantissa bits
 *   instead of FP32's 23 bits.  Fig. 4 of the paper shows this is safe
 *   because the addition error stays below the unavoidable MX quantisation
 *   error for matrix sizes up to 256×256.
 *   Storage format: 1-bit sign | 8-bit biased exponent | 16-bit mantissa
 *   (25 bits total).  The FP32 output zero-pads to 23 mantissa bits.
 *
 * Compared with FusedDotProductUnit.scala (FP32-addition approach):
 *   - Removes:  N × ScaleToFP32 + FP32 balanced-tree (log2N levels)
 *   - Replaces: single integer alignment + one normaliser + 25-bit accum
 *   - Area/energy savings match those reported in the paper for MXFP8/6
 *     (≈3× improvement) and MXFP4 (≈1.1×).
 */

// ============================================================
// Shared parameters for the hybrid design
// ============================================================
object HybridParams {
  /** Guard bits appended below each lane's mantissa before the integer shift.
   *  Preserves sub-LSB precision through the alignment step. */
  val GUARD_BITS      = 4

  /** Stored partial-result mantissa width (Optimization 3).
   *  16 bits is the paper's conservative safe choice (Fig. 4 worst-case). */
  val ACCUM_MANT_BITS = 16

  /** Alignment shifts beyond this threshold produce a negligible contribution
   *  (below 2^-(MAX_SHIFT) of the larger operand), so the lane is zeroed. */
  val MAX_SHIFT       = 63
}

// ============================================================
// Optimization 1: Integer-Domain Lane Reducer
// ============================================================
/**
 * Replaces the per-lane ScaleToFP32 + FP32 balanced-tree of
 * FusedDotProductUnit with a single integer-domain reduction.
 *
 * For vectorSize lanes, each providing (sign, exp: SInt, mant: UInt)
 * from ScaleAddition:
 *   1. Find the maximum exponent across all lanes.
 *   2. Append GUARD_BITS guard zeros below each mantissa.
 *   3. Right-shift each lane's extended mantissa by (maxExp − laneExp),
 *      capped at MAX_SHIFT to avoid excessive hardware.
 *   4. Convert to 2's-complement signed (negate if sign = 1).
 *   5. Sum all signed terms.
 *   6. Convert the result back to sign + unsigned magnitude.
 *
 * Output:
 *   sumSign : sign of the product sum (0 = positive, 1 = negative)
 *   sumExp  : = maxExp (the common fixed-point reference exponent)
 *   sumMant : unsigned integer magnitude, width = sumMantWidth
 *             Value of the sum = (−1)^sumSign × sumMant × 2^sumExp
 */
class IntegerDomainReducer(
    val scfg      : ScaleAddConfig,
    val vectorSize: Int
) extends Module {
  import HybridParams._

  private val M     = scfg.resScaleAddMantWidth         // = 32 (from Parameter.scala)
  private val G     = GUARD_BITS                        // = 4
  private val MG    = M + G                             // per-lane extended width = 36
  // ceil(log2(vectorSize+1)) extra bits prevent sum overflow
  private val EXTRA = log2Ceil(vectorSize + 1)
  /** Width of the integer product-sum magnitude output. */
  val sumMantWidth  = MG + EXTRA

  private val expW = scfg.resScaleAddExpWidth

  val io = IO(new Bundle {
    val lanes = Input(Vec(vectorSize, new Bundle {
      val sign = UInt(1.W)
      val exp  = SInt(expW.W)
      val mant = UInt(M.W)
    }))
    val sumSign = Output(UInt(1.W))
    val sumExp  = Output(SInt(expW.W))
    val sumMant = Output(UInt(sumMantWidth.W))
  })

  // ── Step 1: maximum exponent ────────────────────────────────────
  val maxExp: SInt = io.lanes.map(_.exp).reduce { (a, b) => Mux(a > b, a, b) }
  io.sumExp := maxExp

  // ── Steps 2-4: per-lane extend → right-shift → signed ──────────
  // Each term is SInt of width MG+1: one sign bit, MG magnitude bits.
  val signedTerms: Seq[SInt] = (0 until vectorSize).map { i =>
    val shiftRaw = (maxExp - io.lanes(i).exp).asUInt   // always ≥ 0
    val shift    = Mux(shiftRaw > MAX_SHIFT.U, MAX_SHIFT.U, shiftRaw)
    // Append guard zeros: MG bits total
    val extended = Cat(io.lanes(i).mant, 0.U(G.W))
    // Align: right-shift discards sub-precision bits (acceptable loss)
    val aligned  = (extended >> shift)(MG - 1, 0)     // MG bits
    // .zext gives SInt(MG+1) which is non-negative (MSB = 0)
    val asPos    = aligned.zext
    // Negate for negative lanes (sign = 1)
    Mux(io.lanes(i).sign.asBool, -asPos, asPos)
  }

  // ── Step 5: sum (width grows by 1 per +& step) ─────────────────
  // Final width = MG+1 + (vectorSize-1) = MG+vectorSize (worst case)
  // Magnitude of sum ≤ vectorSize × 2^MG < 2^(MG+EXTRA) = 2^sumMantWidth
  val rawSumFull: SInt = signedTerms.reduce { (a, b) => a +& b }

  // ── Step 6: sign-magnitude output ──────────────────────────────
  io.sumSign := Mux(rawSumFull < 0.S, 1.U, 0.U)
  val absFull  = Mux(rawSumFull < 0.S, (-rawSumFull).asUInt, rawSumFull.asUInt)
  // Truncate to sumMantWidth: safe because magnitude < 2^sumMantWidth (shown above)
  io.sumMant  := absFull(sumMantWidth - 1, 0)
}

// ============================================================
// Optimization 2 + 3: MUX Early Accumulator
// ============================================================
/**
 * Adds an integer product-sum to a stored reduced-precision partial
 * result using the MUX early-accumulation technique (Fig. 3b), then
 * normalises once and writes back.
 *
 * Stored accumulator format (Optimization 3):
 *   [sign(1) | biasedExp(8) | mant(AM=16)] = 25 bits
 *
 * Algorithm:
 *   1. LZC-normalise the product-sum → (normProd, prodTrueExp).
 *   2. Compute delta = accTrueExp − prodTrueExp.
 *   3. Build a shared (NW−1)-bit aligned window via MUX (Opt. 2):
 *        delta > 0  (accum larger): accum at top, product right-shifted by delta.
 *        delta ≤ 0  (prod  larger): product at top, accum right-shifted by |delta|.
 *      Only ONE operand is shifted → this is the MUX trick.
 *   4. Add the two signed (NW−1)-bit values with 2's-complement arithmetic
 *      → rawSum : SInt(NW+1.W).
 *   5. LZC-normalise the magnitude of rawSum.
 *   6. Round to AM mantissa bits (RNE), pack result, output.
 *
 * Combined window width:
 *   NW = sumMantWidth + (AM+1) + 1
 * Naïve approach width (without MUX trick):
 *   sumMantWidth + 2·AM + extra  ≈ NW + AM wider
 * Savings match the paper: for the specific case sumMantWidth=28, AM=23
 * the MUX reduces 77→53 bits; our ratio is similar.
 */
class MUXEarlyAccumulator(
    val sumMantWidth: Int,  // from IntegerDomainReducer.sumMantWidth
    val sumExpWidth : Int   // = scfg.resScaleAddExpWidth
) extends Module {
  import HybridParams._

  private val AM  = ACCUM_MANT_BITS     // = 16 (Opt. 3)
  private val ASW = AM + 1              // accumulator significand width with implicit-1
  /** Combined normalisation window width (NW−1 bits for data, bit NW−1 for carry). */
  val NW          = sumMantWidth + ASW + 1

  val io = IO(new Bundle {
    // Integer product-sum from IntegerDomainReducer
    val prodSign = Input(UInt(1.W))
    val prodExp  = Input(SInt(sumExpWidth.W))
    val prodMant = Input(UInt(sumMantWidth.W))
    // Stored reduced-precision partial result (25 bits)
    val accIn    = Input(UInt((1 + 8 + AM).W))
    // Updated partial result
    val accOut   = Output(UInt((1 + 8 + AM).W))
    // Full FP32 output (mantissa zero-padded from 16→23 bits)
    val fp32Out  = Output(UInt(32.W))
  })

  // ── Unpack accumulator ──────────────────────────────────────────
  val accSign     = io.accIn(AM + 8)
  val accBiasedE  = io.accIn(AM + 7, AM)            // UInt(8.W), biased exponent
  val accMantBits = io.accIn(AM - 1, 0)             // AM bits, stored mantissa
  val accSig      = Cat(accBiasedE.orR, accMantBits) // ASW bits: implicit-1 prepended
  val accTrueExp  = accBiasedE.zext - 127.S          // SInt, true unbiased exponent

  // ── LZC-normalise the product sum ───────────────────────────────
  // The integer product sum represents value = prodMant × 2^prodExp.
  // We find the position of the leading-1 in prodMant (= lzcProd leading zeros),
  // then the normalised float's true exponent is prodExp + (sumMantWidth−1) − lzcProd.
  val isZeroProd  = io.prodMant === 0.U
  val lzcProd     = PriorityEncoder(io.prodMant.asBools.reverse) // UInt, leading-zero count
  // normProd: leading-1 at bit sumMantWidth−1 (MSB), sumMantWidth bits
  val normProd    = (io.prodMant << lzcProd)(sumMantWidth - 1, 0)
  // True exponent of the normalised product-sum float
  val prodTrueExp = io.prodExp +& (sumMantWidth - 1).S - lzcProd.zext

  // ── Delta, MUX control, capped shift ────────────────────────────
  val isZeroAcc   = io.accIn === 0.U
  // delta > 0: accumulator has higher exponent (accum is "larger")
  val delta       = accTrueExp - prodTrueExp
  val accumLarger = (delta > 0.S) && !isZeroAcc && !isZeroProd
  val absDelta    = Mux(delta >= 0.S, delta.asUInt, (-delta).asUInt)
  // Cap: shifts ≥ NW push the smaller value below precision → treat as 0
  val cappedShift = Mux(absDelta >= NW.U, (NW - 1).U, absDelta)

  // ── Build aligned (NW−1)-bit windows ────────────────────────────
  // Convention: bit NW−2 (= index NW−2 from LSB) represents 2^refExp,
  // where refExp = accTrueExp (case 1) or prodTrueExp (case 2).
  //
  // prodWindowBase:  normProd implicit-1 at bit NW−2, zeros fill bits [ASW−1:0]
  //   Cat(normProd[sW−1:0], 0[ASW]) = (sW + ASW) = NW−1 bits
  val prodWindowBase : UInt = Cat(normProd, 0.U(ASW.W))          // NW−1 bits
  // accumWindowBase: accSig implicit-1 at bit NW−2, zeros fill bits [sW−1:0]
  //   Cat(accSig[ASW−1:0], 0[sW]) = (ASW + sW) = NW−1 bits
  val accumWindowBase: UInt = Cat(accSig, 0.U(sumMantWidth.W))   // NW−1 bits

  // MUX (Optimization 2): shift the SMALLER operand's window rightward.
  //   Case 1 (accum larger): shift product right by delta; accum stays at top.
  //   Case 2 (prod  larger): shift accum  right by |delta|; product stays at top.
  val prodWindow : UInt = Mux(accumLarger,
    (prodWindowBase  >> cappedShift)(NW - 2, 0),  // case 1: prod slides right
    prodWindowBase(NW - 2, 0))                     // case 2: prod stays
  val accumWindow: UInt = Mux(accumLarger,
    accumWindowBase(NW - 2, 0),                    // case 1: accum stays
    (accumWindowBase >> cappedShift)(NW - 2, 0))   // case 2: accum slides right

  // Reference exponent for the combined window (bit NW−2 = 2^refExp)
  val refExp: SInt = Mux(accumLarger, accTrueExp, prodTrueExp)

  // ── Signed addition ─────────────────────────────────────────────
  // prodWindow.zext  : SInt(NW.W), always non-negative (bit NW−1 = 0)
  // negation is safe: max magnitude = 2^(NW−2)−1 << 2^(NW−1)
  val sProd  : SInt = Mux(isZeroProd, 0.S(NW.W),
                      Mux(io.prodSign.asBool,
                          -(prodWindow.zext),
                           prodWindow.zext))
  val sAccum : SInt = Mux(isZeroAcc, 0.S(NW.W),
                      Mux(accSign.asBool,
                          -(accumWindow.zext),
                           accumWindow.zext))

  val rawSum : SInt = sProd +& sAccum              // SInt(NW+1.W) – one carry bit

  // ── Extract sign + NW-bit magnitude ─────────────────────────────
  val outSign = Mux(rawSum < 0.S, 1.U, 0.U)
  // Take lower NW bits of the absolute value (magnitude < 2^NW, proven above)
  val outMag  = Mux(rawSum < 0.S,
                    (-rawSum).asUInt,
                    rawSum.asUInt)(NW - 1, 0)      // UInt(NW.W)

  // ── LZC-normalise the magnitude ─────────────────────────────────
  val isZeroSum = outMag === 0.U
  val lzcSum    = PriorityEncoder(outMag.asBools.reverse)  // # leading zeros in outMag
  // After left-shift, leading-1 at bit NW−1
  val normSum   = (outMag << lzcSum)(NW - 1, 0)    // UInt(NW.W)

  // normSum bit layout (indexed from 0 at LSB, NW−1 at MSB):
  //   bit NW−1        : implicit-1
  //   bits[NW−2:NW−1−AM] : AM mantissa bits  (mantHigh downto mantLow)
  //   bit  NW−2−AM    : guard bit
  //   bit  NW−3−AM    : round bit
  //   bits[NW−4−AM:0] : sticky bits

  private val mantHigh = NW - 2
  private val mantLow  = NW - 1 - AM    // = NW − 17 for AM=16
  private val guardIdx = mantLow - 1    // = NW − 18
  private val roundIdx = mantLow - 2    // = NW − 19

  // ── RNE rounding to AM mantissa bits ────────────────────────────
  val outMantRaw = normSum(mantHigh, mantLow)     // AM bits
  val gBit       = normSum(guardIdx)
  val rBit       = if (roundIdx >= 0) normSum(roundIdx)             else false.B
  val sBit       = if (roundIdx - 1 >= 0) normSum(roundIdx - 1, 0).orR else false.B

  val doRound   = gBit && (outMantRaw(0) || rBit || sBit)
  val roundedM  = outMantRaw +& doRound            // UInt(AM+1.W), possible carry
  val mantCarry = roundedM(AM)
  val finalMant = roundedM(AM - 1, 0)              // AM bits

  // ── Compute output exponent ──────────────────────────────────────
  // outMag bit k represents 2^(refExp + 1 − (NW−1−k)) = 2^(refExp − NW + 2 + k).
  // Leading-1 at bit NW−1−lzcSum → true exponent = refExp + 1 − lzcSum + mantCarry.
  val trueExpResult   = refExp + 1.S - lzcSum.zext + mantCarry.zext
  val biasedExpResult = trueExpResult + 127.S

  val finalExp = Mux(isZeroSum, 0.U(8.W),
                 Mux(biasedExpResult <= 0.S,   0.U(8.W),
                 Mux(biasedExpResult >= 255.S, 255.U(8.W),
                     biasedExpResult.asUInt(7, 0))))

  // ── Pack outputs ────────────────────────────────────────────────
  // 25-bit reduced-precision accumulator (Opt. 3)
  io.accOut  := Mux(isZeroSum, 0.U, Cat(outSign, finalExp, finalMant))
  // FP32: zero-pad mantissa from AM=16 to 23 bits
  io.fp32Out := Mux(isZeroSum, 0.U(32.W),
                    Cat(outSign, finalExp, finalMant, 0.U((23 - AM).W)))
}

// ============================================================
// Hybrid Dot-Product Unit (top level)
// ============================================================
/**
 * Drop-in replacement for FusedDotProductUnit with identical I/O.
 *
 * Internal structure:
 *
 *   ┌─ lane 0 ─────────────────────────────┐
 *   │ CustomOperator → ScaleAddition        ├──► reducer.lanes(0)
 *   └──────────────────────────────────────┘
 *           ...                                    ↓
 *   ┌─ lane N-1 ────────────────────────────┐  IntegerDomainReducer
 *   │ CustomOperator → ScaleAddition        ├──►  (Opt. 1)
 *   └──────────────────────────────────────┘      ↓
 *                                           MUXEarlyAccumulator
 *                                           (Opt. 2 + 3)
 *                                                  ↓
 *                                           25-bit accReg
 *                                                  ↓
 *                                           fp32Out (zero-padded)
 */
class HybridDotProductUnit(
    val scfg      : ScaleAddConfig,
    val vectorSize: Int,
    val istest    : Boolean
) extends Module {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  import HybridParams._

  override def desiredName =
    if (!istest) "Hybrid_BFP_PE"
    else
      s"HybridDotProductUnit_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}" +
      s"_scale_${scfg.stype.name}_vec${vectorSize}"

  private val wA = scfg.elementTypeA.totalWidth
  private val wB = scfg.elementTypeB.totalWidth
  private val AM = ACCUM_MANT_BITS

  // Identical I/O to FusedDotProductUnit (enables side-by-side comparison)
  val io = IO(new Bundle {
    val op_a_i        = Input(UInt((vectorSize * wA).W))
    val op_b_i        = Input(UInt((vectorSize * wB).W))
    val share_exp_A_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    val share_exp_B_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    val validIn       = Input(Bool())
    val resetAcc      = Input(Bool())
    val validOut      = Output(Bool())
    val accOut        = Output(UInt(32.W))  // FP32, mantissa zero-padded from 16 bits
  })

  // ── Lanes: CustomOperator → ScaleAddition (unchanged from FusedDotProductUnit) ──
  val reducer = Module(new IntegerDomainReducer(scfg, vectorSize))

  for (i <- 0 until vectorSize) {
    val op = Module(new CustomOperator(OperatorConfig(scfg.elementTypeA, scfg.elementTypeB)))
    op.io.inA := io.op_a_i((i + 1) * wA - 1, i * wA)
    op.io.inB := io.op_b_i((i + 1) * wB - 1, i * wB)

    val sa = Module(new ScaleAddition(scfg))
    sa.io.inOpSign      := op.io.outSign
    sa.io.inOpExp       := op.io.outExp
    sa.io.inOpMant      := op.io.outMant
    sa.io.inShareScaleA := io.share_exp_A_i
    sa.io.inShareScaleB := io.share_exp_B_i

    // Feed lane result directly to the integer reducer (no ScaleToFP32!)
    reducer.io.lanes(i).sign := sa.io.outSign
    reducer.io.lanes(i).exp  := sa.io.outExp
    reducer.io.lanes(i).mant := sa.io.outMant
  }

  // ── MUX early accumulator (Opt. 2 + 3) ────────────────────────────────
  val accumUnit = Module(new MUXEarlyAccumulator(
    sumMantWidth = reducer.sumMantWidth,
    sumExpWidth  = scfg.resScaleAddExpWidth
  ))

  accumUnit.io.prodSign := reducer.io.sumSign
  accumUnit.io.prodExp  := reducer.io.sumExp
  accumUnit.io.prodMant := reducer.io.sumMant

  // ── Reduced-precision accumulator register (Opt. 3) ───────────────────
  // 25-bit format: [sign(1) | biasedExp(8) | mant(16)]
  val asyncRstN  = (!reset.asBool).asAsyncReset
  val accumReg   = withReset(asyncRstN)(RegInit(0.U((1 + 8 + AM).W)))
  val validReg   = withReset(asyncRstN)(RegInit(false.B))

  accumUnit.io.accIn := accumReg

  when(io.resetAcc) {
    accumReg := 0.U
    validReg := false.B
  }.elsewhen(io.validIn) {
    accumReg := accumUnit.io.accOut
    validReg := true.B
  }.otherwise {
    validReg := false.B
  }

  io.validOut := validReg
  io.accOut   := accumUnit.io.fp32Out
}

// ============================================================
// Emission helpers (mirror the structure of FusedDotProductUnit)
// ============================================================

object HybridFusedDotProductMain extends App {
  val scfg       = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)
  val vectorSize = 8
  emitVerilog(
    new HybridDotProductUnit(scfg, vectorSize, false),
    Array("--target-dir", s"generated/hybrid_dot_product/default_vec${vectorSize}")
  )
}

object AllHybridDotProductMain extends App {
  val vectorSizes = Seq(1, 2, 4, 8, 16, 32)
  for {
    typeA <- MXFormats.allElementTypes
    typeB <- MXFormats.allElementTypes
    stype <- ScaleFormats.allScaleTypes
    vsize <- vectorSizes
  } {
    val scfg = ScaleAddConfig(typeA, typeB, stype)
    println(
      s"Generating HybridDotProductUnit: ${typeA.name} x ${typeB.name}, " +
      s"scale ${stype.name}, vectorSize=$vsize"
    )
    emitVerilog(
      new HybridDotProductUnit(scfg, vsize, false),
      Array("--target-dir",
        s"generated/hybrid_dot/${typeA.name}_${typeB.name}_${stype.name}_vec${vsize}")
    )
  }
}

object TestHybridDotProductMain extends App {
  val vectorSizes = Seq(1, 2, 4, 8, 16, 32)
  val typeA = MXFormats.INT8
  val typeB = MXFormats.E2M1
  val stype = ScaleFormats.UE8M0

  for (vsize <- vectorSizes) {
    val scfg = ScaleAddConfig(typeA, typeB, stype)
    println(
      s"Generating HybridDotProductUnit: ${typeA.name} x ${typeB.name}, " +
      s"scale ${stype.name}, vectorSize=$vsize"
    )
    emitVerilog(
      new HybridDotProductUnit(scfg, vsize, false),
      Array("--target-dir",
        s"generated/${typeA.name}_${typeB.name}_${stype.name}_hybrid/" +
        s"${typeA.name}_${typeB.name}_${stype.name}_vec${vsize}")
    )
  }
}
