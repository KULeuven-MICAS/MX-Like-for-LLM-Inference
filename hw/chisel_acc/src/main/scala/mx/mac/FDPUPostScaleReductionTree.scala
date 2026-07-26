package mx.mac

import chisel3._
import chisel3.util.Cat

// ============================================================
// Fused Dot-Product Unit — Post-Scale Reduction Tree
// ============================================================
/** Fused dot-product unit: MAC → FixedFP Reduction Tree → single ScaleAddition → FP Accumulator.
 *
 *  Pipeline (fully combinational within each cycle):
 *
 *    ┌─ lane 0 ─┐
 *    │ CustomOp ├──►┐
 *    └──────────┘   │   ┌───────────────────┐   ┌────────────────────┐   ┌────────────┐
 *        ...        ├──►│  FixedFP Reduction ├──►│ Single ScaleAdd +  ├──►│ FP Accum   │
 *    ┌─ lane N-1 ─┐ │   │  Tree              │   │  ScaleToFP32       │   │ (accMantBits)├──► accOut
 *    │ CustomOp  ├──►┘   └───────────────────┘   └────────────────────┘   └────────────┘
 *    └───────────┘
 *
 *  Reduction tree design (replaces per-node CustomFPAdder tree):
 *    - One maxExp comparator tree across N lanes
 *    - N bounded alignment right-shifts (≤ productExpRange positions)
 *    - (N−1) plain 2's-complement integer adders
 *    - One LZC + normalisation shift + RNE round at the output
 *
 *  Correctness: sa/sb are shared across all lanes, so by the distributive property:
 *    Σᵢ(macᵢ × sa × sb) = (Σᵢ macᵢ) × sa × sb
 *
 *  Accumulator precision:
 *    The accumulator register stores a reduced-precision FP value with `accMantBits`
 *    mantissa bits (default: computed by AccPrecision.recommended from K and scfg).
 *    This is sufficient because accumulation noise stays below the requant noise floor
 *    when accMantBits ≥ rqFloor + ½·log₂(K) (see AccPrecision for derivation).
 *    accOut zero-extends to 32 bits for downstream compatibility.
 *
 *  @param scfg        ScaleAddConfig describing element and scale types.
 *  @param vectorSize  Number of parallel MACs per cycle (>= 1).
 *  @param K           Accumulation depth (number of cycles per output element).
 *                     Used to derive the default accMantBits at elaboration time.
 *  @param accMantBits Accumulator mantissa bits. -1 = auto (AccPrecision.recommended).
 *                     Override to 23 for full FP32 accumulation.
 *  @param istest      Enable debug ports for simulation visibility.
 */
class FDPUPostScaleReductionTree(
  val scfg: ScaleAddConfig,
  val vectorSize: Int,
  val K: Int = 32,
  val accMantBits: Int = -1,
  val istest: Boolean = false
) extends Module {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  require(K >= 1, "K must be >= 1")

  val actualAccMantBits: Int =
    if (accMantBits == -1) AccPrecision.recommended(scfg, K)
    else { require(accMantBits >= 1 && accMantBits <= 23); accMantBits }

  // Internal FP word width used throughout the post-tree pipeline
  private val fpNW = 1 + 8 + actualAccMantBits

  private val wA = scfg.elementTypeA.totalWidth
  private val wB = scfg.elementTypeB.totalWidth

  override def desiredName =
    if (!istest)
      s"BFP_PE"
    else
      s"FDPUPostScale_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}" +
      s"_scale_${scfg.stype.name}_vec${vectorSize}_K${K}_acc${actualAccMantBits}b"

  val io = IO(new Bundle {
    val op_a_i        = Input(UInt((vectorSize * wA).W))
    val op_b_i        = Input(UInt((vectorSize * wB).W))
    val share_exp_A_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    val share_exp_B_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    val validIn  = Input(Bool())
    val resetAcc = Input(Bool())
    val validOut = Output(Bool())
    // FP32 output: {sign[1], exp[8], mant[23]}. The accumulator register stores
    // (1+8+actualAccMantBits) bits; the mantissa is zero-extended (right-padded)
    // to 23 bits so downstream requant blocks receive a valid IEEE-754 single
    // value with no rounding loss beyond what FPNAdder already produced.
    val accOut   = Output(UInt(32.W))

    // ── Debug ports (test mode only) ────────────────────────────────────────
    val debug = if (istest) Some(new Bundle {
      // Per-lane CustomOperator outputs (tree inputs)
      val all_lanes_op_sign = Output(Vec(vectorSize, UInt(1.W)))
      val all_lanes_op_exp  = Output(Vec(vectorSize, SInt(scfg.resOperatorExpWidth.W)))
      val all_lanes_op_mant = Output(Vec(vectorSize, UInt(scfg.resOperatorMantWidth.W)))
      // FixedFP reduction tree output (single CustomFP at resOperatorMantWidth precision)
      val tree_out_sign     = Output(UInt(1.W))
      val tree_out_exp      = Output(SInt(scfg.resOperatorExpWidth.W))
      val tree_out_mant     = Output(UInt(scfg.resOperatorMantWidth.W))
      // ScaleAddition output
      val sa_out_sign       = Output(UInt(1.W))
      val sa_out_exp        = Output(SInt(scfg.resScaleAddExpWidth.W))
      val sa_out_mant       = Output(UInt(scfg.resScaleAddMantWidth.W))
      // FP32 result after ScaleToFP32
      val reducedSum        = Output(UInt(32.W))
    }) else None
  })

  // ── Per-lane MAC ─────────────────────────────────────────────────────────
  val laneOp = Wire(Vec(vectorSize, new CustomFP(scfg.resOperatorExpWidth, scfg.resOperatorMantWidth)))

  for (i <- 0 until vectorSize) {
    val op = Module(new CustomOperator(OperatorConfig(scfg.elementTypeA, scfg.elementTypeB)))
    op.io.inA := io.op_a_i((i + 1) * wA - 1, i * wA)
    op.io.inB := io.op_b_i((i + 1) * wB - 1, i * wB)

    laneOp(i).sign := op.io.outSign
    laneOp(i).mant := op.io.outMant
    laneOp(i).exp  := op.io.outExp

    io.debug.foreach { d =>
      d.all_lanes_op_sign(i) := op.io.outSign
      d.all_lanes_op_exp(i)  := op.io.outExp
      d.all_lanes_op_mant(i) := op.io.outMant
    }
  }

  // ── FixedFP reduction tree ────────────────────────────────────────────────
  val tree = Module(new FixedFPReductionTree(
    expW           = scfg.resOperatorExpWidth,
    inMantW        = scfg.resOperatorMantWidth,
    outMantW       = scfg.resOperatorMantWidth,
    vectorSize     = vectorSize,
    productExpRange = scfg.productExpRange
  ))
  tree.io.inputs := laneOp
  val treeOut = tree.io.out  // CustomFP(resOperatorExpWidth, resOperatorMantWidth)

  io.debug.foreach { d =>
    d.tree_out_sign := treeOut.sign
    d.tree_out_exp  := treeOut.exp
    d.tree_out_mant := treeOut.mant
  }

  // ── ScaleAdd + FPn conversion: path selected at elaboration time ─────────
  // All intermediate values use fpNW = 1+8+actualAccMantBits bits throughout.
  // UE8M0: DirectToFPn — pure exponent arithmetic, no multipliers.
  // Non-UE8M0: ScaleAddition + ScaleToFPn — LZC normalise + RNE round.
  val reducedSum = Wire(UInt(fpNW.W))

  if (scfg.stype.mantScaleWidth == 0) {
    // ── Path A: UE8M0 — DirectToFPn ──────────────────────────────────────
    val d2fpn = Module(new DirectToFPn(scfg, actualAccMantBits))
    d2fpn.io.inOpSign      := treeOut.sign
    d2fpn.io.inOpExp       := treeOut.exp
    d2fpn.io.inOpMant      := treeOut.mant
    d2fpn.io.inShareScaleA := io.share_exp_A_i
    d2fpn.io.inShareScaleB := io.share_exp_B_i
    reducedSum             := d2fpn.io.out

    io.debug.foreach { d =>
      val adjA = io.share_exp_A_i.zext - scfg.stype.bias.S
      val adjB = io.share_exp_B_i.zext - scfg.stype.bias.S
      val expDebug = Wire(SInt(scfg.resScaleAddExpWidth.W))
      expDebug := treeOut.exp + adjA + adjB
      d.sa_out_sign := treeOut.sign
      d.sa_out_exp  := expDebug
      d.sa_out_mant := Cat(0.U(2.W), treeOut.mant)
    }

  } else {
    // ── Path B: non-UE8M0 — ScaleAddition + ScaleToFPn ───────────────────
    val sa = Module(new ScaleAddition(scfg))
    sa.io.inOpSign      := treeOut.sign
    sa.io.inOpExp       := treeOut.exp
    sa.io.inOpMant      := treeOut.mant
    sa.io.inShareScaleA := io.share_exp_A_i
    sa.io.inShareScaleB := io.share_exp_B_i

    io.debug.foreach { d =>
      d.sa_out_sign := sa.io.outSign
      d.sa_out_exp  := sa.io.outExp
      d.sa_out_mant := sa.io.outMant
    }

    val conv = Module(new ScaleToFPn(scfg, actualAccMantBits))
    conv.io.inSign := sa.io.outSign
    conv.io.inExp  := sa.io.outExp
    conv.io.inMant := sa.io.outMant
    reducedSum     := conv.io.out
  }

  // Debug reducedSum: zero-extend to 32 bits for log continuity
  io.debug.foreach { d =>
    d.reducedSum := (if (actualAccMantBits < 23) Cat(reducedSum, 0.U((23 - actualAccMantBits).W))
                     else reducedSum)
  }

  // ── Reduced-precision FP accumulator register ────────────────────────────
  // Register stores fpNW = (1 + 8 + actualAccMantBits) bits.
  // FPNAdder natively operates at fpNW precision — no post-add truncation needed.
  // accOut is FP32: accReg's mantissa is zero-extended to 23 bits (no rounding).
  val accRegW   = fpNW
  val asyncRstN = (!reset.asBool).asAsyncReset
  val accReg    = withReset(asyncRstN)(RegInit(0.U(accRegW.W)))
  val validReg  = withReset(asyncRstN)(RegInit(false.B))

  val accAdder = Module(new FPNAdder(actualAccMantBits))
  accAdder.io.a := accReg
  accAdder.io.b := reducedSum

  when(io.resetAcc) {
    accReg   := 0.U
    validReg := false.B
  }.elsewhen(io.validIn) {
    accReg   := accAdder.io.out
    validReg := true.B
  }.otherwise {
    validReg := false.B
  }

  io.validOut := validReg
  // FP32 output: accReg holds {sign[1], exp[8], mant[actualAccMantBits]}.
  // Zero-extend (right-pad) the mantissa to 23 bits to form a valid IEEE-754
  // single-precision word. This is exact — no rounding — because FPNAdder
  // already produced the value at actualAccMantBits precision.
  io.accOut := (
    if (actualAccMantBits >= 23) accReg
    else Cat(accReg, 0.U((23 - actualAccMantBits).W))
  )
}

// ============================================================
// Emission helpers
// ============================================================

object FDPUPostScaleMain extends App {
  val scfg       = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)
  val vectorSize = 8
  emitVerilog(
    new FDPUPostScaleReductionTree(scfg, vectorSize, istest = false),
    Array("--target-dir", s"generated/fdpu_post_scale/default_vec${vectorSize}")
  )
}

object AllFDPUPostScaleMain extends App {
  val vectorSizes  = Seq(4, 16)
  val elementTypes = MXFormats.allElementTypes
  val scaleTypes   = Seq(ScaleFormats.UE8M0, ScaleFormats.UE6M2, ScaleFormats.UE4M4)
  val lowPrecision = Set(MXFormats.E3M2, MXFormats.E2M3, MXFormats.E2M1)

  for {
    (typeA, i) <- elementTypes.zipWithIndex
    typeB      <- elementTypes.drop(i)
    if !(lowPrecision(typeA) && lowPrecision(typeB))
    stype      <- scaleTypes
    vsize      <- vectorSizes
  } {
    val scfg = ScaleAddConfig(typeA, typeB, stype)
    println(
      s"Generating FDPUPostScale: ${typeA.name} x ${typeB.name}, " +
      s"scale ${stype.name}, vectorSize=$vsize"
    )
    emitVerilog(
      new FDPUPostScaleReductionTree(scfg, vsize, istest = false),
      Array("--target-dir",
        s"generated/post_scale/${typeA.name}_${typeB.name}_${stype.name}_vec${vsize}")
    )
  }
}
