package mx.mac

import chisel3._

// ============================================================
// Fused Dot-Product Unit
// ============================================================
/** Fused dot-product unit that performs vectorSize MAC operations per cycle.
 *
 *  Pipeline (fully combinational within each cycle):
 *
 *    ┌─ lane 0 ─┐   ┌──────────────┐
 *    │ CustomOp │   │              │
 *    │ ScaleAdd ├──►│              │
 *    │ ToFP32   │   │  FP32        │   ┌─────────┐
 *    └──────────┘   │  Reduction   ├──►│  FP32   │
 *        ...        │  Tree        │   │  Accum  ├──► accOut
 *    ┌─ lane N-1 ─┐ │  (balanced)  │   │  Reg    │
 *    │ CustomOp  │  │              │   └─────────┘
 *    │ ScaleAdd  ├──►│              │
 *    │ ToFP32    │  └──────────────┘
 *    └───────────┘
 *
 *  The shared scale factors (share_exp_A_i / share_exp_B_i) are
 *  broadcast to every lane, matching the MX block-floating-point
 *  convention where one scale covers an entire vector.
 *
 *  @param scfg       ScaleAddConfig describing element and scale types.
 *  @param vectorSize Number of parallel MACs per cycle (>= 1).
 */
class FusedDotProductUnit(val scfg: ScaleAddConfig, val vectorSize: Int, val istest: Boolean) extends Module {
  require(vectorSize >= 1, "vectorSize must be >= 1")

  override def desiredName =
    if (!istest)
      s"BFP_PE"
    else
      s"FusedDotProductUnit_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}" +
      s"_scale_${scfg.stype.name}_vec${vectorSize}"

  private val wA = scfg.elementTypeA.totalWidth
  private val wB = scfg.elementTypeB.totalWidth

  val io = IO(new Bundle {
    // Packed vector inputs: [vectorSize*wA-1:0] ≡ logic [vectorSize-1:0][wA-1:0]
    val op_a_i        = Input(UInt((vectorSize * wA).W))
    val op_b_i        = Input(UInt((vectorSize * wB).W))
    // Shared scale factors (broadcast to all lanes)
    val share_exp_A_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    val share_exp_B_i = Input(UInt(scfg.stype.totalScaleWidth.W))
    // Control
    val validIn  = Input(Bool())
    val resetAcc = Input(Bool())
    // Output
    val validOut = Output(Bool())
    val accOut   = Output(UInt(32.W))

    // ============================================================
    // 专门为 Trace 增加的测试端口 (仅观察 Lane 0)
    // ============================================================
    val debug = if (istest) Some(new Bundle {
      val all_lanes_fp32    = Output(Vec(vectorSize, UInt(32.W)))
      val reducedSum        = Output(UInt(32.W))
      val all_lanes_sa_mant = Output(Vec(vectorSize, UInt(scfg.resScaleAddMantWidth.W)))
    }) else None
  })

  // ------------------------------------------------------------------
  // Lanes: vectorSize × (CustomOperator → ScaleAddition → ScaleToFP32)
  // ------------------------------------------------------------------
  val laneResults = Wire(Vec(vectorSize, UInt(32.W)))

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

    val conv = Module(new ScaleToFP32(scfg))
    conv.io.inSign := sa.io.outSign
    conv.io.inExp  := sa.io.outExp
    conv.io.inMant := sa.io.outMant

    laneResults(i) := conv.io.out

    // 只有在测试模式下连接调试信号
    io.debug.foreach { d =>
      d.all_lanes_fp32(i)    := conv.io.out
      d.all_lanes_sa_mant(i) := sa.io.outMant
    }
  }

  // ------------------------------------------------------------------
  // FP32 balanced reduction tree
  // ------------------------------------------------------------------
  /** Recursively halve the list using FP32Adder instances until one value remains. */
  def fp32ReduceTree(inputs: Seq[UInt]): UInt = {
    if (inputs.length == 1) {
      inputs.head
    } else {
      val nextLevel = inputs.grouped(2).map { group =>
        if (group.length == 2) {
          val adder = Module(new FP32Adder())
          adder.io.a := group(0)
          adder.io.b := group(1)
          adder.io.out
        } else {
          group(0) // odd lane passes through without an adder
        }
      }.toSeq
      fp32ReduceTree(nextLevel)
    }
  }

  val reducedSum = fp32ReduceTree(laneResults.toSeq)
  
  //for debug------------------------------------------------------------
  io.debug.foreach(_.reducedSum := reducedSum)
  // ------------------------------------------------------------------
  // FP32 accumulator register
  // ------------------------------------------------------------------
  val asyncRstN = (!reset.asBool).asAsyncReset
  val accReg   = withReset(asyncRstN)(RegInit(0.U(32.W)))
  val validReg = withReset(asyncRstN)(RegInit(false.B))

  val accAdder = Module(new FP32Adder())
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
  io.accOut   := accReg
}

// ============================================================
// Emission helpers
// ============================================================

/** Emit a single FusedDotProductUnit for the default E4M3×E2M1/UE5M3 config
 *  with a configurable vector size. */
object FusedDotProductMain extends App {
  val scfg       = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)
  val vectorSize = 8

  emitVerilog(
    new FusedDotProductUnit(scfg, vectorSize, false),
    Array("--target-dir", s"generated/fused_dot_product/default_vec${vectorSize}")
  )
}
/** Emit Verilog for selected element-type × scale-type × vector-size combinations.
 *  Only unordered pairs are generated (A ≤ B in list order).
 *  Pairs where both types are lower precision than E4M3 (E3M2, E2M3, E2M1) are skipped. */
object AllFusedDotProductMain extends App {
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
      s"Generating FusedDotProductUnit: ${typeA.name} x ${typeB.name}, " +
      s"scale ${stype.name}, vectorSize=$vsize"
    )
    emitVerilog(
      new FusedDotProductUnit(scfg, vsize, false),
      Array("--target-dir",
        s"generated/fused_dot/${typeA.name}_${typeB.name}_${stype.name}_vec${vsize}")
    )
  }
}

object TestFusedDotProductMain extends App {
  val vectorSizes = Seq(1, 2, 4, 8, 16, 32)
  val typeA = MXFormats.INT8
  val typeB = MXFormats.E2M1
  val stype = ScaleFormats.UE8M0

  for {
    vsize <- vectorSizes
  } {
    val scfg = ScaleAddConfig(typeA, typeB, stype)
    println(
      s"Generating FusedDotProductUnit: ${typeA.name} x ${typeB.name}, " +
      s"scale ${stype.name}, vectorSize=$vsize"
    )
    emitVerilog(
      new FusedDotProductUnit(scfg, vsize, false),
      Array("--target-dir",
        s"generated/${typeA.name}_${typeB.name}_${stype.name}/${typeA.name}_${typeB.name}_${stype.name}_vec${vsize}")
    )
  }
}