package mx.array

import chisel3._
import chisel3.util._
import mx_template.{TemplateMACUnit, MACUnitConfig, ElemOperand, ScaleOperand, BF16, Elem, Scale}
import mx.requant.{RequantFP8, RequantConfig, RequantINT8, RequantINT8Config}

sealed trait TemplateOut
object TemplateOut {
  final case class FP8(outType: Elem) extends TemplateOut
  case object INT8 extends TemplateOut
  case object BF16 extends TemplateOut
  case object FP32 extends TemplateOut
}

/** Config for the TemplateMACUnit array.  MACUnitConfig (A: Elem, W: Elem, S: Scale, N)
 *  is the single format+config source — the former mx.mac.ScaleAddConfig and its
 *  parallel ElementType/ScaleType were unified into mx_template.{Elem,Scale}. */
case class TemplateArrayConfig(
  macUnitCfg:    MACUnitConfig,
  tileRows:  Int,
  tileCols:  Int,
  out:       TemplateOut,
  blockSize: Int = 16,
) {
  val vectorSize = macUnitCfg.N
  val A: Elem = macUnitCfg.A; val W: Elem = macUnitCfg.W; val S: Scale = macUnitCfg.S

  val srcWidthA  = A.totalWidth * vectorSize
  val srcWidthB  = W.totalWidth * vectorSize
  val scaleWidth = S.totalScaleWidth
  val bf16W      = 1 + BF16.expBits + BF16.mantBits           // 16

  
  val rqInMantW: Int = out match {
    case TemplateOut.INT8 => 8                                  
    case _              => BF16.mantBits                     
  }
  val hasSharedScale: Boolean = out match {
    case TemplateOut.FP8(_) | TemplateOut.INT8 => true
    case _                                 => false
  }
  val resultW: Int = out match {
    case TemplateOut.FP8(t) => tileRows * blockSize * t.totalWidth
    case TemplateOut.INT8   => tileRows * blockSize * 8
    case TemplateOut.BF16   => tileRows * tileCols * 16
    case TemplateOut.FP32   => tileRows * tileCols * 32
  }
  require(blockSize % vectorSize == 0,
    s"blockSize ($blockSize) must be divisible by vectorSize ($vectorSize)")
}

class TemplatePEArray(cfg: TemplateArrayConfig) extends Module {
  override def desiredName = "PE_Array"

  private val A  = cfg.macUnitCfg.A
  private val W  = cfg.macUnitCfg.W
  private val S  = cfg.macUnitCfg.S
  private val N  = cfg.vectorSize

  val io = IO(new Bundle {
    // CSR controller
    val A_mode           = Input(UInt(3.W))
    val B_mode           = Input(UInt(3.W))
    val result_mode_quan = Input(UInt(2.W))
    val group_size       = Input(UInt(2.W))
    val shared_format_i  = Input(UInt(4.W))
    val acc_reset_i      = Input(Bool())
    val send_output_i    = Input(Bool())
    val accumulation_count_i = Input(UInt(32.W))

    val A_valid_i = Input(Bool())
    val B_valid_i = Input(Bool())
    val A_ready_o = Output(Bool())
    val B_ready_o = Output(Bool())

    // Operands and scales
    val op_a_i         = Input(Vec(cfg.tileRows, UInt(cfg.srcWidthA.W)))
    val op_b_i         = Input(Vec(cfg.tileCols, UInt(cfg.srcWidthB.W)))
    val shared_exp_A_i = Input(Vec(cfg.tileRows, UInt(cfg.scaleWidth.W)))
    val shared_exp_B_i = Input(Vec(cfg.tileCols, UInt(cfg.scaleWidth.W)))

    // Output
    val shared_scale_out =
      if (cfg.hasSharedScale) Some(Output(UInt((cfg.tileRows * cfg.scaleWidth).W))) else None
    val result    = Output(UInt(cfg.resultW.W))
    val valid_out = Output(Bool())
  })


  io.A_ready_o := !io.send_output_i
  io.B_ready_o := !io.send_output_i
  val internal_valid = io.A_valid_i && io.B_valid_i

  private def unpackElem(bits: UInt, el: Elem, lane: Int): ElemOperand = {
    val w = 1 + el.e + el.m
    val slice = bits(w * (lane + 1) - 1, w * lane)
    val o = Wire(new ElemOperand(el))
    o.sign := slice(w - 1)
    o.mant := slice(el.m - 1, 0)
    o.exp  := (if (el.e > 0) slice(w - 2, el.m) else 0.U(0.W))
    o
  }
  private def unpackScale(bits: UInt, sc: Scale): ScaleOperand = {
    val o = Wire(new ScaleOperand(sc))
    o.exp  := bits(sc.e + sc.m - 1, sc.m)
    o.mant := (if (sc.m > 0) bits(sc.m - 1, 0) else 0.U(0.W))
    o
  }

  val peValidOut = Wire(Bool())
  val accBF16    = Wire(Vec(cfg.tileRows, Vec(cfg.tileCols, UInt(cfg.bf16W.W))))

  for (r <- 0 until cfg.tileRows) {
    for (c <- 0 until cfg.tileCols) {
      // Active-low reconciliation: TemplateMACUnit's internal reset.asAsyncReset now
      // asserts when the wrapper's reset port is 0 (cluster active-low rst_ni).
      val pe = withReset((!reset.asBool).asAsyncReset)(Module(new TemplateMACUnit(cfg.macUnitCfg)))
      for (i <- 0 until N) {
        pe.io.a(i) := unpackElem(io.op_a_i(r), A, i)
        pe.io.w(i) := unpackElem(io.op_b_i(c), W, i)
      }
      pe.io.scaleA   := unpackScale(io.shared_exp_A_i(r), S)
      pe.io.scaleW   := unpackScale(io.shared_exp_B_i(c), S)
      pe.io.enable   := internal_valid
      pe.io.clearAcc := io.acc_reset_i

      accBF16(r)(c) := Cat(pe.io.accOut.sign, pe.io.accOut.exp, pe.io.accOut.mant)
    }
  }


  private val validReg = withReset((!reset.asBool).asAsyncReset)(RegInit(false.B))
  when(io.acc_reset_i)        { validReg := false.B }
    .elsewhen(internal_valid) { validReg := true.B }
    .otherwise                { validReg := false.B }
  peValidOut := validReg

  private val rqAsyncRstN = (!reset.asBool).asAsyncReset
  private val accCnt      = withReset(rqAsyncRstN)(RegInit(0.U(32.W)))
  private val resultDone  = WireDefault(false.B)
  when(peValidOut) {
    when(accCnt === io.accumulation_count_i - 1.U) {
      accCnt     := 0.U
      resultDone := true.B
    }.otherwise {
      accCnt := accCnt + 1.U
    }
  }
  when(io.acc_reset_i) { accCnt := 0.U }


  private val accFlat: Seq[UInt] =
    for (r <- 0 until cfg.tileRows; c <- 0 until cfg.tileCols) yield accBF16(r)(c)

  cfg.out match {
    case TemplateOut.FP8(outType) =>
      val rqCfg = RequantConfig(cfg.blockSize, cfg.tileRows, cfg.tileCols,
                                outType, cfg.S, inputMantWidth = cfg.rqInMantW)
      val rq = Module(new RequantFP8(rqCfg))
      rq.io.fp32_in  := Cat(accFlat)                          // 16-bit each = inputWidth
      rq.io.valid_in := resultDone
      io.shared_scale_out.get := rq.io.shared_scale_out
      io.result               := rq.io.elem_out
      io.valid_out            := rq.io.valid_out

    case TemplateOut.INT8 =>
      val rqCfg = RequantINT8Config(cfg.blockSize, cfg.tileRows, cfg.tileCols,
                                    cfg.S, inputMantWidth = cfg.rqInMantW)
      val rq = Module(new RequantINT8(rqCfg))
      rq.io.fp32_in  := Cat(accFlat.map(x => Cat(x, 0.U(1.W))))  // 17-bit (mant 7→8)
      rq.io.valid_in := resultDone
      io.shared_scale_out.get := rq.io.shared_scale_out
      io.result               := rq.io.int8_out
      io.valid_out            := rq.io.valid_out

    case TemplateOut.BF16 =>
      val outReg   = withReset(rqAsyncRstN)(RegInit(0.U(cfg.resultW.W)))
      val validOut = withReset(rqAsyncRstN)(RegInit(false.B))
      when(resultDone) { outReg := Cat(accFlat); validOut := true.B }
        .otherwise     { validOut := false.B }
      io.result    := outReg
      io.valid_out := validOut

    case TemplateOut.FP32 =>
      val fp32Flat = accFlat.map { bf =>
        val sign = bf(15); val exp = bf(14, 7); val mant7 = bf(6, 0)
        Cat(sign, exp, mant7, 0.U(16.W))                      // 32-bit
      }
      val outReg   = withReset(rqAsyncRstN)(RegInit(0.U(cfg.resultW.W)))
      val validOut = withReset(rqAsyncRstN)(RegInit(false.B))
      when(resultDone) { outReg := Cat(fp32Flat); validOut := true.B }
        .otherwise     { validOut := false.B }
      io.result    := outReg
      io.valid_out := validOut
  }
}
