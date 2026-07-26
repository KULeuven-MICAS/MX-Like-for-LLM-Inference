package mx.mac

import chisel3._
import chisel3.util._

class CustomOperator(val cfg: OperatorConfig) extends Module {
  override def desiredName = s"CustomOperator_${cfg.elementTypeA.name}_to_${cfg.elementTypeB.name}"


  val io = IO(new Bundle {
    val inA = Input(UInt(cfg.elementTypeA.totalWidth.W))
    val inB = Input(UInt(cfg.elementTypeB.totalWidth.W))
    val outSign = Output(UInt(1.W))
    val outExp = Output(SInt(cfg.resOperatorExpWidth.W))
    val outMant = Output(UInt(cfg.resOperatorMantWidth.W))
  })

  def getExtendedMantissa(in:UInt, etype:ElementType):(UInt,UInt,UInt) = {
    val sign= in(etype.totalWidth-1)
    if(etype.name == "INT8"){
      //2sComplement version: INT8 totalWidth=8 → sign at bit7, magnitude in bits[6:0] (7 bits)
      val raw7 = in(etype.totalWidth-2, 0)
      val negMag = (~raw7 + 1.U)(etype.totalWidth-2, 0)  // 7-bit 2's complement negate
      val magnitude = Mux(sign.asBool, negMag, raw7)
      (sign, 0.U, magnitude)
    }else{
      val exp = in(etype.elementWidthMant+etype.elementWidthExp-1,etype.elementWidthMant)
      val mant= in(etype.elementWidthMant-1,0)
      val implicitBit = exp.orR// 1 if exp>0 (normal), else exp==0 (subnormal)
      (sign,exp,Cat(implicitBit,mant))
    }
  }

  val(signA,expA,fullMantA) = getExtendedMantissa(io.inA,cfg.elementTypeA)
  val(signB,expB,fullMantB) = getExtendedMantissa(io.inB,cfg.elementTypeB)

  // 扩展位宽防止溢出，加上各格式的隐含缩放指数（INT8 为 −6）
  val adjExpA = Mux(expA === 0.U && cfg.elementTypeA.elementWidthExp.U > 0.U,
                    1.S - cfg.elementTypeA.bias.S, //subnormal fixed to 1-bias
                    expA.zext - cfg.elementTypeA.bias.S) +
                cfg.elementTypeA.implicitScaleExp.S

  val adjExpB = Mux(expB === 0.U && cfg.elementTypeB.elementWidthExp.U > 0.U,
                    1.S - cfg.elementTypeB.bias.S,
                    expB.zext - cfg.elementTypeB.bias.S) +
                cfg.elementTypeB.implicitScaleExp.S

  //+& means keep the carry bit
  val exp_sum = adjExpA +& adjExpB
  io.outExp  := exp_sum
  io.outSign := signA ^ signB
  val product = fullMantA * fullMantB
  io.outMant := product
}

// 供生成 Verilog 使用的入口
object OperatorMain extends App {
  def gen(cfg: OperatorConfig): Unit = {
    // 构造一个文件夹名称
    val folderName = s"gen_${cfg.elementTypeA.name}_to_${cfg.elementTypeB.name}"
    
    // 执行生成
    emitVerilog(
      new CustomOperator(cfg), 
      Array("--target-dir", s"generated/$folderName")
    )
  }

  gen(MXFormats.e5m2_e4m3_config)
  gen(MXFormats.e4m3_e2m1_config)

}
  
object AllOperatorMain extends App{
  // 遍历所有 A 和 B 的组合 (嵌套循环)
  for {
    typeA <- MXFormats.allElementTypes
    typeB <- MXFormats.allElementTypes
  } {
    val currentCfg = OperatorConfig(typeA, typeB)
    val dirName = s"generated/${typeA.name}_${typeB.name}"
    
    println(s"正在为 ${typeA.name} x ${typeB.name} 生成硬件...")
    
    emitVerilog(
      new CustomOperator(currentCfg), 
      Array("--target-dir", dirName) 
    )
  }

}