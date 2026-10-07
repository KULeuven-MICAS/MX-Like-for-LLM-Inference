// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Ruichu Ao <ruichuao@gmail.com>

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

  // Extend the exponent range and apply each format's implicit scale.
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

// Verilog generator.
object OperatorMain extends App {
  def gen(cfg: OperatorConfig): Unit = {
    // Build the output path.
    val folderName = s"gen_${cfg.elementTypeA.name}_to_${cfg.elementTypeB.name}"
    
    // Emit Verilog.
    emitVerilog(
      new CustomOperator(cfg), 
      Array("--target-dir", s"generated/$folderName")
    )
  }

  gen(MXFormats.e5m2_e4m3_config)
  gen(MXFormats.e4m3_e2m1_config)

}
  
object AllOperatorMain extends App{
  // Generate all format pairs.
  for {
    typeA <- MXFormats.allElementTypes
    typeB <- MXFormats.allElementTypes
  } {
    val currentCfg = OperatorConfig(typeA, typeB)
    val dirName = s"generated/${typeA.name}_${typeB.name}"
    
    println(s"Generating ${typeA.name} x ${typeB.name} hardware...")
    
    emitVerilog(
      new CustomOperator(currentCfg), 
      Array("--target-dir", dirName) 
    )
  }

}
