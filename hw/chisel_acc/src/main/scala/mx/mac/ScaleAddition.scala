// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Ruichu Ao <ruichuao@gmail.com>

package mx.mac

import chisel3._
import chisel3.util._
/*ScaleAddition 
  Add Scaled parts with Operator Inputs -> sends to FP32 Accumulator

*/

class ScaleAddition(val scfg: ScaleAddConfig) extends Module {
  override def desiredName = s"ScaleAddition_${scfg.elementTypeA.name}_to_${scfg.elementTypeB.name}_scale_${scfg.stype.name}"


  val io = IO(new Bundle {
    //input
    val inOpSign = Input(UInt(1.W))
    val inOpExp  = Input(SInt(scfg.resOperatorExpWidth.W))
    val inOpMant = Input(UInt(scfg.resOperatorMantWidth.W))
    val inShareScaleA = Input(UInt(scfg.stype.totalScaleWidth.W))
    val inShareScaleB = Input(UInt(scfg.stype.totalScaleWidth.W))
    //output
    val outSign = Output(UInt(1.W))
    val outExp  = Output(SInt(scfg.resScaleAddExpWidth.W))
    val outMant = Output(UInt(scfg.resScaleAddMantWidth.W))
  })

  def getScaledParts(in:UInt, stype:ScaleType):(UInt,UInt) = {
    val exp = in(stype.totalScaleWidth - 1, stype.mantScaleWidth)
    if (stype.mantScaleWidth == 0) {
      (exp, 1.U(1.W)) // Implicit bit only
    } else {
      val mant = in(stype.mantScaleWidth - 1, 0)
      val implicitBit = exp.orR
      (exp, Cat(implicitBit, mant))
    }
  }

  val(expScaleA,fullMantScaleA) = getScaledParts(io.inShareScaleA,scfg.stype)
  val(expScaleB,fullMantScaleB) = getScaledParts(io.inShareScaleB,scfg.stype)

  // Use signed bias arithmetic.
  val safeBias = scfg.stype.bias.S

  // Subnormal scales use 1 - bias, matching CustomOperator.
  val adjExpScaleA = Mux(expScaleA === 0.U && scfg.stype.mantScaleWidth.U > 0.U,
                         1.S - safeBias,
                         expScaleA.zext - safeBias)
  val adjExpScaleB = Mux(expScaleB === 0.U && scfg.stype.mantScaleWidth.U > 0.U,
                         1.S - safeBias,
                         expScaleB.zext - safeBias)

  //+& means keep the carry bit
  val scaleExpSum = adjExpScaleA +& adjExpScaleB
  val scaleMantProduct = fullMantScaleA * fullMantScaleB

  val ExpAdd = scaleExpSum +& io.inOpExp
  val MantProduct = scaleMantProduct * io.inOpMant

  io.outSign := io.inOpSign
  io.outExp  := ExpAdd
  io.outMant := MantProduct
 
}

object OperatorWithScaleMain extends App {
  def gen(elementTypeA: ElementType, elementTypeB: ElementType, stype: ScaleType): Unit = {
    // Wrap into the config object
    val scfg = ScaleAddConfig(elementTypeA, elementTypeB, stype)
    val folderName = s"gen_scale_${stype.name}_${elementTypeA.name}_to_${elementTypeB.name}"
    
    emitVerilog(
      new ScaleAddition(scfg), 
      Array("--target-dir", s"generated/$folderName")
    )
  }

  // Example call
  gen(MXFormats.E5M2, MXFormats.E4M3, ScaleFormats.UE5M3)
}

object AllOperatorWithScaleMain extends App{
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
