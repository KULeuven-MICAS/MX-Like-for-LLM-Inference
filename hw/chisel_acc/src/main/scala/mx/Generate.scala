// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Ruichu Ao <ruichuao@gmail.com>

package mx

import chisel3._
import mx.mac._
import mx.requant.{RequantConfig, RequantFP8, RequantINT8Config, RequantINT8, RequantBF16}

/** Generates a FusedDotProductUnit from command-line options.
 *
 * Usage:
 *   sbt "runMain mx.GenerateFusedDotProduct --type-a E5M2 --type-b E4M3 --scale UE8M0 --vec 16 --out-dir generated/fused_dot"
 *
 * Options:
 *   --type-a  <name>   Input A format (default: E5M2)
 *   --type-b  <name>   Input B format (default: E5M2)
 *   --scale   <name>   Shared-scale format (default: UE8M0)
 *   --vec     <int>    Parallel MAC count (default: 8)
 *   --out-dir <path>   Output directory (default: generated path)
 *
 * Valid formats:
 *   type-a / type-b : INT8 E5M2 E4M3 E3M2 E2M3 E2M1
 *   scale           : UE8M0 UE7M1 UE6M2 UE5M3 UE4M4 UE3M5 UE2M6
 */
object GenerateFusedDotProduct extends App {

  def getArg(flag: String, default: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }

  val typeAName  = getArg("--type-a", "E5M2")
  val typeBName  = getArg("--type-b", "E5M2")
  val scaleName  = getArg("--scale",  "UE8M0")
  val vectorSize = getArg("--vec",    "8").toInt

  val elementA = MXFormats.allElementTypes.find(_.name == typeAName)
    .getOrElse(sys.error(s"Unknown type-a: '$typeAName', Valid: ${MXFormats.allElementTypes.map(_.name).mkString(", ")}"))

  val elementB = MXFormats.allElementTypes.find(_.name == typeBName)
    .getOrElse(sys.error(s"Unknown type-b: '$typeBName', Valid: ${MXFormats.allElementTypes.map(_.name).mkString(", ")}"))

  val scale = ScaleFormats.allScaleTypes.find(_.name == scaleName)
    .getOrElse(sys.error(s"Unknown scale: '$scaleName', Valid: ${ScaleFormats.allScaleTypes.map(_.name).mkString(", ")}"))

  val outDir = getArg("--out-dir",
    s"generated/fused_dot/${typeAName}_${typeBName}_${scaleName}_vec${vectorSize}")

  val scfg = ScaleAddConfig(elementA, elementB, scale)

  println(s"[Generate] A=$typeAName  B=$typeBName  Scale=$scaleName  vec=$vectorSize  → $outDir")

  emitVerilog(
    new FusedDotProductUnit(scfg, vectorSize, istest = false),
    Array("--target-dir", outDir)
  )

  println(s"[Generate] Done → $outDir/BFP_PE.sv")
}

/**
 * Generate RequantFP8 (supports FP8 and FP6 output formats) via command-line arguments.
 *
 * Usage (via sbt):
 *   sbt "runMain mx.GenerateRequantFP8 --out-type E5M2 --scale UE8M0 --block-size 32 --tile-rows 4 --tile-cols 4 --out-dir generated/requant"
 *
 * Parameters (all optional, have defaults):
 *   --out-type   <name>  MX float output element format, default E5M2
 *                        FP8: E5M2, E4M3 | FP6: E3M2, E2M3
 *   --scale      <name>  Shared-scale format, default UE8M0
 *   --block-size <int>   MX block size (16, 32, or 64), default 32
 *   --tile-rows  <int>   PE tile rows (4 or 8), default 4
 *   --tile-cols  <int>   PE tile columns (4 or 8), default 4
 *   --out-dir    <path>  Output directory, default auto-generated path
 */
object GenerateRequantFP8or6 extends App {

  def getArg(flag: String, default: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }

  val outTypeName = getArg("--out-type",   "E5M2")
  val scaleName   = getArg("--scale",      "UE8M0")
  val blockSize   = getArg("--block-size", "32").toInt
  val tileRows    = getArg("--tile-rows",  "4").toInt
  val tileCols    = getArg("--tile-cols",  "4").toInt

  val validMxFloatTypes = Seq("E5M2", "E4M3", "E3M2", "E2M3")
  val outputType = MXFormats.allElementTypes.find(_.name == outTypeName)
    .getOrElse(sys.error(s"Unknown --out-type: '$outTypeName', valid: ${validMxFloatTypes.mkString(", ")}"))

  val scale = ScaleFormats.allScaleTypes.find(_.name == scaleName)
    .getOrElse(sys.error(s"Unknown --scale: '$scaleName', valid: ${ScaleFormats.allScaleTypes.map(_.name).mkString(", ")}"))

  val outDir = getArg("--out-dir",
    s"generated/requant/${outTypeName}_${scaleName}_blk${blockSize}_${tileRows}x${tileCols}")

  val cfg = RequantConfig(blockSize, tileRows, tileCols, outputType, scale)

  println(s"[GenerateRequantFP8] outType=$outTypeName  scale=$scaleName  blk=$blockSize  ${tileRows}x${tileCols}  → $outDir")

  emitVerilog(new RequantFP8(cfg), Array("--target-dir", outDir))

  println(s"[GenerateRequantFP8] Done → $outDir")
}

/**
 * Generate RequantINT8 via command-line arguments.
 *
 * Usage (via sbt):
 *   sbt "runMain mx.GenerateRequantINT8 --block-size 32 --tile-rows 4 --tile-cols 4 --out-dir generated/requant"
 *
 * Parameters (all optional, have defaults):
 *   --block-size <int>   MX block size (16, 32, or 64), default 32
 *   --tile-rows  <int>   PE tile rows (4 or 8), default 4
 *   --tile-cols  <int>   PE tile columns (4 or 8), default 4
 *   --out-dir    <path>  Output directory, default auto-generated path
 *
 * Scale is always UE8M0 (max biased FP32 exponent) for INT8.
 */
object GenerateRequantINT8 extends App {

  def getArg(flag: String, default: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }

  val blockSize = getArg("--block-size", "32").toInt
  val tileRows  = getArg("--tile-rows",  "4").toInt
  val tileCols  = getArg("--tile-cols",  "4").toInt
  val outDir    = getArg("--out-dir",
    s"generated/requant/INT8_UE8M0_blk${blockSize}_${tileRows}x${tileCols}")

  val cfg = RequantINT8Config(blockSize, tileRows, tileCols)

  println(s"[GenerateRequantINT8] blk=$blockSize  ${tileRows}x${tileCols}  → $outDir")

  emitVerilog(new RequantINT8(cfg), Array("--target-dir", outDir))

  println(s"[GenerateRequantINT8] Done → $outDir")
}

/**
 * Generate RequantBF16 via command-line arguments.
 *
 * Usage (via sbt):
 *   sbt "runMain mx.GenerateRequantBF16 --tile-rows 4 --tile-cols 4 --out-dir generated/requant"
 *
 * Parameters (all optional, have defaults):
 *   --tile-rows  <int>   PE tile rows (4 or 8), default 4
 *   --tile-cols  <int>   PE tile columns (4 or 8), default 4
 *   --out-dir    <path>  Output directory, default auto-generated path
 *
 * No block size or scale: BF16 is a per-element combinational pass-through.
 */
object GenerateRequantBF16 extends App {

  def getArg(flag: String, default: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }

  val tileRows = getArg("--tile-rows", "4").toInt
  val tileCols = getArg("--tile-cols", "4").toInt
  val outDir   = getArg("--out-dir", s"generated/requant/BF16_${tileRows}x${tileCols}")

  println(s"[GenerateRequantBF16] ${tileRows}x${tileCols}  → $outDir")

  emitVerilog(new RequantBF16(tileRows, tileCols), Array("--target-dir", outDir))

  println(s"[GenerateRequantBF16] Done → $outDir")
}
