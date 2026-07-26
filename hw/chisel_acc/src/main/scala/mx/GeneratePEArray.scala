package mx

import chisel3._
import mx.array.{
  PEArrayBF16Config,
  PEArrayConfig,
  PEArrayFP32Config,
  PEArrayINT8Config,
  PEArrayWrapper,
  PEArrayWrapperBF16,
  PEArrayWrapperFP32,
  PEArrayWrapperINT8
}
import mx.mac.{MXFormats, ScaleAddConfig, ScaleFormats}
import mx.requant.{RequantConfig, RequantINT8Config}

/**
 * Parametrised generator for the combined PE-Array + Requant RTL.
 *
 * Picks the right wrapper from PEArray.scala based on `--requant-mode`:
 *   0 → PEArrayWrapperFP32   (FP32 pass-through, no requant)
 *   1 → PEArrayWrapperBF16   (BF16 element output, no MX scale, no block)
 *   2 → PEArrayWrapper       (FP8  E5M2 output via RequantFP8)
 *   3 → PEArrayWrapper       (FP8  E4M3 output via RequantFP8)
 *   4 → PEArrayWrapperINT8   (mxint8 output via RequantINT8)
 *   5 → PEArrayWrapper       (FP6  E2M3 output via RequantFP8)
 *   6 → PEArrayWrapper       (FP6  E3M2 output via RequantFP8)
 *
 * Mapping mirrors orchestrator.py's `_REQUNAT_TYPE_MAP`.
 *
 * Usage:
 *   sbt "runMain mx.GeneratePEArray \
 *        --type-a E2M3 --type-b INT8 --scale UE5M3 \
 *        --vec 4 --tile-rows 4 --tile-cols 4 \
 *        --block-size 16 --requant-mode 2 \
 *        --out-dir generated/pe_array/test"
 */
object GeneratePEArray extends App {

  private def getArg(flag: String, default: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1) else default
  }
  private def reqArg(flag: String): String = {
    val idx = args.indexOf(flag)
    if (idx >= 0 && idx + 1 < args.length) args(idx + 1)
    else sys.error(s"Missing required argument: $flag")
  }

  val typeAName   = getArg("--type-a",     "E5M2")
  val typeBName   = getArg("--type-b",     "E5M2")
  val scaleName   = getArg("--scale",      "UE8M0")
  val vectorSize  = getArg("--vec",        "4").toInt
  val tileRows    = getArg("--tile-rows",  "4").toInt
  val tileCols    = getArg("--tile-cols",  "4").toInt
  val blockSize   = getArg("--block-size", "32").toInt
  val requantMode = reqArg("--requant-mode").toInt

  val elementA = MXFormats.allElementTypes.find(_.name == typeAName)
    .getOrElse(sys.error(
      s"Unknown --type-a: '$typeAName', valid: " +
      MXFormats.allElementTypes.map(_.name).mkString(", ")
    ))
  val elementB = MXFormats.allElementTypes.find(_.name == typeBName)
    .getOrElse(sys.error(
      s"Unknown --type-b: '$typeBName', valid: " +
      MXFormats.allElementTypes.map(_.name).mkString(", ")
    ))
  val scale = ScaleFormats.allScaleTypes.find(_.name == scaleName)
    .getOrElse(sys.error(
      s"Unknown --scale: '$scaleName', valid: " +
      ScaleFormats.allScaleTypes.map(_.name).mkString(", ")
    ))

  val macCfg = ScaleAddConfig(elementA, elementB, scale)

  // Default output directory mirrors EmitDir.* naming from PEArray.scala.
  val outTagDefault: String = requantMode match {
    case 0 => "FP32"
    case 1 => "BF16"
    case 2 => s"E5M2_blk$blockSize"
    case 3 => s"E4M3_blk$blockSize"
    case 4 => s"INT8_blk$blockSize"
    case 5 => s"E2M3_blk$blockSize"
    case 6 => s"E3M2_blk$blockSize"
    case other =>
      sys.error(s"--requant-mode must be 0..6 (0=FP32, 1=BF16, 2=E5M2, 3=E4M3, 4=mxint8, 5=E2M3, 6=E3M2); got $other")
  }
  val outDirDefault =
    s"generated/pe_array/${tileRows}x${tileCols}_${typeAName}_${typeBName}" +
    s"_${scaleName}_vec${vectorSize}_$outTagDefault"
  val outDir = getArg("--out-dir", outDirDefault)

  println(
    s"[GeneratePEArray] A=$typeAName×B=$typeBName scale=$scaleName vec=$vectorSize " +
    s"${tileRows}x${tileCols} blk=$blockSize requantMode=$requantMode ($outTagDefault) → $outDir"
  )

  requantMode match {
    case 0 =>
      val cfg = PEArrayFP32Config(macCfg, vectorSize, tileRows, tileCols)
      emitVerilog(new PEArrayWrapperFP32(cfg), Array("--target-dir", outDir))

    case 1 =>
      val cfg = PEArrayBF16Config(macCfg, vectorSize, tileRows, tileCols)
      emitVerilog(new PEArrayWrapperBF16(cfg), Array("--target-dir", outDir))

    case 4 =>
      val cfg = PEArrayINT8Config(
        macCfg     = macCfg,
        vectorSize = vectorSize,
        tileRows   = tileRows,
        tileCols   = tileCols,
        requantCfg = RequantINT8Config(blockSize, tileRows, tileCols, scale)
      )
      emitVerilog(new PEArrayWrapperINT8(cfg), Array("--target-dir", outDir))

    case _ =>
      val outType = requantMode match {
        case 2 => MXFormats.E5M2
        case 3 => MXFormats.E4M3
        case 5 => MXFormats.E2M3
        case 6 => MXFormats.E3M2
      }
      val cfg = PEArrayConfig(
        macCfg     = macCfg,
        vectorSize = vectorSize,
        tileRows   = tileRows,
        tileCols   = tileCols,
        requantCfg = RequantConfig(blockSize, tileRows, tileCols, outType, scale)
      )
      emitVerilog(new PEArrayWrapper(cfg), Array("--target-dir", outDir))
  }

  println(s"[GeneratePEArray] Done → $outDir")
}
