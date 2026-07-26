package mx.array

import mx.mac.{ScaleAddConfig, MXFormats, ScaleFormats}
import mx.requant.{RequantConfig, RequantINT8Config}

/**
 * Combined elaboration-time configuration for a PEArrayWrapper (FP8/FP6 output).
 *
 * @param macCfg     ScaleAddConfig for element/scale types driving each PE.
 * @param vectorSize Number of parallel MACs per PE (>= 1).
 * @param tileRows   Number of PE rows.
 * @param tileCols   Number of PE columns.
 * @param requantCfg RequantConfig for the FP32→MXFP8/FP6 requantization block.
 *                   Its tileRows/tileCols must match the array dimensions.
 */
case class PEArrayConfig(
  macCfg:        ScaleAddConfig,
  vectorSize:    Int,
  tileRows:      Int,
  tileCols:      Int,
  requantCfg:    RequantConfig,
  K:             Int     = 16384,
  exposeResults: Boolean = false  // expose FP32 results_o IO for testing
) {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  require(K >= 1, "K must be >= 1")
  require(tileRows == requantCfg.tileRows,
    s"PEArrayConfig tileRows ($tileRows) must match requantCfg.tileRows (${requantCfg.tileRows})")
  require(tileCols == requantCfg.tileCols,
    s"PEArrayConfig tileCols ($tileCols) must match requantCfg.tileCols (${requantCfg.tileCols})")

  /** Total packed input width for operand A (vectorSize elements). */
  val srcWidthA  = macCfg.elementTypeA.totalWidth * vectorSize
  /** Total packed input width for operand B (vectorSize elements). */
  val srcWidthB  = macCfg.elementTypeB.totalWidth * vectorSize
  /** Shared scale factor bit width. */
  val scaleWidth = macCfg.stype.totalScaleWidth
  /** PE accumulator output width (FP32). */
  val dstWidth   = 32
  /** Single output element bit width. */
  val fp8Width   = requantCfg.outputType.totalWidth
}

/**
 * Combined elaboration-time configuration for a PEArrayWrapperINT8.
 *
 * @param macCfg     ScaleAddConfig for element/scale types driving each PE.
 * @param vectorSize Number of parallel MACs per PE (>= 1).
 * @param tileRows   Number of PE rows.
 * @param tileCols   Number of PE columns.
 * @param requantCfg RequantINT8Config for FP32→INT8 requantization.
 */
case class PEArrayINT8Config(
  macCfg:        ScaleAddConfig,
  vectorSize:    Int,
  tileRows:      Int,
  tileCols:      Int,
  requantCfg:    RequantINT8Config,
  K:             Int     = 16384,
  exposeResults: Boolean = false  // expose FP32 results_o IO for testing
) {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  require(K >= 1, "K must be >= 1")
  require(tileRows == requantCfg.tileRows,
    s"PEArrayINT8Config tileRows ($tileRows) must match requantCfg.tileRows (${requantCfg.tileRows})")
  require(tileCols == requantCfg.tileCols,
    s"PEArrayINT8Config tileCols ($tileCols) must match requantCfg.tileCols (${requantCfg.tileCols})")

  val srcWidthA  = macCfg.elementTypeA.totalWidth * vectorSize
  val srcWidthB  = macCfg.elementTypeB.totalWidth * vectorSize
  val scaleWidth = macCfg.stype.totalScaleWidth
  val dstWidth   = 32
}

/**
 * Combined elaboration-time configuration for a PEArrayWrapperBF16.
 *
 * No requant config needed: BF16 is a per-element combinational pass-through.
 *
 * @param macCfg     ScaleAddConfig for element/scale types driving each PE.
 * @param vectorSize Number of parallel MACs per PE (>= 1).
 * @param tileRows   Number of PE rows (4 or 8).
 * @param tileCols   Number of PE columns (4 or 8).
 */
/**
 * Combined elaboration-time configuration for a PEArrayWrapperFP32 (no requant).
 *
 * Used by quantize_mode = 0: the PE array's FP32 accumulator output IS the
 * primary result, packed [tileRows × tileCols × 32]. No requant block, no
 * shared-scale output.
 */
case class PEArrayFP32Config(
  macCfg:     ScaleAddConfig,
  vectorSize: Int,
  tileRows:   Int,
  tileCols:   Int,
  K:          Int = 16384
) {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  require(K >= 1, "K must be >= 1")

  val srcWidthA  = macCfg.elementTypeA.totalWidth * vectorSize
  val srcWidthB  = macCfg.elementTypeB.totalWidth * vectorSize
  val scaleWidth = macCfg.stype.totalScaleWidth
  val dstWidth   = 32
}

case class PEArrayBF16Config(
  macCfg:        ScaleAddConfig,
  vectorSize:    Int,
  tileRows:      Int,
  tileCols:      Int,
  K:             Int     = 16384,
  exposeResults: Boolean = false  // expose FP32 results_o IO for testing
) {
  require(vectorSize >= 1, "vectorSize must be >= 1")
  require(K >= 1, "K must be >= 1")
  require(Seq(4, 8, 16).contains(tileRows), s"tileRows must be 4 or 8; got $tileRows")
  require(Seq(4, 8, 16).contains(tileCols), s"tileCols must be 4 or 8; got $tileCols")

  val srcWidthA  = macCfg.elementTypeA.totalWidth * vectorSize
  val srcWidthB  = macCfg.elementTypeB.totalWidth * vectorSize
  val scaleWidth = macCfg.stype.totalScaleWidth
  val dstWidth   = 32
}

object DefaultPEArrayConfigs {
  // ── Symmetric FP8 ──────────────────────────────────────────────────────────

  /** 4×4 tile, vec4, E5M2×E5M2 / UE8M0, block-32, E5M2 output */
  val e5m2_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E5M2, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(
      blockSize  = 32,
      tileRows   = 4,
      tileCols   = 4,
      outputType = MXFormats.E5M2
    )
  )

  /** 8×8 tile, vec4, E5M2×E5M2 / UE8M0, block-16, E5M2 output.
   *  K = 16 (small for test simulation speed; production users should set K
   *  to their model's worst-case accumulation depth so AccPrecision sizes
   *  the FP32 accumulator mantissa accordingly). */
  val e5m2_8x8 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E5M2, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 8,
    tileCols   = 8,
    requantCfg = RequantConfig(
      blockSize  = 16,
      tileRows   = 8,
      tileCols   = 8,
      outputType = MXFormats.E5M2
    ),
    K = 16
  )

  /** 4×4 tile, vec4, E4M3×E4M3 / UE8M0, block-32, E4M3 output */
  val e4m3_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(
      blockSize  = 32,
      tileRows   = 4,
      tileCols   = 4,
      outputType = MXFormats.E4M3
    )
  )

  /** 4×4 tile, vec4, E3M2×E3M2 / UE8M0, block-32, E3M2 (FP6) output */
  val e3m2_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E3M2, MXFormats.E3M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(
      blockSize  = 32,
      tileRows   = 4,
      tileCols   = 4,
      outputType = MXFormats.E3M2
    )
  )

  /** 8×8 tile, vec4, E4M3×E4M3 / UE8M0, block-16, E4M3 output */
  val e4m3_8x8 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 8,
    tileCols   = 8,
    requantCfg = RequantConfig(
      blockSize  = 16,
      tileRows   = 8,
      tileCols   = 8,
      outputType = MXFormats.E4M3
    )
  )

  // ── Asymmetric FP8/FP6 ────────────────────────────────────────────────────

  /** 4×4 tile, vec4, INT8×E5M2 / UE8M0, block-32, E5M2 output */
  val int8_e5m2_fp8_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.INT8, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(32, 4, 4, MXFormats.E5M2)
  )

  /** 4×4 tile, vec4, INT8×E4M3 / UE8M0, block-32, E4M3 output */
  val int8_e4m3_fp8_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.INT8, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(32, 4, 4, MXFormats.E4M3)
  )

  /** 4×4 tile, vec4, E5M2×E4M3 / UE8M0, block-32, E4M3 output */
  val e5m2_e4m3_fp8_4x4 = PEArrayConfig(
    macCfg     = ScaleAddConfig(MXFormats.E5M2, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantConfig(32, 4, 4, MXFormats.E4M3)
  )

  // ── INT8 Output ────────────────────────────────────────────────────────────

  /** 4×4 tile, vec4, E5M2×E5M2 / UE8M0 → INT8 output */
  val e5m2_4x4_int8 = PEArrayINT8Config(
    macCfg     = ScaleAddConfig(MXFormats.E5M2, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantINT8Config(32, 4, 4)
  )

  /** 4×4 tile, vec4, E4M3×E4M3 / UE8M0 → INT8 output */
  val e4m3_4x4_int8 = PEArrayINT8Config(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantINT8Config(32, 4, 4)
  )

  /** 4×4 tile, vec4, INT8×E5M2 / UE8M0 → INT8 output */
  val int8_e5m2_4x4_int8 = PEArrayINT8Config(
    macCfg     = ScaleAddConfig(MXFormats.INT8, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4,
    requantCfg = RequantINT8Config(32, 4, 4)
  )

  /** 8×8 tile, vec4, E4M3×E4M3 / UE8M0 → INT8 output */
  val e4m3_8x8_int8 = PEArrayINT8Config(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 8,
    tileCols   = 8,
    requantCfg = RequantINT8Config(16, 8, 8)
  )

  // ── BF16 Output ────────────────────────────────────────────────────────────

  /** 4×4 tile, vec4, E5M2×E5M2 / UE8M0 → BF16 output */
  val e5m2_4x4_bf16 = PEArrayBF16Config(
    macCfg     = ScaleAddConfig(MXFormats.E5M2, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4
  )

  /** 4×4 tile, vec4, E4M3×E4M3 / UE8M0 → BF16 output */
  val e4m3_4x4_bf16 = PEArrayBF16Config(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4
  )

  /** 4×4 tile, vec4, INT8×E5M2 / UE8M0 → BF16 output */
  val int8_e5m2_4x4_bf16 = PEArrayBF16Config(
    macCfg     = ScaleAddConfig(MXFormats.INT8, MXFormats.E5M2, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4
  )

  /** 4×4 tile, vec4, INT8×E4M3 / UE8M0 → BF16 output */
  val int8_e4m3_4x4_bf16 = PEArrayBF16Config(
    macCfg     = ScaleAddConfig(MXFormats.INT8, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 4,
    tileCols   = 4
  )

  /** 8×8 tile, vec4, E4M3×E4M3 / UE8M0 → BF16 output */
  val e4m3_8x8_bf16 = PEArrayBF16Config(
    macCfg     = ScaleAddConfig(MXFormats.E4M3, MXFormats.E4M3, ScaleFormats.UE8M0),
    vectorSize = 4,
    tileRows   = 8,
    tileCols   = 8
  )
}
