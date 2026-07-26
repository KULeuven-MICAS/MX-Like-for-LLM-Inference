package mx.requant

import mx.mac.{ElementType, MXFormats, ScaleType, ScaleFormats}

/**
 * Configuration for the FP32 → MXFP8 requantization block.
 *
 * @param blockSize  Number of elements sharing one MX scale: 16, 32, or 64.
 * @param tileRows   Tile rows of the MAC array: 4 or 8.
 * @param tileCols   Tile columns of the MAC array: 4 or 8.
 * @param outputType Target MXFP8 element format: E5M2 or E4M3.
 * @param scaleType  Shared-scale encoding format (UE8M0 … UE2M6).  All
 *                   variants are exactly 8 bits wide.  Default: UE8M0
 *                   (pure power-of-2, backward-compatible behaviour).
 */
case class RequantConfig(
  blockSize:  Int,
  tileRows:   Int,
  tileCols:   Int,
  outputType: ElementType,
  scaleType:  ScaleType = ScaleFormats.UE8M0
) {
  require(Seq(16, 32, 64).contains(blockSize),
    s"blockSize must be 16, 32, or 64; got $blockSize")
  require(Seq(4, 8, 16).contains(tileRows),
    s"tileRows must be 4, 8, or 16; got $tileRows")
  require(Seq(4, 8, 16).contains(tileCols),
    s"tileCols must be 4, 8, or 16; got $tileCols")
  require(Seq("E5M2", "E4M3", "E3M2", "E2M3").contains(outputType.name),
    s"outputType must be E5M2, E4M3, E3M2, or E2M3; got ${outputType.name}")
  require(blockSize % tileCols == 0,
    s"blockSize ($blockSize) must be divisible by tileCols ($tileCols)")

  /** How many PE-array outputs (each of width tileCols) fill one MX block. */
  val batchesPerBlock: Int = blockSize / tileCols
}

object DefaultRequantConfigs {
  // ── UE8M0 (pure power-of-2 scale) ──────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE8M0 scale */
  val e5m2_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E5M2
  )

  /** 8×8 tile, block-16, E4M3 output, UE8M0 scale */
  val e4m3_block16_8x8 = RequantConfig(
    blockSize  = 16,
    tileRows   = 8,
    tileCols   = 8,
    outputType = MXFormats.E4M3
  )

  /** 4×4 tile, block-64, E4M3 output, UE8M0 scale */
  val e4m3_block64_4x4 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E4M3
  )

  // ── UE7M1 (1-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE7M1 scale */
  val e5m2_block32_4x4_ue7m1 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E5M2,
    scaleType  = ScaleFormats.UE7M1
  )

  /** 4×4 tile, block-64, E4M3 output, UE7M1 scale */
  val e4m3_block64_4x4_ue7m1 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E4M3,
    scaleType  = ScaleFormats.UE7M1
  )

  // ── UE6M2 (2-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE6M2 scale */
  val e5m2_block32_4x4_ue6m2 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E5M2,
    scaleType  = ScaleFormats.UE6M2
  )

  /** 4×4 tile, block-64, E4M3 output, UE6M2 scale */
  val e4m3_block64_4x4_ue6m2 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E4M3,
    scaleType  = ScaleFormats.UE6M2
  )

  // ── UE4M4 (4-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x4_ue4m4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E5M2,
    scaleType  = ScaleFormats.UE4M4
  )

    /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x16_e4m4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 16,
    outputType = MXFormats.E5M2,
    scaleType  = ScaleFormats.UE4M4
  )

    /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x16_e6m2 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 16,
    outputType = MXFormats.E5M2,
    scaleType  = ScaleFormats.UE6M2
  )



  // ── FP6 E3M2 (1+3+2=6 bits) ────────────────────────────────
  /** 4×4 tile, block-32, E3M2 output, UE8M0 scale */
  val e3m2_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E3M2
  )

  /** 8×8 tile, block-32, E3M2 output, UE8M0 scale */
  val e3m2_block32_8x8 = RequantConfig(
    blockSize  = 32,
    tileRows   = 8,
    tileCols   = 8,
    outputType = MXFormats.E3M2
  )

  // ── FP6 E2M3 (1+2+3=6 bits) ────────────────────────────────
  /** 4×4 tile, block-32, E2M3 output, UE8M0 scale */
  val e2m3_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = MXFormats.E2M3
  )

  /** 8×8 tile, block-32, E2M3 output, UE8M0 scale */
  val e2m3_block32_8x8 = RequantConfig(
    blockSize  = 32,
    tileRows   = 8,
    tileCols   = 8,
    outputType = MXFormats.E2M3
  )

  /** Enumerate all non-UE8M0 scale configs for a given element type. */
  def allScaleVariants(
    blockSize: Int, tileRows: Int, tileCols: Int, outputType: ElementType
  ): Seq[RequantConfig] =
    ScaleFormats.allScaleTypes.map(st =>
      RequantConfig(blockSize, tileRows, tileCols, outputType, st)
    )
}
