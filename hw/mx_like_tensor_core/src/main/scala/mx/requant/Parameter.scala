package mx.requant

import mx_template.{Elem, Scale}

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
  outputType: Elem,
  scaleType:  Scale = Scale.UE8M0,
  /** Mantissa width (explicit, no implicit-1) of the FP input that the PE
   *  produces.  Default 23 reproduces full IEEE-754 FP32 (32-bit input).
   *  When the upstream PE outputs a narrower FP format (e.g. FP20 for
   *  M_acc=11, FP21 for M_acc=12), set this to that M_acc.  The exponent
   *  field width is fixed at 8 bits, so total input bit-width per element
   *  becomes  1 + 8 + inputMantWidth. */
  inputMantWidth: Int = 23
) {
  require(Seq(16, 32, 64).contains(blockSize),
    s"blockSize must be 16, 32, or 64; got $blockSize")
  require(Seq(4, 8, 16).contains(tileRows),
    s"tileRows must be 4, 8, or 16; got $tileRows")
  require(Seq(4, 8, 16).contains(tileCols),
    s"tileCols must be 4, 8, or 16; got $tileCols")
  require(Seq("E5M2", "E4M3", "E3M2", "E2M3", "E2M1").contains(outputType.name),
    s"outputType must be E5M2, E4M3, E3M2, E2M3, or E2M1; got ${outputType.name}")
  require(blockSize % tileCols == 0,
    s"blockSize ($blockSize) must be divisible by tileCols ($tileCols)")
  require(inputMantWidth >= outputType.elementWidthMant + 2 && inputMantWidth <= 23,
    s"inputMantWidth must be in [${outputType.elementWidthMant + 2}, 23] " +
    s"(need >= outMantBits+2 for guard/sticky); got $inputMantWidth")

  /** How many PE-array outputs (each of width tileCols) fill one MX block. */
  val batchesPerBlock: Int = blockSize / tileCols
  /** Total per-element FP input width (sign + 8-bit exp + mantissa). */
  val inputWidth: Int = 1 + 8 + inputMantWidth
}

object DefaultRequantConfigs {
  // ── UE8M0 (pure power-of-2 scale) ──────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE8M0 scale */
  val e5m2_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E5M2
  )

  /** 8×8 tile, block-16, E4M3 output, UE8M0 scale */
  val e4m3_block16_8x8 = RequantConfig(
    blockSize  = 16,
    tileRows   = 8,
    tileCols   = 8,
    outputType = Elem.E4M3
  )

  /** 4×4 tile, block-64, E4M3 output, UE8M0 scale */
  val e4m3_block64_4x4 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E4M3
  )

  // ── UE7M1 (1-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE7M1 scale */
  val e5m2_block32_4x4_ue7m1 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E5M2,
    scaleType  = Scale.UE7M1
  )

  /** 4×4 tile, block-64, E4M3 output, UE7M1 scale */
  val e4m3_block64_4x4_ue7m1 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E4M3,
    scaleType  = Scale.UE7M1
  )

  // ── UE6M2 (2-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE6M2 scale */
  val e5m2_block32_4x4_ue6m2 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E5M2,
    scaleType  = Scale.UE6M2
  )

  /** 4×4 tile, block-64, E4M3 output, UE6M2 scale */
  val e4m3_block64_4x4_ue6m2 = RequantConfig(
    blockSize  = 64,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E4M3,
    scaleType  = Scale.UE6M2
  )

  // ── UE4M4 (4-bit mantissa scale) ────────────────────────────
  /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x4_ue4m4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E5M2,
    scaleType  = Scale.UE4M4
  )

    /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x16_e4m4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 16,
    outputType = Elem.E5M2,
    scaleType  = Scale.UE4M4
  )

    /** 4×4 tile, block-32, E5M2 output, UE4M4 scale */
  val e5m2_block32_4x16_e6m2 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 16,
    outputType = Elem.E5M2,
    scaleType  = Scale.UE6M2
  )



  // ── FP6 E3M2 (1+3+2=6 bits) ────────────────────────────────
  /** 4×4 tile, block-32, E3M2 output, UE8M0 scale */
  val e3m2_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E3M2
  )

  /** 8×8 tile, block-32, E3M2 output, UE8M0 scale */
  val e3m2_block32_8x8 = RequantConfig(
    blockSize  = 32,
    tileRows   = 8,
    tileCols   = 8,
    outputType = Elem.E3M2
  )

  // ── FP6 E2M3 (1+2+3=6 bits) ────────────────────────────────
  /** 4×4 tile, block-32, E2M3 output, UE8M0 scale */
  val e2m3_block32_4x4 = RequantConfig(
    blockSize  = 32,
    tileRows   = 4,
    tileCols   = 4,
    outputType = Elem.E2M3
  )

  /** 8×8 tile, block-32, E2M3 output, UE8M0 scale */
  val e2m3_block32_8x8 = RequantConfig(
    blockSize  = 32,
    tileRows   = 8,
    tileCols   = 8,
    outputType = Elem.E2M3
  )

  /** Enumerate all non-UE8M0 scale configs for a given element type. */
  def allScaleVariants(
    blockSize: Int, tileRows: Int, tileCols: Int, outputType: Elem
  ): Seq[RequantConfig] =
    Scale.allScales.map(st =>
      RequantConfig(blockSize, tileRows, tileCols, outputType, st)
    )
}
