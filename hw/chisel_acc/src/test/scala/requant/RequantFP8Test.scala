package mx.requant

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random
import java.lang.Float.{floatToRawIntBits, intBitsToFloat}
import mx.mac.{ElementType, MXFormats}

class RequantFP8Test extends AnyFunSuite with ChiselScalatestTester {

  // =========================================================================
  // Software golden model
  // =========================================================================

  /** Max biased FP32 exponent across a block = UE8M0 shared scale. */
  def swSharedScale(fp32s: Seq[Float]): Int =
    fp32s.map(f => (floatToRawIntBits(f) >>> 23) & 0xFF).max

  /**
   * FP32 → MXFP8 for a single element given the UE8M0 shared exponent.
   * Matches the hardware implementation exactly:
   *   - Subnormal / zero FP32 → 0
   *   - Re-biased exp <= 0    → underflow → 0
   *   - Re-biased exp > max   → saturate to ±max-normal
   *   - Otherwise             → truncate mantissa with RNE rounding
   */
  def swFP32toFP8(f: Float, sharedExp: Int, t: ElementType): Int = {
    val bits    = floatToRawIntBits(f)
    val sign    = (bits >>> 31) & 1
    val fp32Exp = (bits >>> 23) & 0xFF
    val fp32Man = bits & 0x7FFFFF

    val fp8ExpBits      = t.elementWidthExp
    val fp8MantBits     = t.elementWidthMant
    val fp8Bias         = t.bias
    val fp8MaxNormalExp = (1 << fp8ExpBits) - 2   // 30 for E5M2, 14 for E4M3

    if (fp32Exp == 0) return 0                     // zero / subnormal → 0

    val fp8ExpFull = fp32Exp - sharedExp + fp8Bias

    if (fp8ExpFull <= 0) return 0                  // underflow → 0

    if (fp8ExpFull > fp8MaxNormalExp) {            // saturate to ±max-normal
      val maxMant = (1 << fp8MantBits) - 1
      return (sign << 7) | (fp8MaxNormalExp << fp8MantBits) | maxMant
    }

    // Truncate mantissa with RNE rounding.
    // fp32_man[22 : 23-fp8MantBits]  — fp8 mantissa bits
    // fp32_man[22-fp8MantBits]        — guard bit
    // fp32_man[21-fp8MantBits : 0]    — sticky bits
    val fp8MantRaw = fp32Man >>> (23 - fp8MantBits)
    val guardBit   = (fp32Man >>> (22 - fp8MantBits)) & 1
    val stickyBits =
      if (22 - fp8MantBits > 0) (fp32Man & ((1 << (22 - fp8MantBits)) - 1)) != 0
      else false
    val roundUp = guardBit == 1 && ((fp8MantRaw & 1) == 1 || stickyBits)
    val fp8Mant = (fp8MantRaw + (if (roundUp) 1 else 0)) & ((1 << fp8MantBits) - 1)
    val fp8Exp  = fp8ExpFull & ((1 << fp8ExpBits) - 1)

    (sign << 7) | (fp8Exp << fp8MantBits) | fp8Mant
  }

  /**
   * Full SW model for a complete block.
   * @param block  Indexed as block(row)(col), dimensions tileRows × blockSize.
   * @return  (shared scale per row,  fp8 element per row per col)
   */
  def swRequantBlock(
    block: Seq[Seq[Float]],
    t: ElementType
  ): (Seq[Int], Seq[Seq[Int]]) = {
    val scales = block.map(row => swSharedScale(row))
    val fp8s   = block.zip(scales).map { case (row, sc) =>
      row.map(f => swFP32toFP8(f, sc, t))
    }
    (scales, fp8s)
  }

  // =========================================================================
  // DUT helpers
  // =========================================================================

  /**
   * Pack a tileRows × tileCols matrix of FP32 values into the flat input bus.
   *
   * SV convention [0:R-1][0:C-1][31:0]:
   *   element [i][j] occupies MSB-slot k = i*C + j.
   *   Flat bit range: hi = (nIn-k-1)*32 + 31, lo = (nIn-k-1)*32
   */
  def packFP32(tileRows: Int, tileCols: Int, data: Seq[Seq[Float]]): BigInt = {
    val nIn = tileRows * tileCols
    var result = BigInt(0)
    for (row <- 0 until tileRows; col <- 0 until tileCols) {
      val k     = row * tileCols + col
      val bits  = floatToRawIntBits(data(row)(col)).toLong & 0xFFFFFFFFL
      val shift = (nIn - k - 1) * 32
      result |= BigInt(bits) << shift
    }
    result
  }

  /**
   * Extract the shared scale for one row from the packed shared_scale_out bus.
   * Row 0 is at MSB (SV [0:tileRows-1][7:0]).
   */
  def extractScale(packed: BigInt, tileRows: Int, row: Int): Int =
    ((packed >> ((tileRows - row - 1) * 8)) & 0xFF).toInt

  /**
   * Extract fp8_out[row][col] from the packed fp8_out bus.
   * SV [0:tileRows-1][0:blockSize-1][7:0]: element [row][col] at MSB-slot
   * k = row*blockSize + col.
   */
  def extractFP8(
    packed: BigInt, tileRows: Int, blockSize: Int, row: Int, col: Int
  ): Int = {
    val k     = row * blockSize + col
    val nOut  = tileRows * blockSize
    val shift = (nOut - k - 1) * 8
    ((packed >> shift) & 0xFF).toInt
  }

  /**
   * Drive one tile-column batch: poke valid_in=1 + fp32_in, step, then
   * de-assert valid_in.  The batch data is a tileRows × tileCols matrix.
   */
  def driveBatch(dut: RequantFP8, batchData: Seq[Seq[Float]]): Unit = {
    val cfg    = dut.cfg
    val fp32Bus = packFP32(cfg.tileRows, cfg.tileCols, batchData)
    dut.io.fp32_in.poke(fp32Bus.U)
    dut.io.valid_in.poke(true.B)
    dut.clock.step()
    dut.io.valid_in.poke(false.B)
  }

  /**
   * Drive a complete block of blockSize columns split across
   * batchesPerBlock pulses of tileCols columns each.
   * @param block  Indexed block(row)(col), col in 0..blockSize-1.
   */
  def driveBlock(dut: RequantFP8, block: Seq[Seq[Float]]): Unit = {
    val cfg = dut.cfg
    for (b <- 0 until cfg.batchesPerBlock) {
      val slice = block.map(row =>
        row.slice(b * cfg.tileCols, (b + 1) * cfg.tileCols))
      driveBatch(dut, slice)
    }
  }

  // =========================================================================
  // Shared test configs
  // =========================================================================

  // blockSize=16, tileRows=2, tileCols=4  →  batchesPerBlock=4
  val cfgE5M2 = RequantConfig(16, 4, 4, MXFormats.E5M2)
  val cfgE4M3 = RequantConfig(16, 4, 4, MXFormats.E4M3)

  // =========================================================================
  // Test 1: valid_out fires after exactly batchesPerBlock pulses
  // =========================================================================
  test("RequantFP8: valid_out timing — fires only after full block") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg       = dut.cfg
      val zeroSlice = Seq.fill(cfg.tileRows)(Seq.fill(cfg.tileCols)(0.0f))

      dut.io.valid_in.poke(false.B)
      dut.clock.step()
      dut.io.valid_out.expect(false.B, "valid_out must be false before any input")

      // First B-1 batches: valid_out stays low
      for (b <- 0 until cfg.batchesPerBlock - 1) {
        driveBatch(dut, zeroSlice)
        dut.io.valid_out.expect(false.B, s"valid_out must be false after batch $b")
      }

      // B-th batch: valid_out goes high the cycle after the clock edge
      driveBatch(dut, zeroSlice)
      dut.io.valid_out.expect(true.B, "valid_out must be true after batchesPerBlock pulses")

      // De-asserts after one idle cycle
      dut.clock.step()
      dut.io.valid_out.expect(false.B, "valid_out must de-assert after 1 cycle")
    }
  }

  // =========================================================================
  // Test 2: all-zeros block → all-zero outputs
  // =========================================================================
  test("RequantFP8 E5M2: all-zeros block → shared_scale=0 and fp8=0") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg       = dut.cfg
      val zeroBlock = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(0.0f))

      driveBlock(dut, zeroBlock)
      dut.io.valid_out.expect(true.B)

      val scaleOut = dut.io.shared_scale_out.peek().litValue
      val fp8Out   = dut.io.fp8_out.peek().litValue

      for (row <- 0 until cfg.tileRows)
        assert(extractScale(scaleOut, cfg.tileRows, row) == 0,
          s"row $row: shared_scale should be 0")

      for (row <- 0 until cfg.tileRows; col <- 0 until cfg.blockSize)
        assert(extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col) == 0,
          s"fp8[$row][$col] should be 0")
    }
  }

  // =========================================================================
  // Test 3: block of 1.0 → shared_scale=127, every element = 0x3C (E5M2)
  // =========================================================================
  test("RequantFP8 E5M2: block of 1.0 → shared_scale=127, fp8=0x3C") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg   = dut.cfg
      // E5M2 encoding of 1.0:
      //   fp32 biased exp = 127, fp8 exp = 127 - 127 + 15 = 15, mant = 0
      //   → (0 << 7) | (15 << 2) | 0 = 0x3C
      val block = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(1.0f))

      driveBlock(dut, block)
      dut.io.valid_out.expect(true.B)

      val scaleOut = dut.io.shared_scale_out.peek().litValue
      val fp8Out   = dut.io.fp8_out.peek().litValue

      for (row <- 0 until cfg.tileRows) {
        val sc = extractScale(scaleOut, cfg.tileRows, row)
        assert(sc == 127, s"row $row: shared_scale=$sc, expected 127")
        for (col <- 0 until cfg.blockSize) {
          val fp8 = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
          assert(fp8 == 0x3C, f"fp8[$row][$col]=0x$fp8%02X, expected 0x3C")
        }
      }
    }
  }

  // =========================================================================
  // Test 4: max-exponent selection — row with larger value sets the scale
  // =========================================================================
  test("RequantFP8 E5M2: max-exponent selection with [1.0, 2.0] mix") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg = dut.cfg
      // Row 0: alternating 1.0 / 2.0 → max biased exp = 128
      // Row 1: all 1.0              → max biased exp = 127
      val block = Seq(
        Seq.tabulate(cfg.blockSize)(i => if (i % 2 == 0) 1.0f else 2.0f),
        Seq.fill(cfg.blockSize)(1.0f),
        Seq.fill(cfg.blockSize)(1.0f),
        Seq.fill(cfg.blockSize)(1.0f)
      )
      val (expScales, expFP8) = swRequantBlock(block, cfg.outputType)

      driveBlock(dut, block)
      dut.io.valid_out.expect(true.B)

      val scaleOut = dut.io.shared_scale_out.peek().litValue
      val fp8Out   = dut.io.fp8_out.peek().litValue

      for (row <- 0 until cfg.tileRows) {
        val hwScale = extractScale(scaleOut, cfg.tileRows, row)
        assert(hwScale == expScales(row),
          s"row $row scale: hw=$hwScale sw=${expScales(row)}")
        for (col <- 0 until cfg.blockSize) {
          val hw = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
          val sw = expFP8(row)(col)
          assert(hw == sw, f"fp8[$row][$col]: hw=0x$hw%02X sw=0x$sw%02X")
        }
      }
    }
  }

  // =========================================================================
  // Test 5: negative values preserve sign bit, magnitude matches positive
  // =========================================================================
  test("RequantFP8 E5M2: negative values preserve sign bit") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg = dut.cfg
      // Row 0: all -1.0  →  sign bit should be 1
      // Row 1: all +1.0  →  sign bit should be 0
      val block = Seq(
        Seq.fill(cfg.blockSize)(-1.0f),
        Seq.fill(cfg.blockSize)(1.0f),
        Seq.fill(cfg.blockSize)(-1.0f),
        Seq.fill(cfg.blockSize)(1.0f)
      )

      driveBlock(dut, block)
      dut.io.valid_out.expect(true.B)

      val fp8Out = dut.io.fp8_out.peek().litValue

      for (col <- 0 until cfg.blockSize) {
        val hwNeg = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, 0, col)
        val hwPos = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, 1, col)

        assert((hwNeg & 0x80) != 0,
          f"row0 col$col: sign bit should be 1, got 0x$hwNeg%02X")
        assert((hwPos & 0x80) == 0,
          f"row1 col$col: sign bit should be 0, got 0x$hwPos%02X")
        assert((hwNeg & 0x7F) == (hwPos & 0x7F),
          f"magnitude mismatch: neg=0x$hwNeg%02X pos=0x$hwPos%02X")
      }
    }
  }

  // =========================================================================
  // Test 6: underflow and saturation
  // =========================================================================
  test("RequantFP8 E5M2: underflow → 0, overflow → max-normal") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg = dut.cfg
      // Huge: 2^125 (fp32 biased exp=252), very near E5M2 representable maximum.
      // Tiny: 2^-126 (minimum normal FP32 — tiny relative to huge).
      val huge = intBitsToFloat(0x7E800000)   // fp32 biased exp = 253
      val tiny = intBitsToFloat(0x00800000)   // fp32 biased exp = 1
      val half = cfg.blockSize / 2
      // Rows: first half huge, second half tiny
      val block = Seq.fill(cfg.tileRows)(
        Seq.fill(half)(huge) ++ Seq.fill(half)(tiny)
      )
      val (expScales, expFP8) = swRequantBlock(block, cfg.outputType)

      driveBlock(dut, block)
      dut.io.valid_out.expect(true.B)

      val fp8Out = dut.io.fp8_out.peek().litValue

      for (row <- 0 until cfg.tileRows) {
        // Huge values: compare against SW model (may saturate to max-normal)
        for (col <- 0 until half) {
          val hw = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
          val sw = expFP8(row)(col)
          assert(hw == sw,
            f"huge[$row][$col]: hw=0x$hw%02X sw=0x$sw%02X")
        }
        // Tiny values: must underflow to 0
        for (col <- half until cfg.blockSize) {
          val hw = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
          assert(hw == 0x00,
            f"tiny[$row][$col]=0x$hw%02X, expected 0x00 (underflow)")
        }
      }
    }
  }

  // =========================================================================
  // Test 7: second consecutive block is fully independent (counter wraps)
  // =========================================================================
  test("RequantFP8 E5M2: consecutive blocks are independent") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg    = dut.cfg
      val block1 = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(1.0f))
      val block2 = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(2.0f))

      // Block 1: 1.0  → shared_scale = 127
      driveBlock(dut, block1)
      dut.io.valid_out.expect(true.B)
      val sc1 = extractScale(dut.io.shared_scale_out.peek().litValue, cfg.tileRows, 0)
      assert(sc1 == 127, s"block1 scale=$sc1, expected 127")

      // Block 2: 2.0  → shared_scale = 128
      driveBlock(dut, block2)
      dut.io.valid_out.expect(true.B)
      val sc2 = extractScale(dut.io.shared_scale_out.peek().litValue, cfg.tileRows, 0)
      assert(sc2 == 128, s"block2 scale=$sc2, expected 128")
    }
  }

  // =========================================================================
  // Test 8: valid_in gaps — block fills correctly with non-contiguous pulses
  // =========================================================================
  test("RequantFP8 E5M2: gaps between valid_in pulses do not corrupt buffer") {
    test(new RequantFP8(cfgE5M2)) { dut =>
      val cfg   = dut.cfg
      val block = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(1.0f))
      val (expScales, expFP8) = swRequantBlock(block, cfg.outputType)

      // Drive each batch with 2 idle cycles between pulses
      for (b <- 0 until cfg.batchesPerBlock) {
        val slice = block.map(row =>
          row.slice(b * cfg.tileCols, (b + 1) * cfg.tileCols))
        dut.io.fp32_in.poke(packFP32(cfg.tileRows, cfg.tileCols, slice).U)
        dut.io.valid_in.poke(true.B)
        dut.clock.step()
        dut.io.valid_in.poke(false.B)
        // valid_out is a one-cycle pulse: RegNext(blockDone) fires on the
        // same edge as the last valid_in, so check it immediately after.
        if (b == cfg.batchesPerBlock - 1) dut.io.valid_out.expect(true.B)
        // Two idle cycles between batches
        dut.clock.step()
        dut.clock.step()
      }

      val scaleOut = dut.io.shared_scale_out.peek().litValue
      val fp8Out   = dut.io.fp8_out.peek().litValue

      for (row <- 0 until cfg.tileRows) {
        val hwSc = extractScale(scaleOut, cfg.tileRows, row)
        assert(hwSc == expScales(row), s"row $row scale: hw=$hwSc sw=${expScales(row)}")
        for (col <- 0 until cfg.blockSize) {
          val hw = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
          val sw = expFP8(row)(col)
          assert(hw == sw, f"fp8[$row][$col]: hw=0x$hw%02X sw=0x$sw%02X")
        }
      }
    }
  }

  // =========================================================================
  // Test 9 & 10: randomized SW vs HW — E5M2 and E4M3
  // =========================================================================
  def randomizedTest(cfg: RequantConfig, seed: Long, nBlocks: Int): Unit = {
    val rng = new Random(seed)
    // Random normal FP32 in [-16, 16] — avoids denormals and extremes
    def randFP32(): Float = (rng.nextFloat() * 32.0f - 16.0f)

    test(new RequantFP8(cfg)) { dut =>
      for (blockIdx <- 0 until nBlocks) {
        val block = Seq.fill(cfg.tileRows)(Seq.fill(cfg.blockSize)(randFP32()))
        val (expScales, expFP8) = swRequantBlock(block, cfg.outputType)

        driveBlock(dut, block)
        dut.io.valid_out.expect(true.B, s"block $blockIdx: valid_out not asserted")

        val scaleOut = dut.io.shared_scale_out.peek().litValue
        val fp8Out   = dut.io.fp8_out.peek().litValue

        for (row <- 0 until cfg.tileRows) {
          val hwSc = extractScale(scaleOut, cfg.tileRows, row)
          assert(hwSc == expScales(row),
            s"block$blockIdx row$row scale: hw=$hwSc sw=${expScales(row)}")

          for (col <- 0 until cfg.blockSize) {
            val hw = extractFP8(fp8Out, cfg.tileRows, cfg.blockSize, row, col)
            val sw = expFP8(row)(col)
            assert(hw == sw,
              f"block$blockIdx fp8[$row][$col]: hw=0x$hw%02X sw=0x$sw%02X " +
              f"(input=${block(row)(col)}%.4f scale=${expScales(row)})")
          }
        }
      }
    }
  }

  test("RequantFP8 E5M2: randomized 20-block SW vs HW") {
    randomizedTest(cfgE5M2, seed = 1234L, nBlocks = 20)
  }

  test("RequantFP8 E4M3: randomized 20-block SW vs HW") {
    randomizedTest(cfgE4M3, seed = 5678L, nBlocks = 20)
  }

  // =========================================================================
  // Test 11: blockSize=32 with 8x8 tile, E4M3
  // =========================================================================
  test("RequantFP8 E4M3: blockSize=32, 8x8 tile, 4 random blocks") {
    val cfg = RequantConfig(32, 4, 8, MXFormats.E4M3)  // B = 4 batches
    randomizedTest(cfg, seed = 9999L, nBlocks = 4)
  }
}
