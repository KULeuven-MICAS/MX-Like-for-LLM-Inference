package mx.mac

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import java.lang.Float.intBitsToFloat
import java.lang.Integer.toHexString

/**
 * Unit tests for ScaleToFP32 RNE (Round-to-Nearest-Even) rounding logic.
 *
 * WHY INT8 x INT8 / UE2M6?
 *   mantWidth = resScaleAddMantWidth = 28 (> 26), so EXTRA = 0 and the
 *   rounding bits (guard / round / sticky) fall into real data bits —
 *   not into the zero-padding region.  For smaller configs (e.g. E5M2 x
 *   E2M3 / UE8M0, mantWidth = 9) guard/round/sticky are always 0 and the
 *   synthesiser prunes the entire RNE tree statically.
 *
 *   Bit layout of a 28-bit normalised mantissa (lzc = 0, bit 27 = implicit 1):
 *     bit 27  : implicit 1
 *     bits 26:4 (23 bits) : mant23  ← stored as FP32 mantissa
 *     bit 3   : guard
 *     bit 2   : round
 *     bits 1:0: sticky
 *
 * RNE rule:
 *   roundUp = guard && (LSB || round || sticky)
 *
 * Four test cases:
 *   1. guard=0                  → always truncate
 *   2. guard=1, round=1         → above halfway → round up
 *   3. guard=1, r=0, s=0, LSB=0 → tie, already even → truncate
 *   4. guard=1, r=0, s=0, LSB=1 → tie, odd → round up (to even)
 */
class ScaleToFP32Test extends AnyFunSuite with ChiselScalatestTester {

  // ── configuration ─────────────────────────────────────────────────────────
  // INT8 x INT8 / UE2M6 → mantWidth = 28, resScaleAddExpWidth = 6
  val cfg = ScaleAddConfig(MXFormats.INT8, MXFormats.INT8, ScaleFormats.UE2M6)

  // Sanity-check the width calculation so the test is self-documenting.
  assert(cfg.resScaleAddMantWidth == 28,
    s"Expected mantWidth=28, got ${cfg.resScaleAddMantWidth}")

  // ── Scala golden model ─────────────────────────────────────────────────────
  /**
   * Software reference for ScaleToFP32 with the given config.
   * Mirrors the Chisel implementation exactly (LZC → shift → pad → RNE →
   * exponent bias).
   */
  def goldenScaleToFP32(inSign: Int, inExpSigned: Int, inMant: Long): Int = {
    val mantWidth = cfg.resScaleAddMantWidth  // 28

    // ① LZC: find MSB position
    val lzc = (mantWidth - 1 to 0 by -1)
      .find(i => ((inMant >> i) & 1L) == 1L)
      .map(i => mantWidth - 1 - i)
      .getOrElse(mantWidth)

    val shiftedMant = inMant << lzc           // still Long, MSB is implicit 1

    // ② Padding (EXTRA = (27 - mantWidth) max 0 = 0 for mantWidth=28)
    val EXTRA = (27 - mantWidth) max 0
    val paddedShift = shiftedMant << EXTRA    // = shiftedMant when EXTRA=0
    val safeExtractPos = EXTRA + mantWidth - 2  // = 26

    // ③ Extract fields
    val mant23    = (paddedShift >> (safeExtractPos - 22)) & 0x7FFFFFL
    val guardBit  = (paddedShift >> (safeExtractPos - 23)) & 1L
    val roundBit  = (paddedShift >> (safeExtractPos - 24)) & 1L
    val stickyLow = paddedShift & ((1L << (safeExtractPos - 24)) - 1L)
    val stickyBit = if (stickyLow != 0L) 1L else 0L

    // ④ RNE decision
    val roundUp  = guardBit == 1L && ((mant23 & 1L) == 1L || roundBit == 1L || stickyBit == 1L)
    val finalMant = (if (roundUp) mant23 + 1L else mant23) & 0x7FFFFFL

    // ⑤ Exponent: adjustedExp = inExp - lzc + expBias
    def elemFrac(t: ElementType) = if (t.name == "INT8") 0 else t.elementWidthMant
    val fracBits = elemFrac(cfg.elementTypeA) + elemFrac(cfg.elementTypeB) +
                   2 * cfg.stype.mantScaleWidth
    val expBias     = 127 + mantWidth - 1 - fracBits
    val adjustedExp = inExpSigned - lzc + expBias
    val finalExp    = if (adjustedExp >= 255) 255 else if (adjustedExp <= 0) 0 else adjustedExp

    (inSign << 31) | (finalExp << 23) | finalMant.toInt
  }

  // ── helper ─────────────────────────────────────────────────────────────────
  /**
   * Build a 28-bit inMant with implicit-1 at bit 27 (lzc=0) and the
   * supplied rounding fields.
   *
   * @param mant23 23-bit mantissa value (will be placed at bits[26:4])
   * @param guard  1 bit
   * @param round  1 bit
   * @param sticky 2-bit sticky (bits[1:0])
   */
  def buildMant(mant23: Long, guard: Int, round: Int, sticky: Int): Long =
    (1L << 27) | ((mant23 & 0x7FFFFFL) << 4) | ((guard & 1) << 3) |
    ((round & 1) << 2) | (sticky & 3)

  // Use inExp=8 (signed 6-bit) so adjustedExp = 8 + 0 + 142 = 150 (valid FP32)
  val IN_EXP   = 8
  val IN_SIGN  = 0

  // ── tests ──────────────────────────────────────────────────────────────────
  test("RNE case 1: guard=0 — always truncate") {
    val mant23 = 0x7FFFFFL  // all ones
    val inMant = buildMant(mant23, guard=0, round=1, sticky=3)  // guard=0 overrides

    val expected = goldenScaleToFP32(IN_SIGN, IN_EXP, inMant)

    test(new ScaleToFP32(cfg)) { dut =>
      dut.io.inSign.poke(IN_SIGN.U)
      dut.io.inExp .poke(IN_EXP.S)
      dut.io.inMant.poke(inMant.U)
      dut.clock.step()

      val got = dut.io.out.peek().litValue.toInt
      assert(got == expected,
        s"case1 FAIL: got 0x${toHexString(got)}  expected 0x${toHexString(expected)}" +
        s"  (${intBitsToFloat(got)} vs ${intBitsToFloat(expected)})")
      println(f"[RNE case1] guard=0  mant23=0x${mant23}%07X → no round → " +
              f"FP32=0x${expected}%08X  (${intBitsToFloat(expected)})")
    }
  }

  test("RNE case 2: guard=1, round=1 — above halfway, always round up") {
    val mant23 = 0x2AAAAL  // arbitrary, LSB=0
    val inMant = buildMant(mant23, guard=1, round=1, sticky=0)

    val expected = goldenScaleToFP32(IN_SIGN, IN_EXP, inMant)

    test(new ScaleToFP32(cfg)) { dut =>
      dut.io.inSign.poke(IN_SIGN.U)
      dut.io.inExp .poke(IN_EXP.S)
      dut.io.inMant.poke(inMant.U)
      dut.clock.step()

      val got = dut.io.out.peek().litValue.toInt
      assert(got == expected,
        s"case2 FAIL: got 0x${toHexString(got)}  expected 0x${toHexString(expected)}" +
        s"  (${intBitsToFloat(got)} vs ${intBitsToFloat(expected)})")
      println(f"[RNE case2] guard=1 round=1  mant23=0x${mant23}%07X → round up → " +
              f"FP32=0x${expected}%08X  (${intBitsToFloat(expected)})")
    }
  }

  test("RNE case 3: exact tie, LSB=0 — truncate (already even)") {
    val mant23 = 0x555554L  // LSB=0
    val inMant = buildMant(mant23, guard=1, round=0, sticky=0)

    val expected = goldenScaleToFP32(IN_SIGN, IN_EXP, inMant)

    test(new ScaleToFP32(cfg)) { dut =>
      dut.io.inSign.poke(IN_SIGN.U)
      dut.io.inExp .poke(IN_EXP.S)
      dut.io.inMant.poke(inMant.U)
      dut.clock.step()

      val got = dut.io.out.peek().litValue.toInt
      assert(got == expected,
        s"case3 FAIL: got 0x${toHexString(got)}  expected 0x${toHexString(expected)}" +
        s"  (${intBitsToFloat(got)} vs ${intBitsToFloat(expected)})")
      println(f"[RNE case3] guard=1 r=0 s=0 LSB=0 (tie→even) mant23=0x${mant23}%07X → no round → " +
              f"FP32=0x${expected}%08X  (${intBitsToFloat(expected)})")
    }
  }

  test("RNE case 4: exact tie, LSB=1 — round up (to even)") {
    val mant23 = 0x555555L  // LSB=1
    val inMant = buildMant(mant23, guard=1, round=0, sticky=0)

    val expected = goldenScaleToFP32(IN_SIGN, IN_EXP, inMant)

    test(new ScaleToFP32(cfg)) { dut =>
      dut.io.inSign.poke(IN_SIGN.U)
      dut.io.inExp .poke(IN_EXP.S)
      dut.io.inMant.poke(inMant.U)
      dut.clock.step()

      val got = dut.io.out.peek().litValue.toInt
      assert(got == expected,
        s"case4 FAIL: got 0x${toHexString(got)}  expected 0x${toHexString(expected)}" +
        s"  (${intBitsToFloat(got)} vs ${intBitsToFloat(expected)})")
      println(f"[RNE case4] guard=1 r=0 s=0 LSB=1 (tie→odd→roundup) mant23=0x${mant23}%07X → round up → " +
              f"FP32=0x${expected}%08X  (${intBitsToFloat(expected)})")
    }
  }

  test("RNE case 5: sticky=1, guard=1, round=0 — above halfway via sticky") {
    val mant23 = 0x2AAAAL  // LSB=0
    val inMant = buildMant(mant23, guard=1, round=0, sticky=1)

    val expected = goldenScaleToFP32(IN_SIGN, IN_EXP, inMant)

    test(new ScaleToFP32(cfg)) { dut =>
      dut.io.inSign.poke(IN_SIGN.U)
      dut.io.inExp .poke(IN_EXP.S)
      dut.io.inMant.poke(inMant.U)
      dut.clock.step()

      val got = dut.io.out.peek().litValue.toInt
      assert(got == expected,
        s"case5 FAIL: got 0x${toHexString(got)}  expected 0x${toHexString(expected)}" +
        s"  (${intBitsToFloat(got)} vs ${intBitsToFloat(expected)})")
      println(f"[RNE case5] guard=1 r=0 sticky=1 mant23=0x${mant23}%07X → round up → " +
              f"FP32=0x${expected}%08X  (${intBitsToFloat(expected)})")
    }
  }
}
