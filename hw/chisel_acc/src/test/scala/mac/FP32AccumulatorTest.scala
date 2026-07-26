package mx.mac

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import java.lang.Float.intBitsToFloat

class FP32AccumulatorTest extends AnyFunSuite with ChiselScalatestTester {

  val scfg      = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)
  val mantWidth = scfg.resScaleAddMantWidth  // 14
  val expWidth  = scfg.resScaleAddExpWidth   // 9

  // fracBits mirrors the hardware: element mantissa bits + scale mantissa bits from all multiplications.
  private def elemFrac(t: ElementType): Int = if (t.name == "INT8") 0 else t.elementWidthMant
  val fracBits = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB) +
                 2 * scfg.stype.mantScaleWidth  // = 3 + 1 + 6 = 10

  // --- Software golden model ---
  // ScaleAddition output convention: value = inMant × 2^(inExp − fracBits)
  def toDouble(sign: Int, exp: Int, mant: Long): Double = {
    val mag = mant.toDouble * Math.pow(2.0, exp - fracBits)
    if (sign == 1) -mag else mag
  }

  def peekFloat(dut: ScaleAccumulatorFP32): Float =
    intBitsToFloat(dut.io.accOut.peek().litValue.toInt)

  // Drive one accumulation step (1 cycle: assert validIn, step, de-assert).
  def driveOne(dut: ScaleAccumulatorFP32, sign: Int, exp: Int, mant: Long): Unit = {
    dut.io.inSign.poke(sign.U)
    dut.io.inExp.poke(exp.S)
    dut.io.inMant.poke(mant.U)
    dut.io.validIn.poke(true.B)
    dut.clock.step()           // accReg updated at posedge; validOut will be true after this
    dut.io.validIn.poke(false.B)
  }

  // --- Test 1: Reset clears the accumulator ---
  test("FP32Accumulator: Reset clears accumulator") {
    println("\n[TEST 1] Reset clears accumulator")
    try {
      test(new ScaleAccumulatorFP32(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)
        dut.io.accOut.expect(0.U, "initial accOut should be 0")

        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.accOut.expect(0.U, "accOut should remain 0 after reset")
        dut.io.resetAcc.poke(false.B)
      }
      println("[PASSED] Reset clears accumulator")
    } catch {
      case e: Exception =>
        println(s"[FAILED] Reset clears accumulator: ${e.getMessage}")
        throw e
    }
  }

  // --- Test 2: valid handshake timing ---
  // validOut is validIn delayed by exactly 1 cycle.
  test("FP32Accumulator: valid handshake timing") {
    println("\n[TEST 2] valid handshake timing")
    try {
      test(new ScaleAccumulatorFP32(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)
        dut.io.inSign.poke(0.U)
        dut.io.inExp.poke(0.S)
        dut.io.inMant.poke((1L << (mantWidth - 1)).U)

        dut.io.validOut.expect(false.B, "validOut should be false when validIn=0")

        dut.io.validIn.poke(true.B)
        dut.clock.step()                        // accReg updated at posedge
        dut.io.validIn.poke(false.B)
        dut.io.validOut.expect(true.B, "validOut should be true one cycle after validIn")

        dut.clock.step()
        dut.io.validOut.expect(false.B, "validOut should return false when validIn=0")
      }
      println("[PASSED] valid handshake timing")
    } catch {
      case e: Exception =>
        println(s"[FAILED] valid handshake timing: ${e.getMessage}")
        throw e
    }
  }

  // --- Test 3: Accumulate three positive values ---
  test("FP32Accumulator: Accumulate positive values") {
    println("\n[TEST 3] Accumulate positive values")
    try {
      test(new ScaleAccumulatorFP32(scfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        var swAcc    = 0.0f
        val baseMant = 1L << (mantWidth - 1)  // 2^13 = 8192

        // Accumulate 1.0, 2.0, 0.5  (exponents shifted by -fracBits+mantWidth-1 = -3 vs old convention)
        val inputs = Seq((0, -3), (0, -2), (0, -4))  // (sign, exp) → 1.0, 2.0, 0.5
        for ((sign, exp) <- inputs) {
          val value = toDouble(sign, exp, baseMant).toFloat
          driveOne(dut, sign, exp, baseMant)
          swAcc += value

          val hwResult = peekFloat(dut)
          println("-" * 50)
          println(s"Added value : $value  (sign=$sign, exp=$exp, mant=$baseMant)")
          println(f"SW acc      : $swAcc%.6f")
          println(f"HW acc      : $hwResult%.6f")

          val tol = math.abs(swAcc) * 0.05f + 1e-37f
          assert(
            math.abs(hwResult - swAcc) <= tol,
            s"Accumulation mismatch: hw=$hwResult expected≈$swAcc (tol=$tol)"
          )
        }
      }
      println("[PASSED] Accumulate positive values")
    } catch {
      case e: Exception =>
        println(s"[FAILED] Accumulate positive values: ${e.getMessage}")
        throw e
    }
  }

  // --- Test 4: Accumulate a negative value ---
  test("FP32Accumulator: Accumulate negative value") {
    println("\n[TEST 4] Accumulate negative value")
    try {
      test(new ScaleAccumulatorFP32(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val baseMant = 1L << (mantWidth - 1)

        // Add +1.0  (exp=-3 with fracBits convention: 8192×2^(-3-10)=1.0)
        driveOne(dut, 0, -3, baseMant)
        val afterFirst = peekFloat(dut)
        println(s"After +1.0: hw=$afterFirst")

        // Add -0.5  (sign=1, exp=-4: 8192×2^(-4-10)=0.5)
        driveOne(dut, 1, -4, baseMant)
        val afterSecond = peekFloat(dut)
        println(s"After -0.5: hw=$afterSecond")

        // 1.0 + (-0.5) = 0.5: result must be positive and smaller than the first result
        assert(afterSecond > 0.0f, s"Expected positive result (0.5), got $afterSecond")
        assert(afterSecond < afterFirst, s"Expected result < $afterFirst, got $afterSecond")
      }
      println("[PASSED] Accumulate negative value")
    } catch {
      case e: Exception =>
        println(s"[FAILED] Accumulate negative value: ${e.getMessage}")
        throw e
    }
  }

  // --- Test 5: Reset mid-accumulation ---
  test("FP32Accumulator: Reset mid-accumulation") {
    println("\n[TEST 5] Reset mid-accumulation")
    try {
      test(new ScaleAccumulatorFP32(scfg)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(false.B)

        val baseMant = 1L << (mantWidth - 1)

        // Accumulate 1.0 so accReg is non-zero
        driveOne(dut, 0, -3, baseMant)
        val beforeReset = dut.io.accOut.peek().litValue.toInt
        println(s"Before reset: accOut = 0x${beforeReset.toHexString}")
        assert(beforeReset != 0, "accOut should be non-zero after accumulation")

        // Reset
        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.accOut.expect(0.U, "accOut should be 0 after reset")
        dut.io.resetAcc.poke(false.B)

        // Accumulate 2.0 after reset
        driveOne(dut, 0, -2, baseMant)
        val afterReset = dut.io.accOut.peek().litValue.toInt
        println(s"After reset + add 2.0: accOut = 0x${afterReset.toHexString}")
        assert(afterReset != 0, "accOut should be non-zero after post-reset accumulation")
      }
      println("[PASSED] Reset mid-accumulation")
    } catch {
      case e: Exception =>
        println(s"[FAILED] Reset mid-accumulation: ${e.getMessage}")
        throw e
    }
  }

  // --- Test 6: Repeated accumulation with VCD ---
  test("FP32Accumulator: Repeated accumulation") {
    println("\n[TEST 6] Repeated accumulation")
    try {
      test(new ScaleAccumulatorFP32(scfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.io.validIn.poke(false.B)
        dut.io.resetAcc.poke(true.B)
        dut.clock.step()
        dut.io.resetAcc.poke(false.B)

        val baseMant = 1L << (mantWidth - 1)
        var swAcc    = 0.0f

        println("=" * 60)
        println("Repeated accumulation: 10 values from 2^-5 to 2^4")
        println("=" * 60)

        for (i <- 0 until 10) {
          val exp   = i - 8   // exponents shifted by -3 (fracBits convention): -8..1 → values 2^-5..2^4
          val value = toDouble(0, exp, baseMant).toFloat
          driveOne(dut, 0, exp, baseMant)
          swAcc += value

          val hwResult = peekFloat(dut)
          println(f"Iter $i%2d | added $value%10.5f | HW = $hwResult%12.5f | SW = $swAcc%12.5f")

          val tol = math.abs(swAcc) * 0.05f + 1e-37f
          assert(
            math.abs(hwResult - swAcc) <= tol,
            s"Mismatch at iter $i: hw=$hwResult expected≈$swAcc"
          )
        }
      }
      println("[PASSED] Repeated accumulation")
    } catch {
      case e: Exception =>
        println(s"[FAILED] Repeated accumulation: ${e.getMessage}")
        throw e
    }
  }
}
