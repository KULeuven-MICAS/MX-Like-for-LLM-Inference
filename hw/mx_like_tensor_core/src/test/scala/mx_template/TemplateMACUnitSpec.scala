package mx_template

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** A small self-checking testbench for a single `TemplateMACUnit`.
 *
 *  The MAC unit is a 1-cycle unit: on a cycle with `enable=1`, the BF16
 *  accumulator register updates to `round_bf16(acc + term)`, where
 *
 *      term = (Σ_i  a_i · w_i) · scaleA · scaleW.
 *s
 *  Run:  sbt "testOnly mx_template.TemplateMACUnitSpec"
 */
object ScalarModel {
  /** Decode one element operand (sign, exp, mant fields) to a real value. */
  def decodeElem(el: Elem, sign: Int, exp: Int, mant: Int): Double = {
    val v =
      if (el.isInt) mant.toDouble * math.pow(2.0, el.impSc)
      else {
        val hidden = if (exp != 0) 1.0 else 0.0
        val sig    = hidden + mant.toDouble / (1 << el.m)
        val eEff   = (if (exp == 0) 1 else exp) - el.bias
        sig * math.pow(2.0, eEff)
      }
    if (sign != 0) -v else v
  }

  /** Decode one scale operand (exp, mant fields) to a real multiplier. */
  def decodeScale(sc: Scale, exp: Int, mant: Int): Double = {
    val eEff = (if (exp == 0) 1 else exp) - sc.bias
    if (sc.m == 0) math.pow(2.0, eEff) // UE8M0: implicit significand 1.0
    else {
      val hidden = if (exp != 0) 1.0 else 0.0
      val sig    = hidden + mant.toDouble / (1 << sc.m)
      sig * math.pow(2.0, eEff)
    }
  }

  /** Decode the BF16 accumulator register (sign, 8b exp, 7b mant) to Double. */
  def decodeBF16(sign: Int, exp: Int, mant: Int): Double = {
    if (exp == 0 && mant == 0) return 0.0
    val v =
      if (exp == 0) (mant.toDouble / 128.0) * math.pow(2.0, 1 - 127)
      else (1.0 + mant.toDouble / 128.0) * math.pow(2.0, exp - 127)
    if (sign != 0) -v else v
  }

  /** Round a real value to the nearest BF16 (round-to-nearest-even). */
  def bf16Round(x: Double): Double = {
    if (x == 0.0 || x.isNaN) return 0.0
    val sign = if (x < 0) -1.0 else 1.0
    val a    = math.abs(x)
    val e    = math.getExponent(a)          // floor(log2 a), exact for normals
    val biasedExp = e + 127
    if (biasedExp < 1) return 0.0           // underflow → flush to zero (matches HW)
    val frac   = a / math.pow(2.0, e) - 1.0 // significand fraction in [0,1)
    var mant   = math.rint(frac * 128.0).toInt
    var exp    = biasedExp
    if (mant == 128) { mant = 0; exp += 1 } // rounding carried into a new binade
    if (exp >= 255) return sign * (2.0 - math.pow(2.0, -7)) * math.pow(2.0, 254 - 127)
    sign * (1.0 + mant.toDouble / 128.0) * math.pow(2.0, exp - 127)
  }
}

final case class Operand(sign: Int, exp: Int, mant: Int)
final case class ScaleOp(exp: Int, mant: Int)
final case class MacStep(a: Seq[Operand], w: Seq[Operand], sa: ScaleOp, sw: ScaleOp)

class TemplateMACUnitSpec extends AnyFlatSpec with ChiselScalatestTester {
  import ScalarModel._

  private val CONFIGS = Seq(
    MACUnitConfig(Elem.E4M3, Elem.E4M3, Scale.UE6M2),
    MACUnitConfig(Elem.E5M2, Elem.E5M2, Scale.UE8M0),
    MACUnitConfig(Elem.INT8, Elem.INT8, Scale.UE8M0),
    MACUnitConfig(Elem.E2M1, Elem.E2M1, Scale.UE4M3),
  )

  /** Run a sequence of MAC steps on the DUT and return the decoded BF16 acc. */
  private def runOnHw(dut: TemplateMACUnit, cfg: MACUnitConfig, steps: Seq[MacStep]): Double = {
    dut.io.clearAcc.poke(true.B)
    dut.io.enable.poke(false.B)
    dut.clock.step(1)
    dut.io.clearAcc.poke(false.B)

    for (st <- steps) {
      for (i <- 0 until cfg.N) {
        dut.io.a(i).sign.poke((st.a(i).sign != 0).B)
        if (cfg.A.e > 0) dut.io.a(i).exp.poke(st.a(i).exp.U)
        dut.io.a(i).mant.poke(st.a(i).mant.U)
        dut.io.w(i).sign.poke((st.w(i).sign != 0).B)
        if (cfg.W.e > 0) dut.io.w(i).exp.poke(st.w(i).exp.U)
        dut.io.w(i).mant.poke(st.w(i).mant.U)
      }
      dut.io.scaleA.exp.poke(st.sa.exp.U)
      dut.io.scaleW.exp.poke(st.sw.exp.U)
      if (cfg.S.m > 0) {
        dut.io.scaleA.mant.poke(st.sa.mant.U)
        dut.io.scaleW.mant.poke(st.sw.mant.U)
      }
      dut.io.enable.poke(true.B)
      dut.clock.step(1)
    }
    dut.io.enable.poke(false.B)

    decodeBF16(
      if (dut.io.accOut.sign.peek().litToBoolean) 1 else 0,
      dut.io.accOut.exp.peek().litValue.toInt,
      dut.io.accOut.mant.peek().litValue.toInt,
    )
  }

  private def runModel(cfg: MACUnitConfig, steps: Seq[MacStep]): Double = {
    var acc = 0.0
    for (st <- steps) {
      val sop = (0 until cfg.N).map { i =>
        decodeElem(cfg.A, st.a(i).sign, st.a(i).exp, st.a(i).mant) *
        decodeElem(cfg.W, st.w(i).sign, st.w(i).exp, st.w(i).mant)
      }.sum
      val term = sop * decodeScale(cfg.S, st.sa.exp, st.sa.mant) *
                       decodeScale(cfg.S, st.sw.exp, st.sw.mant)
      acc = bf16Round(acc + term)
    }
    acc
  }

  private def assertClose(hw: Double, ref: Double, ctx: String): Unit = {
    if (ref == 0.0) {
      assert(hw == 0.0, s"$ctx: expected 0, got hw=$hw")
    } else {
      val ulp = math.pow(2.0, math.getExponent(ref) - 7)
      assert(math.abs(hw - ref) <= 1.5 * ulp,
        s"$ctx: hw=$hw ref=$ref |Δ|=${math.abs(hw - ref)} ulp=$ulp")
    }
  }

  it should "clear the accumulator on clearAcc" in {
    val cfg = CONFIGS.head
    test(new TemplateMACUnit(cfg)) { dut =>
      val hw = runOnHw(dut, cfg, Seq.empty)  // clear then read
      assert(hw == 0.0)
    }
  }

  it should "compute a known dot product and accumulate across steps" in {
    def one(el: Elem): Operand =
      if (el.isInt) Operand(0, 0, 1 << (-el.impSc))            // mant·2^impSc = 1
      else          Operand(0, el.bias, 0)                     // sig 1.0 · 2^0
    def oneScale(sc: Scale): ScaleOp = ScaleOp(sc.bias, 0)     // 1.0

    for (cfg <- CONFIGS) {
      test(new TemplateMACUnit(cfg)) { dut =>
        val step = MacStep(
          a  = Seq.fill(cfg.N)(one(cfg.A)),
          w  = Seq.fill(cfg.N)(one(cfg.W)),
          sa = oneScale(cfg.S), sw = oneScale(cfg.S))
        val hw1 = runOnHw(dut, cfg, Seq(step))
        assertClose(hw1, runModel(cfg, Seq(step)), s"${cfg.A.name}/${cfg.W.name}/${cfg.S.name} 1-step")
        val hw2 = runOnHw(dut, cfg, Seq(step, step))
        assertClose(hw2, runModel(cfg, Seq(step, step)), s"${cfg.A.name}/${cfg.W.name}/${cfg.S.name} 2-step")
      }
    }
  }

  it should "hold the accumulator when enable is low" in {
    val cfg = CONFIGS.head
    test(new TemplateMACUnit(cfg)) { dut =>
      def one(el: Elem) = Operand(0, el.bias, 0)
      val step = MacStep(Seq.fill(cfg.N)(one(cfg.A)), Seq.fill(cfg.N)(one(cfg.W)),
                         ScaleOp(cfg.S.bias, 0), ScaleOp(cfg.S.bias, 0))
      // one enabled step
      dut.io.clearAcc.poke(true.B); dut.io.enable.poke(false.B); dut.clock.step(1)
      dut.io.clearAcc.poke(false.B)
      for (i <- 0 until cfg.N) {
        dut.io.a(i).sign.poke(false.B); dut.io.a(i).exp.poke(step.a(i).exp.U); dut.io.a(i).mant.poke(0.U)
        dut.io.w(i).sign.poke(false.B); dut.io.w(i).exp.poke(step.w(i).exp.U); dut.io.w(i).mant.poke(0.U)
      }
      dut.io.scaleA.exp.poke(cfg.S.bias.U); dut.io.scaleW.exp.poke(cfg.S.bias.U)
      if (cfg.S.m > 0) { dut.io.scaleA.mant.poke(0.U); dut.io.scaleW.mant.poke(0.U) }
      dut.io.enable.poke(true.B); dut.clock.step(1)
      val held = dut.io.accOut.exp.peek().litValue.toInt
      // now disable and drive garbage; value must not move
      dut.io.enable.poke(false.B)
      for (i <- 0 until cfg.N) dut.io.a(i).mant.poke(((1 << cfg.A.m) - 1).U)
      dut.clock.step(3)
      assert(dut.io.accOut.exp.peek().litValue.toInt == held, "acc moved while disabled")
    }
  }

  
  it should "match the reference on random operand sequences" in {
    val rng = new scala.util.Random(0xC0FFEE)

    // Pick a valid, non-reserved element exponent near the bias (values ~O(1)).
    def randElem(el: Elem): Operand = {
      if (el.isInt) Operand(rng.nextInt(2), 0, rng.nextInt(1 << el.m))
      else {
        val lo = math.max(0, el.bias - 1)
        val hi = el.bias + 1                       // well below the reserved code
        val exp = lo + rng.nextInt(hi - lo + 1)
        Operand(rng.nextInt(2), exp, rng.nextInt(1 << el.m))
      }
    }
    // Scale exponent near the bias (multiplier ~O(1)), exp ≥ 1.
    def randScale(sc: Scale): ScaleOp = {
      val exp = (sc.bias - 1) + rng.nextInt(3)
      ScaleOp(exp, if (sc.m == 0) 0 else rng.nextInt(1 << sc.m))
    }

    for (cfg <- CONFIGS) {
      test(new TemplateMACUnit(cfg)) { dut =>
        for (t <- 0 until 120) {
          val nSteps = 1 + rng.nextInt(4)
          val steps = Seq.fill(nSteps)(MacStep(
            a  = Seq.fill(cfg.N)(randElem(cfg.A)),
            w  = Seq.fill(cfg.N)(randElem(cfg.W)),
            sa = randScale(cfg.S), sw = randScale(cfg.S)))
          val hw  = runOnHw(dut, cfg, steps)
          val ref = runModel(cfg, steps)
          assertClose(hw, ref, s"${cfg.A.name}/${cfg.W.name}/${cfg.S.name} rand#$t")
        }
      }
    }
  }
}
