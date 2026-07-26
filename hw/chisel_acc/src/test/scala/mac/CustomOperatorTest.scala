package mx.mac

import chisel3._
import chiseltest._
import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

class CustomOperatorTest extends AnyFunSuite with ChiselScalatestTester {

  // --- 1. 基于数学定义的真实值转换 (真正的 Golden Standard) ---
  def toRealValue(raw: Int, t: ElementType): Double = {
    val sign = if (((raw >> (t.totalWidth - 1)) & 0x1) == 1) -1.0 else 1.0
    
    if (t.name == "INT8") {
      // 匹配硬件：将 INT8 视为补码处理 (2's Complement)
      // sign = bit[7], magnitude = 2's-complement-negate of bits[6:0] when sign=1
      val raw7      = raw & 0x7F
      val signBit   = (raw >> 7) & 1
      val magnitude = if (signBit == 1) (-raw7) & 0x7F else raw7
      (if (signBit == 1) -1.0 else 1.0) * magnitude.toDouble
    } else {
      val expMask = (1 << t.elementWidthExp) - 1
      val mantMask = (1 << t.elementWidthMant) - 1
      val exp = (raw >> t.elementWidthMant) & expMask
      val mant = raw & mantMask
      
      if (exp == 0) {
        // 次正规数 (Subnormal): (-1)^s * (0.mant) * 2^(1 - bias)
        sign * (mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, 1 - t.bias)
      } else {
        // 正规数 (Normal): (-1)^s * (1.mant) * 2^(exp - bias)
        sign * (1.0 + mant.toDouble / (1 << t.elementWidthMant)) * Math.pow(2, exp - t.bias)
      }
    }
  }

  // 计算理想的输出分量
  def calculateExpected(typeA: ElementType, typeB: ElementType, rawA: Int, rawB: Int): (Int, Int, BigInt) = {
    def getLogicalParts(raw: Int, t: ElementType) = {
      val sign = (raw >> (t.totalWidth - 1)) & 0x1
      if (t.name == "INT8") {
        // 2's complement: negate lower 7 bits when sign bit is set
        val raw7      = raw & 0x7F
        val magnitude = if (sign == 1) (-raw7) & 0x7F else raw7
        (sign, 0, BigInt(magnitude))
      } else {
        val exp = (raw >> t.elementWidthMant) & ((1 << t.elementWidthExp) - 1)
        val mant = raw & ((1 << t.elementWidthMant) - 1)
        val implicitBit = if (exp > 0) 1 else 0
        val adjExp = if (exp == 0) 1 - t.bias else exp - t.bias
        (sign, adjExp, (BigInt(implicitBit) << t.elementWidthMant) | BigInt(mant))
      }
    }

    val (sA, eA, mA) = getLogicalParts(rawA, typeA)
    val (sB, eB, mB) = getLogicalParts(rawB, typeB)
    (sA ^ sB, eA + eB, mA * mB)
  }

  // --- 2. 自动化全面测试 ---
  test("CustomOperator Full Mathematical Verification") {
    val allTypes = MXFormats.allElementTypes 

    for (typeA <- allTypes; typeB <- allTypes) {
      val cfg = OperatorConfig(typeA, typeB)
      
      test(new CustomOperator(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        val rng = new Random(42)
        val numRandomTests = 50

        for (i <- 0 until numRandomTests) {
          val rawA = rng.nextInt(1 << typeA.totalWidth)
          val rawB = rng.nextInt(1 << typeB.totalWidth)

          // 1. 获取数学真实期望值
          val realA = toRealValue(rawA, typeA)
          val realB = toRealValue(rawB, typeB)
          val expectedMathResult = realA * realB

          // 2. 获取分量期望值
          val (expSign, expExp, expMant) = calculateExpected(typeA, typeB, rawA, rawB)

          // 3. 硬件激励
          dut.io.inA.poke(rawA.U)
          dut.io.inB.poke(rawB.U)
          dut.clock.step()

          // 4. 获取硬件输出并还原为数学数值
          val hwSign = dut.io.outSign.peek().litValue.toInt
          val hwExp  = dut.io.outExp.peek().litValue.toInt
          val hwMant = dut.io.outMant.peek().litValue
          
          val totalMantWidth = (if(typeA.name == "INT8") 0 else typeA.elementWidthMant) + 
                               (if(typeB.name == "INT8") 0 else typeB.elementWidthMant)
          
          // 还原公式: (-1)^s * (hwMant / 2^combinedMantWidth) * 2^hwExp
          val hwRealValue = (if(hwSign == 1) -1.0 else 1.0) * (hwMant.toDouble / (1 << totalMantWidth)) * Math.pow(2, hwExp)

          // 4. --- 打印详细调试信息 ---
          println("-" * 50)
          println(s"Test Instance: ${typeA.name} x ${typeB.name} | Loop: $i")
          println(s"Inputs:  RawA=0x${rawA.toHexString} ($realA), RawB=0x${rawB.toHexString} ($realB)")
          
          // 打印分量对比
          println(s"Output Components [Sign | Exp | Mant]:")
          println(s"  Expected: [ $expSign | $expExp | $expMant ]")
          println(s"  Hardware: [ $hwSign | $hwExp | $hwMant ]")
          
          // 打印最终数学值对比
          println(s"Values:")
          println(f"  Math Result:     $expectedMathResult%.10f")
          println(f"  Hardware Result: $hwRealValue%.10f")

          // 5. 双重断言
          // A. hardware check (ensuring wiring is correct)
          dut.io.outSign.expect(expSign.U, s"Sign Error: A=0x${rawA.toHexString}, B=0x${rawB.toHexString}")
          dut.io.outExp.expect(expExp.S, s"Exp Error: A=0x${rawA.toHexString}, B=0x${rawB.toHexString}")
          dut.io.outMant.expect(expMant.U, s"Mant Error: A=0x${rawA.toHexString}, B=0x${rawB.toHexString}")

          // B. mathematical check (ensuring result is correct)
          val tolerance = 1e-12
          assert(Math.abs(hwRealValue - expectedMathResult) < tolerance, 
            s"\n[Math Mismatch] ${typeA.name} x ${typeB.name}\n" +
            s"Input A: $realA (0x${rawA.toHexString})\n" +
            s"Input B: $realB (0x${rawB.toHexString})\n" +
            s"HW Result: $hwRealValue\n" +
            s"Math Result: $expectedMathResult")
        }
      }
    }
  }

  // --- 3. 边界值专项测试 ---
  test("CustomOperator Critical Corner Cases") {
    val allTypes = MXFormats.allElementTypes
    for (tA <- allTypes; tB <- allTypes) {
      val cfg = OperatorConfig(tA, tB)
      test(new CustomOperator(cfg)) { dut =>
        val cases = Seq(
          (0, 0),                                      // 全零
          (1 << (tA.totalWidth - 1), 0),              // -0 * 0
          (1, 1),                                      // 最小次正规数相乘
          ((1 << tA.totalWidth) - 1, (1 << tB.totalWidth) - 1) // 最大值相乘
        )
        
        cases.foreach { case (a, b) =>
          val (expS, expE, expM) = calculateExpected(tA, tB, a, b)
          dut.io.inA.poke(a.U)
          dut.io.inB.poke(b.U)
          dut.clock.step()
          
          dut.io.outSign.expect(expS.U)
          dut.io.outExp.expect(expE.S)
          dut.io.outMant.expect(expM.U)
        }
      }
    }
  }
}