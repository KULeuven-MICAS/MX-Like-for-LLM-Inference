package mx.mac

import chisel3._
import chisel3.util._

// ============================================================
// Combinational FP32 adder (shared submodule)
// ============================================================
/** Purely-combinational IEEE-754 single-precision adder.
 *
 *  Supports both addition and subtraction (sign-magnitude).
 *  Uses 3 extra guard/round/sticky bits for RNE rounding.
 *  No special-case handling for NaN/Inf (consistent with the
 *  rest of the mx.mac pipeline).
 */
class FP32Adder extends Module {
  val io = IO(new Bundle {
    val a   = Input(UInt(32.W))
    val b   = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })

  val valA_S = io.a(31)
  val valA_E = io.a(30, 23)
  val valA_M = Cat(valA_E.orR, io.a(22, 0)) // 24-bit with implicit-1

  val valB_S = io.b(31)
  val valB_E = io.b(30, 23)
  val valB_M = Cat(valB_E.orR, io.b(22, 0))

  // Determine which operand has larger magnitude
  val expDiff  = valA_E.zext - valB_E.zext
  val aGreater = expDiff > 0.S || (expDiff === 0.S && valA_M >= valB_M)

  val farExp = Mux(aGreater, valA_E, valB_E)
  val nearM  = Mux(aGreater, valB_M, valA_M)
  val farM   = Mux(aGreater, valA_M, valB_M)

  // Align smaller operand: 3 guard bits below mantissa, then right-shift
  val absExpDiff      = Mux(aGreater, expDiff.asUInt, (-expDiff).asUInt)
  val nearM27         = Cat(nearM, 0.U(3.W))            // 27 bits: 24-bit mantissa + 3 zero guard
  val alignedNearM    = nearM27 >> absExpDiff
  // Sticky: any bit of nearM27 shifted past position 0 during alignment
  val stickyFromAlign = (nearM27 & ((1.U << absExpDiff) - 1.U)(26, 0)).orR

  // Add / subtract (28-bit result)
  val isSub  = valA_S ^ valB_S
  val farM27 = Cat(farM, 0.U(3.W))                      // 27 bits: 24-bit mantissa + 3 zero guard
  val resMag: UInt = Mux(
    isSub,
    Cat(0.U(1.W), farM27) - Cat(0.U(1.W), alignedNearM),
    farM27 +& alignedNearM
  )

  val resSign = Mux(aGreater, valA_S, valB_S)

  // Normalize: left-shift until implicit-1 lands at bit 27
  val resLZC    = PriorityEncoder(resMag.asBools.reverse)
  val normShift = resMag << resLZC

  // RNE rounding: bit 27 = implicit-1, [26:4] = mantissa, bit 3 = G, bit 2 = R, [1:0] = S
  val mant23    = normShift(26, 4)
  val guardBit  = normShift(3)
  val roundBit  = normShift(2)
  val stickyBit = normShift(1, 0).orR || stickyFromAlign

  val roundUp  = guardBit && (mant23(0) || roundBit || stickyBit)
  val roundedM = mant23 +& roundUp                      // 24 bits; bit 23 = carry out on 0xFFFFFF→0

  // If rounding overflows the mantissa, increment the exponent (mantissa wraps to 0)
  val mantCarry = roundedM(23)
  val finalM    = roundedM(22, 0)
  val finalE_wide = farExp.zext - resLZC.zext + 1.S + mantCarry.zext

  // Underflow (denormal result) → flush to zero; overflow → clamp to Infinity
  val isZero        = resMag === 0.U || finalE_wide <= 0.S
  val isExpOverflow = finalE_wide >= 255.S

  io.out := Mux(isZero,        0.U(32.W),
             Mux(isExpOverflow, Cat(resSign, 255.U(8.W), 0.U(23.W)),
                                Cat(resSign, finalE_wide.asUInt(7, 0), finalM)))
}

// ============================================================
// Direct (scale-exp-only) tree → IEEE FP32 conversion
// ============================================================
/** Merges ScaleAddition + ScaleToFP32 into a single module for the UE8M0 scale type.
 *
 *  When mantScaleWidth == 0 (UE8M0), both scale factors are pure powers of 2:
 *    scaleA = 2^(fieldA − bias),  scaleB = 2^(fieldB − bias)
 *  Their mantissa product is trivially 1×1 = 1, so ScaleAddition passes the tree
 *  mantissa through unchanged (zero-padded by 2 bits).  Consequently:
 *
 *    • ScaleAddition mantissa path eliminates both multipliers.
 *    • ScaleToFP32 LZC is always 2 (constant), replacing the variable barrel-shifter
 *      with a fixed zero-extension Cat.
 *    • The FP32 mantissa is simply the tree mantissa's fractional bits zero-padded
 *      to 23 bits — pure wiring, no logic.
 *    • The FP32 exponent is a single integer addition of tree exp + scale exp A
 *      + scale exp B + a static elaboration-time bias constant.
 *
 *  Bias derivation:
 *    expBias (ScaleToFP32) = 127 + saMantW − 1 − fracBits  (saMantW = opMantW+2 for UE8M0)
 *    lzc = 2 (always, since FixedFPReductionTree normalises treeMant and UE8M0 pads 2 zeros)
 *    fixedBias = expBias − lzc = 127 + opMantW + 2 − 1 − elemFracA − elemFracB − 2
 *              = 126 + opMantW − elemFracA − elemFracB
 *
 *  Requires: scfg.stype.mantScaleWidth == 0 (UE8M0).
 */
class DirectToFP32(val scfg: ScaleAddConfig) extends Module {
  require(scfg.stype.mantScaleWidth == 0, "DirectToFP32 requires mantScaleWidth == 0 (UE8M0)")
  override def desiredName =
    s"DirectToFP32_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}_${scfg.stype.name}"

  val io = IO(new Bundle {
    val inOpSign      = Input(UInt(1.W))
    val inOpExp       = Input(SInt(scfg.resOperatorExpWidth.W))
    val inOpMant      = Input(UInt(scfg.resOperatorMantWidth.W))
    val inShareScaleA = Input(UInt(scfg.stype.totalScaleWidth.W))  // 8-bit UE8M0 field
    val inShareScaleB = Input(UInt(scfg.stype.totalScaleWidth.W))
    val out           = Output(UInt(32.W))
  })

  private val opMantW   = scfg.resOperatorMantWidth
  // elemFrac: number of fractional mantissa bits baked into the integer product
  // (0 for INT8 whose fractional scaling is absorbed into adjExp; mantBits for FP types)
  private def elemFrac(t: ElementType): Int =
    if (t.elementWidthExp == 0) 0 else t.elementWidthMant
  private val fracBitsElem = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB)
  // Static FP32 exponent offset (constant per configuration)
  private val fixedBias    = 126 + opMantW - fracBitsElem

  // Debias scale exponents: adjExpScale = raw_field − bias (UE8M0: pure integer subtraction)
  private val scaleBias  = scfg.stype.bias  // = 127 for UE8M0
  val adjExpScaleA: SInt = io.inShareScaleA.zext - scaleBias.S
  val adjExpScaleB: SInt = io.inShareScaleB.zext - scaleBias.S

  // Total exponent: tree + both scale exponents + static bias
  val expTotal: SInt = io.inOpExp + adjExpScaleA + adjExpScaleB + fixedBias.S

  val isZero     = io.inOpMant === 0.U
  val isOverflow = expTotal >= 255.S
  val isUnder    = expTotal <= 0.S

  val fp32Exp = Mux(isOverflow, 255.U(8.W),
                Mux(isUnder || isZero, 0.U(8.W), expTotal.asUInt(7, 0)))

  // FP32 mantissa = treeMant[opMantW-2:0] zero-padded to 23 bits.
  // Derivation: lzc=2 always → shiftedMant normalises leading 1 out of the window;
  // the remaining opMantW-1 fractional bits land in fp32[22:24-opMantW], rest = 0.
  // Since opMantW ≤ 14 < 24, no rounding is needed (zero-extension only).
  val fp32Mant = Cat(io.inOpMant(opMantW - 2, 0), 0.U((24 - opMantW).W))  // 23 bits

  io.out := Mux(isZero || isUnder,   0.U(32.W),
            Mux(isOverflow,           Cat(io.inOpSign, 255.U(8.W), 0.U(23.W)),
                                      Cat(io.inOpSign, fp32Exp, fp32Mant)))
}

// ============================================================
// ScaleAddition output → IEEE FP32 conversion (shared submodule)
// ============================================================
/** Converts the (sign, exp, mant) output of ScaleAddition to a
 *  32-bit IEEE-754 float using LZC normalization and RNE rounding.
 *
 *  Replicates the normalization stage that was previously
 *  baked into ScaleAccumulatorFP32.
 */
class ScaleToFP32(val scfg: ScaleAddConfig) extends Module {
  override def desiredName =
    s"ScaleToFP32_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}_${scfg.stype.name}"

  val io = IO(new Bundle {
    val inSign = Input(UInt(1.W))
    val inExp  = Input(SInt(scfg.resScaleAddExpWidth.W))
    val inMant = Input(UInt(scfg.resScaleAddMantWidth.W))
    val out    = Output(UInt(32.W))
  })

  private def elemFrac(t: ElementType): Int =
    if (t.name == "INT8") 0 else t.elementWidthMant

  val fracBits  = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB) +
                  2 * scfg.stype.mantScaleWidth
  val mantWidth = scfg.resScaleAddMantWidth
  val expBias   = 127 + mantWidth - 1 - fracBits

  val isZero = io.inMant === 0.U

  // Normalization: find leading-one position
  val lzc         = PriorityEncoder(io.inMant.asBools.reverse)
  val shiftedMant = io.inMant << lzc

  // Ensure we always have enough bits below the implicit-1 for extraction
  val EXTRA = (27 - mantWidth) max 0
  val paddedShift =
    if (EXTRA > 0) Cat(shiftedMant(mantWidth - 1, 0), 0.U(EXTRA.W))
    else           shiftedMant(mantWidth - 1, 0)
  val safeExtractPos = EXTRA + mantWidth - 2  // bit index of first fractional bit

  // RNE rounding
  val mant23    = paddedShift(safeExtractPos, safeExtractPos - 22)
  val guardBit  = paddedShift(safeExtractPos - 23)
  val roundBit  = paddedShift(safeExtractPos - 24)
  val stickyBit = paddedShift(safeExtractPos - 25, 0).orR || (lzc > (mantWidth - 1).U) //补齐sticky

  val roundUp   = guardBit && (mant23(0) || roundBit || stickyBit)
  val roundedM  = mant23 +& roundUp // 使用 +& 保留进位

  val roundCarry = roundedM(23)
  val finalMant  = roundedM(22, 0)

  // Exponent with bias correction
  val adjustedExp = io.inExp - lzc.zext + expBias.S + roundCarry.zext
  val isOverflow  = adjustedExp >= 255.S
  val isUnderflow = adjustedExp <= 0.S

  val finalExp = Mux(isOverflow,  255.U(8.W),
                 Mux(isUnderflow, 0.U(8.W), adjustedExp.asUInt(7, 0)))

  // 零值强制清零；下溢（次正规结果）同样清零（FTZ）
  io.out := Mux(isZero || isUnderflow, 0.U(32.W), Cat(io.inSign, finalExp, finalMant))
}

// ============================================================
// Parameterized FP adder: FP(1+8+mantBits) in/out
// ============================================================
/** Same algorithm as FP32Adder but with a configurable mantissa width.
 *
 *  Input/output word width W = 1 + 8 + mantBits (sign + 8-bit exp + mantBits mantissa).
 *  When mantBits == 23 this is bit-for-bit equivalent to FP32Adder.
 *  mantBits == 7 gives BF16-precision arithmetic.
 */
class FPNAdder(val mantBits: Int) extends Module {
  require(mantBits >= 1 && mantBits <= 23, s"FPNAdder: mantBits=$mantBits out of range [1,23]")
  private val W = 1 + 8 + mantBits

  val io = IO(new Bundle {
    val a   = Input(UInt(W.W))
    val b   = Input(UInt(W.W))
    val out = Output(UInt(W.W))
  })

  val valA_S = io.a(W - 1)
  val valA_E = io.a(W - 2, mantBits)                       // 8-bit biased exponent
  val valA_M = Cat(valA_E.orR, io.a(mantBits - 1, 0))      // mantBits+1 bits, implicit-1 at MSB

  val valB_S = io.b(W - 1)
  val valB_E = io.b(W - 2, mantBits)
  val valB_M = Cat(valB_E.orR, io.b(mantBits - 1, 0))

  val expDiff  = valA_E.zext - valB_E.zext
  val aGreater = expDiff > 0.S || (expDiff === 0.S && valA_M >= valB_M)

  val farExp = Mux(aGreater, valA_E, valB_E)
  val nearM  = Mux(aGreater, valB_M, valA_M)               // mantBits+1 bits
  val farM   = Mux(aGreater, valA_M, valB_M)               // mantBits+1 bits

  // Align smaller operand: 3 guard bits below mantissa, then right-shift
  val absExpDiff      = Mux(aGreater, expDiff.asUInt, (-expDiff).asUInt)
  val nearM_G         = Cat(nearM, 0.U(3.W))               // mantBits+4 bits
  val alignedNearM    = nearM_G >> absExpDiff
  val stickyFromAlign = (nearM_G & ((1.U << absExpDiff) - 1.U)(mantBits + 3, 0)).orR

  val isSub  = valA_S ^ valB_S
  val farM_G = Cat(farM, 0.U(3.W))                         // mantBits+4 bits
  val resMag: UInt = Mux(
    isSub,
    Cat(0.U(1.W), farM_G) - Cat(0.U(1.W), alignedNearM),  // mantBits+5 bits
    farM_G +& alignedNearM                                   // mantBits+5 bits (carry captured)
  )

  val resSign = Mux(aGreater, valA_S, valB_S)

  // Normalize: left-shift until implicit-1 is at bit mantBits+4
  val resLZC    = PriorityEncoder(resMag.asBools.reverse)
  val normShift = resMag << resLZC

  // [mantBits+4]=implicit-1, [mantBits+3:4]=mantissa, [3]=G, [2]=R, [1:0]=S
  val mantN     = normShift(mantBits + 3, 4)               // mantBits bits
  val guardBit  = normShift(3)
  val roundBit  = normShift(2)
  val stickyBit = normShift(1, 0).orR || stickyFromAlign

  val roundUp   = guardBit && (mantN(0) || roundBit || stickyBit)
  val roundedM  = mantN +& roundUp                          // mantBits+1 bits

  val mantCarry    = roundedM(mantBits)
  val finalM       = roundedM(mantBits - 1, 0)
  val finalE_wide  = farExp.zext - resLZC.zext + 1.S + mantCarry.zext

  val isZero        = resMag === 0.U || finalE_wide <= 0.S
  val isExpOverflow = finalE_wide >= 255.S

  io.out := Mux(isZero,        0.U(W.W),
             Mux(isExpOverflow, Cat(resSign, 255.U(8.W), 0.U(mantBits.W)),
                                Cat(resSign, finalE_wide.asUInt(7, 0), finalM)))
}

// ============================================================
// DirectToFPn: UE8M0 tree → FP(1+8+outMantBits) conversion
// ============================================================
/** Narrowed version of DirectToFP32.
 *
 *  Identical exponent computation; mantissa is either zero-padded (when
 *  outMantBits ≥ opMantW−1) or truncated from the MSB side.
 *  Output width = 1 + 8 + outMantBits.
 *  Requires: scfg.stype.mantScaleWidth == 0 (UE8M0).
 */
class DirectToFPn(val scfg: ScaleAddConfig, val outMantBits: Int) extends Module {
  require(scfg.stype.mantScaleWidth == 0, "DirectToFPn requires UE8M0 (mantScaleWidth == 0)")
  require(outMantBits >= 1 && outMantBits <= 23)
  override def desiredName =
    s"DirectToFP${outMantBits}b_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}_${scfg.stype.name}"

  private val outW   = 1 + 8 + outMantBits
  private val opMantW = scfg.resOperatorMantWidth

  val io = IO(new Bundle {
    val inOpSign      = Input(UInt(1.W))
    val inOpExp       = Input(SInt(scfg.resOperatorExpWidth.W))
    val inOpMant      = Input(UInt(opMantW.W))
    val inShareScaleA = Input(UInt(scfg.stype.totalScaleWidth.W))
    val inShareScaleB = Input(UInt(scfg.stype.totalScaleWidth.W))
    val out           = Output(UInt(outW.W))
  })

  private def elemFrac(t: ElementType): Int =
    if (t.elementWidthExp == 0) 0 else t.elementWidthMant
  private val fracBitsElem = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB)
  private val fixedBias    = 126 + opMantW - fracBitsElem
  private val scaleBias    = scfg.stype.bias

  val adjExpScaleA: SInt = io.inShareScaleA.zext - scaleBias.S
  val adjExpScaleB: SInt = io.inShareScaleB.zext - scaleBias.S
  val expTotal:    SInt  = io.inOpExp + adjExpScaleA + adjExpScaleB + fixedBias.S

  val isZero     = io.inOpMant === 0.U
  val isOverflow = expTotal >= 255.S
  val isUnder    = expTotal <= 0.S

  val fpNExp = Mux(isOverflow, 255.U(8.W),
               Mux(isUnder || isZero, 0.U(8.W), expTotal.asUInt(7, 0)))

  // Fractional bits of the operator mantissa (opMantW-1 bits after dropping implicit-1)
  val fracW = opMantW - 1
  val fpNMant: UInt =
    if (outMantBits >= fracW)
      Cat(io.inOpMant(fracW - 1, 0), 0.U((outMantBits - fracW).W))  // zero-pad
    else
      io.inOpMant(fracW - 1, fracW - outMantBits)                    // truncate to MSBs

  io.out := Mux(isZero || isUnder,   0.U(outW.W),
            Mux(isOverflow,           Cat(io.inOpSign, 255.U(8.W), 0.U(outMantBits.W)),
                                      Cat(io.inOpSign, fpNExp, fpNMant)))
}

// ============================================================
// ScaleToFPn: ScaleAddition output → FP(1+8+outMantBits) conversion
// ============================================================
/** Narrowed version of ScaleToFP32.
 *
 *  Same LZC normalization and RNE rounding, but only outMantBits mantissa
 *  bits are kept.  Output width = 1 + 8 + outMantBits.
 */
class ScaleToFPn(val scfg: ScaleAddConfig, val outMantBits: Int) extends Module {
  require(outMantBits >= 1 && outMantBits <= 23)
  override def desiredName =
    s"ScaleToFP${outMantBits}b_${scfg.elementTypeA.name}_x_${scfg.elementTypeB.name}_${scfg.stype.name}"

  private val outW      = 1 + 8 + outMantBits
  private val mantWidth = scfg.resScaleAddMantWidth

  val io = IO(new Bundle {
    val inSign = Input(UInt(1.W))
    val inExp  = Input(SInt(scfg.resScaleAddExpWidth.W))
    val inMant = Input(UInt(mantWidth.W))
    val out    = Output(UInt(outW.W))
  })

  private def elemFrac(t: ElementType): Int =
    if (t.name == "INT8") 0 else t.elementWidthMant
  val fracBits = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB) +
                 2 * scfg.stype.mantScaleWidth
  val expBias  = 127 + mantWidth - 1 - fracBits

  val isZero = io.inMant === 0.U

  val lzc         = PriorityEncoder(io.inMant.asBools.reverse)
  val shiftedMant = io.inMant << lzc

  val EXTRA = (27 - mantWidth) max 0
  val paddedShift =
    if (EXTRA > 0) Cat(shiftedMant(mantWidth - 1, 0), 0.U(EXTRA.W))
    else           shiftedMant(mantWidth - 1, 0)
  val safeExtractPos = EXTRA + mantWidth - 2

  // Extract outMantBits instead of 23
  val mantN     = paddedShift(safeExtractPos, safeExtractPos - (outMantBits - 1))
  val guardBit  = paddedShift(safeExtractPos - outMantBits)
  val roundBit  = paddedShift(safeExtractPos - outMantBits - 1)
  val stickyBit = (if (safeExtractPos - outMantBits - 2 >= 0)
                     paddedShift(safeExtractPos - outMantBits - 2, 0)
                   else 0.U).orR || (lzc > (mantWidth - 1).U)

  val roundUp    = guardBit && (mantN(0) || roundBit || stickyBit)
  val roundedM   = mantN +& roundUp                          // outMantBits+1 bits

  val roundCarry  = roundedM(outMantBits)
  val finalMant   = roundedM(outMantBits - 1, 0)

  val adjustedExp = io.inExp - lzc.zext + expBias.S + roundCarry.zext
  val isOverflow  = adjustedExp >= 255.S
  val isUnderflow = adjustedExp <= 0.S

  val finalExp = Mux(isOverflow,  255.U(8.W),
                 Mux(isUnderflow, 0.U(8.W), adjustedExp.asUInt(7, 0)))

  io.out := Mux(isZero || isUnderflow, 0.U(outW.W),
            Mux(isOverflow,             Cat(io.inSign, 255.U(8.W), 0.U(outMantBits.W)),
                                        Cat(io.inSign, finalExp, finalMant)))
}
