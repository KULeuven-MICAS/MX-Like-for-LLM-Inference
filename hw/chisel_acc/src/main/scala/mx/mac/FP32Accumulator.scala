package mx.mac

import chisel3._
import chisel3.util._

class ScaleAccumulatorFP32(val scfg: ScaleAddConfig) extends Module {
  val io = IO(new Bundle {
    // Control
    val validIn   = Input(Bool())
    val resetAcc  = Input(Bool())
    val validOut  = Output(Bool())
    // Data from ScaleAddition
    val inSign    = Input(UInt(1.W))
    val inExp     = Input(SInt(scfg.resScaleAddExpWidth.W))
    val inMant    = Input(UInt(scfg.resScaleAddMantWidth.W))
    // Result
    val accOut    = Output(UInt(32.W))
  })

  // --- Registers ---
  val accReg   = RegInit(0.U(32.W))
  val validReg = RegInit(false.B)

  // --- 1. Normalization (Find leading one) ---
  // PriorityEncoder finds the lowest index of a '1'.
  // We reverse the bits to find the Highest Significant Bit.
  val lzc         = PriorityEncoder(io.inMant.asBools.reverse)
  val shiftedMant = (io.inMant << lzc)

  // --- 2. Round-to-Nearest-Even (RNE) Logic ---
  // FP32 has 23 bits of explicit mantissa.
  val mantWidth = scfg.resScaleAddMantWidth

  // ScaleAddition output convention: value = inMant × 2^(inExp − fracBits)
  // fracBits counts fractional bits from element and scale multiplications.
  // After normalization with lzc, leading 1 at bit (mantWidth-1), so:
  //   biasedExp = inExp − lzc + 127 + mantWidth − 1 − fracBits
  private def elemFrac(t: ElementType): Int = if (t.name == "INT8") 0 else t.elementWidthMant
  val fracBits = elemFrac(scfg.elementTypeA) + elemFrac(scfg.elementTypeB) +
                 2 * scfg.stype.mantScaleWidth
  val expBias  = 127 + mantWidth - 1 - fracBits

  // Minimum zero-padding needed so all extraction indices stay non-negative.
  // Constraint: (EXTRA + mantWidth - 2) - 25 >= 0  =>  EXTRA >= 27 - mantWidth
  val EXTRA = (27 - mantWidth) max 0
  val paddedShift = if (EXTRA > 0) Cat(shiftedMant(mantWidth - 1, 0), 0.U(EXTRA.W))
                   else shiftedMant(mantWidth - 1, 0)
  val safeExtractPos = EXTRA + mantWidth - 2  // first fractional bit below implicit leading-1

  val mant23    = paddedShift(safeExtractPos, safeExtractPos - 22)
  val guardBit  = paddedShift(safeExtractPos - 23)
  val roundBit  = paddedShift(safeExtractPos - 24)
  val stickyBit = paddedShift(safeExtractPos - 25, 0).orR

  val roundUp   = guardBit && (mant23(0) || roundBit || stickyBit)
  val finalMant = Mux(roundUp, mant23 + 1.U, mant23)

  // --- 3. Exponent Adjustment ---
  // Adjust based on LZC and the corrected bias (accounts for fracBits in the product)
  val adjustedExp = io.inExp - lzc.zext + expBias.S

  // Basic Overflow/Underflow Handling
  val isOverflow  = adjustedExp >= 255.S
  val isUnderflow = adjustedExp <= 0.S

  val finalExp = Mux(isOverflow, 255.U(8.W),
                 Mux(isUnderflow, 0.U(8.W), adjustedExp.asUInt(7,0)))

  // --- 4. Convert scaled input to FP32 ---
  val currentResult = Cat(io.inSign, finalExp, finalMant)

  // --- 5. FP32 Addition: accReg + currentResult (fully combinational) ---
  val valA_S = accReg(31)
  val valA_E = accReg(30, 23)
  val valA_M = Cat(valA_E.orR, accReg(22, 0)) // Add implicit bit

  val valB_S = currentResult(31)
  val valB_E = currentResult(30, 23)
  val valB_M = Cat(valB_E.orR, currentResult(22, 0))

  // .zext zero-extends 8-bit unsigned biased exponents before signed arithmetic,
  // preventing values > 127 from being misinterpreted as negative.
  val expDiff  = valA_E.zext - valB_E.zext
  val aGreater = expDiff > 0.S || (expDiff === 0.S && valA_M >= valB_M)

  val farExp = Mux(aGreater, valA_E, valB_E)
  val nearM  = Mux(aGreater, valB_M, valA_M)
  val farM   = Mux(aGreater, valA_M, valB_M)

  // Shift the smaller number's mantissa to match the larger exponent
  val absExpDiff   = Mux(aGreater, expDiff.asUInt, (-expDiff).asUInt)
  val alignedNearM = (nearM << 3) >> absExpDiff // 3 extra precision bits for rounding

  // farM is shifted left by 3 to match the x8 scale of alignedNearM,
  // so both operands carry the same 3 extra precision bits.
  // Addition: +& captures the carry at bit 27 -> 28-bit UInt.
  // Subtraction: farM is the larger-exponent operand so resMag >= 0;
  //   Cat(0,x) pads both to 28-bit before unsigned subtraction.
  val isSub  = valA_S ^ valB_S
  val farM27 = farM << 3  // 27-bit static shift; implicit-1 at bit 26
  val resMag: UInt = Mux(isSub,
    Cat(0.U(1.W), farM27) - Cat(0.U(1.W), alignedNearM),  // 28-bit UInt
    farM27 +& alignedNearM)                                 // 28-bit UInt

  // Result sign comes from the far (larger-exponent) operand.
  val resSign = Mux(aGreater, valA_S, valB_S)

  // After <<resLZC the implicit-1 lands at bit 27; explicit mantissa bits are 26..4.
  val resLZC = PriorityEncoder(resMag.asBools.reverse)
  val finalM = (resMag << resLZC)(26, 4) // Re-align to 23 bits
  val finalE = farExp.zext - resLZC.zext + 1.S

  // Exact cancellation (e.g. x + (-x)) yields resMag = 0 -> output 0.0.
  // (Without this, PriorityEncoder on all-zeros returns the last index as
  //  a 5-bit value; 27 = 0b11011 interpreted as SInt = -5, inflating finalE.)
  val isZero = resMag === 0.U
  val newAcc = Mux(isZero, 0.U, Cat(resSign, finalE(7,0), finalM(22,0)))

  // --- 6. Register update (1-cycle pipeline) ---
  // resetAcc takes priority; otherwise accumulate on every validIn pulse.
  when(io.resetAcc) {
    accReg   := 0.U
    validReg := false.B
  }.elsewhen(io.validIn) {
    accReg   := newAcc
    validReg := true.B
  }.otherwise {
    validReg := false.B
  }

  io.validOut := validReg
  io.accOut   := accReg
}

object AccumulatorMain extends App {
  val scfg = ScaleAddConfig(MXFormats.E4M3, MXFormats.E2M1, ScaleFormats.UE5M3)

  emitVerilog(
    new ScaleAccumulatorFP32(scfg),
    Array("--target-dir", "generated/accumulator")
  )
}
