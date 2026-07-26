package mx.requant

import chisel3._
import chisel3.util._

// ============================================================
// FP32 → BF16 single-element converter
// ============================================================
/**
 * Converts one FP32 to BF16 (Brain Float 16).
 *
 * BF16 = top 16 bits of FP32: {sign[1], exp[8], mant[7]}.
 * The lower 16 FP32 mantissa bits are dropped with RNE rounding.
 *
 * NaN / Inf (exponent = 0xFF) are passed through without rounding
 * to prevent mantissa carry-out from corrupting the exponent field.
 *
 * Rounding: round-to-nearest-even (RNE).
 *   round_bit = fp32[15]
 *   sticky    = fp32[14:0].orR
 *   round_up  = round_bit && (fp32[16] || sticky)
 */
class FP32ToBF16 extends Module {
  val io = IO(new Bundle {
    val fp32_in  = Input(UInt(32.W))
    val bf16_out = Output(UInt(16.W))
  })

  val isNanOrInf = io.fp32_in(30, 23).andR       // exponent all-1s
  val roundBit   = io.fp32_in(15)
  val stickyBit  = io.fp32_in(14, 0).orR
  val mantLSB    = io.fp32_in(16)                  // LSB of the BF16 mantissa field
  val roundUp    = roundBit && (mantLSB || stickyBit)

  val fp32Top16   = io.fp32_in(31, 16)
  val fp32Rounded = fp32Top16 + roundUp.asUInt      // carry propagates correctly through mant→exp

  io.bf16_out := Mux(isNanOrInf, fp32Top16, fp32Rounded)
}

// ============================================================
// Top-level: purely combinational BF16 pass-through
// ============================================================
/**
 * RequantBF16: converts tileRows × tileCols FP32 values to BF16 every cycle.
 *
 * No block buffering, no shared scale, no requantization delay.
 * valid_out mirrors valid_in combinationally.
 *
 * Input  : tileRows × tileCols × 32 bits (same packing as RequantFP8 fp32_in)
 * Output : tileRows × tileCols × 16 bits (element order preserved)
 */
class RequantBF16(val tileRows: Int, val tileCols: Int) extends Module {
  override def desiredName = s"requant"

  private val nElems = tileRows * tileCols

  val io = IO(new Bundle {
    val fp32_in   = Input(UInt((nElems * 32).W))
    val valid_in  = Input(Bool())
    val bf16_out  = Output(UInt((nElems * 16).W))
    val valid_out = Output(Bool())
  })

  // Element k is packed at fp32_in[nElems*32 - k*32 - 1 : nElems*32 - (k+1)*32]
  // (same big-endian element ordering as RequantFP8)
  def extractFP32(k: Int): UInt =
    io.fp32_in(nElems * 32 - k * 32 - 1, nElems * 32 - (k + 1) * 32)

  val bf16Elems = Seq.tabulate(nElems) { k =>
    val conv = Module(new FP32ToBF16)
    conv.io.fp32_in := extractFP32(k)
    conv.io.bf16_out
  }

  io.bf16_out  := Cat(bf16Elems)
  io.valid_out := io.valid_in   // purely combinational, no buffering
}

// ============================================================
// Emission helpers
// ============================================================
object RequantBF16Main extends App {
  Seq((4, 4), (8, 8), (4, 8), (8, 4)).foreach { case (rows, cols) =>
    println(s"Generating RequantBF16 ${rows}x${cols}")
    emitVerilog(
      new RequantBF16(rows, cols),
      Array("--target-dir", s"generated/requant/BF16_${rows}x${cols}")
    )
  }
}
