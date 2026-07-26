package mx.requant

import chisel3._
import chisel3.util._
import mx_template.{Scale}

case class RequantINT8Config(
  blockSize: Int,
  tileRows:  Int,
  tileCols:  Int,
  scaleType: Scale = Scale.UE8M0,
  inputMantWidth: Int = 23
) {
  require(Seq(16, 32, 64).contains(blockSize),
    s"blockSize must be 16, 32, or 64; got $blockSize")
  require(Seq(4, 8,16).contains(tileRows),
    s"tileRows must be 4, 8, or 16; got $tileRows")
  require(Seq(4, 8,16).contains(tileCols),
    s"tileCols must be 4, 8, or 16; got $tileCols")
  require(blockSize % tileCols == 0,
    s"blockSize ($blockSize) must be divisible by tileCols ($tileCols)")
  require(inputMantWidth >= 8 && inputMantWidth <= 23,
    s"inputMantWidth must be in [8, 23] for INT8 requant; got $inputMantWidth")

  val batchesPerBlock: Int = blockSize / tileCols
  /** Total per-element FP input width (sign + 8-bit exp + mantissa). */
  val inputWidth: Int = 1 + 8 + inputMantWidth
}


private object INT8Limits {
  val mantBits: Int       = 6
  val emax: Int           = 0
  val maxNormSignifInt: Int = 127
}

class FP32ToMXINT8(val cfg: RequantINT8Config) extends Module {
  override def desiredName = {
    val tag = if (cfg.inputMantWidth == 23) "" else s"_in${cfg.inputWidth}"
    s"FP32ToMXINT8_${cfg.scaleType.name}_blk${cfg.blockSize}${tag}"
  }

  // Input bit layout: [sign | 8-bit exp | inputMantWidth mantissa]
  private val IN_W = cfg.inputWidth
  private val M_IN = cfg.inputMantWidth

  val io = IO(new Bundle {
    val fp32_in      = Input(UInt(IN_W.W))
    val shared_scale = Input(UInt(cfg.scaleType.totalScaleWidth.W))   // UE8M0: max biased FP32 exp; ExMy: {biased_exp[E-1:0], mant[M-1:0]}
    val int8_out     = Output(UInt(8.W))  // two's complement signed INT8
  })

  // ── Unpack narrow-FP input ───────────────────────────────
  val sign     = io.fp32_in(IN_W - 1)            // bit M_IN+8
  val fp32_exp = io.fp32_in(IN_W - 2, M_IN)      // 8 bits
  val fp32_man = io.fp32_in(M_IN - 1, 0)         // M_IN-bit mantissa
  val isZeroOrSubnormal = fp32_exp === 0.U
  val fp32_exp_s = fp32_exp.zext

  if (cfg.scaleType.mantScaleWidth == 0) {
    // ── UE8M0 path (barrel-shift; widths parameterised) ────
    val k = fp32_exp_s - io.shared_scale.zext + 6.S

    // FRAC = 1 + M_IN = width of the implicit-1 + mantissa stack.
    // Below it we pad FRAC zero bits to provide guard + sticky room.
    val FRAC         = 1 + M_IN
    val fp32FullMant = Cat(fp32_exp.orR, fp32_man)           // (1+M_IN) bits
    val mantExt      = Cat(fp32FullMant, 0.U(FRAC.W))        // 2*(1+M_IN) bits

    // shiftAmt = M_IN − k positions the int8 magnitude at bits [FRAC+6:FRAC]
    val shiftAmt = (M_IN.S - k).asUInt
    val shifted  = mantExt >> shiftAmt

    val mag7      = shifted(FRAC + 6, FRAC)
    val guardBit  = shifted(FRAC - 1)
    val stickyBit = shifted(FRAC - 2, 0).orR

    val roundUp     = guardBit && (mag7(0) || stickyBit)
    val mag8        = mag7 +& roundUp
    val magOverflow = mag8(7)

    val mag = Mux(magOverflow, 127.U(7.W), mag8(6, 0))

    val posResult = Cat(0.U(1.W), mag)
    val negResult = (~posResult + 1.U)(7, 0)

    io.int8_out := Mux(isZeroOrSubnormal, 0.U,
                   Mux(sign, negResult, posResult))

  } else {
    val M          = cfg.scaleType.mantScaleWidth
    val scaleBias  = cfg.scaleType.bias
    val correction = scaleBias - 127 + 6   // analog of FP8's "(scale_bias − 127 + outBias)"

    val scale_biased_exp = io.shared_scale(cfg.scaleType.totalScaleWidth - 1, M)
    val scale_mant_raw   = io.shared_scale(M - 1, 0)
    val isZeroScale      = !scale_biased_exp.orR && !scale_mant_raw.orR
    val isSubnormScale   = !scale_biased_exp.orR &&  scale_mant_raw.orR

    val msbPos      = Log2(Mux(scale_mant_raw === 0.U, 1.U(M.W), scale_mant_raw))
    val leftShift   = M.U - msbPos                                          // [1, M]
    val mantSubNorm = (scale_mant_raw << leftShift)(M - 1, 0)
    val effBiasedExpSubn = 1.S - leftShift.zext

    val effBiasedExp_s = Mux(isSubnormScale, effBiasedExpSubn, scale_biased_exp.zext.asSInt)
    val scaleFullMant  = Mux(isSubnormScale,
                              Cat(1.U(1.W), mantSubNorm),
                              Cat(scale_biased_exp.orR, scale_mant_raw))  // (M+1)-bit

    val M_elem = INT8Limits.mantBits           // 6
    val EXTRA  = M_elem + 3                    // 9, mirrors FP8's "outMantBits + 3"
    val IMPL   = M_IN - M + EXTRA              // implicit-1 position when q ≥ 1

    val fp32FullMant = Cat(fp32_exp.orR, fp32_man)                          // (1+M_IN)-bit
    val qNum         = Cat(fp32FullMant, 0.U(EXTRA.W))                      // (1+M_IN+EXTRA)-bit
    val safeDenom    = Mux(scaleFullMant === 0.U, 1.U((M + 1).W), scaleFullMant)
    val q_int        = qNum / safeDenom
    val q_rem        = qNum % safeDenom

    val k_s         = fp32_exp_s - effBiasedExp_s + correction.S
    val shift_amt_s = IMPL.S - k_s

    val MAX_SHIFT_INT8 = IMPL + 2
    val saturateNeg = shift_amt_s < 0.S
    val shift_amt_u = Mux(saturateNeg, 0.U,
                          Mux(shift_amt_s > MAX_SHIFT_INT8.S, MAX_SHIFT_INT8.U,
                              shift_amt_s.asUInt))

    val padded     = q_int ## 0.U(MAX_SHIFT_INT8.W)
    val shiftedPad = padded >> shift_amt_u

    val mag7Pre    = shiftedPad(MAX_SHIFT_INT8 + 6, MAX_SHIFT_INT8)
    val highBits   = shiftedPad(padded.getWidth - 1, MAX_SHIFT_INT8 + 7).orR
    val guardInt8  = shiftedPad(MAX_SHIFT_INT8 - 1)
    val stickyLow  = shiftedPad(MAX_SHIFT_INT8 - 2, 0).orR
    val stickyInt8 = stickyLow || (q_rem =/= 0.U)

    val roundUpInt8 = guardInt8.asBool && (mag7Pre(0) || stickyInt8)
    val magSum      = mag7Pre +& roundUpInt8.asUInt   // 8-bit
    val magOverflow = magSum(7)
    val magUnsat    = magSum(6, 0)

    val saturate = saturateNeg || highBits || magOverflow
    val mag      = Mux(saturate, 127.U(7.W), magUnsat)

    val posResult = Cat(0.U(1.W), mag)
    val negResult = (~posResult + 1.U)(7, 0)

    io.int8_out := Mux(isZeroOrSubnormal || isZeroScale, 0.U,
                   Mux(sign, negResult, posResult))
  }
}

class MaxScaleFinderINT8(val cfg: RequantINT8Config) extends Module {
  override def desiredName = {
    val tag = if (cfg.inputMantWidth == 23) "" else s"_in${cfg.inputWidth}"
    s"MaxScaleFinderINT8_${cfg.scaleType.name}_blk${cfg.blockSize}${tag}"
  }

  private val IN_W = cfg.inputWidth
  private val M_IN = cfg.inputMantWidth

  val io = IO(new Bundle {
    val fp32_in   = Input(Vec(cfg.blockSize, UInt(IN_W.W)))
    val max_scale = Output(UInt(cfg.scaleType.totalScaleWidth.W))
  })

  def maxTree(vals: Seq[UInt]): UInt =
    if (vals.length == 1) vals.head
    else {
      val next = vals.grouped(2).map {
        case Seq(a, b) => Mux(a >= b, a, b)
        case Seq(a)    => a
        case g         => g.head
      }.toSeq
      maxTree(next)
    }

  if (cfg.scaleType.mantScaleWidth == 0) {
    // UE8M0: just the max biased FP exponent (legacy MXINT8).
    io.max_scale := maxTree(io.fp32_in.map(_(IN_W - 2, M_IN)))

  } else {
    // ExMy: NVFP4-style ceil-encode max_abs / (127/64).
    val M             = cfg.scaleType.mantScaleWidth
    val E             = cfg.scaleType.expScaleWidth
    val scaleBias     = cfg.scaleType.bias
    val maxScaleExpV  = (1 << E) - 1

    val M_elem        = INT8Limits.mantBits
    val E_elem        = INT8Limits.emax
    val maxNormSignifInt = INT8Limits.maxNormSignifInt

    val maxMag31     = maxTree(io.fp32_in.map(_(IN_W - 2, 0)))       // (8+M_IN)-bit magnitude
    val maxBiasedExp = maxMag31(IN_W - 2, M_IN)                       // 8-bit biased exp
    val maxMant23    = maxMag31(M_IN - 1, 0)                          // M_IN-bit mantissa
    val maxIsZero    = !maxBiasedExp.orR

    val EXTRA = M + 3
    val IMPL  = M_IN - M_elem + EXTRA
    val fp32FullMant = Cat(maxBiasedExp.orR, maxMant23)
    val qNum         = Cat(fp32FullMant, 0.U(EXTRA.W))
    val divisorU     = maxNormSignifInt.U((M_elem + 1).W)
    val q_int        = qNum / divisorU
    val q_rem        = qNum % divisorU

    val qGeq1 = q_int(IMPL)

    val mant_raw = Mux(qGeq1,
                       q_int(IMPL - 1, IMPL - M),
                       q_int(IMPL - 2, IMPL - 1 - M))
    val resid_geq1 = q_int(IMPL - M - 1, 0).orR
    val resid_lt1  = q_int(IMPL - M - 2, 0).orR
    val ceilInc    = (q_rem =/= 0.U) || Mux(qGeq1, resid_geq1, resid_lt1)

    val mant_sum     = mant_raw +& ceilInc
    val mantOverflow = mant_sum(M)
    val mant_normal  = mant_sum(M - 1, 0)

    val scaleBiasedExp = maxBiasedExp.zext - 127.S - E_elem.S +
                         Mux(qGeq1, 0.S(2.W), (-1).S(2.W)) +
                         mantOverflow.zext +
                         scaleBias.S

    val isSat = scaleBiasedExp >= maxScaleExpV.S
    val isSub = scaleBiasedExp <= 0.S

    val expOut = Mux(maxIsZero, 0.U(E.W),
                 Mux(isSat,    maxScaleExpV.U(E.W),
                 Mux(isSub,    0.U(E.W),
                               scaleBiasedExp.asUInt(E - 1, 0))))
    val mantOut = Mux(maxIsZero, 0.U(M.W),
                  Mux(isSat,    ((1 << M) - 1).U(M.W),
                  Mux(isSub,    1.U(M.W),     // smallest subnormal (per FP8 design)
                                mant_normal)))

    io.max_scale := Cat(expOut, mantOut)
  }
}

class RequantBlockINT8(val cfg: RequantINT8Config) extends Module {
  override def desiredName = {
    val tag = if (cfg.inputMantWidth == 23) "" else s"_in${cfg.inputWidth}"
    s"RequantBlockINT8_${cfg.scaleType.name}_blk${cfg.blockSize}${tag}"
  }

  val io = IO(new Bundle {
    val fp32_in      = Input(Vec(cfg.blockSize, UInt(cfg.inputWidth.W)))
    val shared_scale = Output(UInt(cfg.scaleType.totalScaleWidth.W))
    val int8_out     = Output(Vec(cfg.blockSize, UInt(8.W)))
  })

  val scaleFinder = Module(new MaxScaleFinderINT8(cfg))
  scaleFinder.io.fp32_in := io.fp32_in
  io.shared_scale        := scaleFinder.io.max_scale

  for (i <- 0 until cfg.blockSize) {
    val conv = Module(new FP32ToMXINT8(cfg))
    conv.io.fp32_in      := io.fp32_in(i)
    conv.io.shared_scale := scaleFinder.io.max_scale
    io.int8_out(i)       := conv.io.int8_out
  }
}

class RequantINT8(val cfg: RequantINT8Config) extends Module {
  override def desiredName = {
    val tag = if (cfg.inputMantWidth == 23) "" else s"_in${cfg.inputWidth}"
    s"requant${tag}"
  }

  private val B    = cfg.batchesPerBlock
  private val nIn  = cfg.tileRows * cfg.tileCols
  private val IN_W = cfg.inputWidth

  val io = IO(new Bundle {
    val fp32_in          = Input(UInt((nIn * IN_W).W))
    val valid_in         = Input(Bool())
    val shared_scale_out = Output(UInt((cfg.tileRows * cfg.scaleType.totalScaleWidth).W))
    val int8_out         = Output(UInt((cfg.tileRows * cfg.blockSize * 8).W))
    val valid_out        = Output(Bool())
  })

  def extractFP32(row: Int, col: Int): UInt = {
    val k = row * cfg.tileCols + col
    io.fp32_in(nIn * IN_W - k * IN_W - 1, nIn * IN_W - (k + 1) * IN_W)
  }

  val asyncRstN = (!reset.asBool).asAsyncReset

  val buffer    = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(IN_W.W)))))
  val batchCnt  = withReset(asyncRstN)(RegInit(0.U(6.W)))
  val blockDone = io.valid_in && (batchCnt === (B - 1).U)

  when(io.valid_in) {
    for (batch <- 0 until B) {
      when(batchCnt === batch.U) {
        for (row <- 0 until cfg.tileRows; col <- 0 until cfg.tileCols)
          buffer(row)(batch * cfg.tileCols + col) := extractFP32(row, col)
      }
    }
    batchCnt := Mux(blockDone, 0.U, batchCnt + 1.U)
  }

  val sharedScaleWire = Wire(Vec(cfg.tileRows, UInt(cfg.scaleType.totalScaleWidth.W)))
  val int8Wire        = Wire(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(8.W))))

  for (row <- 0 until cfg.tileRows) {
    val rq = Module(new RequantBlockINT8(cfg))
    for (col <- 0 until cfg.blockSize) {
      val batchIdx   = col / cfg.tileCols
      val colInBatch = col % cfg.tileCols
      rq.io.fp32_in(col) := (
        if (batchIdx == B - 1)
          Mux(blockDone, extractFP32(row, colInBatch), buffer(row)(col))
        else
          buffer(row)(col)
      )
    }
    sharedScaleWire(row) := rq.io.shared_scale
    int8Wire(row)        := rq.io.int8_out
  }

  val validOutReg    = withReset(asyncRstN)(RegNext(blockDone, init = false.B))
  val sharedScaleReg = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, UInt(cfg.scaleType.totalScaleWidth.W))))
  val int8OutReg     = withReset(asyncRstN)(Reg(Vec(cfg.tileRows, Vec(cfg.blockSize, UInt(8.W)))))

  when(blockDone) {
    sharedScaleReg := sharedScaleWire
    int8OutReg     := int8Wire
  }

  io.valid_out        := validOutReg
  // Pack row 0 in LSB so little-endian memory stores [row0, row1, …, rowN-1]
  // in ascending byte address (matches software golden and RequantFP8).
  io.shared_scale_out := Cat(sharedScaleReg.reverse)
  io.int8_out := Cat((
    for (row <- 0 until cfg.tileRows; col <- 0 until cfg.blockSize)
      yield int8OutReg(row)(col)
  ).reverse)
}

object RequantINT8Main extends App {
  Seq(
    RequantINT8Config(blockSize = 32, tileRows = 4, tileCols = 4),
    RequantINT8Config(blockSize = 16, tileRows = 8, tileCols = 8),
    RequantINT8Config(blockSize = 64, tileRows = 4, tileCols = 4),
  ).foreach { cfg =>
    println(s"Generating RequantINT8: ${cfg.scaleType.name} blk${cfg.blockSize} ${cfg.tileRows}x${cfg.tileCols}")
    emitVerilog(
      new RequantINT8(cfg),
      Array("--target-dir",
            s"generated/requant/INT8_${cfg.scaleType.name}_blk${cfg.blockSize}_${cfg.tileRows}x${cfg.tileCols}")
    )
  }
}

/** Emit all ExMy variants for the INT8 requantizer. */
object AllRequantINT8Main extends App {
  val configs = Seq(
    (32, 4, 4, Scale.UE8M0),
    (32, 4, 4, Scale.UE7M1),
    (32, 4, 4, Scale.UE6M2),
    (32, 4, 4, Scale.UE4M4),
    (16, 8, 8, Scale.UE8M0),
    (16, 8, 8, Scale.UE7M1),
  )
  for ((blk, tr, tc, st) <- configs) {
    val cfg = RequantINT8Config(blk, tr, tc, st)
    println(s"Generating RequantINT8: ${st.name} blk$blk ${tr}x$tc")
    emitVerilog(
      new RequantINT8(cfg),
      Array("--target-dir", s"generated/requant/INT8_${st.name}_blk${blk}_${tr}x$tc")
    )
  }
}
