
package mx_template


final case class Elem(
  name:    String,
  e:       Int,      
  m:       Int,      
  resvNaN: Boolean,  
  impSc:   Int = 0, 
) {
  def isInt: Boolean = e == 0
  def hasHiddenBit: Boolean = !isInt

  def bias: Int = if (isInt) 0 else (1 << (e - 1)) - 1

  
  def maxOpValExp: Int = {
    if (isInt) m + impSc
    else {
      val reserved = if (resvNaN) 2 else 1
      ((1 << e) - reserved) - bias
    }
  }

  
  def minOpValExp: Int = {
    if (isInt) impSc
    else 1 - bias - m
  }

  // ── Requant/emit-facing format accessors (unify the former mx.mac.ElementType) ──
  def totalWidth: Int        = 1 + e + m   // sign + exp + mant (INT8: 1+0+7 = 8)
  def elementWidthExp: Int   = e
  def elementWidthMant: Int  = m
}

object Elem {
  val E5M2 = Elem("E5M2", 5, 2, resvNaN = true)
  val E4M3 = Elem("E4M3", 4, 3, resvNaN = false)
  val E3M2 = Elem("E3M2", 3, 2, resvNaN = false)
  val E2M3 = Elem("E2M3", 2, 3, resvNaN = false)
  val E2M1 = Elem("E2M1", 2, 1, resvNaN = false)
  val INT8 = Elem("INT8", 0, 7, resvNaN = false, impSc = -6)  

  val byName: Map[String, Elem] = Seq(E5M2, E4M3, E3M2, E2M3, E2M1, INT8)
                                    .map(e => e.name -> e).toMap
}


final case class Scale(name: String, e: Int, m: Int) {
  def bias: Int = (1 << (e - 1)) - 1

  // ── Requant/emit-facing accessors (unify the former mx.mac.ScaleType) ──
  def totalScaleWidth: Int = e + m
  def expScaleWidth: Int   = e
  def mantScaleWidth: Int  = m
}

object Scale {
  val UE8M0 = Scale("UE8M0", 8, 0)
  val UE7M1 = Scale("UE7M1", 7, 1)
  val UE6M2 = Scale("UE6M2", 6, 2)
  val UE5M3 = Scale("UE5M3", 5, 3)
  val UE4M4 = Scale("UE4M4", 4, 4)
  val UE4M3 = Scale("UE4M3", 4, 3)

  val allScales: Seq[Scale] = Seq(UE8M0, UE7M1, UE6M2, UE5M3, UE4M4, UE4M3)
  val byName: Map[String, Scale] = allScales.map(s => s.name -> s).toMap
}


object BF16 {
  val expBits:  Int = 8
  val mantBits: Int = 7
  val sigBits:  Int = mantBits + 1   
  val bias:     Int = 127
}


final case class MACUnitConfig(A: Elem, W: Elem, S: Scale, N: Int = 4) {
  def log2N: Int = chisel3.util.log2Ceil(N.max(2))
}

/** Datapath widths derived from a MACUnitConfig — mirrors `datapath_widths.py`. */
final case class Widths(cfg: MACUnitConfig) {
  private val A = cfg.A; private val W = cfg.W; private val S = cfg.S
  val prodMantW: Int = (A.m + 1) + (W.m + 1)
  val pMax: Int = A.maxOpValExp + W.maxOpValExp + 1
  val pMinVal: Int = A.minOpValExp + W.minOpValExp
  val aboveAnchor: Int = pMax + cfg.log2N + 1
  val belowAnchor: Int = math.abs(pMinVal)
  val intBits: Int = pMax + 1
  val sopFieldW: Int = 1 + cfg.log2N + intBits + belowAnchor
  val scaleMantProdW: Int =
    if (S.m == 0) 0 else 2 * (S.m + 1)
  
  val scaledTermW: Int =
    if (scaleMantProdW == 0) sopFieldW
    else sopFieldW + scaleMantProdW + 1

  val finalAdderW: Int = 1 + BF16.sigBits + scaledTermW

  val fracBitsA: Int = if (A.hasHiddenBit) A.m else 0
  val fracBitsW: Int = if (W.hasHiddenBit) W.m else 0
  val sopShift: Int = belowAnchor - fracBitsA - fracBitsW
  val signedProdW: Int = prodMantW + 1
  val sopUnitPos: Int = belowAnchor
  val termUnitPos: Int = belowAnchor + 2 * S.m
  val expSignedW: Int = {
    val maxAbs = math.max(math.abs(pMax), math.abs(pMinVal))
    chisel3.util.log2Ceil(maxAbs + 1) + 2  // +1 sign, +1 headroom
  }

  val scaleExpSumW: Int = S.e + 2

  def show(): String = {
    val header = f"config=${A.name}/${W.name}/${S.name} (N=${cfg.N})"
    val rows = Seq(
      f"Mantpre        = $prodMantW%3d   (1.mA)(1.mW) product magnitude",
      f"pMax           = $pMax%3d   max{Pexp}",
      f"pMinVal        = $pMinVal%3d   min product magnitude exp (signed)",
      f"intBits        = $intBits%3d   pMax + 1",
      f"belowAnchor    = $belowAnchor%3d   |pMinVal|",
      f"aboveAnchor    = $aboveAnchor%3d   Lutz Eq 5",
      f"sopFieldW      = $sopFieldW%3d   1 + log2N + intBits + belowAnchor",
      f"sopShift       = $sopShift%3d   const left-shift per product",
      f"scaleMantProdW = $scaleMantProdW%3d   2*(mS+1); 0 for UE8M0",
      f"scaledTermW    = $scaledTermW%3d   after post-tree scale mult",
      f"finalAdderW    = $finalAdderW%3d   with BF16 accreg",
    )
    (header +: rows).mkString("\n  ")
  }
}
