import chisel3._

class HelloWorld extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(8.W))
    val b = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  // Simple hardware logic: an adder
  io.out := io.a + io.b
}

// This object tells Chisel to generate the Verilog
object HelloWorldMain extends App {
  emitVerilog(new HelloWorld(), Array("--target-dir", "generated"))
}