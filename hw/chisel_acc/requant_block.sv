// =============================================================================
// requant_block.sv
//
// Runtime-selectable requantization: FP32 → {BF16, FP16, MXFP8-E5M2, MXFP8-E4M3}
//
// Port summary
//   clk, rst_n    – clock / active-low asynchronous reset
//   valid_in      – input strobe
//   fp32_in[31:0] – IEEE-754 single-precision input
//   mode[1:0]     – output format select (see table below)
//   valid_out     – output strobe (registered; 1-cycle latency)
//   data_out[15:0]– converted value; upper byte is 0 for 8-bit formats
//
// mode encoding
//   2'b00  BF16        (1.8.7,  bias 127) – output in [15:0]
//   2'b01  FP16        (1.5.10, bias 15)  – output in [15:0]
//   2'b10  MXFP8-E5M2 (1.5.2,  bias 15)  – output in [7:0]
//   2'b11  MXFP8-E4M3 (1.4.3,  bias 7)   – output in [7:0]
//
// Rounding  : Round-to-Nearest-Even (RNE) for all formats
// Denormals : FP32 subnormals → ±0 (flush-to-zero)
//             Output subnormals → ±0 (flush-to-zero)
//
// E4M3 specifics (OCP MX spec)
//   No Inf encoding; FP32 Inf → E4M3 NaN
//   NaN = S.1111.111; max normal = S.1111.110 (±448)
//   Overflow clips to max normal (not Inf)
//
// E5M2 specifics
//   Inf = S.11111.00; NaN = S.11111.{01,10,11}
// =============================================================================

module requant_block (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        valid_in,
  input  logic [31:0] fp32_in,
  input  logic [1:0]  mode,

  output logic        valid_out,
  output logic [15:0] data_out    // [15:0] for BF16/FP16; [7:0] for MXFP8
);

  // ---------------------------------------------------------------------------
  // Mode constants
  // ---------------------------------------------------------------------------
  localparam logic [1:0] MODE_BF16 = 2'b00;
  localparam logic [1:0] MODE_FP16 = 2'b01;
  localparam logic [1:0] MODE_E5M2 = 2'b10;
  localparam logic [1:0] MODE_E4M3 = 2'b11;

  // FP32 exponent thresholds for each target format
  //
  // FP16 / E5M2 both use bias 15; offset from FP32 bias: 127-15 = 112
  //   min normal fp32_exp = 1 + 112 = 113
  //   max normal fp32_exp = 30 + 112 = 142  (exp=31 reserved for Inf/NaN)
  //
  // E4M3 uses bias 7; offset: 127-7 = 120
  //   min normal fp32_exp = 1 + 120 = 121
  //   max normal fp32_exp = 15 + 120 = 135  (exp=15, mant=111 is NaN)
  localparam logic [7:0] FP16_E_MIN = 8'd113;
  localparam logic [7:0] FP16_E_MAX = 8'd142;
  localparam logic [7:0] E4M3_E_MIN = 8'd121;
  localparam logic [7:0] E4M3_E_MAX = 8'd135;

  // ---------------------------------------------------------------------------
  // FP32 field extraction
  // ---------------------------------------------------------------------------
  logic        fp_s;
  logic [7:0]  fp_e;
  logic [22:0] fp_m;

  assign fp_s = fp32_in[31];
  assign fp_e = fp32_in[30:23];
  assign fp_m = fp32_in[22:0];

  // Special-case flags
  logic is_zero, is_sub, is_inf, is_nan;

  assign is_zero = (fp_e == 8'h00) & (fp_m == '0);
  assign is_sub  = (fp_e == 8'h00) & (fp_m != '0);
  assign is_inf  = (fp_e == 8'hFF) & (fp_m == '0);
  assign is_nan  = (fp_e == 8'hFF) & (fp_m != '0);

  // ===========================================================================
  // BF16  (1.8.7, bias 127 — same exponent encoding as FP32)
  //
  //   Keep fp_m[22:16] (7 bits); round using fp_m[15:0]
  //   G = fp_m[15]  R = fp_m[14]  S = |fp_m[13:0]  LSB = fp_m[16]
  // ===========================================================================
  logic        bf16_rup;
  logic [7:0]  bf16_mant8;   // {mantissa_carry, 7-bit rounded mantissa}
  logic [7:0]  bf16_exp8;    // exponent after carry propagation
  logic [15:0] r_bf16;

  assign bf16_rup   = fp_m[15] & (fp_m[14] | (|fp_m[13:0]) | fp_m[16]);
  assign bf16_mant8 = {1'b0, fp_m[22:16]} + {7'h0, bf16_rup};
  assign bf16_exp8  = fp_e + {7'h0, bf16_mant8[7]};

  always_comb begin
    priority if (is_nan)
      r_bf16 = {fp_s, 8'hFF, 7'h40};          // quiet NaN
    else if (is_inf)
      r_bf16 = {fp_s, 8'hFF, 7'h00};          // ±Inf
    else if (is_zero | is_sub)
      r_bf16 = {fp_s, 15'h0};                 // ±0  (FTZ subnormals)
    else if (bf16_exp8 == 8'hFF)
      r_bf16 = {fp_s, 8'hFF, 7'h00};          // rounding overflow → ±Inf
    else
      r_bf16 = {fp_s, bf16_exp8, bf16_mant8[6:0]};
  end

  // ===========================================================================
  // FP16  (1.5.10, bias 15)
  //
  //   fp16_exp = fp32_exp − 112
  //   Keep fp_m[22:13] (10 bits); round using fp_m[12:0]
  //   G = fp_m[12]  R = fp_m[11]  S = |fp_m[10:0]  LSB = fp_m[13]
  //
  //   fp32_exp < 113  → underflow → ±0
  //   fp32_exp > 142  → overflow  → ±Inf
  //   fp32_exp ∈ [113,142]: normal conversion, check carry-induced overflow
  // ===========================================================================
  logic        fp16_rup;
  logic [10:0] fp16_mant11;  // {mantissa_carry, 10-bit rounded mantissa}
  logic [8:0]  fp16_eadj;   // fp_e + carry − 112 (9-bit to hold carry safely)
  logic [15:0] r_fp16;

  assign fp16_rup    = fp_m[12] & (fp_m[11] | (|fp_m[10:0]) | fp_m[13]);
  assign fp16_mant11 = {1'b0, fp_m[22:13]} + {10'h0, fp16_rup};
  assign fp16_eadj   = ({1'b0, fp_e} + {8'h0, fp16_mant11[10]}) - 9'd112;

  always_comb begin
    priority if (is_nan)
      r_fp16 = {fp_s, 5'h1F, 10'h200};        // quiet NaN
    else if (is_inf)
      r_fp16 = {fp_s, 5'h1F, 10'h000};        // ±Inf
    else if (is_zero | is_sub)
      r_fp16 = {fp_s, 15'h0};                 // ±0
    else if (fp_e < FP16_E_MIN)
      r_fp16 = {fp_s, 15'h0};                 // underflow → ±0
    else if ((fp_e > FP16_E_MAX) | (fp16_eadj[4:0] == 5'h1F))
      r_fp16 = {fp_s, 5'h1F, 10'h000};        // overflow → ±Inf
    else
      r_fp16 = {fp_s, fp16_eadj[4:0], fp16_mant11[9:0]};
  end

  // ===========================================================================
  // MXFP8-E5M2  (1.5.2, bias 15 — same exponent bias as FP16)
  //
  //   Keep fp_m[22:21] (2 bits); round using fp_m[20:0]
  //   G = fp_m[20]  R = fp_m[19]  S = |fp_m[18:0]  LSB = fp_m[21]
  //
  //   exp=31 (5'h1F): mant=00 → ±Inf; mant≠00 → NaN
  //   Same underflow/overflow thresholds as FP16
  // ===========================================================================
  logic       e5m2_rup;
  logic [2:0] e5m2_mant3;   // {mantissa_carry, 2-bit rounded mantissa}
  logic [8:0] e5m2_eadj;    // fp_e + carry − 112
  logic [7:0] r_e5m2;

  assign e5m2_rup   = fp_m[20] & (fp_m[19] | (|fp_m[18:0]) | fp_m[21]);
  assign e5m2_mant3 = {1'b0, fp_m[22:21]} + {2'h0, e5m2_rup};
  assign e5m2_eadj  = ({1'b0, fp_e} + {8'h0, e5m2_mant3[2]}) - 9'd112;

  always_comb begin
    priority if (is_nan)
      r_e5m2 = {fp_s, 5'h1F, 2'b01};          // quiet NaN
    else if (is_inf)
      r_e5m2 = {fp_s, 5'h1F, 2'b00};          // ±Inf
    else if (is_zero | is_sub)
      r_e5m2 = {fp_s, 7'h00};                 // ±0
    else if (fp_e < FP16_E_MIN)
      r_e5m2 = {fp_s, 7'h00};                 // underflow → ±0
    else if ((fp_e > FP16_E_MAX) | (e5m2_eadj[4:0] == 5'h1F))
      r_e5m2 = {fp_s, 5'h1F, 2'b00};          // overflow → ±Inf
    else
      r_e5m2 = {fp_s, e5m2_eadj[4:0], e5m2_mant3[1:0]};
  end

  // ===========================================================================
  // MXFP8-E4M3  (1.4.3, bias 7)
  //
  //   e4m3_exp = fp32_exp − 120
  //   Keep fp_m[22:20] (3 bits); round using fp_m[19:0]
  //   G = fp_m[19]  R = fp_m[18]  S = |fp_m[17:0]  LSB = fp_m[20]
  //
  //   No Inf encoding:
  //     FP32 Inf → E4M3 NaN     (S.1111.111)
  //     Overflow → max normal   (S.1111.110 = ±448)
  //
  //   fp32_exp < 121 → underflow → ±0
  //   fp32_exp > 135 → overflow  → clip to max normal
  //   After rounding, exp=15 & mant=111 is the NaN encoding → clamp to 110
  // ===========================================================================
  logic       e4m3_rup;
  logic [3:0] e4m3_mant4;   // {mantissa_carry, 3-bit rounded mantissa}
  logic [8:0] e4m3_eadj;    // fp_e + carry − 120
  logic       e4m3_overflow;
  logic [7:0] r_e4m3;

  assign e4m3_rup      = fp_m[19] & (fp_m[18] | (|fp_m[17:0]) | fp_m[20]);
  assign e4m3_mant4    = {1'b0, fp_m[22:20]} + {3'h0, e4m3_rup};
  assign e4m3_eadj     = ({1'b0, fp_e} + {8'h0, e4m3_mant4[3]}) - 9'd120;
  // Overflow: static (fp_e > 135) OR carry pushed a boundary exponent to 16
  // e4m3_eadj[4] set means eadj >= 16; safe to check because underflow (fp_e < 121)
  // is caught earlier in the priority chain before this signal is used.
  assign e4m3_overflow = (fp_e > E4M3_E_MAX) | e4m3_eadj[4];

  always_comb begin
    priority if (is_nan)
      r_e4m3 = {fp_s, 4'hF, 3'b111};          // NaN (S.1111.111)
    else if (is_inf)
      r_e4m3 = {fp_s, 4'hF, 3'b111};          // FP32 Inf → E4M3 NaN (no Inf encoding)
    else if (is_zero | is_sub)
      r_e4m3 = {fp_s, 7'h00};                 // ±0
    else if (fp_e < E4M3_E_MIN)
      r_e4m3 = {fp_s, 7'h00};                 // underflow → ±0
    else if (e4m3_overflow)
      r_e4m3 = {fp_s, 4'hF, 3'b110};          // overflow → max normal ±448
    else if ((e4m3_eadj[3:0] == 4'hF) & (e4m3_mant4[2:0] == 3'b111))
      r_e4m3 = {fp_s, 4'hF, 3'b110};          // avoid NaN encoding, clamp to max
    else
      r_e4m3 = {fp_s, e4m3_eadj[3:0], e4m3_mant4[2:0]};
  end

  // ===========================================================================
  // Output mux + pipeline register
  // ===========================================================================
  logic [15:0] mux_out;

  always_comb begin
    unique case (mode)
      MODE_BF16: mux_out = r_bf16;
      MODE_FP16: mux_out = r_fp16;
      MODE_E5M2: mux_out = {8'h00, r_e5m2};
      MODE_E4M3: mux_out = {8'h00, r_e4m3};
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_out  <= '0;
      valid_out <= 1'b0;
    end else begin
      data_out  <= mux_out;
      valid_out <= valid_in;
    end
  end

endmodule
