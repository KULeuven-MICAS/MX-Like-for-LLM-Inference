package mxfp8_pkg;
    // ---------
    // FP TYPES
    // ---------
    // | Enumerator | Format           | Width  | EXP_BITS | MAN_BITS
    // |:----------:|------------------|-------:|:--------:|:--------:
    // | E5M2       | binary8          |  8 bit | 5        | 2
    // | E4M3       | binary8          |  8 bit | 4        | 3
    // | FP32       | IEEE binary32    | 32 bit | 8        | 23
    // *NOTE:* Add new formats only at the end of the enumeration for backwards compatibilty!

    // Encoding for a format
    typedef struct packed {
    int unsigned exp_bits;
    int unsigned man_bits;
    } fp_encoding_t;

    localparam int unsigned NUM_FP_FORMATS = 3; // change me to add formats
    localparam int unsigned FP_FORMAT_BITS = $clog2(NUM_FP_FORMATS);

    // FP formats
    typedef enum logic [FP_FORMAT_BITS-1:0] {
    E5M2    = 'd0,
    E4M3    = 'd1,
    FP32    = 'd2 
    // add new formats here
    } fp_format_e;

    // Encodings for supported FP formats， mind the order!
    localparam fp_encoding_t [0:NUM_FP_FORMATS-1] FP_ENCODINGS = '{
    '{exp_bits: 5, man_bits:  2},  // E5M2, index 0
    '{exp_bits: 4, man_bits:  3},  // E4M3
    '{exp_bits: 8, man_bits: 23}   // FP32
    };

    typedef logic [0:NUM_FP_FORMATS-1]       fmt_logic_t;
    // ---------
    // INT TYPES
    // ---------
    // | Enumerator | Width  |
    // |:----------:|-------:|
    // | INT8       |  8 bit |

    // -------------------
    // RISC-V FP-SPECIFIC
    // -------------------
    // Rounding modes
    typedef enum logic [2:0] {
    RNE = 3'b000,
    RTZ = 3'b001,
    RDN = 3'b010,
    RUP = 3'b011,
    RMM = 3'b100,
    ROD = 3'b101,  // This mode is not defined in RISC-V FP-SPEC
    RSR = 3'b110,  // This mode is not defined in RISC-V FP-SPEC
    DYN = 3'b111
    } roundmode_e;

    // Status flags
    typedef struct packed {
    logic NV; // Invalid
    logic DZ; // Divide by zero
    logic OF; // Overflow
    logic UF; // Underflow
    logic NX; // Inexact
    } status_t;

    // Information about a floating point value
    typedef struct packed {
    logic is_normal;     // is the value normal
    logic is_subnormal;  // is the value subnormal
    logic is_zero;       // is the value zero
    logic is_inf;        // is the value infinity
    logic is_nan;        // is the value NaN
    logic is_signalling; // is the value a signalling NaN
    logic is_quiet;      // is the value a quiet NaN
    logic is_boxed;      // is the value properly NaN-boxed (RISC-V specific)
    } fp_info_t;

    // -------------------------------------------
    // Helper functions for FP formats and values
    // -------------------------------------------
    // Returns the width of a FP format,automatic sicne there are multiple data types
    function automatic int unsigned fp_width(fp_format_e fmt);
        return FP_ENCODINGS[fmt].exp_bits + FP_ENCODINGS[fmt].man_bits + 1;
    endfunction

    // Returns the number of expoent bits for a format
    function automatic int unsigned exp_bits(fp_format_e fmt);
        return FP_ENCODINGS[fmt].exp_bits;
    endfunction

    // Returns the number of mantissa bits for a format
    function automatic int unsigned man_bits(fp_format_e fmt);
        return FP_ENCODINGS[fmt].man_bits;
    endfunction

    // Returns the bias value for a given format (as per IEEE 754-2008)
    function automatic int unsigned bias(fp_format_e fmt);
        return unsigned'(2**(FP_ENCODINGS[fmt].exp_bits-1)-1); // symmetrical bias
    endfunction
    
    // Returns the format that can hold all the data in config
    // function automatic fp_encoding_t super_format(fmt_logic_t cfg);
    // automatic fp_encoding_t res;
    // res = '0;
    // for (int unsigned fmt = 0; fmt < NUM_FP_FORMATS; fmt++)
    //   if (cfg[fmt]) begin // only active format
    //     res.exp_bits = unsigned'(maximum(res.exp_bits, exp_bits(fp_format_e'(fmt))));
    //     res.man_bits = unsigned'(maximum(res.man_bits, man_bits(fp_format_e'(fmt))));
    //   end
    // return res;
    // endfunction

endpackage