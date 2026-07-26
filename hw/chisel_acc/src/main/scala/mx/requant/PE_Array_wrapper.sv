`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/02/24 03:30:06
// Design Name: 
// Module Name: gemm_engine_4way
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module PE_Array_wrapper #(
    parameter int unsigned SRC_WIDTH   = 8,
    parameter int unsigned DST_WIDTH   = 32,
    parameter int unsigned TileRows    = 2,
    parameter int unsigned TileCols    = 2,
    parameter int unsigned VectorSize   = 1,
    parameter int unsigned SCALE_WIDTH = 8
)(
    input  logic clk_i,
    input  logic rst_ni,

    // --- CSR setting ---
    input logic [2:0] A_mode, // precision mode for input A
    input logic [2:0] B_mode,
    input logic [1:0] Result_mode_quan, // FP mode for quantization
    input logic [1:0] group_size,
    input logic [3:0] shared_format_i, // ExMy format for shared input

    // --- Control Interface ---
    //input  logic        start_i,        // Start a batch of accumulations
    input  logic        acc_reset_i,         // Reset accumulators for a new row/col
    input  logic        send_output_i, //TODO: to indicate when to quantize output
    //output logic        busy_o,         // High while calculating
    //output logic        done_o,         // Pulse when specific count reached
    //output logic [15:0] acc_count_o,    // Current accumulation step

    // ---handshakes ----
    input   logic       A_valid_i,
    input   logic       B_valid_i,
    //input   logic       Res_ready_i,
    output  logic       A_ready_o,
    output  logic       B_ready_o,
    //output  logic       Res_valid_o,


    // --- Configuration ---
    //input  logic [15:0] target_count_i, // How many steps to accumulate before 'done'
    
    // --- Data Interface (Quad-way SIMD) ---
    input  logic [0:TileRows-1][SRC_WIDTH-1:0] op_a_i, // 2 separate A operands
    input  logic [0:TileCols-1][SRC_WIDTH-1:0] op_b_i, // 2 separate B operands
    input  logic [0:TileRows-1][SCALE_WIDTH-1:0]    shared_exp_A_i,
    input  logic [0:TileCols-1][SCALE_WIDTH-1:0]    shared_exp_B_i,
    
    // --- Output Interface ---
    output logic [0:TileRows-1][0:TileCols-1][DST_WIDTH-1:0] results_o
);

    logic [0:TileRows-1][0:TileCols-1] PE_done_bus;
    // logic       internal_valid;

    assign A_ready_o = ~send_output_i; 
    assign B_ready_o = ~send_output_i;

    // --- Control System & Counter ---
    // always_ff @(posedge clk_i or negedge rst_ni) begin
    //     if (!rst_ni) begin
    //         acc_count_o    <= '0;
    //         busy_o         <= 1'b0;
    //         internal_valid <= 1'b0;
    //     end else begin
    //         if (start_i && !busy_o) begin
    //             busy_o         <= 1'b1;
    //             acc_count_o    <= '0;
    //             internal_valid <= 1'b1;
    //         end else if (busy_o) begin
    //             if (dot_done_bus[0]) begin // Use unit 0 as the reference for the counter
    //                 acc_count_o <= acc_count_o + 1;
                    
    //                 // Automatically stop when target is reached
    //                 if (acc_count_o >= target_count_i - 1) begin
    //                     busy_o         <= 1'b0;
    //                     internal_valid <= 1'b0;
    //                 end
    //             end
    //         end
    //     end
    // end

    //assign done_o = (acc_count_o == target_count_i) && !busy_o;
    logic internal_valid;
    assign internal_valid =(A_valid_i&&B_valid_i);

    // --- Array Instantiation ---
       generate
        for (genvar i = 0; i < TileRows; i++) begin : gen_units
            for (genvar j = 0; j < TileCols; j++) begin
                DotProductUnit_E5M2_x_E5M2_scale_UE8M0 u_PE_i_j(
                    .clock(clk_i),
                    .reset(rst_ni),
                    .io_op_a_i(op_a_i[i]),
                    .io_op_b_i(op_b_i[j]),
                    .io_share_exp_A_i(shared_exp_A_i[i]),
                    .io_share_exp_B_i(shared_exp_B_i[j]),
                    .io_validIn(internal_valid),
                    .io_resetAcc(acc_reset_i),
                    .io_validOut(PE_done_bus[i][j]),
                    .io_accOut(results_o[i][j])
                );
            end
        end
    endgenerate
    // generate
    //     for (genvar i = 0; i < TileRows; i++) begin : gen_units
    //         for (genvar j = 0; j < TileCols; j++) begin
    //             onedotproduct #(
    //                 .SRC_WIDTH(SRC_WIDTH),
    //                 .SCALE_WIDTH(SCALE_WIDTH),
    //                 .DST_WIDTH(DST_WIDTH)
    //             ) u_dot (
    //                 .clk_i       (clk_i),
    //                 .rst_ni      (rst_ni),
    //                 .operands_a_i(op_a_i[i]),
    //                 .operands_b_i(op_b_i[j]),
    //                 .src_fmt_i   (mxfp8_pkg::E5M2),
    //                 .dst_fmt_i   (mxfp8_pkg::FP32),
    //                 .scale_a_i     (shared_exp_A_i[i]),
    //                 .scale_b_i   (shared_exp_B_i[j]),
    //                 .a_valid_i   (A_valid_i),
    //                 .b_valid_i   (B_valid_i),
    //                 .init_save_i (acc_reset_i),
    //                 .done_o      (PE_done_bus[i][j]),
    //                 .result_o    (results_o[i][j])
    //             );
    //         end
    //     end
    // endgenerate

endmodule