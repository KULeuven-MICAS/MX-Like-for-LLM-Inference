module CustomOperator_E2M3_to_E3M2 (
	io_inA,
	io_inB,
	io_outSign,
	io_outExp,
	io_outMant
);
	input [5:0] io_inA;
	input [5:0] io_inB;
	output wire io_outSign;
	output wire [4:0] io_outExp;
	output wire [6:0] io_outMant;
	wire [2:0] adjExpA = (io_inA[4:3] == 2'h0 ? 3'h0 : {1'h0, io_inA[4:3]} - 3'h1);
	wire [3:0] adjExpB = (io_inB[4:2] == 3'h0 ? 4'he : {1'h0, io_inB[4:2]} - 4'h3);
	assign io_outSign = io_inA[5] ^ io_inB[5];
	assign io_outExp = {{2 {adjExpA[2]}}, adjExpA} + {adjExpB[3], adjExpB};
	assign io_outMant = {3'h0, |io_inA[4:3], io_inA[2:0]} * {4'h0, |io_inB[4:2], io_inB[1:0]};
endmodule
module ScaleAddition_E2M3_to_E3M2_scale_UE5M3 (
	io_inOpSign,
	io_inOpExp,
	io_inOpMant,
	io_inShareScaleA,
	io_inShareScaleB,
	io_outSign,
	io_outExp,
	io_outMant
);
	input io_inOpSign;
	input [4:0] io_inOpExp;
	input [6:0] io_inOpMant;
	input [7:0] io_inShareScaleA;
	input [7:0] io_inShareScaleB;
	output wire io_outSign;
	output wire [8:0] io_outExp;
	output wire [14:0] io_outMant;
	wire [6:0] _scaleExpSum_T_2 = ({2'h0, io_inShareScaleA[7:3]} + {2'h0, io_inShareScaleB[7:3]}) - 7'h1e;
	wire [7:0] _GEN = {4'h0, |io_inShareScaleA[7:3], io_inShareScaleA[2:0]};
	wire [7:0] ExpAdd = {_scaleExpSum_T_2[6], _scaleExpSum_T_2} + {{3 {io_inOpExp[4]}}, io_inOpExp};
	assign io_outSign = io_inOpSign;
	assign io_outExp = {ExpAdd[7], ExpAdd};
	assign io_outMant = {7'h00, _GEN * _GEN} * {8'h00, io_inOpMant};
endmodule
module ScaleAccumulatorFP32 (
	clock,
	reset,
	io_start,
	io_resetAcc,
	io_done,
	io_inSign,
	io_inExp,
	io_inMant,
	io_accOut
);
	input clock;
	input reset;
	input io_start;
	input io_resetAcc;
	output wire io_done;
	input io_inSign;
	input [8:0] io_inExp;
	input [14:0] io_inMant;
	output wire [31:0] io_accOut;
	reg [1:0] state;
	reg [31:0] accReg;
	wire _GEN = state == 2'h0;
	wire _GEN_0 = state == 2'h1;
	always @(posedge clock)
		if (reset) begin
			state <= 2'h0;
			accReg <= 32'h00000000;
		end
		else begin : sv2v_autoblock_1
			reg [7:0] _GEN_1;
			_GEN_1 = {state, 4'h2, (io_start ? 2'h1 : state)};
			state <= _GEN_1[state * 2+:2];
			if (io_resetAcc)
				accReg <= 32'h00000000;
			else if (_GEN | ~_GEN_0)
				;
			else begin : sv2v_autoblock_2
				reg [3:0] lzc;
				reg [8:0] _adjustedExp_T_4;
				reg [7:0] finalExp;
				reg [23:0] valA_M;
				reg [29:0] shiftedMant;
				reg [23:0] valB_M;
				reg [7:0] _expDiff_T_2;
				reg aGreater;
				reg [23:0] farM;
				reg [26:0] alignedNearM;
				reg [26:0] _resM_T_3;
				reg [27:0] resM;
				reg _finalM_T;
				reg [27:0] _finalM_T_1;
				reg [26:0] _resLZC_T_4;
				reg [4:0] resLZC;
				reg [58:0] _finalM_T_6;
				lzc = (io_inMant[14] ? 4'h0 : (io_inMant[13] ? 4'h1 : (io_inMant[12] ? 4'h2 : (io_inMant[11] ? 4'h3 : (io_inMant[10] ? 4'h4 : (io_inMant[9] ? 4'h5 : (io_inMant[8] ? 4'h6 : (io_inMant[7] ? 4'h7 : (io_inMant[6] ? 4'h8 : (io_inMant[5] ? 4'h9 : (io_inMant[4] ? 4'ha : (io_inMant[3] ? 4'hb : (io_inMant[2] ? 4'hc : (io_inMant[1] ? 4'hd : 4'he))))))))))))));
				_adjustedExp_T_4 = (io_inExp - {{5 {lzc[3]}}, lzc}) + 9'h07f;
				finalExp = (_adjustedExp_T_4 == 9'h0ff ? 8'hff : ($signed(_adjustedExp_T_4) < 9'sh001 ? 8'h00 : _adjustedExp_T_4[7:0]));
				shiftedMant = {15'h0000, io_inMant} << lzc;
				valB_M = {|finalExp, shiftedMant[13:0], 9'h000};
				valA_M = {|accReg[30:23], accReg[22:0]};
				_expDiff_T_2 = accReg[30:23] - finalExp;
				aGreater = $signed(_expDiff_T_2) > -8'sh01;
				farM = (aGreater ? valA_M : valB_M);
				alignedNearM = {(aGreater ? valB_M : valA_M), 3'h0} >> (aGreater ? _expDiff_T_2 : 8'h00 - _expDiff_T_2);
				_resM_T_3 = {farM, 3'h0} - alignedNearM;
				resM = (accReg[31] ^ io_inSign ? {_resM_T_3[26], _resM_T_3} : {1'h0, farM, 3'h0} + {1'h0, alignedNearM});
				_finalM_T = $signed(resM) < 28'sh0000000;
				_finalM_T_1 = 28'h0000000 - resM;
				_resLZC_T_4 = (_finalM_T ? _finalM_T_1[27:1] : resM[27:1]);
				resLZC = (_resLZC_T_4[26] ? 5'h00 : (_resLZC_T_4[25] ? 5'h01 : (_resLZC_T_4[24] ? 5'h02 : (_resLZC_T_4[23] ? 5'h03 : (_resLZC_T_4[22] ? 5'h04 : (_resLZC_T_4[21] ? 5'h05 : (_resLZC_T_4[20] ? 5'h06 : (_resLZC_T_4[19] ? 5'h07 : (_resLZC_T_4[18] ? 5'h08 : (_resLZC_T_4[17] ? 5'h09 : (_resLZC_T_4[16] ? 5'h0a : (_resLZC_T_4[15] ? 5'h0b : (_resLZC_T_4[14] ? 5'h0c : (_resLZC_T_4[13] ? 5'h0d : (_resLZC_T_4[12] ? 5'h0e : (_resLZC_T_4[11] ? 5'h0f : (_resLZC_T_4[10] ? 5'h10 : (_resLZC_T_4[9] ? 5'h11 : (_resLZC_T_4[8] ? 5'h12 : (_resLZC_T_4[7] ? 5'h13 : (_resLZC_T_4[6] ? 5'h14 : (_resLZC_T_4[5] ? 5'h15 : (_resLZC_T_4[4] ? 5'h16 : (_resLZC_T_4[3] ? 5'h17 : (_resLZC_T_4[2] ? 5'h18 : (_resLZC_T_4[1] ? 5'h19 : {4'hd, ~_resLZC_T_4[0]}))))))))))))))))))))))))));
				_finalM_T_6 = {31'h00000000, (_finalM_T ? _finalM_T_1 : resM)} << resLZC;
				accReg <= {(aGreater ? accReg[31] : io_inSign), ((aGreater ? accReg[30:23] : finalExp) - {{3 {resLZC[4]}}, resLZC}) + 8'h01, _finalM_T_6[26:4]};
			end
		end
	initial begin : sv2v_autoblock_3
		reg [31:0] _RANDOM [0:1];
	end
	assign io_done = ~(_GEN | _GEN_0) & (state == 2'h2);
	assign io_accOut = accReg;
endmodule
module DotProductUnit_E2M3_x_E3M2_scale_UE5M3 (
	clock,
	reset,
	io_inA,
	io_inB,
	io_inScaleA,
	io_inScaleB,
	io_start,
	io_resetAcc,
	io_done,
	io_accOut
);
	input clock;
	input reset;
	input [5:0] io_inA;
	input [5:0] io_inB;
	input [7:0] io_inScaleA;
	input [7:0] io_inScaleB;
	input io_start;
	input io_resetAcc;
	output wire io_done;
	output wire [31:0] io_accOut;
	wire _scaleAdd_io_outSign;
	wire [8:0] _scaleAdd_io_outExp;
	wire [14:0] _scaleAdd_io_outMant;
	wire _operator_io_outSign;
	wire [4:0] _operator_io_outExp;
	wire [6:0] _operator_io_outMant;
	CustomOperator_E2M3_to_E3M2 operator(
		.io_inA(io_inA),
		.io_inB(io_inB),
		.io_outSign(_operator_io_outSign),
		.io_outExp(_operator_io_outExp),
		.io_outMant(_operator_io_outMant)
	);
	ScaleAddition_E2M3_to_E3M2_scale_UE5M3 scaleAdd(
		.io_inOpSign(_operator_io_outSign),
		.io_inOpExp(_operator_io_outExp),
		.io_inOpMant(_operator_io_outMant),
		.io_inShareScaleA(io_inScaleA),
		.io_inShareScaleB(io_inScaleB),
		.io_outSign(_scaleAdd_io_outSign),
		.io_outExp(_scaleAdd_io_outExp),
		.io_outMant(_scaleAdd_io_outMant)
	);
	ScaleAccumulatorFP32 accumulator(
		.clock(clock),
		.reset(reset),
		.io_start(io_start),
		.io_resetAcc(io_resetAcc),
		.io_done(io_done),
		.io_inSign(_scaleAdd_io_outSign),
		.io_inExp(_scaleAdd_io_outExp),
		.io_inMant(_scaleAdd_io_outMant),
		.io_accOut(io_accOut)
	);
endmodule
