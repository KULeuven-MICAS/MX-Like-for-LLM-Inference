#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# orchestrator.py — top-level automation for the SNAX-MX data/hw workflow.
#
# Steps:
#   1. Read params.hjson  (sw config)
#   2. Compute HW streamer parameters from params
#   3. Patch snax_mx_cluster.hjson → write to snax_mx_generated_cluster.hjson
#   4. Run datagen.py → emit data.h
#
# Usage (from any directory):
#   python orchestrator.py
#   python orchestrator.py --swcfg path/to/params.hjson
#   python orchestrator.py --swcfg ... --hwcfg ... --output ...

import argparse
import math
import subprocess
import sys
import hjson
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent.resolve()
CHISEL_DIR    = (SCRIPT_DIR / "../../../../../hw/mx_like_tensor_core").resolve()
DEFAULT_SWCFG  = SCRIPT_DIR / "data" / "params.hjson"
DEFAULT_HWCFG  = SCRIPT_DIR / "../../../cfg/snax_mx_cluster_template.hjson"
DEFAULT_GEN_HW = SCRIPT_DIR / "../../../cfg/snax_mx_cluster.hjson"
DEFAULT_GEN_PE = (SCRIPT_DIR / "../../../../../hw/snax_mx_alu/source").resolve()
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "data.h"
DATAGEN_PY     = SCRIPT_DIR / "data" / "datagen.py"


# ---------------------------------------------------------------------------
# HW parameter derivation
# ---------------------------------------------------------------------------

_DTYPE_BITS = {
    'fp8_e4m3': 8, 'fp8_e5m2': 8,
    'fp6_e3m2': 6, 'fp6_e2m3': 6,
    'fp4_e2m1': 4, 'mxint8':   8,
}

# Mappings from params.hjson integer codes to Chisel type names
_ELEMENT_TYPE_MAP = {0: "INT8", 1: "E5M2", 2: "E4M3", 3: "E3M2", 4: "E2M3", 5: "E2M1"}
_SCALE_FORMAT_MAP = {0: "UE8M0", 1: "UE7M1", 2: "UE6M2", 3: "UE5M3", 4: "UE4M4", 5: "UE3M5", 6: "UE2M6"}
# quantize_mode → requant output type string (used for Chisel gen directory naming)
# 0: fp32 (no requant), 1: bf16, 2: fp8_e5m2, 3: fp8_e4m3, 4: mxint8, 5: fp6_e2m3, 6: fp6_e3m2, 7: fp4_e2m1
_REQUNAT_TYPE_MAP = {0: "fp32", 1: "bf16", 2: "fp8_e5m2", 3: "fp8_e4m3", 4: "mxint8", 5: "fp6_e2m3", 6: "fp6_e3m2", 7: "fp4_e2m1"}

# Mapping from params.hjson dtype strings to _ELEMENT_TYPE_MAP integer codes
_DTYPE_TO_ELEMENT_TYPE = {
    'mxint8':   0,  # INT8
    'fp8_e5m2': 1,  # E5M2
    'fp8_e4m3': 2,  # E4M3
    'fp6_e3m2': 3,  # E3M2
    'fp6_e2m3': 4,  # E2M3
    'fp4_e2m1': 5,  # E2M1
}

_REQUANT_OUT_TAG = {
    0: ("FP32", 32),
    1: ("BF16", 16),
    2: ("E5M2",  8),
    3: ("E4M3",  8),
    4: ("INT8",  8),
    5: ("E2M3",  6),
    6: ("E3M2",  6),
    7: ("E2M1",  4),
}


def _o_bitwidth(quantize_mode: int) -> int:
    return {0: 32, 1: 16, 2: 8, 3: 8, 4: 8, 5: 6, 6: 6, 7: 4}.get(quantize_mode, 32)


def compute_hw_cfg(p: dict) -> dict:
    """
    Derive streamer config fields from sw params.

    Reader tile widths (bytes, rounded up to 8-byte TCDM boundary):
      A tile : parfor_M × parfor_K × A_bit_width / 8
      B tile : parfor_K × parfor_N × B_bit_width / 8
      Scale  : ceil((parfor_M + parfor_N) / 8) × 8

    Writer tile width:
      O tile : parfor_M × parfor_N × o_bitwidth / 8

    num_channel for each mover = tile_bytes / 8  (one 64-bit TCDM port each)
    spatial_bounds for readers = [[1]] (one element per 64-bit port)
    spatial_bounds for writer  = [[64 / o_bitwidth]] (pack multiple outputs per port)

    temporal_dim for output writer: 2 for output-stationary, 3 otherwise.
    """
    parfor_M      = p["parfor_M"]
    parfor_N      = p["parfor_N"]
    parfor_K      = p["parfor_K"]
    A_dtype       = p["A_dtype"]
    B_dtype       = p["B_dtype"]
    A_bits        = _DTYPE_BITS[A_dtype]
    B_bits        = _DTYPE_BITS[B_dtype]
    quantize_mode = p.get("quantize_mode", 0)
    o_bits        = _o_bitwidth(quantize_mode)
    stationary    = p.get("stationary", 0)
    block_size    = p.get("block_size", 32)

    # Tile sizes in bytes, aligned to 8-byte TCDM boundary.
    # For mode 2+ the PE outputs a full MX block (parfor_M × block_size elements)
    # per valid_out pulse, so O_tile must cover the entire block, not just one
    # parfor_N-wide sub-tile.
    A_tile = math.ceil(parfor_M * parfor_K * A_bits / 8 / 8) * 8
    B_tile = math.ceil(parfor_K * parfor_N * B_bits / 8 / 8) * 8
    S_tile = math.ceil((parfor_M + parfor_N) / 8) * 8
    if quantize_mode in (0, 1):
        O_tile = math.ceil(parfor_M * parfor_N  * o_bits / 8 / 8) * 8
    else:
        O_tile = math.ceil(parfor_M * block_size * o_bits / 8 / 8) * 8

    ch_A   = A_tile // 8
    ch_B   = B_tile // 8
    ch_S   = S_tile // 8
    ch_out = O_tile // 8
    total_channels = ch_A + ch_B + ch_S + ch_out

    writer_spatial_bounds = [[ch_out]]
    writer_num_channel = [ch_out]
    writer_fifo_depth  = [8]
    # Scale reader: 4 loops (within-block stride=0 / k_block / n_tile / m_tile)
    # output_stationary writer: 2 temporal loops; other modes: 3
    writer_tdim = 2 if stationary == 0 else 3
    writer_temporal_dim = [writer_tdim]

    #for requantizaiton
    if quantize_mode in [2, 3, 4, 5, 6, 7]:
        O_shared_tile = math.ceil((parfor_M) / 8) * 8
        ch_share_out = O_shared_tile // 8
        total_channels += ch_share_out
        writer_spatial_bounds.append([ch_share_out])
        writer_num_channel.append(ch_share_out)
        writer_temporal_dim.append(writer_tdim)
        writer_fifo_depth.append(8)


    portwidth = 64
    if quantize_mode in (0, 1):
        out_width = o_bits * parfor_M * parfor_N
    else:
        out_width = o_bits * parfor_M * block_size
    stream_c_data_width = math.ceil(out_width / portwidth) * portwidth

    return {
        "reader_spatial_bounds": [[ch_A], [ch_B], [ch_S]],
        "reader_num_channel":    [ch_A, ch_B, ch_S],
        "reader_temporal_dim":   [3, 3, 4],
        "writer_spatial_bounds": writer_spatial_bounds,
        "writer_num_channel":    writer_num_channel,
        "writer_temporal_dim":   writer_temporal_dim,
        "writer_fifo_depth":     writer_fifo_depth,
        "snax_tcdm_ports":       total_channels,
        "sparse_interconnect_config": [[total_channels, 1]],
        "TileRows":           parfor_M,
        "TileCols":           parfor_N,
        "VectorSize":         parfor_K,
        "A_Width":            A_bits,
        "B_Width":            B_bits,
        "OutputDataWidth":    o_bits,
        "StreamCDataWidth":   stream_c_data_width,
    }


# ---------------------------------------------------------------------------
# hjson patching
# ---------------------------------------------------------------------------

def patch_hwcfg(src: Path, dst: Path, hw: dict) -> None:
    """
    Load src hjson, patch streamer fields, write to dst.
    dst is a generated file so original comments in src are preserved.
    """
    with open(src, encoding="utf-8") as f:
        cfg = hjson.load(f)

    # --- Streamer template ---
    st = cfg["snax_mx_alu_streamer_template"]

    rp = st["data_reader_params"]
    rp["spatial_bounds"] = hw["reader_spatial_bounds"]
    rp["num_channel"]    = hw["reader_num_channel"]
    rp["temporal_dim"]   = hw["reader_temporal_dim"]

    wp = st["data_writer_params"]
    wp["spatial_bounds"] = hw["writer_spatial_bounds"]
    wp["num_channel"]    = hw["writer_num_channel"]
    wp["temporal_dim"]   = hw["writer_temporal_dim"]
    wp["fifo_depth"]     = hw["writer_fifo_depth"]

    # --- snax_tcdm_ports in the core template (resolved via $ref at build time) ---
    for acc in cfg["snax_alu_core_template"]["snax_acc_cfg"]:
        if acc.get("snax_acc_name") == "snax_mx_alu":
            acc["snax_tcdm_ports"]                    = hw["snax_tcdm_ports"]
            acc["sparse_interconnect_config"]         = hw["sparse_interconnect_config"]
            acc["snax_acc_params"]["TileRows"]        = hw["TileRows"]
            acc["snax_acc_params"]["TileCols"]        = hw["TileCols"]
            acc["snax_acc_params"]["VectorSize"]      = hw["VectorSize"]
            acc["snax_acc_params"]["OutputDataWidth"] = hw["OutputDataWidth"]
            acc["snax_acc_params"]["A_Width"]           = hw["A_Width"]
            acc["snax_acc_params"]["B_Width"]           = hw["B_Width"]
            acc["snax_acc_params"]["StreamCDataWidth"]  = hw["StreamCDataWidth"]

    with open(dst, "w", encoding="utf-8") as f:
        hjson.dump(cfg, f, indent=4)

    print(f"[orchestrator] HW config written → {dst}")


# ---------------------------------------------------------------------------
# Chisel RTL generation
# ---------------------------------------------------------------------------


def emit_pe_array_wrapper_sv(p: dict, out_dir: Path) -> Path:
    """Emit a thin SV adaptor module (PE_Array_wrapper) that exposes packed
    multi-dim IO around the Chisel-emitted PE_Array module, matching the
    connection style used by snax_mx_alu_shell_wrapper.

    The wrapper exposes only the requant outputs (shared_scale_out + result),
    not the FP32 accumulator path — that's internal to PE_Array.
    """
    rqmode  = p["quantize_mode"]
    rows    = p["parfor_M"]
    cols    = p["parfor_N"]
    vec     = p["parfor_K"]
    blk     = p.get("block_size", 32)
    A_W     = _DTYPE_BITS[p["A_dtype"]]
    B_W     = _DTYPE_BITS[p["B_dtype"]]
    scale_W = 8       # all ScaleFormats are 8 bits total (UE8M0..UE2M6)
    out_tag, elem_W = _REQUANT_OUT_TAG[rqmode]
    scale_name = _SCALE_FORMAT_MAP[p.get("shared_format", 0)]

    # Mode 0 (FP32) and mode 1 (BF16): per-element pass-through, no shared
    # scale. Output shape [TileRows][TileCols][ELEM_WIDTH] (32 or 16 bits).
    # Modes 2/3/4/5/6: blocked output [TileRows][BlockSize][ELEM_WIDTH] plus
    # one 8-bit shared scale per row.
    is_passthrough = rqmode in (0, 1)

    lines: list[str] = [
        "// AUTO-GENERATED by orchestrator.py — DO NOT EDIT.",
        "// Adaptor wrapper around Chisel-emitted PE_Array. Re-packs the flat",
        "// scalar Chisel ports (io_op_a_i_0, io_op_a_i_1, …) into packed",
        "// multi-dim arrays expected by snax_mx_alu_shell_wrapper.",
        f"// Config: A={p['A_dtype']} B={p['B_dtype']} scale={scale_name}"
        f" {rows}x{cols} vec{vec} blk{blk} requant_mode={rqmode} ({_REQUNAT_TYPE_MAP[rqmode]})",
        "`timescale 1ns / 1ps",
        "",
        "module PE_Array_wrapper #(",
        f"    parameter int unsigned A_WIDTH     = {A_W},",
        f"    parameter int unsigned B_WIDTH     = {B_W},",
        f"    parameter int unsigned ELEM_WIDTH  = {elem_W},",
        f"    parameter int unsigned TileRows    = {rows},",
        f"    parameter int unsigned TileCols    = {cols},",
        f"    parameter int unsigned VectorSize  = {vec},",
        f"    parameter int unsigned BlockSize   = {blk},",
        f"    parameter int unsigned SCALE_WIDTH = {scale_W}",
        ")(",
        "    input  logic clk_i,",
        "    input  logic rst_ni,",
        "",
        "    // CSR / control",
        "    input  logic [2:0] A_mode,",
        "    input  logic [2:0] B_mode,",
        "    input  logic [1:0] Result_mode_quan,",
        "    input  logic [1:0] group_size,",
        "    input  logic [3:0] shared_format_i,",
        "",
        "    input  logic        acc_reset_i,",
        "    input  logic        send_output_i,",
        "    input  logic [31:0] accumulation_count_i,  // K/vec — runtime threshold",
        "    input  logic        A_valid_i,",
        "    input  logic        B_valid_i,",
        "    output logic        A_ready_o,",
        "    output logic        B_ready_o,",
        "",
        "    // Packed operand / shared-exp inputs",
        "    input  logic [0:TileRows-1][A_WIDTH*VectorSize-1:0] op_a_i,",
        "    input  logic [0:TileCols-1][B_WIDTH*VectorSize-1:0] op_b_i,",
        "    input  logic [0:TileRows-1][SCALE_WIDTH-1:0]        shared_exp_A_i,",
        "    input  logic [0:TileCols-1][SCALE_WIDTH-1:0]        shared_exp_B_i,",
    ]

    if is_passthrough:
        lines += [
            "",
            f"    // {out_tag} pass-through output: [TileRows][TileCols][{elem_W}] packed; no shared scale.",
            "    output logic [0:TileRows-1][0:TileCols-1][ELEM_WIDTH-1:0] result_o,",
        ]
    else:
        lines += [
            "",
            f"    // {out_tag} requantised output: [TileRows][BlockSize][{elem_W}] packed,"
            " plus one 8-bit shared scale per row.",
            "    output logic [0:TileRows-1][7:0]                          shared_scale_out,",
            "    output logic [0:TileRows-1][0:BlockSize-1][ELEM_WIDTH-1:0] result_o,",
        ]

    lines += [
        "    output logic                                              valid_out",
        ");",
        "",
        "    // Chisel emits positive-asserted reset; shell uses active-low async rst_ni.",
        "    PE_Array u_pe (",
        "        .clock              (clk_i),",
        "        .reset              (rst_ni),",
        "        .io_A_mode          (A_mode),",
        "        .io_B_mode          (B_mode),",
        "        .io_result_mode_quan(Result_mode_quan),",
        "        .io_group_size      (group_size),",
        "        .io_shared_format_i (shared_format_i),",
        "        .io_acc_reset_i         (acc_reset_i),",
        "        .io_send_output_i       (send_output_i),",
        "        .io_accumulation_count_i(accumulation_count_i),",
        "        .io_A_valid_i           (A_valid_i),",
        "        .io_B_valid_i           (B_valid_i),",
        "        .io_A_ready_o           (A_ready_o),",
        "        .io_B_ready_o           (B_ready_o),",
    ]

    for r in range(rows):
        lines.append(f"        .io_op_a_i_{r}        (op_a_i[{r}]),")
    for c in range(cols):
        lines.append(f"        .io_op_b_i_{c}        (op_b_i[{c}]),")
    for r in range(rows):
        lines.append(f"        .io_shared_exp_A_i_{r}(shared_exp_A_i[{r}]),")
    for c in range(cols):
        lines.append(f"        .io_shared_exp_B_i_{c}(shared_exp_B_i[{c}]),")

    if not is_passthrough:
        lines.append("        .io_shared_scale_out(shared_scale_out),")
    lines += [
        "        .io_result          (result_o),",
        "        .io_valid_out       (valid_out)",
        "    );",
        "",
        "endmodule",
        "",
    ]

    sv_path = out_dir / "PE_Array_wrapper.sv"
    sv_path.write_text("\n".join(lines), encoding="utf-8")
    return sv_path


def run_pe_array_gen(p: dict, out_dir: Path) -> Path:
    """
    Generate the combined PE-Array + Requant RTL via `mx.GeneratePEArray`,
    and emit the SV adaptor wrapper next to it.

    Mapping from params.hjson:
      A_dtype, B_dtype           → MAC element formats
      shared_format (int 0..6)   → block-scale format
      parfor_M / parfor_N        → tileRows / tileCols
      parfor_K                   → vectorSize (parallel MACs per PE per cycle)
      block_size  (default 32)   → MX block size (16, 32, or 64)
      quantize_mode (int 1..6)   → output requant variant:
                                    1=bf16, 2=fp8_e5m2, 3=fp8_e4m3,
                                    4=mxint8, 5=fp6_e2m3, 6=fp6_e3m2
    """
    requant_mode = p.get("quantize_mode")
    if requant_mode not in _REQUNAT_TYPE_MAP:
        sys.exit(f"[orchestrator] quantize_mode={requant_mode} is not a requant mode. "
                 f"Valid: {list(_REQUNAT_TYPE_MAP)}")

    A_dtype       = p["A_dtype"]
    B_dtype       = p["B_dtype"]
    shared_format = p.get("shared_format", 0)

    type_a = _ELEMENT_TYPE_MAP.get(_DTYPE_TO_ELEMENT_TYPE.get(A_dtype))
    type_b = _ELEMENT_TYPE_MAP.get(_DTYPE_TO_ELEMENT_TYPE.get(B_dtype))
    scale  = _SCALE_FORMAT_MAP.get(shared_format)

    if type_a is None:
        sys.exit(f"[orchestrator] unknown A_dtype={A_dtype!r}, "
                 f"valid: {list(_DTYPE_TO_ELEMENT_TYPE)}")
    if type_b is None:
        sys.exit(f"[orchestrator] unknown B_dtype={B_dtype!r}, "
                 f"valid: {list(_DTYPE_TO_ELEMENT_TYPE)}")
    if scale is None:
        sys.exit(f"[orchestrator] unknown shared_format={shared_format}, "
                 f"valid: {_SCALE_FORMAT_MAP}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sbt_cmd = (
        f"runMain mx.EmitTensorCore"
        f" --act    {type_a}"
        f" --weight {type_b}"
        f" --scale  {scale}"
        f" --vec    {p['parfor_K']}"
        f" --tile-rows  {p['parfor_M']}"
        f" --tile-cols  {p['parfor_N']}"
        f" --block-size {p.get('block_size', 32)}"
        f" --quantize-mode {requant_mode}"
        f" --outdir {out_dir}"
        f" --no-standalone-requant"
    )

    print(f"[orchestrator] PE-Array RTL: {type_a}×{type_b} scale={scale} "
          f"{p['parfor_M']}x{p['parfor_N']} vec{p['parfor_K']} "
          f"blk{p.get('block_size', 32)} → {_REQUNAT_TYPE_MAP[requant_mode]}")
    print(f"[orchestrator] sbt {sbt_cmd}")
    result = subprocess.run(["sbt", sbt_cmd], cwd=CHISEL_DIR)
    if result.returncode != 0:
        sys.exit(f"[orchestrator] PE-Array RTL generation failed (exit {result.returncode})")

    sv_path = out_dir / "PE_Array.sv"
    print(f"[orchestrator] PE-Array RTL written → {sv_path}")

    wrapper_path = emit_pe_array_wrapper_sv(  p, out_dir)
    print(f"[orchestrator] SV adaptor wrapper → {wrapper_path}")
    return out_dir


# ---------------------------------------------------------------------------
# datagen invocation
# ---------------------------------------------------------------------------

def run_datagen(swcfg: Path, hwcfg: Path, output: Path) -> None:
    cmd = [sys.executable, str(DATAGEN_PY),
           "--swcfg", str(swcfg),
           "--hwcfg", str(hwcfg)]
    print(f"[orchestrator] {' '.join(cmd)} > {output}")
    with open(output, "w", encoding="utf-8") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"[orchestrator] datagen.py failed (exit {result.returncode})")
    print(f"[orchestrator] test data written → {output}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SNAX-MX workflow orchestrator")
    ap.add_argument("--swcfg",  default=DEFAULT_SWCFG,  help="params.hjson (sw config)")
    ap.add_argument("--hwcfg",  default=DEFAULT_HWCFG,  help="snax_mx_cluster.hjson template")
    ap.add_argument("--genhw",  default=DEFAULT_GEN_HW, help="generated hw config output path")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="data.h output path")
    ap.add_argument("--gen-pe-dir", default=DEFAULT_GEN_PE,
                    help="output directory for PE_Array.sv + PE_Array_wrapper.sv")
    ap.add_argument("--defines-mk", default=None,
                    help="path for auto-generated snax_mx_defines.mk "
                         "(default: <genhw>.parent/snax_mx_defines.mk)")
    ap.add_argument("--skip-rtl", action="store_true",    help="skip Chisel RTL generation step")
    args = ap.parse_args()

    swcfg     = Path(args.swcfg).resolve()
    hwcfg     = Path(args.hwcfg).resolve()
    genhw     = Path(args.genhw).resolve()
    output    = Path(args.output).resolve()
    gen_pe    = Path(args.gen_pe_dir).resolve()
    for d in (genhw.parent, output.parent, gen_pe):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load params
    print(f"[orchestrator] loading params  ← {swcfg}")
    with open(swcfg, encoding="utf-8") as f:
        params = hjson.load(f)

    # 2. Compute HW config
    hw = compute_hw_cfg(params)
    print(f"[orchestrator] computed HW config:")
    for k, v in hw.items():
        print(f"               {k} = {v}")

    # 3. Generate Chisel RTL
    if not args.skip_rtl:
        # run_mac_gen(params)
        # if params.get("quantize_mode", 0) in [1, 2, 3, 4, 5, 6]:
        #     run_requant_gen(params)
        if params.get("quantize_mode", 0) in _REQUNAT_TYPE_MAP:
            run_pe_array_gen(params, gen_pe)
        else:
            sys.exit(f"[orchestrator] quantize_mode={params.get('quantize_mode')} "
                     f"has no requant mode; valid: {list(_REQUNAT_TYPE_MAP)}")
    else:
        print("[orchestrator] skipping Chisel RTL generation (--skip-rtl)")

    # 4. Patch hwcfg → genhw
    patch_hwcfg(hwcfg, genhw, hw)

    # 4.5. Write RTL defines Makefile fragment for conditional wrapper ports
    defines_mk = (Path(args.defines_mk).resolve()
                  if args.defines_mk else genhw.parent / "snax_mx_defines.mk")
    defines_mk.parent.mkdir(parents=True, exist_ok=True)
    qmode = params.get("quantize_mode", 0)
    with open(defines_mk, "w") as f:
        f.write("# Auto-generated by orchestrator.py — do not edit manually\n")
        if qmode >= 2:
            f.write("VLT_FLAGS  += +define+SCALE_OUTPUT_EN\n")
            f.write("VLOG_FLAGS += +define+SCALE_OUTPUT_EN\n")
        else:
            f.write("# quantize_mode < 2: SCALE_OUTPUT_EN not defined\n")
    print(f"[orchestrator] wrote RTL defines  → {defines_mk}")

    # 5. Run datagen
    run_datagen(swcfg, genhw, output)

    print("[orchestrator] done.")


if __name__ == "__main__":
    main()
