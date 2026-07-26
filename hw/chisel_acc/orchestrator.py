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

# Mappings from params.hjson integer codes to Chisel type names
_ELEMENT_TYPE_MAP = {0: "INT8", 1: "E5M2", 2: "E4M3", 3: "E3M2", 4: "E2M3", 5: "E2M1"}
_SCALE_FORMAT_MAP = {0: "UE8M0", 1: "UE7M1", 2: "UE6M2", 3: "UE5M3", 4: "UE4M4", 5: "UE3M5", 6: "UE2M6"}

# ---------------------------------------------------------------------------
# Default paths (relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent.resolve()
DEFAULT_SWCFG  = SCRIPT_DIR / "data" / "params.hjson"
DEFAULT_HWCFG  = SCRIPT_DIR / "../../../cfg/snax_mx_cluster_template.hjson"
DEFAULT_GEN_HW = SCRIPT_DIR / "../../../cfg/snax_mx_cluster.hjson"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "data.h"
DATAGEN_PY     = SCRIPT_DIR / "data" / "datagen.py"


# ---------------------------------------------------------------------------
# HW parameter derivation
# ---------------------------------------------------------------------------

def _o_bitwidth(quantize_mode: int) -> int:
    return {0: 32, 1: 16, 2: 8}.get(quantize_mode, 32)


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
    parfor_M   = p["parfor_M"]
    parfor_N   = p["parfor_N"]
    parfor_K   = p["parfor_K"]
    A_bits     = p["A_bit_width"]
    B_bits     = p["B_bit_width"]
    o_bits     = _o_bitwidth(p.get("quantize_mode", 0))
    stationary = p.get("stationary", 0)

    # Tile sizes in bytes, aligned to 8
    A_tile = math.ceil(parfor_M * parfor_K * A_bits / 8 / 8) * 8
    B_tile = math.ceil(parfor_K * parfor_N * B_bits / 8 / 8) * 8
    S_tile = math.ceil((parfor_M + parfor_N) / 8) * 8
    O_tile = parfor_M * parfor_N * o_bits // 8

    ch_A   = A_tile // 8
    ch_B   = B_tile // 8
    ch_S   = S_tile // 8
    ch_out = O_tile // 8
    total_channels = ch_A + ch_B + ch_S + ch_out


    # Scale reader: 4 loops (within-block stride=0 / k_block / n_tile / m_tile)
    # output_stationary writer: 2 temporal loops; other modes: 3
    writer_tdim = 2 if stationary == 0 else 3

    return {
        "reader_spatial_bounds": [[ch_A], [ch_B], [ch_S]],
        "reader_num_channel":    [ch_A, ch_B, ch_S],
        "reader_temporal_dim":   [3, 3, 4],
        "writer_spatial_bounds": [[ch_out]],
        "writer_num_channel":    [ch_out],
        "writer_temporal_dim":   [writer_tdim],
        "snax_tcdm_ports":       total_channels,
        "sparse_interconnect_config": [[total_channels, 1]],
        "TileRows":    parfor_M,
        "TileCols":    parfor_N,
        "VectorSize":  parfor_K,
        "InputDataWidth": 8,
        "OutputDataWidth": 32,
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

    # --- snax_tcdm_ports in the core template (resolved via $ref at build time) ---
    for acc in cfg["snax_alu_core_template"]["snax_acc_cfg"]:
        if acc.get("snax_acc_name") == "snax_mx_alu":
            acc["snax_tcdm_ports"]                    = hw["snax_tcdm_ports"]
            acc["sparse_interconnect_config"]         = hw["sparse_interconnect_config"]
            acc["snax_acc_params"]["TileRows"]        = hw["TileRows"]
            acc["snax_acc_params"]["TileCols"]        = hw["TileCols"]
            acc["snax_acc_params"]["VectorSize"]      = hw["VectorSize"]
            acc["snax_acc_params"]["InputDataWidth"]  = hw["InputDataWidth"]
            acc["snax_acc_params"]["OutputDataWidth"] = hw["OutputDataWidth"]

    with open(dst, "w", encoding="utf-8") as f:
        hjson.dump(cfg, f, indent=4)

    print(f"[orchestrator] HW config written → {dst}")


# ---------------------------------------------------------------------------
# Chisel RTL generation
# ---------------------------------------------------------------------------

def run_chisel_gen(p: dict) -> Path:
    """
    Derive FusedDotProductUnit parameters from params and invoke sbt to
    generate SystemVerilog.

    Mapping from params.hjson:
      data_type     (int, default 0) → type_a and type_b element format
      shared_format (int, default 0) → scale format
      parfor_K                       → vectorSize (parallel MACs per cycle)
    """
    data_type     = p.get("data_type",     0)
    shared_format = p.get("shared_format", 0)
    vec           = p["parfor_K"]

    type_a = _ELEMENT_TYPE_MAP.get(data_type)
    type_b = _ELEMENT_TYPE_MAP.get(data_type)
    scale  = _SCALE_FORMAT_MAP.get(shared_format)

    if type_a is None:
        sys.exit(f"[orchestrator] unknown data_type={data_type}, "
                 f"valid: {_ELEMENT_TYPE_MAP}")
    if scale is None:
        sys.exit(f"[orchestrator] unknown shared_format={shared_format}, "
                 f"valid: {_SCALE_FORMAT_MAP}")

    out_dir = SCRIPT_DIR / "generated" / "fused_dot" / f"{type_a}_{type_b}_{scale}_vec{vec}"

    sbt_cmd = (
        f"runMain mx.GenerateFusedDotProduct"
        f" --type-a {type_a}"
        f" --type-b {type_b}"
        f" --scale  {scale}"
        f" --vec    {vec}"
        f" --out-dir {out_dir}"
    )

    print(f"[orchestrator] Chisel RTL: {type_a} x {type_b}, scale={scale}, vec={vec}")
    print(f"[orchestrator] sbt {sbt_cmd}")
    result = subprocess.run(["sbt", sbt_cmd], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        sys.exit(f"[orchestrator] Chisel generation failed (exit {result.returncode})")

    sv_path = out_dir / "BFP_PE.sv"
    print(f"[orchestrator] RTL written → {sv_path}")
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
    ap.add_argument("--swcfg",    default=DEFAULT_SWCFG,  help="params.hjson (sw config)")
    ap.add_argument("--hwcfg",    default=DEFAULT_HWCFG,  help="snax_mx_cluster.hjson template")
    ap.add_argument("--genhw",    default=DEFAULT_GEN_HW, help="generated hw config output path")
    ap.add_argument("--output",   default=DEFAULT_OUTPUT, help="data.h output path")
    ap.add_argument("--skip-rtl", action="store_true",    help="skip Chisel RTL generation step")
    args = ap.parse_args()

    swcfg  = Path(args.swcfg).resolve()
    hwcfg  = Path(args.hwcfg).resolve()
    genhw  = Path(args.genhw).resolve()
    output = Path(args.output).resolve()

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
        run_chisel_gen(params)
    else:
        print("[orchestrator] skipping Chisel RTL generation (--skip-rtl)")

    # 4. Patch hwcfg → genhw
    patch_hwcfg(hwcfg, genhw, hw)

    # 5. Run datagen
    run_datagen(swcfg, genhw, output)

    print("[orchestrator] done.")


if __name__ == "__main__":
    main()