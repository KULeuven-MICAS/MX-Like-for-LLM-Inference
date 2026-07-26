#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Zijun Zhao <zijun.zhao@student.kuleuven.be>

import sys
import argparse
import numpy as np
import pathlib
import hjson
import math
import struct
import os
import quantize
from math import ceil

sys.path.append(os.path.join(os.path.dirname(__file__),
                "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, \
                       format_vector_definition  # noqa: E402

_SCALE_FORMAT_MAP = {0: "UE8M0", 1: "UE7M1", 2: "UE6M2", 3: "UE5M3", 4: "UE4M4", 5: "UE3M5", 6: "UE2M6"}

_DTYPE_TO_HW_ETYPE = {
    'fp8_e5m2': 'E5M2', 'fp8_e4m3': 'E4M3',
    'fp6_e3m2': 'E3M2', 'fp6_e2m3': 'E2M3',
    'fp4_e2m1': 'E2M1', 'mxint8':   'INT8',
}

_O_DTYPE_MAP = {2: 'fp8_e5m2', 3: 'fp8_e4m3', 4: 'mxint8', 5: 'fp6_e2m3', 6: 'fp6_e3m2'}


def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32
        bit_position  = i % 32
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)
    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR


def data_file_emit(**kwargs):
    data_str = []

    M = kwargs["M"]; K = kwargs["K"]; N = kwargs["N"]
    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]

    parfor_M      = kwargs["parfor_M"]
    parfor_N      = kwargs["parfor_N"]
    parfor_K      = kwargs["parfor_K"]
    block_size    = kwargs["block_size"]
    quantize_mode = kwargs["quantize_mode"]

    data_str += [format_scalar_definition("uint32_t", "PARFOR_M",      parfor_M)]
    data_str += [format_scalar_definition("uint32_t", "PARFOR_N",      parfor_N)]
    data_str += [format_scalar_definition("uint32_t", "PARFOR_K",      parfor_K)]
    data_str += [format_scalar_definition("uint32_t", "BLOCK_SIZE",    block_size)]

    variance = 3
    A_fp32 = (np.random.randn(M, K) * variance).astype(np.float32)
    B_fp32 = (np.random.randn(K, N) * variance).astype(np.float32)

    m_padded = ceil(M / parfor_M) * parfor_M
    n_padded = ceil(N / block_size) * block_size if quantize_mode not in (0, 1) else ceil(N / parfor_N) * parfor_N
    k_padded = ceil(K / parfor_K) * parfor_K
    k_aligned_blocksize = ceil(k_padded / block_size) * block_size

    A_fp32_padded = np.zeros((m_padded, k_aligned_blocksize), dtype=np.float32)
    B_fp32_padded = np.zeros((k_aligned_blocksize, n_padded),  dtype=np.float32)
    A_fp32_padded[:M, :K] = A_fp32
    B_fp32_padded[:K, :N] = B_fp32

    A_dtype      = kwargs["A_dtype"]
    B_dtype      = kwargs["B_dtype"]
    share_format = kwargs["shared_format"]
    _qmx = quantize.quantize_mx

    data_str += [format_scalar_definition("const char*", "A_dtype",      f'"{A_dtype}"')]
    data_str += [format_scalar_definition("const char*", "B_dtype",      f'"{B_dtype}"')]
    data_str += [format_scalar_definition("uint32_t",    "share_format", share_format)]

    A_quantized, exp_A_readable, exp_A, A_input = _qmx(
        A_fp32_padded, A_dtype, block_size=block_size, axis=1,
        scale_format=_SCALE_FORMAT_MAP.get(share_format, 'UE8M0'))
    B_quantized, exp_B_readable, exp_B, B_input = _qmx(
        B_fp32_padded, B_dtype, block_size=block_size, axis=0,
        scale_format=_SCALE_FORMAT_MAP.get(share_format, 'UE8M0'))

    acc_cnt = np.ceil(k_aligned_blocksize / parfor_K)
    out_cnt = (np.ceil(M / parfor_M) * np.ceil(N / parfor_N) if quantize_mode in (0, 1)
               else np.ceil(M / parfor_M) * np.ceil(N / block_size))

    data_str += [format_scalar_definition("uint32_t", "ACC_CNT",       int(acc_cnt))]
    data_str += [format_scalar_definition("uint32_t", "OUT_CNT",       int(out_cnt))]
    data_str += [format_scalar_definition("uint32_t", "MODE",          int(0))]
    data_str += [format_scalar_definition("uint32_t", "QUANTIZE_MODE", int(quantize_mode))]
    data_str += [format_scalar_definition("uint32_t", "stationary",    0)]

    a_bitwidth      = quantize._DTYPE_BITS[kwargs["A_dtype"]]
    b_bitwidth      = quantize._DTYPE_BITS[kwargs["B_dtype"]]
    bankwidth       = 64
    shared_bitwidth = 8
    bits_per_byte   = 8
    bank_bytes      = bankwidth // bits_per_byte  # = 8

    if quantize_mode == 0:
        o_bitwidth = 32
    elif quantize_mode == 1:
        o_bitwidth = 16
    elif quantize_mode in (2, 3, 4):
        o_bitwidth = 8
    else:
        o_bitwidth = 6

    m_tiles  = ceil(m_padded / parfor_M)
    n_tiles  = ceil(n_padded / parfor_N)
    k_tiles  = ceil(k_aligned_blocksize / parfor_K)
    k_blocks = ceil(k_padded / block_size)

    # Streamer A
    data_str += [format_scalar_definition("int32_t", "Aslstride0", bank_bytes)]
    A_tile_bytes         = parfor_M * parfor_K * a_bitwidth / bits_per_byte
    A_tile_bytes_aligned = int(ceil(A_tile_bytes / bank_bytes) * bank_bytes)
    Atlbound0  = k_tiles;  Atlstride0 = A_tile_bytes_aligned
    Atlbound1  = n_tiles;  Atlstride1 = 0
    Atlbound2  = m_tiles;  Atlstride2 = k_tiles * A_tile_bytes_aligned
    data_str += [format_scalar_definition("int32_t", "Atlbound0",  Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "Atlstride0", Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound1",  Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2",  Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "Atlstride2", Atlstride2)]

    # Streamer B
    data_str += [format_scalar_definition("int32_t", "Bslstride0", bank_bytes)]
    B_tile_bytes         = parfor_K * parfor_N * b_bitwidth / bits_per_byte
    B_tile_bytes_aligned = int(ceil(B_tile_bytes / bank_bytes) * bank_bytes)
    Btlbound0  = k_tiles;  Btlstride0 = B_tile_bytes_aligned
    Btlbound1  = n_tiles;  Btlstride1 = k_tiles * B_tile_bytes_aligned
    Btlbound2  = m_tiles;  Btlstride2 = 0
    data_str += [format_scalar_definition("int32_t", "Btlbound0",  Btlbound0)]
    data_str += [format_scalar_definition("int32_t", "Btlstride0", Btlstride0)]
    data_str += [format_scalar_definition("int32_t", "Btlbound1",  Btlbound1)]
    data_str += [format_scalar_definition("int32_t", "Btlstride1", Btlstride1)]
    data_str += [format_scalar_definition("int32_t", "Btlbound2",  Btlbound2)]
    data_str += [format_scalar_definition("int32_t", "Btlstride2", Btlstride2)]

    # Streamer Scale (combined A+B, bank-aligned)
    steps_per_block  = block_size // parfor_K
    combined_raw     = parfor_M + parfor_N
    combined_tile    = int(ceil(combined_raw / bank_bytes) * bank_bytes)
    pad              = combined_tile - combined_raw
    data_str += [format_scalar_definition("int32_t", "SHslstride0", bank_bytes)]
    SHtlbound0  = steps_per_block;              SHtlstride0 = 0
    SHtlbound1  = k_blocks;                    SHtlstride1 = combined_tile
    SHtlbound2  = n_tiles;                     SHtlstride2 = k_blocks * combined_tile
    SHtlbound3  = m_tiles;                     SHtlstride3 = n_tiles * k_blocks * combined_tile
    assert steps_per_block * k_blocks == k_tiles
    data_str += [format_scalar_definition("int32_t", "SHtlbound0",  SHtlbound0)]
    data_str += [format_scalar_definition("int32_t", "SHtlstride0", SHtlstride0)]
    data_str += [format_scalar_definition("int32_t", "SHtlbound1",  SHtlbound1)]
    data_str += [format_scalar_definition("int32_t", "SHtlstride1", SHtlstride1)]
    data_str += [format_scalar_definition("int32_t", "SHtlbound2",  SHtlbound2)]
    data_str += [format_scalar_definition("int32_t", "SHtlstride2", SHtlstride2)]
    data_str += [format_scalar_definition("int32_t", "SHtlbound3",  SHtlbound3)]
    data_str += [format_scalar_definition("int32_t", "SHtlstride3", SHtlstride3)]

    # Streamer Output
    data_str += [format_scalar_definition("int32_t", "Oslstride0", bank_bytes)]
    if quantize_mode == 0 or quantize_mode == 1:
        O_tile_bytes = parfor_M * parfor_N * o_bitwidth / bits_per_byte
        Otlbound0  = n_tiles;  Otlstride0 = O_tile_bytes
        Otlbound1  = m_tiles;  Otlstride1 = n_tiles * O_tile_bytes
    else:
        block_tiles  = n_padded // block_size
        O_tile_bytes = parfor_M * block_size * o_bitwidth / bits_per_byte
        Otlbound0  = block_tiles;  Otlstride0 = O_tile_bytes
        Otlbound1  = m_tiles;      Otlstride1 = block_tiles * O_tile_bytes
    data_str += [format_scalar_definition("int32_t", "Otlbound0",  Otlbound0)]
    data_str += [format_scalar_definition("int32_t", "Otlstride0", Otlstride0)]
    data_str += [format_scalar_definition("int32_t", "Otlbound1",  Otlbound1)]
    data_str += [format_scalar_definition("int32_t", "Otlstride1", Otlstride1)]

    # Streamer Scale Out (BFP requantization modes only)
    data_str += [format_scalar_definition("int32_t", "SHOutslstride0", bank_bytes)]
    if not (quantize_mode == 0 or quantize_mode == 1):
        SHOut_tile_bytes         = parfor_M * shared_bitwidth // bits_per_byte
        SHOut_tile_bytes_aligned = int(ceil(SHOut_tile_bytes / bank_bytes) * bank_bytes)
        block_tiles      = n_padded // block_size
        SHOuttlbound0  = block_tiles;  SHOuttlstride0 = SHOut_tile_bytes_aligned
        SHOuttlbound1  = m_tiles;      SHOuttlstride1 = block_tiles * SHOut_tile_bytes_aligned
    else:
        SHOuttlbound0 = 0; SHOuttlstride0 = 0
        SHOuttlbound1 = 0; SHOuttlstride1 = 0
    data_str += [format_scalar_definition("int32_t", "SHOuttlbound0",  SHOuttlbound0)]
    data_str += [format_scalar_definition("int32_t", "SHOuttlstride0", SHOuttlstride0)]
    data_str += [format_scalar_definition("int32_t", "SHOuttlbound1",  SHOuttlbound1)]
    data_str += [format_scalar_definition("int32_t", "SHOuttlstride1", SHOuttlstride1)]

    # Base addresses
    def align_addr(addr, align):
        return int(ceil(addr / align) * align)

    A_data_size       = m_tiles * k_tiles * A_tile_bytes_aligned
    B_data_size       = k_tiles * n_tiles * B_tile_bytes_aligned
    scale_data_size   = m_tiles * n_tiles * k_blocks * combined_tile
    O_data_size       = m_padded * n_padded * o_bitwidth // bits_per_byte
    O_scale_data_size = m_padded * (n_padded // block_size) * shared_bitwidth // bits_per_byte

    data_str += [format_scalar_definition("int32_t", "A_data_size",       A_data_size)]
    data_str += [format_scalar_definition("int32_t", "B_data_size",       B_data_size)]
    data_str += [format_scalar_definition("int32_t", "scale_data_size",   scale_data_size)]
    data_str += [format_scalar_definition("int32_t", "O_data_size",       O_data_size)]
    data_str += [format_scalar_definition("int32_t", "O_scale_data_size", O_scale_data_size)]

    delta_local_a       = 0
    delta_local_b       = align_addr(delta_local_a + A_data_size, bank_bytes)
    delta_local_scale   = align_addr(delta_local_b + B_data_size, bank_bytes)
    delta_local_o       = align_addr(delta_local_scale + scale_data_size, bank_bytes)
    delta_local_o_scale = align_addr(delta_local_o + O_data_size, bank_bytes)

    data_str += [format_scalar_definition("int32_t", "delta_local_a",       delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b",       delta_local_b)]
    data_str += [format_scalar_definition("int32_t", "delta_local_scale",   delta_local_scale)]
    data_str += [format_scalar_definition("int32_t", "delta_local_o",       delta_local_o)]
    data_str += [format_scalar_definition("int32_t", "delta_local_o_scale", delta_local_o_scale)]

    # Data arrays: A, B, combined scale
    A_tile_pad  = int(A_tile_bytes_aligned - A_tile_bytes)
    A_reordered = (A_input
                   .reshape(m_tiles, parfor_M, k_tiles, parfor_K)
                   .transpose(0, 2, 1, 3)
                   .reshape(m_tiles * k_tiles, parfor_M * parfor_K))
    A_tiles = []
    for row in A_reordered:
        packed = quantize._pack_raw_1d(row, A_dtype)
        if A_tile_pad > 0:
            packed = np.concatenate([packed, np.zeros(A_tile_pad, dtype=np.uint8)])
        A_tiles.append(packed)
    A_tiled = np.concatenate(A_tiles).view(np.int8)
    data_str += [format_vector_definition("int8_t", "A", A_tiled)]

    B_tile_pad  = int(B_tile_bytes_aligned - B_tile_bytes)
    B_reordered = (B_input
                   .reshape(k_tiles, parfor_K, n_tiles, parfor_N)
                   .transpose(2, 0, 3, 1)
                   .reshape(k_tiles * n_tiles, parfor_K * parfor_N))
    B_tiles = []
    for row in B_reordered:
        packed = quantize._pack_raw_1d(row, B_dtype)
        if B_tile_pad > 0:
            packed = np.concatenate([packed, np.zeros(B_tile_pad, dtype=np.uint8)])
        B_tiles.append(packed)
    B_tiled = np.concatenate(B_tiles).view(np.int8)
    data_str += [format_vector_definition("int8_t", "B", B_tiled)]

    exp_A_u8 = exp_A.astype(np.uint8)  # (m_padded, k_blocks)
    exp_B_u8 = exp_B.astype(np.uint8)  # (k_blocks, n_padded)

    loop = [(m, n, kb)
            for m in range(m_tiles)
            for n in range(n_tiles)
            for kb in range(k_blocks)]

    chunks = []
    for (m, n, kb) in loop:
        a_sc = exp_A_u8[m * parfor_M:(m + 1) * parfor_M, kb]
        b_sc = exp_B_u8[kb, n * parfor_N:(n + 1) * parfor_N]
        chunks.append(np.concatenate([a_sc, b_sc, np.zeros(pad, dtype=np.uint8)]))
    combined_scale = np.concatenate(chunks).view(np.int8)
    data_str += [format_vector_definition("int8_t", "scale", combined_scale)]

    # Golden output
    use_hw_gemm = bool(kwargs.get("use_hw_gemm", 0))
    if use_hw_gemm:
        etype_a  = _DTYPE_TO_HW_ETYPE[A_dtype]
        etype_b  = _DTYPE_TO_HW_ETYPE[B_dtype]
        stype    = _SCALE_FORMAT_MAP.get(share_format, 'UE8M0')
        A_raw_3d = A_input.reshape(m_padded, k_blocks, block_size)
        B_raw_3d = B_input.reshape(k_blocks, block_size, n_padded).transpose(2, 0, 1)
        D_golden = quantize.hw_snax_gemm(
            A_raw_3d, B_raw_3d, exp_A, exp_B.T,
            etype_a=etype_a, etype_b=etype_b, stype=stype,
        ).astype(np.float32)
    else:
        D_golden = (A_quantized @ B_quantized).astype(np.float32)

    D_tiled = (D_golden
               .reshape(m_tiles, parfor_M, n_tiles, parfor_N)
               .transpose(0, 2, 1, 3)
               .flatten())

    # Verify streamer alignment and tiling correctness
    _a_aln  = int(A_tile_bytes_aligned); _b_aln = int(B_tile_bytes_aligned)
    _a_elem = int(A_tile_bytes);         _b_elem = int(B_tile_bytes)
    A_mem   = A_tiled.reshape(m_tiles * k_tiles, _a_aln)
    B_mem   = B_tiled.reshape(k_tiles * n_tiles, _b_aln)
    a_ok = all(
        np.array_equal(
            A_mem[m * k_tiles + k, :_a_elem].astype(np.int8),
            A_input[m*parfor_M:(m+1)*parfor_M, k*parfor_K:(k+1)*parfor_K].flatten().astype(np.int8)
        )
        for m in range(m_tiles) for k in range(k_tiles)
    )
    b_ok = all(
        np.array_equal(
            B_mem[n * k_tiles + k, :_b_elem].astype(np.int8),
            B_input[k*parfor_K:(k+1)*parfor_K, n*parfor_N:(n+1)*parfor_N].flatten().astype(np.int8)
        )
        for n in range(n_tiles) for k in range(k_tiles)
    )
    D_untiled = (D_tiled
                 .reshape(m_tiles, n_tiles, parfor_M, parfor_N)
                 .transpose(0, 2, 1, 3)
                 .reshape(m_padded, n_padded))
    d_ok = np.allclose(D_untiled, D_golden, rtol=0, atol=0)

    if not (quantize_mode == 0 or quantize_mode == 1):
        o_dtype      = _O_DTYPE_MAP[quantize_mode]
        scale_fmt_s  = _SCALE_FORMAT_MAP.get(share_format, 'UE8M0')
        block_tiles  = n_padded // block_size

        _, _, exp_O, D_req_raw = _qmx(
            D_golden, o_dtype, block_size=block_size, axis=1, scale_format=scale_fmt_s
        )

        O_quant_tiles = []
        for m in range(m_tiles):
            for kb in range(block_tiles):
                elems  = D_req_raw[m*parfor_M:(m+1)*parfor_M, kb*block_size:(kb+1)*block_size]
                packed = quantize._pack_raw_1d(elems.flatten(), o_dtype)
                O_quant_tiles.append(packed)
        O_quant_golden = np.concatenate(O_quant_tiles).view(np.int8)
        data_str += [format_vector_definition("int8_t", "O_quant_golden", O_quant_golden)]

        exp_O_u8                 = exp_O.astype(np.uint8)
        SHOut_tile_bytes         = parfor_M * shared_bitwidth // bits_per_byte
        SHOut_tile_bytes_aligned = int(ceil(SHOut_tile_bytes / bank_bytes) * bank_bytes)
        SHOut_pad                = SHOut_tile_bytes_aligned - SHOut_tile_bytes
        O_scale_chunks = []
        for m in range(m_tiles):
            for kb in range(block_tiles):
                sc = exp_O_u8[m*parfor_M:(m+1)*parfor_M, kb]
                if SHOut_pad > 0:
                    sc = np.concatenate([sc, np.zeros(SHOut_pad, dtype=np.uint8)])
                O_scale_chunks.append(sc)
        O_scale_golden = np.concatenate(O_scale_chunks).view(np.int8)
        data_str += [format_vector_definition("int8_t", "O_scale_golden", O_scale_golden)]
    else:
        dummy = np.zeros(1, dtype=np.int8)
        data_str += [format_vector_definition("int8_t", "O_quant_golden", dummy)]
        data_str += [format_vector_definition("int8_t", "O_scale_golden", dummy)]

    data_str += [format_vector_definition("uint32_t", "O_golden",        D_tiled.view(np.uint32))]
    data_str += [format_vector_definition("float",    "O_human_readable", D_tiled)]
    data_str = "\n\n".join(data_str)
    return data_str


def main():
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument("--swcfg", type=pathlib.Path, required=True,
                        help="Select param config file kernel")
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True,
                        help="Select hardware config file kernel")
    args = parser.parse_args()

    with args.swcfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    print(data_file_emit(**{**param, **hw}))


if __name__ == '__main__':
    main()
