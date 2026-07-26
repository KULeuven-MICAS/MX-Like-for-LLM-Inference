// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Zijun Zhao <zijun.zhao@student.kuleuven.be>

#include <stdint.h>
#include "snrt.h"
#include "streamer_csr_addr_map.h"

#pragma once

// MX Accelerator CSRs (start after streamer at STREAMER_PERFORMANCE_COUNTER_CSR)
#define MX_CSR_ADDR_BASE  (STREAMER_PERFORMANCE_COUNTER_CSR + 1)
#define MX_CSR_MODE       (MX_CSR_ADDR_BASE + 0)  // rw: accumulator mode
#define MX_CSR_ACC_COUNT  (MX_CSR_ADDR_BASE + 1)  // rw: k_blocks per output tile
#define MX_CSR_OUT_COUNT  (MX_CSR_ADDR_BASE + 2)  // rw: total output tiles
#define MX_CSR_START      (MX_CSR_ADDR_BASE + 3)  // rw: write 1 to start
#define MX_CSR_BUSY       (MX_CSR_ADDR_BASE + 4)  // ro: 1 while running
#define MX_CSR_PERF_COUNT (MX_CSR_ADDR_BASE + 5)  // ro: cycle counter

// Configure all 4 streamers in one call.
// Each mover takes: delta_local (byte offset from snrt_l1_next()),
//   slstride[S_STRIDE_NUM] (spatial strides),
//   tlbound[T_BOUND_NUM]   (temporal loop bounds, innermost first),
//   tlstride[T_STRIDE_NUM] (temporal strides in bytes, innermost first).
void set_mx_streamer_csr(
    int32_t delta_local_a,
    int32_t* Aslstride, int32_t* Atlbound, int32_t* Atlstride,

    int32_t delta_local_b,
    int32_t* Bslstride, int32_t* Btlbound, int32_t* Btlstride,

    int32_t delta_local_shared,
    int32_t* SHslstride, int32_t* SHtlbound, int32_t* SHtlstride,

    int32_t delta_local_o,
    int32_t* Oslstride, int32_t* Otlbound, int32_t* Otlstride);

// Configure WRITER_1 (SHOut – output shared scale) streamer.
// Only needed when quantize_mode >= 2 (BFP requantization output).
// delta_local_o_scale: byte offset from snrt_l1_next() for output scale buffer.
// SHOutslstride[S_STRIDE_NUM_WRITER_1], SHOuttlbound[T_BOUND_NUM_WRITER_1],
// SHOuttlstride[T_STRIDE_NUM_WRITER_1]: streamer loop parameters from data.h.
void set_mx_shout_streamer_csr(
    int32_t delta_local_o_scale,
    int32_t* SHOutslstride, int32_t* SHOuttlbound, int32_t* SHOuttlstride);

// Trigger streamer
inline void set_mx_streamer_start() { csrw_ss(STREAMER_START_CSR, 1); }

// Configure MX accelerator CSRs
// inline void set_mx_csr(uint32_t mode, uint32_t acc_count, uint32_t out_count) {
//     csrw_ss(MX_CSR_MODE,      mode);
//     csrw_ss(MX_CSR_ACC_COUNT, acc_count);
//     csrw_ss(MX_CSR_OUT_COUNT, out_count);
// }
inline void set_mx_csr(uint32_t mode, uint32_t acc_count, uint32_t out_count);

// Trigger MX accelerator
inline void set_mx_start() { csrw_ss(MX_CSR_START, 1); }

// Poll until both MX accelerator and streamer finish
void wait_mx_and_streamer();

// Poll until MX accelerator alone finishes
void wait_mx();

// Read streamer performance counter (read-only CSR)
uint32_t read_mx_streamer_perf_counter();

// Read MX accelerator performance counter (read-only CSR)
uint32_t read_mx_perf_counter();

// Compare output vs golden, return number of mismatches.
// is_fp32 = 1: decode each word as fp32 and print its value.
// is_fp32 = 0: print raw 0x%08x (use for requantized int8/fp6/fp8 output).
uint32_t check_mx_result(uint32_t* output, uint32_t* output_golden,
                         int32_t out_len, int is_fp32);
