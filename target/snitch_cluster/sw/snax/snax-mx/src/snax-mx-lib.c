// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Zijun Zhao <zijun.zhao@student.kuleuven.be>

#include "snax-mx-lib.h"
#include <stdint.h>
#include "snrt.h"
#include "streamer_csr_addr_map.h"

void set_mx_streamer_csr(
    int32_t delta_local_a,
    int32_t* Aslstride, int32_t* Atlbound, int32_t* Atlstride,

    int32_t delta_local_b,
    int32_t* Bslstride, int32_t* Btlbound, int32_t* Btlstride,

    int32_t delta_local_shared,
    int32_t* SHslstride, int32_t* SHtlbound, int32_t* SHtlstride,

    int32_t delta_local_o,
    int32_t* Oslstride, int32_t* Otlbound, int32_t* Otlstride) {

    // ----------------------------------A (Reader 0)-----------------------------------
    // ----------------------------------A (Reader 0)-----------------------------------
    // ----------------------------------A (Reader 0)-----------------------------------
    // base ptr for A
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(delta_local_a + snrt_l1_next()));

    // spatial strides for A
    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Aslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover A
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);
    }

    // temporal strides for A
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
    }

    // ----------------------------------B (Reader 1)-----------------------------------
    // ----------------------------------B (Reader 1)-----------------------------------
    // ----------------------------------B (Reader 1)-----------------------------------
    // base ptr for B
    csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(delta_local_b + snrt_l1_next()));

    // spatial strides for B
    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, Bslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover B
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, Btlbound[i]);
    }

    // temporal strides for B
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, Btlstride[i]);
    }

    // ----------------------------------Shared (Reader 2)------------------------------
    // ----------------------------------Shared (Reader 2)------------------------------
    // ----------------------------------Shared (Reader 2)------------------------------
    // base ptr for Shared exponent
    csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(delta_local_shared + snrt_l1_next()));

    // spatial strides for Shared
    for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) {
        csrw_ss(S_STRIDE_BASE_READER_2 + i, SHslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover Shared
    for (int i = 0; i < T_BOUND_NUM_READER_2; i++) {
        csrw_ss(T_BOUND_BASE_READER_2 + i, SHtlbound[i]);
    }

    // temporal strides for Shared
    for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) {
        csrw_ss(T_STRIDE_BASE_READER_2 + i, SHtlstride[i]);
    }

    // ----------------------------------Out (Writer 0)---------------------------------
    // ----------------------------------Out (Writer 0)---------------------------------
    // ----------------------------------Out (Writer 0)---------------------------------
    // base ptr for output
    csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)(delta_local_o + snrt_l1_next()));

    // spatial strides for output
    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, Oslstride[i]);
    }

    // loop bounds, from innermost to outermost, for data mover Out
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, Otlbound[i]);
    }

    // temporal strides for output
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, Otlstride[i]);
    }
}

// Configure WRITER_1 (SHOut – output shared scale, modes 2-5)
// Only compiled when the hardware has a second writer (BASE_PTR_WRITER_1_LOW defined).
void set_mx_shout_streamer_csr(
    int32_t delta_local_o_scale,
    int32_t* SHOutslstride, int32_t* SHOuttlbound, int32_t* SHOuttlstride) {

#ifdef BASE_PTR_WRITER_1_LOW
    // ----------------------------------SHOut (Writer 1)-------------------------------
    // base ptr for output shared scale
    csrw_ss(BASE_PTR_WRITER_1_LOW, (uint32_t)(delta_local_o_scale + snrt_l1_next()));

    // spatial strides for SHOut
    for (int i = 0; i < S_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_1 + i, SHOutslstride[i]);
    }

    // loop bounds, from innermost to outermost, for SHOut
    for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_1 + i, SHOuttlbound[i]);
    }

    // temporal strides for SHOut
    for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_1 + i, SHOuttlstride[i]);
    }
#else
    (void)delta_local_o_scale;
    (void)SHOutslstride;
    (void)SHOuttlbound;
    (void)SHOuttlstride;
#endif
}

// Stall until MX accelerator and streamer finish
void wait_mx_and_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(MX_CSR_START, 0);
    printf("Waiting for MX Core to end...\n");
    while (csrr_ss(MX_CSR_BUSY)) {
    }
    printf("Waiting for Streamer to end...\n");
    while (csrr_ss(STREAMER_BUSY_CSR)) {
    }
}

void set_mx_csr(uint32_t mode, uint32_t acc_count, uint32_t out_count) {
    csrw_ss(MX_CSR_MODE,      mode);
    csrw_ss(MX_CSR_ACC_COUNT, acc_count);
    csrw_ss(MX_CSR_OUT_COUNT, out_count);
}

void wait_mx() {
    csrw_ss(MX_CSR_START, 0);
    csrw_ss(MX_CSR_START, 0);
    while (csrr_ss(MX_CSR_BUSY)) {
    }
}

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_mx_streamer_perf_counter() {
    return csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
}

// Read performance counter of MX accelerator, a read-only CSR
uint32_t read_mx_perf_counter() {
    return csrr_ss(MX_CSR_PERF_COUNT);
}

// Print an IEEE-754 FP32 value using only integer arithmetic,
// avoiding float variables in variadic calls (which emit FMV.W.X on Snitch).
static void snax_print_f32(uint32_t bits) {
    int sign   = (bits >> 31) & 1;
    int biased = (bits >> 23) & 0xFF;
    uint32_t frac = bits & 0x7FFFFFu;

    if (biased == 0xFF) {
        if (frac) { printf("NaN"); return; }
        printf(sign ? "-Inf" : "Inf"); return;
    }
    if (sign) printf("-");
    if (biased == 0 && frac == 0) { printf("0.000000"); return; }

    // sig * 2^e = |value|, sig is the 24-bit (or 23-bit denorm) significand
    uint64_t sig = biased ? ((uint64_t)1 << 23 | frac) : (uint64_t)frac;
    int e = biased ? (biased - 127 - 23) : (-126 - 23);

    uint64_t int_part, frac_part;
    if (e >= 0) {
        if (e > 30) { printf("large"); return; }
        int_part  = sig << e;
        frac_part = 0;
    } else {
        int ne = -e;
        if (ne > 60) { printf("0.000000"); return; }
        if (ne <= 23) {
            int_part  = sig >> ne;
            uint64_t rem = sig & (((uint64_t)1 << ne) - 1);
            frac_part = (rem * 1000000ULL) >> ne;
        } else {
            int_part  = 0;
            frac_part = (sig * 1000000ULL) >> ne;
        }
    }
    printf("%u.%06u", (unsigned)int_part, (unsigned)frac_part);
}

// Check the result of MX accelerator output against golden model.
// is_fp32 = 1: decode each uint32 word as IEEE-754 fp32 and print the value.
// is_fp32 = 0: print the raw 0x%08x word (for requantized mxint8/fp8/fp6 output).
uint32_t check_mx_result(uint32_t* output, uint32_t* output_golden,
                         int32_t out_len, int is_fp32) {
    uint32_t err = 0;
    for (int i = 0; i < out_len; i++) {
        int mismatch = (output[i] != output_golden[i]);
        if (is_fp32) {
            printf("[%d]: expected ", i);
            snax_print_f32(output_golden[i]);
            printf(" (0x%08x), got ", output_golden[i]);
            snax_print_f32(output[i]);
            printf(" (0x%08x)%s\n", output[i], mismatch ? " <-- MISMATCH" : "");
        } else {
            printf("[%d]: expected 0x%08x, got 0x%08x%s\n",
                   i, output_golden[i], output[i],
                   mismatch ? " <-- MISMATCH" : "");
        }
        if (mismatch) err++;
    }
    return err;
}
