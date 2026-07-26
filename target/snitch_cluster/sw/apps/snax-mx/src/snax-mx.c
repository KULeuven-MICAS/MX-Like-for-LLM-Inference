// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

#include "data.h"
#include "snax-mx-lib.h"

int main() {
    // Set err value for checking
    int err = 0;
    printf("Starting SNAX MX Accelerator...\n");

    // Allocates space in TCDM
    // A, B, SHARED are uint8 data; output is uint32 (float32 results)
    uint8_t *local_a, *local_b, *local_shared;
    uint32_t* local_o;
    uint8_t* local_o_scale;  // output shared scale (modes 2-5 only)

    local_a = (uint8_t*)(snrt_l1_next() + delta_local_a);
    local_b = (uint8_t*)(snrt_l1_next() + delta_local_b);
    local_o = (uint32_t*)(snrt_l1_next() + delta_local_o);
    local_shared = (uint8_t*)(snrt_l1_next() + delta_local_scale);
    if (QUANTIZE_MODE >= 2) {
        local_o_scale = (uint8_t*)(snrt_l1_next() + delta_local_o_scale);
    }

    // Start of pre-loading data from L2 memory
    // towards the L1 TCDM memory
    // Use the Snitch core with a DMA
    // to move the data from L2 to L1
    if (snrt_is_dm_core()) {
        // This measures the start of cycle count
        // for preloading the data to the L1 memory
        uint32_t start_dma_load = snrt_mcycle();

        // The DATA_LEN is found in data.h
        printf("Start DMA loading...\n");
        // TODO: make it compatible this with the low-bit precision data format
        snrt_dma_start_1d(local_a, A, A_data_size);
        //                addr_L1, addr_L2/3   , transfer size in bytes
        snrt_dma_start_1d(local_b, B, B_data_size);
        snrt_dma_start_1d(local_shared, scale, scale_data_size);
        snrt_dma_wait_all();
        // Measure the end of the transfer process
        // uint32_t end_dma_load = snrt_mcycle();
    }

    // Synchronize cores by setting up a
    // fence barrier for the DMA and accelerator core
    snrt_cluster_hw_barrier();

    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2};

    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};

    int32_t SHslstride[] = {SHslstride0};
    int32_t SHtlbound[] = {SHtlbound0, SHtlbound1, SHtlbound2, SHtlbound3};
    int32_t SHtlstride[] = {SHtlstride0, SHtlstride1, SHtlstride2, SHtlstride3};

    int32_t Oslstride[] = {Oslstride0};
    int32_t Otlbound[] = {Otlbound0, Otlbound1};
    int32_t Otlstride[] = {Otlstride0, Otlstride1};

    // This assigns the tasks inside the condition
    // to the core controlling the accelerator
    if (snrt_is_compute_core()) {  // can also be snrt_global_core_idx() == 0
        // This marks the start of the
        // setting of CSRs for the accelerator
        uint32_t start_csr_setup = snrt_mcycle();

        // Configure streamer settings

        // set_mx_streamer_csr
        set_mx_streamer_csr(delta_local_a, Aslstride, Atlbound, Atlstride,
                            delta_local_b, Bslstride, Btlbound, Btlstride,
                            delta_local_scale, SHslstride, SHtlbound,
                            SHtlstride, delta_local_o, Oslstride, Otlbound,
                            Otlstride);

        // Configure SHOut (Writer 1) for BFP requantization output scale
        if (QUANTIZE_MODE >= 2) {
            int32_t SHOutslstride[] = {SHOutslstride0};
            int32_t SHOuttlbound[] = {SHOuttlbound0, SHOuttlbound1};
            int32_t SHOuttlstride[] = {SHOuttlstride0, SHOuttlstride1};
            set_mx_shout_streamer_csr(delta_local_o_scale, SHOutslstride,
                                      SHOuttlbound, SHOuttlstride);
        }

        set_mx_csr(MODE, ACC_CNT, OUT_CNT);
        printf("CSR setup done...\n");

        // Start streamer then start ALU
        set_mx_streamer_start();
        printf("Streamer started...\n");
        set_mx_start();
        printf("mx started...\n");

        // Mark the end of the CSR setup cycles
        // uint32_t end_csr_setup = snrt_mcycle();

        wait_mx_and_streamer();

        //     printf("checkpoint 2\n");

        //     // TODO: change the checking method for mx
        //     //  Compare results and check if the
        //     //  accelerator returns correct answers
        //     //  For every incorrect answer, increment err
        //     if (QUANTIZE_MODE >= 2) {
        //         // Check quantized element output (packed mxint8/fp8/fp6
        //         codes) err +=
        //             check_mx_result((uint32_t*)local_o,
        //             (uint32_t*)O_quant_golden,
        //                             O_data_size / 4, /*is_fp32=*/0);
        //         // Check output shared scale
        //         err += check_mx_result((uint32_t*)local_o_scale,
        //                                (uint32_t*)O_scale_golden,
        //                                O_scale_data_size / 4, /*is_fp32=*/0);
        //     } else {
        //         int32_t O_length = O_data_size / 4;
        //         err = check_mx_result(local_o, O_golden, O_length,
        //         /*is_fp32=*/1);
        //     }

        //     // Read performance counter
        //     printf("======================================\n");
        //     printf("SNAX MX Accelerator Finished!\n");
        //     printf("Hardware Cycles: %d \n", read_mx_perf_counter());
        //     if (err == 0) {
        //         printf(">>> SUCCESS! Golden model matched! <<<\n");
        //     } else {
        //         printf(">>> FAILED! Errors found: %d <<<\n", err);
        //     }
        //     printf("======================================\n");
    };

    return err;
}
