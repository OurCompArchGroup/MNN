// xsai_ame_probe.h — XSAI AME hardware probe & cycle counter for MNN tools.
// Ported from llama.cpp ggml ame.h hw-config CSR probe (0xcc1/0xcc2/0xcc3);
// header-only, no ggml dependencies. All timing uses rdcycle (FPGA 50MHz).
#ifndef XSAI_AME_PROBE_H
#define XSAI_AME_PROBE_H

#include <stdint.h>

#if defined(__riscv)

static inline uint64_t xsai_ame_rdcycle(void) {
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

typedef struct {
    uint64_t tlenb;
    uint64_t trlenb;
    uint64_t alenb;
    uint64_t rows;
    uint64_t melen_bits;
    uint64_t arlenb;
    int int8_m_max;
    int int8_k_max;
    int int32_n_max;
    int valid;
} xsai_ame_hw_config;

static inline const xsai_ame_hw_config * xsai_ame_hw_probe(void) {
    static xsai_ame_hw_config cfg;
    static int initialized = 0;
    if (initialized) {
        return &cfg;
    }
    initialized = 1;

    __asm__ volatile("csrr %0, 0xcc1" : "=r"(cfg.tlenb));
    __asm__ volatile("csrr %0, 0xcc2" : "=r"(cfg.trlenb));
    __asm__ volatile("csrr %0, 0xcc3" : "=r"(cfg.alenb));

    if (cfg.tlenb == 0 || cfg.trlenb == 0 || cfg.alenb == 0 ||
            cfg.tlenb == UINT64_MAX || cfg.trlenb == UINT64_MAX ||
            cfg.alenb == UINT64_MAX || cfg.tlenb % cfg.trlenb != 0) {
        return &cfg;  // invalid: cfg.valid stays 0
    }

    cfg.rows = cfg.tlenb / cfg.trlenb;
    if (cfg.rows == 0 || cfg.alenb % cfg.rows != 0 ||
            cfg.rows > UINT64_MAX / cfg.rows || cfg.alenb > UINT64_MAX / 8) {
        return &cfg;
    }

    cfg.arlenb = cfg.alenb / cfg.rows;
    const uint64_t elems = cfg.rows * cfg.rows;
    const uint64_t alen_bits = cfg.alenb * 8;
    if (elems == 0 || alen_bits % elems != 0) {
        return &cfg;
    }

    cfg.melen_bits = alen_bits / elems;
    cfg.int8_m_max = (int)cfg.rows;
    cfg.int8_k_max = (int)cfg.trlenb;
    cfg.int32_n_max = (int)(cfg.arlenb / sizeof(int32_t));
    cfg.valid = 1;
    return &cfg;
}

#else  // !__riscv

typedef struct {
    uint64_t tlenb, trlenb, alenb, rows, melen_bits, arlenb;
    int int8_m_max, int8_k_max, int32_n_max, valid;
} xsai_ame_hw_config;

static inline uint64_t xsai_ame_rdcycle(void) { return 0; }
static inline const xsai_ame_hw_config * xsai_ame_hw_probe(void) {
    static xsai_ame_hw_config cfg;
    cfg.valid = 0;
    return &cfg;
}

#endif  // __riscv

#endif  // XSAI_AME_PROBE_H
