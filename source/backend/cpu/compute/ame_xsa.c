// ame_xsa.c — XSAI AME implementation (the only TU with AME instructions).
// Compiled with the zames/zmasync march (see cpu/CMakeLists.txt).
// Ported from llama.cpp ggml arch/riscv/ame (non-STC macro variant).
#include "ame_xsa.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AME_TILE_M 64
#define AME_TILE_K 64
#define AME_TILE_N 64
// The AME K extent is expressed in bytes for mla/mlb; bf16 rows therefore hold
// AME_TILE_K/sizeof(bf16) = 32 elements per k-tile block.
#define AME_TILE_K_BF16_ELEMS 32

#define AME_MCFG_INT8  0x02
#define AME_MCFG_INT32 0x04
#define AME_MCFG_BF16  0x0a
#define AME_MCFG_FP32  0x0c

#define XSAI_MSETTILEM(RD, VAL) \
    asm volatile ("msettilem %0" : : "r"(VAL) : ); (RD) = (VAL);
#define XSAI_MSETTILEK(RD, VAL) \
    asm volatile ("msettilek %0" : : "r"(VAL) : ); (RD) = (VAL);
#define XSAI_MSETTILEN(RD, VAL) \
    asm volatile ("msettilen %0" : : "r"(VAL) : ); (RD) = (VAL);
#define XSAI_MZERO(ACC) \
    asm volatile ("mzero " #ACC : : :)
#define XSAI_MLA(REG, SRC, N) \
    asm volatile ("mla " #REG ", (%0), %1" : : "r"(SRC), "r"(N) :)
#define XSAI_MLB(REG, SRC, N) \
    asm volatile ("mlb " #REG ", (%0), %1" : : "r"(SRC), "r"(N) :)
#define XSAI_MSC(REG, DST, N) \
    asm volatile ("msc " #REG ", (%0), %1" : : "r"(DST), "r"(N) : "memory")
#define XSAI_MMACC(ACC, TR0, TR2) \
    asm volatile ("mmacc " #ACC ", " #TR2 ", " #TR0 "\n" : : : "memory")
#define XSAI_MSETCFG(CFG, VAL) \
    asm volatile ("msetcfg " #CFG ", %0" : : "r"(VAL) : "memory")
#define XSAI_MSYNC_RESET() \
    asm volatile ("msyncregreset sync0" ::: "memory")
#define XSAI_MSYNC_RELEASE() \
    asm volatile ("mrelease sync0" ::: "memory")
#define XSAI_MSYNC_ACQUIRE(target) \
    do { asm volatile ("macquire sync0, %0" :: "r"(target) : "memory"); \
         asm volatile ("mfence" ::: "memory"); } while (0)

static inline void xsai_ame_config_i8_i32(void) {
    const unsigned long cfg_i8 = AME_MCFG_INT8;
    const unsigned long cfg_i32 = AME_MCFG_INT32;
    XSAI_MSETCFG(mcfg0, cfg_i8);
    XSAI_MSETCFG(mcfg1, cfg_i8);
    XSAI_MSETCFG(mcfg2, cfg_i8);
    XSAI_MSETCFG(mcfg3, cfg_i8);
    XSAI_MSETCFG(mcfg4, cfg_i32);
    XSAI_MSETCFG(mcfg5, cfg_i32);
    XSAI_MSETCFG(mcfg6, cfg_i32);
    XSAI_MSETCFG(mcfg7, cfg_i32);
}

// bf16 operands (mcfg0-3) accumulating into an fp32 C tile (mcfg4-7).
static inline void xsai_ame_config_bf16_fp32(void) {
    const unsigned long cfg_bf16 = AME_MCFG_BF16;
    const unsigned long cfg_fp32 = AME_MCFG_FP32;
    XSAI_MSETCFG(mcfg0, cfg_bf16);
    XSAI_MSETCFG(mcfg1, cfg_bf16);
    XSAI_MSETCFG(mcfg2, cfg_bf16);
    XSAI_MSETCFG(mcfg3, cfg_bf16);
    XSAI_MSETCFG(mcfg4, cfg_fp32);
    XSAI_MSETCFG(mcfg5, cfg_fp32);
    XSAI_MSETCFG(mcfg6, cfg_fp32);
    XSAI_MSETCFG(mcfg7, cfg_fp32);
}

static inline uint64_t xsai_ame_rdcycle_impl(void) {
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// fp16 -> bf16 element conversion (ggml GGML_AME_F16_BF16 semantics): widen to
// fp32 exactly, then truncate the low 16 mantissa bits, which is the same
// rounding MNN's MNNFp32ToLowp / test convertFP32ToBF16 use.
static inline uint16_t xsai_f16_to_bf16_scalar(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t man        = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // Subnormal half: renormalize into an fp32 exponent.
            int e = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                --e;
            }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)e << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    return (uint16_t)(bits >> 16);
}

void xsai_ame_f16_to_bf16(const uint16_t * src, uint16_t * dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = xsai_f16_to_bf16_scalar(src[i]);
    }
}
const xsai_ame_hw_config * xsai_ame_hw_probe(void) {
    return xsai_ame_hw_config_get();
}

size_t xsai_ame_min_macs(void) {
    static size_t cached = 0;
    if (cached == 0) {
        const char * v = getenv("XSAI_MNN_AME_MIN_MACS");
        cached = (v != NULL) ? (size_t)strtoul(v, NULL, 0) : (size_t)16384;
        if (cached == 0) {
            cached = 1;  // 0 disables the threshold check entirely
        }
    }
    return cached;
}

// caller 侧缓冲上限（MNN/ncnn 的 atile/btile/ctile 按 128 x kAmeKMax / 128x128 字节分配）。
// 探测值超过上限时截断：用不满只是慢，用超了才是错。
#define XSAI_AME_TILE_M_MAX 128
#define XSAI_AME_TILE_N_MAX 128
#define XSAI_AME_TILE_K_MAX 8192

const xsai_ame_tile_spec * xsai_ame_tile_spec_get(void) {
    static xsai_ame_tile_spec spec;
    static int inited = 0;
    if (inited) {
        return &spec;
    }
    inited = 1;

    // 兜底值：只在探测无效时用（qemu-user / NEMU 没有 AME CSR）。
    spec.tile_m = 64;
    spec.tile_n = 64;
    spec.tile_k = 64;
    spec.enabled = 1;

    const xsai_ame_hw_config * hw = xsai_ame_hw_config_get();
    if (hw->valid) {
        // tlenb / trlenb / alenb 是**硬件探测**出来的，而且不止一种配置：
        //   tile_m = tlenb / trlenb        (行数)
        //   tile_k = trlenb                (字节，一个 k-block)
        //   tile_n = (alenb / rows) / 4    (int32 元素数)
        // 这里直接采用探测值；写成 64 / 128 的常量在别的配置上就是错的
        // （ame_xsa.h 里记着：v13 位流上用 64 宽 tile 每个多 tile 形状都算错）。
        int tm = hw->int8_m_max;
        int tn = hw->int32_n_max;
        int tk = hw->int8_k_max;
        if (tm > XSAI_AME_TILE_M_MAX) tm = XSAI_AME_TILE_M_MAX;
        if (tn > XSAI_AME_TILE_N_MAX) tn = XSAI_AME_TILE_N_MAX;
        if (tk > XSAI_AME_TILE_K_MAX) tk = XSAI_AME_TILE_K_MAX;
        if (tm > 0 && tn > 0 && tk > 0) {
            spec.tile_m = tm;
            spec.tile_n = tn;
            spec.tile_k = tk;
        } else {
            spec.enabled = 0;
        }
    }

    // 显式实验覆盖（在探测值之上覆盖，保留原有 env 旋钮语义）
    int force = 0;  // 0 = auto
    const char * v = getenv("XSAI_MNN_AME_TILE");
    if (v != NULL) {
        force = atoi(v);
    }
    if (force == 0 && getenv("XSAI_MNN_AME_128") != NULL) {
        force = 128;  // legacy alias of XSAI_MNN_AME_TILE=128
    }
    if (force > 0) {
        spec.tile_m = force;
        spec.tile_n = force;
    }

    fprintf(stderr, "[XSAI-AME] tile spec: m=%d n=%d k=%d enabled=%d (%s)\n",
            spec.tile_m, spec.tile_n, spec.tile_k, spec.enabled,
            hw->valid ? "probed" : "fallback(no AME CSR)");
    return &spec;
}

uint64_t xsai_ame_rdcycle(void) {
    return xsai_ame_rdcycle_impl();
}

const xsai_ame_hw_config * xsai_ame_hw_config_get(void) {
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
            cfg.alenb == UINT64_MAX || cfg.trlenb == 0 ||
            cfg.tlenb % cfg.trlenb != 0) {
        return &cfg;
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
    // 和 llama 侧 [AME] hw probe 同款一行，方便板上直接 grep 对照三套栈
    fprintf(stderr,
            "[XSAI-AME] hw probe: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu melen_bits=%llu "
            "arlenb=%llu capacity=m%dn%dk%d.i8i32\n",
            (unsigned long long) cfg.tlenb, (unsigned long long) cfg.trlenb,
            (unsigned long long) cfg.alenb, (unsigned long long) cfg.rows,
            (unsigned long long) cfg.melen_bits, (unsigned long long) cfg.arlenb,
            cfg.int8_m_max, cfg.int32_n_max, cfg.int8_k_max);
    return &cfg;
}

// One 64x64 output tile: K-loop over k_tiles with 2-deep software pipelining,
// accumulate in acc0, store once (rows of TILE_N int32), then drain MSC.
void xsai_ame_gemm_tile_kloop(const int8_t * a_tiles, ptrdiff_t a_tile_stride,
                              const int8_t * b_tiles, ptrdiff_t b_tile_stride,
                              int k_tiles, int32_t * C,
                              int tile_m, int tile_k, int tile_n) {
    int tmp;
    XSAI_MSETTILEM(tmp, tile_m);
    XSAI_MSETTILEK(tmp, tile_k);
    XSAI_MSETTILEN(tmp, tile_n);
    xsai_ame_config_i8_i32();

    XSAI_MSYNC_RESET();
    XSAI_MZERO(acc0);
    __asm__ volatile("mfence" ::: "memory");

    // CUTE consumes the MSC base register asynchronously after the scalar
    // core has issued the store. Pin it in a callee-saved register and keep
    // it live until acquire, or the compiler may recycle it and the tile is
    // stored to a stale address.
    register int32_t * c_live __asm__("s11") = C;

    const int8_t * addr_a = a_tiles;
    const int8_t * addr_b = b_tiles;
    int kb = 0;
    for (; kb + 1 < k_tiles; kb += 2) {
        XSAI_MLA(tr0, addr_a, tile_k);
        XSAI_MLB(tr1, addr_b, tile_k);
        XSAI_MMACC(acc0, tr0, tr1);
        addr_a += a_tile_stride;
        addr_b += b_tile_stride;
        XSAI_MLA(tr2, addr_a, tile_k);
        XSAI_MLB(tr3, addr_b, tile_k);
        XSAI_MMACC(acc0, tr2, tr3);
        addr_a += a_tile_stride;
        addr_b += b_tile_stride;
    }
    if (kb < k_tiles) {
        XSAI_MLA(tr0, addr_a, tile_k);
        XSAI_MLB(tr1, addr_b, tile_k);
        XSAI_MMACC(acc0, tr0, tr1);
    }

    XSAI_MSC(acc0, c_live, tile_n * (int)sizeof(int32_t));
    // ggml begin_op pattern: rebase the sync epoch per tile so the hardware
    // counter stays small and the acquire target is exact.
    XSAI_MSYNC_RELEASE();
    unsigned long acquire_target = 1;
    XSAI_MSYNC_ACQUIRE(acquire_target);
    __asm__ volatile("" : "+r"(c_live) : : "memory");
}

// One tile_m x tile_n bf16 tile with an fp32 accumulator. Same byte-granular K
// model as the int8 path: mla/mlb consume AME_TILE_K bytes of a row, i.e.
// AME_TILE_K_BF16_ELEMS bf16 elements, while msettilek takes the element count.
void xsai_ame_bf16_gemm_tile_kloop(const void * a_tiles, ptrdiff_t a_tile_stride,
                                   const void * b_tiles, ptrdiff_t b_tile_stride,
                                   int k_tiles, int reg_pairs, float * C,
                                   int tile_m, int tile_n) {
    if (k_tiles <= 0) {
        return;
    }
    int tmp;
    // k 宽度取自硬件探测（trlenb），不是常量：msettilek 收元素数，mla/mlb 收字节数。
    const int bf16_tk_bytes = xsai_ame_tile_spec_get()->tile_k;
    const int bf16_tk_elems = bf16_tk_bytes / (int) sizeof(uint16_t);
    XSAI_MSETTILEM(tmp, tile_m);
    XSAI_MSETTILEK(tmp, bf16_tk_elems);
    XSAI_MSETTILEN(tmp, tile_n);
    xsai_ame_config_bf16_fp32();

    XSAI_MSYNC_RESET();
    XSAI_MZERO(acc0);
    __asm__ volatile("mfence" ::: "memory");

    // See the int8 kernel: the MSC base register is consumed asynchronously by
    // CUTE, so it stays pinned in a callee-saved register until acquire.
    register float * c_live __asm__("s11") = C;

    const char * addr_a = (const char *)a_tiles;
    const char * addr_b = (const char *)b_tiles;
    if (reg_pairs >= 2) {
        int kb = 0;
        for (; kb + 1 < k_tiles; kb += 2) {
            XSAI_MLA(tr0, addr_a, bf16_tk_bytes);
            XSAI_MLB(tr1, addr_b, bf16_tk_bytes);
            XSAI_MMACC(acc0, tr0, tr1);
            addr_a += a_tile_stride;
            addr_b += b_tile_stride;
            XSAI_MLA(tr2, addr_a, bf16_tk_bytes);
            XSAI_MLB(tr3, addr_b, bf16_tk_bytes);
            XSAI_MMACC(acc0, tr2, tr3);
            addr_a += a_tile_stride;
            addr_b += b_tile_stride;
        }
        if (kb < k_tiles) {
            XSAI_MLA(tr0, addr_a, bf16_tk_bytes);
            XSAI_MLB(tr1, addr_b, bf16_tk_bytes);
            XSAI_MMACC(acc0, tr0, tr1);
        }
    } else {
        for (int kb = 0; kb < k_tiles; ++kb) {
            XSAI_MLA(tr0, addr_a, bf16_tk_bytes);
            XSAI_MLB(tr1, addr_b, bf16_tk_bytes);
            XSAI_MMACC(acc0, tr0, tr1);
            addr_a += a_tile_stride;
            addr_b += b_tile_stride;
        }
    }

    XSAI_MSC(acc0, c_live, tile_n * (int)sizeof(float));
    XSAI_MSYNC_RELEASE();
    unsigned long acquire_target = 1;
    XSAI_MSYNC_ACQUIRE(acquire_target);
    __asm__ volatile("" : "+r"(c_live) : : "memory");
}
