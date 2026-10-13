// ame_xsa.h — XSAI AME matrix-extension interface for the MNN CPU backend.
// Declarations only: the implementation (compute/ame_xsa.c) is the single
// translation unit compiled with the AME march (zames/zmasync/zvl128b).
#ifndef AME_XSA_H
#define AME_XSA_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint64_t tlenb;
    uint64_t trlenb;
    uint64_t alenb;
    uint64_t rows;
    uint64_t melen_bits;
    uint64_t arlenb;
    int int8_m_max;
    int int8_k_max;
    int32_t int32_n_max;
    int valid;
} xsai_ame_hw_config;

// CSR 0xcc1/0xcc2/0xcc3 probe + derived fields. valid=0 if AME unavailable.
#ifdef __cplusplus
extern "C" {
#endif

const xsai_ame_hw_config * xsai_ame_hw_config_get(void);
const xsai_ame_hw_config * xsai_ame_hw_probe(void);

uint64_t xsai_ame_rdcycle(void);

// Tile-spec selection — the single source of truth for which AME tile shape a
// caller may use. Hardware measurement (harness on the v13 bit):
//   int8 64x64x64    -> wrong results for every multi-tile shape
//   int8 128x64x128  -> bit-exact for all shapes
// so a tile wider than 64 is mandatory where the hardware offers it, and the
// 64-wide spec is treated as emulator-only. tile_k counts bytes (64 == one
// k-block; 32 bf16 elements).
typedef struct {
    int tile_m;
    int tile_n;
    int tile_k;
    int enabled;   // 0 -> do not use the AME at all on this target
} xsai_ame_tile_spec;

const xsai_ame_tile_spec * xsai_ame_tile_spec_get(void);

// Minimum useful work per AME dispatch, expressed as output-channels x K
// (multiply-accumulate columns). The executor hands the kernel DST_XUNIT (=8)
// pixels at a time, so a call only fills 8 rows of a 128-row MMA tile; below
// this many oc*K products the padding overhead dominates and MNN's own RVV
// kernel is the better path. Overridable with XSAI_MNN_AME_MIN_MACS.
size_t xsai_ame_min_macs(void);

// One output tile: C = A(tile_m x k_tiles*tile_k) x BT(tile_n x k_tiles*tile_k)^T.
// a_tiles/b_tiles: k_tiles blocks of tile_m(=64)xtile_k / tile_n x tile_k int8
// (k-padded, row-major tile_k-byte rows).
void xsai_ame_gemm_tile_kloop(const int8_t * a_tiles, ptrdiff_t a_tile_stride,
                              const int8_t * b_tiles, ptrdiff_t b_tile_stride,
                              int k_tiles, int32_t * C,
                              int tile_m, int tile_k, int tile_n);

// One bf16 output tile with an fp32 accumulator: C = A x BT^T, A/B in bf16.
// The AME K extent is byte-based: one k-tile block is 64 bytes = 32 bf16
// elements per row, so k_tiles counts 32-element blocks. a_tiles/b_tiles hold
// k_tiles row-major blocks of tile_m x 32 / tile_n x 32 bf16 (k-padded).
// reg_pairs >= 2 selects the 2-deep pipelined inner loop.
void xsai_ame_bf16_gemm_tile_kloop(const void * a_tiles, ptrdiff_t a_tile_stride,
                                   const void * b_tiles, ptrdiff_t b_tile_stride,
                                   int k_tiles, int reg_pairs, float * C,
                                   int tile_m, int tile_n);

// fp16 -> bf16 element conversion. The AME has no fp16 MMA mode, so fp16
// operands take the ggml GGML_AME_F16_BF16 route: convert (weights once,
// activations per call) and run the bf16 kernel above.
void xsai_ame_f16_to_bf16(const uint16_t * src, uint16_t * dst, size_t size);

#ifdef __cplusplus
}
#endif

#endif  // AME_XSA_H
