// BF16FunctionsRISCV.cpp — portable bf16 core functions for RISC-V.
//
// Upstream BF16Functions.cpp implements only the NEON path ("TODO: raw cpu
// version of bf16"); this file is that raw version for RISC-V, plus the XSAI
// AME fast path.
//
// Convention (same as the ARM bf16 table): the backend keeps bytes == 4 —
// activations, C, bias and postParameters stay fp32 — and only matmulBytes == 2,
// i.e. the packed weights (and the packed A tile) are bf16. Packing converts
// fp32 -> bf16; the GEMM accumulates in fp32.
//
//   A: [l][eP px] bf16, element (ic, px) at ic * eP + px
//   B: [h/4][l][4 oc] bf16 (hP = 4, lP = 1), row stride bStride
//   C: NC4HW4 fp32, cStride = plane * unit * 4 bytes, 4 oc per pixel
#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#include <limits>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__riscv)

#include "../ame/xsai_alloc.h"
#include "../compute/ame_xsa.h"
#include <stdio.h>
#include <stdlib.h>

namespace MNN {
namespace {

// Truncate an fp32 value onto the bf16 grid (matches the test reference's
// convertFP32ToBF16 final rounding), so strict 5e-3 tolerances compare like
// with like.
inline float _fp32ToBf16RoundTrip(float f) {
    uint32_t bits;
    ::memcpy(&bits, &f, 4);
    bits &= 0xffff0000u;
    float g;
    ::memcpy(&g, &bits, 4);
    return g;
}

inline float _bf16ToFp32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    ::memcpy(&f, &bits, 4);
    return f;
}

// Truncating conversion, matching MNNFP32ToBF16 / the test's convertFP32ToBF16.
inline uint16_t _fp32ToBf16(float f) {
    uint32_t bits;
    ::memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

void _MNNFp32ToLowp(const float* src, int16_t* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = (int16_t)_fp32ToBf16(src[i]);
    }
}

void _MNNLowpToFp32(const int16_t* src, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = _bf16ToFp32((uint16_t)src[i]);
    }
}

// bf16 version of MNNAttenPackAndScaleSingleHead (CommonOptFunction.cpp):
// packs one query head [seqLen, headDim] into [l][eP] bf16 tiles for the
// packed matmul. The caller (CPUAttention) advances tiles by
// qStride0 = ROUND_UP(headDim, lP) * eP * bytes with bytes == 4, so each tile
// block keeps the fp32-unit spacing: element (d, sInner) of tile sOuter lives
// at byte offset sOuter * tileStride * 2 (uint16 units below).
void _bf16AttenPackAndScaleSingleHead(float* dstF, const float* srcHeadBase, size_t srcRowStride,
                                      const float* scale, const int32_t* units, size_t seqLen, size_t headDim) {
    const int32_t eP = units[0];
    const int32_t lP = units[1];
    const float scaleVal = scale[0];
    uint16_t* dst = (uint16_t*)dstF;
    // fp32-unit tile stride, counted in uint16 elements: headDimUp * eP * 4 / 2
    const size_t tileStrideEl = (size_t)ROUND_UP((int)headDim, lP) / lP * eP * lP * 2;
    for (size_t s = 0; s < seqLen; ++s) {
        const size_t sOuter = s / eP;
        const size_t sInner = s % eP;
        const float* srcRow = srcHeadBase + s * srcRowStride;
        uint16_t* tile = dst + sOuter * tileStrideEl;
        for (size_t d = 0; d < headDim; ++d) {
            tile[(d / lP) * eP * lP + sInner * lP + (d % lP)] = _fp32ToBf16(srcRow[d] * scaleVal);
        }
    }
}

void _MNNGetMatMulPackMode(int* eP, int* lP, int* hP) {
    *eP = 64;  // one executor call == one AME m-tile
    *lP = 1;
    *hP = 4;
}

// Depthwise conv otherwise runs entirely on the fp32 core functions (with the
// bytes==4 convention the packed weights stay fp32), which silently upgrades
// the effective precision versus the bf16 backend contract. Same loop nest as
// ConvOpt.cpp's MNNConvRunForLineDepthwise, with bf16-grid operands and a
// bf16-grid result so low-precision references match.
void _bf16ConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width,
                                  size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step,
                                  size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep,
                                  const float* bias, const float* parameters) {
    const float minF = parameters[0];
    const float maxF = parameters[1];
    for (int y = 0; y < (int)height; ++y) {
        const float* srcY = src + y * srcHStep;
        float* dstY = dst + y * dstHStep;
        for (size_t dx = 0; dx < width; ++dx) {
            float* dst_x = dstY + dx * 4;
            const float* src_z = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            float summer[4];
            for (int v = 0; v < 4; ++v) {
                summer[v] = bias[v];
            }
            for (size_t fy = 0; fy < fh; ++fy) {
                const float* src_y = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (size_t fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 4 * fx;
                    const float* src_x = src_y + fx * dilateX_step;
                    for (int v = 0; v < 4; ++v) {
                        summer[v] += _fp32ToBf16RoundTrip(src_x[v]) * _fp32ToBf16RoundTrip(weight_x[v]);
                    }
                }
            }
            for (int v = 0; v < 4; ++v) {
                float value = summer[v];
                if (value < minF) value = minF;
                if (value > maxF) value = maxF;
                dst_x[v] = _fp32ToBf16RoundTrip(value);
            }
        }
    }
}

// B: fp32 [h][l] (transpose) or [l][h] -> bf16 [h/4][l][4].
void _MNNPackForMatMul_B(float* destF, const float* sourceF, size_t h, size_t kernelsize, size_t ic,
                         bool transpose) {
    uint16_t* dest = (uint16_t*)destF;
    const size_t l = kernelsize * ic;
    const size_t hC = UP_DIV((int)h, 4);
    ::memset(dest, 0, hC * l * 4 * sizeof(uint16_t));
    for (size_t y = 0; y < h; ++y) {
        uint16_t* destY = dest + (y / 4) * l * 4 + (y % 4);
        const float* sourceY = sourceF + (transpose ? y * l : y);
        for (size_t x = 0; x < l; ++x) {
            destY[x * 4] = _fp32ToBf16(sourceY[transpose ? x : x * h]);
        }
    }
}

// A: fp32 C4HW4 source blits -> bf16 [l][eP], element (ic, px) at ic*eP + px.
void _MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int number = info[0];
    const int eReal = info[1];
    const int eDest = info[2];
    const int offset = info[3];
    if (1 == number) {
        // lP == 1, so the packed A rows are exactly eDest elements wide and the
        // buffer is l * eDest. Clear just the pixel tail of each row; zeroing a
        // 4-aligned l extent would write past the executor's buffer.
        const int e = el[0];
        const int l = el[1];
        if (e < eDest) {
            uint16_t* rows = (uint16_t*)destOrigin;
            for (int x = 0; x < l; ++x) {
                ::memset(rows + (size_t)x * eDest + e, 0, (size_t)(eDest - e) * sizeof(uint16_t));
            }
        }
    }
    for (int n = 0; n < number; ++n) {
        const int e = el[4 * n + 0];
        const int l = el[4 * n + 1];
        const int eOffset = el[4 * n + 2];
        const int lOffset = el[4 * n + 3];
        const float* source = sourceGroup[n];
        uint16_t* dest = (uint16_t*)destOrigin + lOffset * eDest + eOffset;
        for (int y = 0; y < e; ++y) {
            const int yR = y % eDest;
            for (int x = 0; x < l; ++x) {
                const int xR = x % 4;
                const int xC = x / 4;
                dest[x * eDest + yR] = _fp32ToBf16(source[xC * eReal * 4 + y * 4 * offset + xR]);
            }
        }
    }
}

struct Bf16Params {
    int h;
    int l;
    int eP;          // A rows (pixels) per call
    size_t cStride;  // fp32 units
    size_t bStride;  // bf16 units
    float minValue;
    float maxValue;
};

Bf16Params _decodeParams(const size_t* parameter, const float* postParameters) {
    Bf16Params p;
    p.h = (int)parameter[2];
    p.l = (int)parameter[1];
    p.eP = (int)(parameter[0] / sizeof(float));  // parameter[0] = eP * lP * bytes
    p.cStride = parameter[3] / sizeof(float);
    p.bStride = parameter[5] / sizeof(uint16_t) + (size_t)p.l * 4;
    p.minValue = -std::numeric_limits<float>().max();
    p.maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        p.minValue = postParameters[2];
        p.maxValue = postParameters[3];
    }
    return p;
}

// Scalar reference GEMM (also the fallback when the AME pool is unavailable).
void _bf16MatmulRef(float* C, const float* A, const float* B, size_t eSize, const Bf16Params& p,
                    const float* bias) {
    const uint16_t* a = (const uint16_t*)A;
    const uint16_t* b = (const uint16_t*)B;
    const int hC4 = UP_DIV(p.h, 4);
    for (size_t x = 0; x < eSize; ++x) {
        for (int y = 0; y < hC4; ++y) {
            float summer[4] = {0.f, 0.f, 0.f, 0.f};
            if (nullptr != bias) {
                for (int v = 0; v < 4; ++v) {
                    summer[v] = bias[4 * y + v];
                }
            }
            for (int z = 0; z < p.l; ++z) {
                const float aZ = _bf16ToFp32(a[(size_t)z * p.eP + x]);
                const uint16_t* wZ = b + (size_t)y * p.bStride + (size_t)z * 4;
                for (int v = 0; v < 4; ++v) {
                    summer[v] += aZ * _bf16ToFp32(wZ[v]);
                }
            }
            for (int v = 0; v < 4; ++v) {
                float value = summer[v];
                if (value < p.minValue) value = p.minValue;
                if (value > p.maxValue) value = p.maxValue;
                C[(size_t)y * p.cStride + x * 4 + v] = _fp32ToBf16RoundTrip(value);
            }
        }
    }
}

// Maximum K (bf16 elements) a single AME dispatch may pack. Larger K falls
// back to the scalar bf16 reference (correct, slower) — never silently wrong.
static const int kAmeBf16KMax = 8192;

struct AmeBf16State {
    bool ready = false;
    bool usable = false;
    int tileM = 64;
    int tileN = 64;
    uint16_t* aTile = nullptr;
    uint16_t* bTile = nullptr;
    float* cTile = nullptr;
};

AmeBf16State& _ameState() {
    static AmeBf16State state;
    return state;
}

bool _ameEnsure() {
    AmeBf16State& s = _ameState();
    if (s.ready) {
        return s.usable;
    }
    s.ready = true;
    static int wanted = -1;
    if (wanted < 0) {
        const char* v = getenv("XSAI_MNN_AME");
        wanted = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    if (wanted != 1) {
        return false;
    }
    // One shared tile decision with the int8 kernel (ame_xsa.c): the 64-wide
    // spec is emulator-only, real hardware needs the native 128 tile.
    const xsai_ame_tile_spec* spec = xsai_ame_tile_spec_get();
    if (!spec->enabled) {
        fprintf(stderr, "[xsai-ame] bf16: no validated tile for this target, scalar bf16 path\n");
        return false;
    }
    s.tileM = spec->tile_m;
    s.tileN = spec->tile_n;
    // Tile buffers sized for the maximum supported K (see kAmeBf16KMax): one
    // k-block is 32 bf16 elements per row, so a tile is tileM/tileN x K bf16.
    // 8192 matches the int8 side's kAmeKMax and covers mainstream LLM shapes
    // (K = 4096 hidden, 2880..8192 conv ic*kx*ky); each buffer is 2 MB at
    // tile 128.
    s.aTile = (uint16_t*)xsai_malloc((size_t)s.tileM * kAmeBf16KMax * sizeof(uint16_t));
    s.bTile = (uint16_t*)xsai_malloc((size_t)s.tileN * kAmeBf16KMax * sizeof(uint16_t));
    s.cTile = (float*)xsai_malloc((size_t)s.tileM * s.tileN * sizeof(float));
    if (s.aTile == nullptr || s.bTile == nullptr || s.cTile == nullptr) {
        fprintf(stderr, "[xsai-ame] bf16: xsai_malloc failed, scalar bf16 path\n");
        s.aTile = nullptr;
        s.bTile = nullptr;
        s.cTile = nullptr;
        return false;
    }
    // Physically contiguous operands are an AMU requirement only on real
    // hardware; qemu-user's matrix ops go through guest virtual addresses, so
    // the host-test allocator mode is accepted (see xsai_alloc.h).
    if (!xsai_pool_phys_contiguous() && !xsai_alloc_host_test_mode()) {
        fprintf(stderr, "[xsai-ame] bf16: pool not physically contiguous, scalar bf16 path\n");
        return false;
    }
    s.usable = true;
    fprintf(stderr, "[xsai-ame] bf16 GEMM via AME (tile %dx32x%d)\n", s.tileM, s.tileN);
    return true;
}

bool _ameBf16Matmul(float* C, const float* A, const float* B, size_t eSize, const Bf16Params& p,
                    const float* bias) {
    if (p.l <= 0 || p.l > kAmeBf16KMax || eSize > (size_t)p.eP || !_ameEnsure()) {
        return false;
    }
    // Same work threshold as the int8 path: tiny oc x K products go to the
    // scalar bf16 reference instead of a mostly-empty AME tile.
    if ((size_t)p.h * (size_t)p.l < xsai_ame_min_macs()) {
        return false;
    }
    AmeBf16State& s = _ameState();
    const uint16_t* a = (const uint16_t*)A;
    const uint16_t* b = (const uint16_t*)B;
    const int kTiles = UP_DIV(p.l, 32);
    for (int oc0 = 0; oc0 < p.h; oc0 += s.tileN) {
        const int ocN = (p.h - oc0 < s.tileN) ? (p.h - oc0) : s.tileN;
        ::memset(s.bTile, 0, (size_t)s.tileN * kTiles * 32 * sizeof(uint16_t));
        for (int oc = 0; oc < ocN; ++oc) {
            const uint16_t* row = b + (size_t)(oc0 + oc) / 4 * p.bStride + ((oc0 + oc) % 4);
            for (int k = 0; k < p.l; ++k) {
                s.bTile[(size_t)(k / 32) * s.tileN * 32 + (size_t)oc * 32 + (k % 32)] = row[(size_t)k * 4];
            }
        }
        for (int px0 = 0; px0 < (int)eSize; px0 += s.tileM) {
            const int pxN = ((int)eSize - px0 < s.tileM) ? ((int)eSize - px0) : s.tileM;
            ::memset(s.aTile, 0, (size_t)s.tileM * kTiles * 32 * sizeof(uint16_t));
            for (int px = 0; px < pxN; ++px) {
                for (int ic = 0; ic < p.l; ++ic) {
                    s.aTile[(size_t)(ic / 32) * s.tileM * 32 + (size_t)px * 32 + (ic % 32)] =
                        a[(size_t)ic * p.eP + px0 + px];
                }
            }
            xsai_ame_bf16_gemm_tile_kloop(s.aTile, (ptrdiff_t)s.tileM * 32 * sizeof(uint16_t), s.bTile,
                                          (ptrdiff_t)s.tileN * 32 * sizeof(uint16_t), kTiles, 2, s.cTile,
                                          s.tileM, s.tileN);
            for (int px = 0; px < pxN; ++px) {
                for (int oc = 0; oc < ocN; ++oc) {
                    float value = s.cTile[(size_t)px * s.tileN + oc];
                    if (bias != nullptr) {
                        value += bias[oc0 + oc];
                    }
                    if (value < p.minValue) value = p.minValue;
                    if (value > p.maxValue) value = p.maxValue;
                    C[(size_t)(oc0 + oc) / 4 * p.cStride + (size_t)(px0 + px) * 4 + ((oc0 + oc) % 4)] =
                        _fp32ToBf16RoundTrip(value);
                }
            }
        }
    }
    return true;
}

void _MNNPackedMatMulImpl(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                          const float* postParameters, const float* bias) {
    const Bf16Params p = _decodeParams(parameter, postParameters);
    static int dump = -1;
    if (dump < 0) {
        const char* v = getenv("XSAI_MNN_BF16_DUMP");
        dump = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    if (dump) {
        fflush(stdout);
        fprintf(stderr, "[xsai-bf16-dump] h=%d l=%d eP=%d eSize=%zu bStride=%zu cStride=%zu\n", p.h, p.l, p.eP,
                eSize, p.bStride, p.cStride);
        const uint16_t* a = (const uint16_t*)A;
        const uint16_t* b = (const uint16_t*)B;
        for (int z = 0; z < p.l && z < 4; ++z) {
            fprintf(stderr, "  a[%d][0..3]=%04x %04x %04x %04x\n", z, a[(size_t)z * p.eP], a[(size_t)z * p.eP + 1],
                    a[(size_t)z * p.eP + 2], a[(size_t)z * p.eP + 3]);
        }
        for (int y = 0; y < 4 && y < UP_DIV(p.h, 4); ++y) {
            fprintf(stderr, "  b[%d][0..3]=%04x %04x %04x %04x\n", y, b[(size_t)y * p.bStride],
                    b[(size_t)y * p.bStride + 1], b[(size_t)y * p.bStride + 2], b[(size_t)y * p.bStride + 3]);
        }
    }
    if (_ameBf16Matmul(C, A, B, eSize, p, bias)) {
        if (dump) {
            fprintf(stderr, "  ame C[0..3]=%g %g %g %g\n", C[0], C[1], C[2], C[3]);
        }
        return;
    }
    _bf16MatmulRef(C, A, B, eSize, p, bias);
    if (dump) {
        fprintf(stderr, "  ref C[0..3]=%g %g %g %g\n", C[0], C[1], C[2], C[3]);
    }
}

void _MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                      const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulImpl(C, A, B, parameter[0] / sizeof(float), parameter, postParameters, bias);
}

void _MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                            const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulImpl(C, A, B, eSize, parameter, postParameters, bias);
}

// CPUMatMul dispatches e==1 / h==1 shapes to the GEMV entry points
// (MNNComputeMatMulForE_1 / ForH_1) instead of the packed GEMM table. Those
// come from the fp32 core copy and would compute at full fp32 precision while
// the bf16 backend contract (and the test reference) truncates the operands to
// bf16. Wrap them: round the operands to bf16-in-fp32, then run the fp32 GEMV.
// AME has no GEMV benefit (M or N == 1), so the scalar fp32 kernel is fine.
static void (*s_fp32MatMulForH_1)(const float*, const float*, float*, const float*, const MatMulParam*, size_t);
static void (*s_fp32MatMulForE_1)(const float*, const float*, float*, const float*, const MatMulParam*, size_t);

static void _bf16RoundBuffer(float* dst, const float* src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits;
        ::memcpy(&bits, &src[i], 4);
        bits &= 0xffff0000u;
        ::memcpy(&dst[i], &bits, 4);
    }
}

static void _bf16MatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr,
                              const MatMulParam* param, size_t tId) {
    if (s_fp32MatMulForH_1 == nullptr) {
        return;
    }
    const size_t sizeA = (size_t)param->e * param->l;
    const size_t sizeB = (size_t)param->l * param->h;
    float* aR = (float*)malloc(sizeA * sizeof(float));
    float* bR = (float*)malloc(sizeB * sizeof(float));
    if (aR == nullptr || bR == nullptr) {
        free(aR);
        free(bR);
        s_fp32MatMulForH_1(A, B, C, biasPtr, param, tId);
        return;
    }
    _bf16RoundBuffer(aR, A, sizeA);
    _bf16RoundBuffer(bR, B, sizeB);
    s_fp32MatMulForH_1(aR, bR, C, biasPtr, param, tId);
    _bf16RoundBuffer(C, C, (size_t)param->e * param->h);
    free(aR);
    free(bR);
}

static void _bf16MatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr,
                              const MatMulParam* param, size_t tId) {
    if (s_fp32MatMulForE_1 == nullptr) {
        return;
    }
    const size_t sizeA = (size_t)param->e * param->l;
    const size_t sizeB = (size_t)param->l * param->h;
    float* aR = (float*)malloc(sizeA * sizeof(float));
    float* bR = (float*)malloc(sizeB * sizeof(float));
    if (aR == nullptr || bR == nullptr) {
        free(aR);
        free(bR);
        s_fp32MatMulForE_1(A, B, C, biasPtr, param, tId);
        return;
    }
    _bf16RoundBuffer(aR, A, sizeA);
    _bf16RoundBuffer(bR, B, sizeB);
    s_fp32MatMulForE_1(aR, bR, C, biasPtr, param, tId);
    _bf16RoundBuffer(C, C, (size_t)param->e * param->h);
    free(aR);
    free(bR);
}

}  // namespace

// Called from BF16Functions::init() on the __riscv branch.
void RISCVBF16FillCoreFunctions(CoreFunctions* gInstance) {
    s_fp32MatMulForH_1 = gInstance->MNNComputeMatMulForH_1;
    s_fp32MatMulForE_1 = gInstance->MNNComputeMatMulForE_1;
    gInstance->MNNConvRunForLineDepthwise = _bf16ConvRunForLineDepthwise;
    gInstance->MNNAttenPackAndScaleSingleHead = _bf16AttenPackAndScaleSingleHead;
    gInstance->MNNFp32ToLowp = _MNNFp32ToLowp;
    gInstance->MNNLowpToFp32 = _MNNLowpToFp32;
    gInstance->matmulBytes = 2;
    gInstance->MNNGetMatMulPackMode = _MNNGetMatMulPackMode;
    gInstance->MNNPackForMatMul_B = _MNNPackForMatMul_B;
    gInstance->MNNPackC4ForMatMul_A = _MNNPackC4ForMatMul_A;
    gInstance->MNNPackedMatMul = _MNNPackedMatMul;
    gInstance->MNNPackedMatMulRemain = _MNNPackedMatMulRemain;
    gInstance->MNNComputeMatMulForH_1 = _bf16MatMulForH_1;
    gInstance->MNNComputeMatMulForE_1 = _bf16MatMulForE_1;
}

}  // namespace MNN

#endif  // defined(__riscv)
