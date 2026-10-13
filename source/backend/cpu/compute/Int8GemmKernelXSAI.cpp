// Int8GemmKernelXSAI.cpp — XSAI AME drop-in for the MNN int8 GEMM kernel.
//
// Signature-compatible with CoreInt8Functions::Int8GemmKernel. Enabled via
// env XSAI_MNN_AME=1 (selection happens in DenseConvInt8TiledExecutor).
//
// Constraints (checked by the caller):
//   symmetric static int8 (mBlockNum == 1, !mWeightAsymmetricQuant),
//   int8 output (post->useInt8 == 1), gcore->bytes == 1.
//
// Semantics replicated from MNNGemmInt8AddBiasScale_16x4_Unit_RVV:
//   value = acc*scale_dz[j] + srcSumPtr[w]*weightBias_dz[j]
//         (+ inputScale / inputBias / bias terms per the same conditionals)
//   dst[w*16+j] = (int8_t)roundf(clamp(value, minValue, maxValue))
//
// AME hardware notes: operands live in the XSAI direct-map reserved pool
// (0x880000000, 1:1 virtual == physical via /dev/mem — STRICT_DEVMEM off).
#include "ame_xsa.h"
#include "../ame/xsai_alloc.h"
#include "Int8FunctionsOpt.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <cstddef>
#include <cstdio>
#include <csignal>
#include <string.h>

namespace {

// The executor's int8 gemm units (from CoreInt8Functions::MNNGetGemmUnit).
// Defaults match the classic 4-ic layout; the executor sets the real values
// via xsai_ame_set_gemm_units() before the first call.
static size_t XSAI_UNIT = 16;          // oc per dz unit
static size_t XSAI_SRC_UNIT = 4;       // ic per sz block
static size_t XSAI_DST_XUNIT = 16;     // px per executor call
constexpr uint64_t XSAI_POOL_PHYS = 0x100000000ULL;
constexpr size_t XSAI_POOL_SIZE = 0x1000000ULL;  // 16MB tile workspace

// AME operand tiles MUST live in the XSAI pool (physically contiguous,
// allocated via xsai_malloc) — mla/mlb/msc address memory by physical base.
// Maximum K (in bytes, i.e. int8 k-elements) a single call may pack: 8192
// covers every convolution in the models under test (ic*kx*ky <= 8192). Larger
// shapes are rejected up front by xsai_ame_int8_supported() so the caller keeps
// using MNN's own kernel instead of getting a silently zeroed tile.
static constexpr int kAmeKMax = 8192;
static constexpr size_t kAmeTileBytes = 128 * (size_t)kAmeKMax;  // tile_m(128) x K
static int8_t  * g_atile = nullptr;
static int8_t  * g_btile = nullptr;
static int32_t * g_ctile = nullptr;
static int g_hw_m_max = 64;
static int g_hw_n_max = 64;

static bool xsai_ame_tiles_init(void) {
    if (g_atile != nullptr) return true;
    g_atile = (int8_t *)xsai_malloc(kAmeTileBytes);
    g_btile = (int8_t *)xsai_malloc(kAmeTileBytes);
    g_ctile = (int32_t *)xsai_malloc(128 * 128 * 4);
    if (g_atile == nullptr || g_btile == nullptr || g_ctile == nullptr) {
        fprintf(stderr, "[xsai-ame] fatal: xsai_malloc failed for tiles\n");
        return false;
    }
    // The tile spec carries the capacity decision; the probe itself may be
    // unavailable (qemu-user), which the spec handles by staying at 64.
    const xsai_ame_hw_config * hw = xsai_ame_hw_config_get();
    g_hw_m_max = (hw->valid && hw->int8_m_max >= 64) ? hw->int8_m_max : 64;
    g_hw_n_max = (hw->valid && hw->int32_n_max >= 64) ? hw->int32_n_max : 64;
    // Contiguity is an AMU requirement on real hardware; qemu-user addresses
    // operands through guest virtual addresses, so accept the host-test mode.
    if (!xsai_pool_phys_contiguous() && !xsai_alloc_host_test_mode()) {
        fprintf(stderr, "[xsai-ame] fatal: xsai pool not physically contiguous\n");
        return false;
    }
    return true;
}

}  // namespace

void xsai_ame_set_gemm_units(int unit, int src_unit, int dst_xunit) {
    XSAI_UNIT = (size_t)unit;
    XSAI_SRC_UNIT = (size_t)src_unit;
    XSAI_DST_XUNIT = (size_t)dst_xunit;
}

// Shape gate for the executor hook: the AME kernel may only be installed for
// calls it can actually handle (see kAmeKMax). Keeping the decision here means
// an unsupported layer silently falls back to MNN's own kernel rather than
// producing wrong (zeroed) output.
extern "C" bool xsai_ame_int8_supported(size_t src_depth_quad, size_t dst_depth_quad, size_t real_count) {
    const size_t k = src_depth_quad * XSAI_SRC_UNIT;
    // k 块宽度来自硬件探测（trlenb），不是常量 64：tlenb/trlenb 有多种配置，
    // 写死会和发射端、打包端对不上。
    const int spec_tile_k = xsai_ame_tile_spec_get()->tile_k;
    if (spec_tile_k <= 0) {
        return false;
    }
    const size_t kPad = ((k + (size_t)spec_tile_k - 1) / (size_t)spec_tile_k) * (size_t)spec_tile_k;
    if (kPad == 0 || kPad > (size_t)kAmeKMax) {
        return false;
    }
    if (real_count == 0 || real_count > XSAI_DST_XUNIT) {
        return false;
    }
    // Work threshold: keep shapes whose oc x K product is too small for the MMA
    // to pay off (1x1 pointwise layers with few channels, tiny oc) on MNN's own
    // RVV kernel instead of padding them into a mostly-empty AME tile.
    const size_t oc = dst_depth_quad * XSAI_UNIT;
    if (oc * k < xsai_ame_min_macs()) {
        // Report each distinct (K, oc) once so the board log shows which layers
        // stayed on the RVV path without flooding.
        static size_t seenK[24];
        static size_t seenOc[24];
        static int seenN = 0;
        bool known = false;
        for (int i = 0; i < seenN; ++i) {
            if (seenK[i] == k && seenOc[i] == oc) {
                known = true;
                break;
            }
        }
        if (!known && seenN < 24) {
            seenK[seenN] = k;
            seenOc[seenN] = oc;
            ++seenN;
            fprintf(stderr, "[xsai-ame] skip K=%zu oc=%zu (below work threshold %zu, using MNN kernel)\n",
                    k, oc, xsai_ame_min_macs());
        }
        return false;
    }
    return true;
}

bool xsai_ame_gemm_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const bool wanted = getenv("XSAI_MNN_AME") != nullptr;
        // No validated tile shape for this target (e.g. a 64-wide-only part)
        // means the AME cannot be trusted; keep the caller on its own kernel.
        cached = (wanted && xsai_ame_tile_spec_get()->enabled) ? 1 : 0;
        if (wanted && cached == 0) {
            fprintf(stderr, "[xsai-ame] tile spec unavailable on this target, AME path disabled\n");
        }
    }
    return cached == 1;
}

// Exported entry point (declared at file scope, defined in the anonymous
// namespace below is not possible -> keep this one outside).
// Crash forensics: record the last two kernel invocations; a SIGSEGV handler
// prints them so the failing call's parameters are visible on the console.
struct XsaiCallInfo {
    const void * src; const void * dst; const void * weight;
    unsigned long long sq, dq, rc, dstep;
};
static XsaiCallInfo g_cur_call = {};
static XsaiCallInfo g_prev_call = {};
static volatile int g_in_kernel = 0;

extern "C" void xsai_ame_segv_handler(int sig, siginfo_t * info, void *) {
    const XsaiCallInfo & c = g_cur_call;
    fprintf(stderr,
            "[xsai-ame] SEGFAULT sig=%d addr=%p\n"
            "  cur : src=%p dst=%p w=%p sq=%llu dq=%llu rc=%llu dstep=%llu\n"
            "  prev: src=%p dst=%p w=%p sq=%llu dq=%llu rc=%llu dstep=%llu\n",
            sig, info->si_addr,
            c.src, c.dst, c.weight, c.sq, c.dq, c.rc, c.dstep,
            g_prev_call.src, g_prev_call.dst, g_prev_call.weight,
            g_prev_call.sq, g_prev_call.dq, g_prev_call.rc, g_prev_call.dstep);
    _exit(66);
}

static void xsai_ame_segv_install() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = xsai_ame_segv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
}

// Per-call differential dump (XSAI_MNN_AME_DUMP=1): one line per AME GEMM call
// with the accumulator and requant operands for the first and last dz unit of
// the call. Running the same binary with XSAI_MNN_AME=0 (MNN's own int8 kernel)
// and diffing the two dumps localizes the first divergent call.
static bool xsai_ame_dump_enabled() {
    static int cached = -1;
    if (cached < 0) cached = getenv("XSAI_MNN_AME_DUMP") != nullptr ? 1 : 0;
    return cached == 1;
}
static unsigned long long g_dump_call = 0;
static unsigned long long g_extern_call = ~0ULL;

extern "C" void xsai_ame_set_call_id(unsigned long long id) {
    g_extern_call = id;
}

// rdcycle profiling (FPGA 50MHz: cycles/50 = us), RAII-guarded per call.
static uint64_t g_prof_cycles = 0;
static size_t g_prof_calls = 0;
static bool g_prof_atexit_done = false;
static bool g_tintr_disabled = false;
static bool g_warmed = false;
static int32_t g_selfcheck_ref = 0;
static bool g_selfcheck_pending = false;

void xsai_ame_prof_summary() {
    fprintf(stderr, "[xsai-ame] profiler: calls=%zu cycles=%llu (50MHz: %llu us)\n",
            g_prof_calls, (unsigned long long)g_prof_cycles,
            (unsigned long long)(g_prof_cycles / 50));
}

struct XsaiProf {
    uint64_t t0;
    XsaiProf() : t0(xsai_ame_rdcycle()) {
        if (!g_prof_atexit_done) {
            atexit(xsai_ame_prof_summary);
            g_prof_atexit_done = true;
        }
        ++g_prof_calls;
    }
    ~XsaiProf() { g_prof_cycles += xsai_ame_rdcycle() - t0; }
};

// AME int8 GEMM for one executor call: realCount (<= DST_XUNIT) pixels x
// dst_depth_quad 16-oc blocks, K = src_depth_quad * 4 (zero-padded to 64s).
// XSAI_MNN_AME_NOCALC=1: install the AME kernel but make it bail out before any
// MMA work (writes zeros), separating "the AME compute changes later inputs"
// from "merely installing the kernel does".
static bool xsai_ame_nocalc(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("XSAI_MNN_AME_NOCALC") != nullptr ? 1 : 0;
    return cached == 1;
}

void xsai_ame_int8_gemm_kernel(int8_t * dst, const int8_t * src, const int8_t * weight,
                               size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
                               const QuanPostTreatParameters * post, size_t realCount) {
    const size_t U  = XSAI_UNIT;        // oc per dz unit
    const size_t SU = XSAI_SRC_UNIT;    // ic per sz block
    const size_t EX = XSAI_DST_XUNIT;   // px per executor unit
    const size_t weight_step_Y = U * SU;
    const size_t weight_step_Z = src_depth_quad * weight_step_Y + 2 * U * sizeof(float);

    static bool segv_installed = false;
    if (!segv_installed) { segv_installed = true; xsai_ame_segv_install(); }
    {
        static int pshown = 0;
        if (getenv("XSAI_MNN_AME_DUMP") != nullptr && pshown < 3) {
            ++pshown;
            fprintf(stderr,
                    "[xsai-ame-ptr] call=%llu post=%p scale=%p biasFloat=%p sum=%p wks=%p fp32minmax=%p "
                    "is=%p ib=%p accumbuf=%p blockNum=%zd useInt8=%d min=%d max=%d\n",
                    g_extern_call, (void*)post, (void*)post->scale, (void*)post->biasFloat,
                    (void*)post->srcKernelSum, (void*)post->weightKernelSum, (void*)post->fp32minmax,
                    (void*)post->inputScale, (void*)post->inputBias, (void*)post->accumBuffer,
                    (ssize_t)post->blockNum, (int)post->useInt8, (int)post->minValue, (int)post->maxValue);
        }
    }
    {
        static int lay = 0;
        if (lay == 0 && getenv("XSAI_MNN_AME_DUMP") != nullptr) {
            lay = 1;
            fprintf(stderr, "[layout-AME] size=%zu scale=%zu biasFloat=%zu sum=%zu wks=%zu fp32minmax=%zu is=%zu ib=%zu accumbuf=%zu blockNum=%zu useInt8=%zu min=%zu max=%zu\n",
                    sizeof(QuanPostTreatParameters),
                    offsetof(QuanPostTreatParameters, scale),
                    offsetof(QuanPostTreatParameters, biasFloat),
                    offsetof(QuanPostTreatParameters, srcKernelSum),
                    offsetof(QuanPostTreatParameters, weightKernelSum),
                    offsetof(QuanPostTreatParameters, fp32minmax),
                    offsetof(QuanPostTreatParameters, inputScale),
                    offsetof(QuanPostTreatParameters, inputBias),
                    offsetof(QuanPostTreatParameters, accumBuffer),
                    offsetof(QuanPostTreatParameters, blockNum),
                    offsetof(QuanPostTreatParameters, useInt8),
                    offsetof(QuanPostTreatParameters, minValue),
                    offsetof(QuanPostTreatParameters, maxValue));
        }
    }
    {
        // Quantization-parameter probe: same source + different per-pixel
        // activation scales would explain a whole-tensor shift in the int8
        // codes without any change in the upstream tensor.
        if (getenv("XSAI_MNN_AME_DUMP") != nullptr && src_depth_quad == 2 && realCount == 8) {
            static int q = 0;
            if (q < 4) {
                ++q;
                fprintf(stderr, "[xsai-ame-quant] call=%llu is=%p ib=%p", g_extern_call,
                        (void*)post->inputScale, (void*)post->inputBias);
                if (post->inputScale) {
                    fprintf(stderr, " is[0..3]=");
                    for (int i = 0; i < 4; ++i) fprintf(stderr, "%.7g ", (double)post->inputScale[i]);
                }
                fprintf(stderr, " min=%d max=%d\n", (int)post->minValue, (int)post->maxValue);
            }
        }
    }
    {
        static int u0 = 0, u1 = 0;
        if (post->useInt8 == 0) {
            if (u0++ < 3) fprintf(stderr, "[xsai-ame-useInt8] call=%llu USEINT8=0 dq=%zu rc=%zu\n",
                                  g_extern_call, dst_depth_quad, realCount);
        } else if (u1++ < 1) {
            fprintf(stderr, "[xsai-ame-useInt8] first call with useInt8=1 (call=%llu)\n", g_extern_call);
        }
    }
    XsaiProf xsai_prof;
    g_cur_call = {src, dst, weight, (unsigned long long)src_depth_quad,
                  (unsigned long long)dst_depth_quad, (unsigned long long)realCount,
                  (unsigned long long)dst_step};
    g_prev_call = g_cur_call;

    // Per-shape self-check: verify output element (w=0, dz=0, j=0) against a
    // scalar dot product of the same inputs on the first call per shape.
    {
        static long long seen[64][3] = {}; static int seen_n = 0;
        bool known = false;
        for (int i = 0; i < seen_n; ++i)
            if (seen[i][0] == (long long)src_depth_quad && seen[i][1] == (long long)dst_depth_quad &&
                seen[i][2] == (long long)realCount) { known = true; break; }
        if (!known && seen_n < 64) {
            seen[seen_n][0] = src_depth_quad; seen[seen_n][1] = dst_depth_quad;
            seen[seen_n][2] = realCount; ++seen_n;
            g_selfcheck_pending = true;
            fprintf(stderr, "[xsai-ame] shape K=%zu dq=%zu rc=%zu (U=%zu SU=%zu EX=%zu)\n",
                    src_depth_quad * SU, dst_depth_quad, realCount, U, SU, EX);
        }
    }
    // Mask timer interrupts for the whole AME region (llama.cpp discipline).
    // The 0x6B nemu_trap instruction only exists on NEMU/QEMU-system — gate it
    // behind XSAI_MNN_TINTR so qemu-user and other targets skip it.
    if (!g_tintr_disabled) {
        static int tintr = -1;
        if (tintr < 0) tintr = getenv("XSAI_MNN_TINTR") != nullptr ? 1 : 0;
        if (tintr == 1) {
            register long a0 asm("a0") = 0x100;  // DISABLE_TIME_INTR
            __asm__ volatile(".insn r 0x6B, 0, 0, x0, x0, x0" : : "r"(a0) : "memory");
        }
        g_tintr_disabled = true;
    }

    // Tile-spec dispatch：tile_m/n/k 全部来自硬件探测（见 xsai_ame_tile_spec_get）。
    // k 块宽度不是常量 64 —— tlenb/trlenb 有多种配置，写死就会和发射端对不上。
    const xsai_ame_tile_spec * spec = xsai_ame_tile_spec_get();
    const int tile_m = spec->tile_m;
    const int tile_n = spec->tile_n;
    const int tile_k = spec->tile_k;
    const int K = (int)(src_depth_quad * SU);
    if (tile_k <= 0 || xsai_ame_nocalc() || !xsai_ame_tiles_init()) {
        memset(dst, 0, dst_depth_quad * realCount * U);
        return;
    }
    const int k_tiles = (K + tile_k - 1) / tile_k;
    const int K_pad = k_tiles * tile_k;
    if (K_pad > kAmeKMax) {
        memset(dst, 0, dst_depth_quad * realCount * U);
        return;
    }
    int8_t * atile = g_atile;
    int8_t * btile = g_btile;
    int32_t * ctile = g_ctile;

    // A tile: rows = realCount px (zero pad to 64), K cols = sz*SU + i.
    // im2col layout: src[sz * (bagPx*SU) + px*SU + i]. The per-sz stride uses the
    // **bag's** pixel count, not DST_XUNIT: for the last bag of a layer MNN packs
    // the im2col buffer compactly with `elast = realDstCount % DST_XUNIT` pixels
    // (see _ArmBasicMNNPackC4ForMatMul_A's lastBag branch), so reading such a bag
    // with the full DST_XUNIT stride pulls in the wrong elements and corrupts the
    // last pixels of the layer's output.
    const size_t bagPx = (realCount < (size_t)EX) ? (size_t)realCount : (size_t)EX;
    memset(atile, 0, (size_t)tile_m * K_pad);
    const size_t sz_per_tile = (size_t)tile_k / SU;
    for (int r = 0; r < (int)realCount; ++r) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int8_t * row = atile + (size_t)kt * (size_t)tile_m * tile_k + (size_t)r * tile_k;
            for (int c = 0; c < tile_k; ++c) {
                const size_t sz = (size_t)kt * sz_per_tile + (size_t)c / SU;
                const size_t i = (size_t)c % SU;
                if (sz < src_depth_quad) {
                    row[c] = src[sz * (bagPx * SU) + (size_t)r * SU + i];
                }
            }
        }
    }

    // B tiles batched: dz_batch dz-units per kloop so the 64 tile rows are
    // fully occupied (dz_batch * U = 64 oc).
    const size_t dz_batch = (size_t)tile_n / U;
    for (size_t dz0 = 0; dz0 < dst_depth_quad; dz0 += dz_batch) {
        const size_t dzn = (dst_depth_quad - dz0 < dz_batch) ? (dst_depth_quad - dz0) : dz_batch;
        memset(btile, 0, (size_t)tile_n * K_pad);
        const size_t sz_per_tile = (size_t)tile_k / SU;
        for (size_t d = 0; d < dzn; ++d) {
            const int8_t * w_dz = weight + (dz0 + d) * weight_step_Z;
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (size_t szl = 0; szl < sz_per_tile; ++szl) {
                    const size_t sz = (size_t)kt * sz_per_tile + szl;
                    if (sz >= src_depth_quad) break;
                    for (size_t j = 0; j < U; ++j) {
                        const int8_t * w_j = w_dz + weight_step_Y * sz + j * SU;
                        int8_t * b_row = btile + (size_t)kt * (size_t)tile_n * tile_k +
                                         (size_t)(d * U + j) * tile_k + szl * SU;
                        for (size_t i = 0; i < SU; ++i) b_row[i] = w_j[i];
                    }
                }
            }
        }

        // Kernel-side view of the K == 32 shape: dump the first k-block of the
        // A tile row 0 and the B tile row 0 so they can be diffed byte for byte
        // against the executor's packed bytes ([int8-ab]).
        {
            static int tv = -1;
            if (tv < 0) tv = (getenv("XSAI_MNN_AME_DUMP") != nullptr) ? 1 : 0;
            static int shown = 0;
            if (tv == 1 && src_depth_quad == 2 && realCount == 8 && dz0 == 0 && shown < 2) {
                ++shown;
                fprintf(stderr, "[xsai-ame-tile] call=%llu K=%d tile=%dx%dx%d", g_extern_call, K, tile_m, tile_k, tile_n);
                for (int r = 0; r < 4; ++r) {
                    fprintf(stderr, " row%d=", r);
                    for (int i = 0; i < 32; ++i)
                        fprintf(stderr, "%02x", (unsigned char)atile[(size_t)r * tile_k + i]);
                }
                fprintf(stderr, "\n");
            }
        }
        xsai_ame_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tile_k,
                                 btile, (ptrdiff_t)tile_n * tile_k, k_tiles, ctile,
                                 tile_m, tile_k, tile_n);
        // Cold-start warm-up: the process's first AME kloop has produced a
        // mismatching result under QEMU-user; re-run once and keep the second
        // result. Identical inputs, so this is a no-op when already warm.
        if (!g_warmed) {
            g_warmed = true;
            xsai_ame_gemm_tile_kloop(atile, (ptrdiff_t)tile_m * tile_k,
                                     btile, (ptrdiff_t)tile_n * tile_k, k_tiles, ctile,
                                     tile_m, tile_k, tile_n);
        }

        // Full-tile self-check (XSAI_MNN_SELFCHECK=full): the single-element
        // check below cannot see per-row/per-unit layout errors, so verify every
        // element of the first output tile against a scalar reference instead.
        {
            static int full = -1;
            if (full < 0) {
                const char * v = getenv("XSAI_MNN_SELFCHECK");
                full = (v != nullptr && strcmp(v, "full") == 0) ? 1 : 0;
            }
            if (full == 1 && dz0 == 0) {
                int bad = 0;
                for (size_t w = 0; w < realCount && bad < 6; ++w)
                    for (size_t d = 0; d < dzn && bad < 6; ++d)
                        for (size_t j = 0; j < U && bad < 6; ++j) {
                            int32_t ref = 0;
                            for (size_t sz = 0; sz < src_depth_quad; ++sz)
                                for (size_t i = 0; i < SU; ++i)
                                    ref += (int32_t)src[sz * (EX * SU) + w * SU + i] *
                                           (int32_t)weight[(dz0 + d) * weight_step_Z + sz * weight_step_Y + j * SU + i];
                            const int32_t got = ctile[w * tile_n + d * U + j];
                            if (ref != got) {
                                fprintf(stderr, "[xsai-ame] FULLCHK MISMATCH w=%zu d=%zu j=%zu ame=%d ref=%d\n",
                                        w, d, j, got, ref);
                                ++bad;
                            }
                        }
                if (bad == 0) {
                    static int okShown = 0;
                    if (okShown < 40) {
                        ++okShown;
                        fprintf(stderr, "[xsai-ame] fullcheck ok call=%llu px=%zu oc=%zu K=%d\n",
                                g_extern_call, realCount, dzn * U, K);
                    }
                }
            }
        }
        if (g_selfcheck_pending && dz0 == 0) {
            g_selfcheck_pending = false;
            int32_t ref = 0;
            for (size_t sz = 0; sz < src_depth_quad; ++sz)
                for (size_t i = 0; i < SU; ++i)
                    ref += (int32_t)src[sz * (EX * SU) + i] *
                           (int32_t)weight[sz * weight_step_Y + i];
            g_selfcheck_ref = ref;
            if (ctile[0] != g_selfcheck_ref) {
                fprintf(stderr, "[xsai-ame] SELFCHECK MISMATCH K=%zu: ame=%d ref=%d\n",
                        src_depth_quad * SU, ctile[0], g_selfcheck_ref);
                fprintf(stderr, "[xsai-ame] src[0..47]:");
                for (int i = 0; i < 48; ++i)
                    fprintf(stderr, " %d", (int)src[i]);
                fprintf(stderr, "\n[xsai-ame] w[0..47]:");
                for (int i = 0; i < 48; ++i)
                    fprintf(stderr, " %d", (int)weight[i]);
                fprintf(stderr, "\n[xsai-ame] atile r0 c0..63:");
                for (int i = 0; i < 64; ++i)
                    fprintf(stderr, " %d", (int)atile[i]);
                fprintf(stderr, "\n[xsai-ame] btile r0 c0..63:");
                for (int i = 0; i < 64; ++i)
                    fprintf(stderr, " %d", (int)btile[i]);
                int32_t partial = 0;
                fprintf(stderr, "\n[xsai-ame] per-kt partial:");
                for (int kt = 0; kt < k_tiles; ++kt) {
                    for (int c = 0; c < 64; ++c)
                        partial += (int32_t)atile[kt * (size_t)tile_m * tile_k + c] * (int32_t)btile[kt * (size_t)tile_n * tile_k + c];
                    fprintf(stderr, " %d", partial);
                }
                fprintf(stderr, "\n");
                for (size_t sz = 0; sz < src_depth_quad && sz < 12; ++sz) {
                    int32_t rp = 0, tp = 0;
                    for (size_t i = 0; i < SU; ++i) {
                        rp += (int32_t)src[sz * (EX * SU) + i] * (int32_t)weight[sz * weight_step_Y + i];
                        tp += (int32_t)atile[(sz / sz_per_tile) * (size_t)tile_m * tile_k + (sz % sz_per_tile) * SU + i] *
                              (int32_t)btile[(sz / sz_per_tile) * (size_t)tile_m * tile_k + (sz % sz_per_tile) * SU + i];
                    }
                    if (rp != tp)
                        fprintf(stderr,
                                "[xsai-ame] sz=%zu rp=%d tp=%d src_bytes:", sz, rp, tp);
                    if (rp != tp)
                        for (size_t i = 0; i < SU; ++i)
                            fprintf(stderr, " %d", (int)src[sz * (EX * SU) + i]);
                    if (rp != tp) fprintf(stderr, "\n");
                }
            } else {
                fprintf(stderr, "[xsai-ame] selcheck ok K=%zu val=%d\n",
                        src_depth_quad * SU, ctile[0]);
            }
        }

        // requant: per dz-unit scale tail [scale[U], weightBias[U]].
        for (size_t d = 0; d < dzn; ++d) {
            const int8_t * w_dz = weight + (dz0 + d) * weight_step_Z;
            const float * scale_dz = (const float *)(w_dz + src_depth_quad * weight_step_Y);
            const float * weightBias_dz = scale_dz + U;
            const float * bias_dz = (post->biasFloat != nullptr)
                                        ? (const float *)(post->biasFloat) + (dz0 + d) * U
                                        : nullptr;
            int8_t * dst_z = dst + (dz0 + d) * dst_step;
            for (size_t w = 0; w < realCount; ++w) {
                const float srcSum = (post->srcKernelSum != nullptr) ? post->srcKernelSum[w] : 0.f;
                int8_t * dst_px = dst_z + w * U;
                for (size_t j = 0; j < U; ++j) {
                    float value = (float)ctile[(size_t)w * tile_n + d * U + j] * scale_dz[j];
                    value += srcSum * weightBias_dz[j];
                    if (post->inputScale != nullptr) {
                        value = (float)ctile[(size_t)w * tile_n + d * U + j] * scale_dz[j] *
                                post->inputScale[w] + srcSum * weightBias_dz[j];
                    }
                    if (post->inputBias != nullptr) {
                        const float * weightKernelSum = post->weightKernelSum + (dz0 + d) * U;
                        value += post->inputBias[w] * weightKernelSum[j];
                    }
                    if (bias_dz != nullptr) value += bias_dz[j];
                    float lo = (float)post->minValue, hi = (float)post->maxValue;
                    if (value < lo) value = lo;
                    if (value > hi) value = hi;
                    dst_px[j] = (int8_t)roundf(value);
                    if (w == 0 && d == 0) {
                        static int shown = 0;
                        if (shown < 24 || (g_extern_call == 1568 && shown < 40)) {
                            ++shown;
                            fprintf(stderr,
                                    "[xsai-ame-rq] call=%llu K=%d dz0=%zu acc=%d sc=%.6g wb=%.6g "
                                    "sum=%.6g is=%s ib=%s wks=%s bias=%.6g min=%.6g max=%.6g -> value=%.6g "
                                    "dst=%d useInt8=%d\n",
                                    g_extern_call, K, dz0, ctile[(size_t)w * tile_n + d * U + j], (double)scale_dz[j],
                                    (double)weightBias_dz[j], (double)srcSum,
                                    post->inputScale ? "yes" : "no", post->inputBias ? "yes" : "no",
                                    post->weightKernelSum ? "yes" : "no",
                                    bias_dz ? (double)bias_dz[j] : 0.0, (double)post->minValue,
                                    (double)post->maxValue, (double)value,
                                    (int)dst_px[j], (int)post->useInt8);
                        }
                    }
                    if (w == 0 && d == 0 && j < 1) {
                        fprintf(stderr, "  [rq-detail] call=%llu j=%zu clamped=%.6g stored=%d\n",
                                g_extern_call, (size_t)j, (double)value, (int)dst_px[j]);
                    }
                }
            }
        }
        if (xsai_ame_dump_enabled() && g_dump_call < 4000) {
            const int8_t * w0 = weight + dz0 * weight_step_Z;
            const float * sc0 = (const float *)(w0 + src_depth_quad * weight_step_Y);
            const float * wb0 = sc0 + U;
            const int8_t * dlast = weight + (dz0 + dzn - 1) * weight_step_Z;
            const float * sclast = (const float *)(dlast + src_depth_quad * weight_step_Y);
            const int8_t * dst0 = dst + dz0 * dst_step;
            const int8_t * dstl = dst + (dz0 + dzn - 1) * dst_step;
            fprintf(stderr,
                    "[xsai-ame-dump] call=%llu K=%d dq=%zu rc=%zu dz0=%zu dzn=%zu "
                    "ctile0=%d dst0=[%d %d %d %d] sc0=%.6g wb0=%.6g sum0=%.6g "
                    "dstL0=%d scL0=%.6g\n",
                    (g_extern_call != ~0ULL) ? g_extern_call : g_dump_call, K, dst_depth_quad,
                    realCount, dz0, dzn, ctile[0],
                    (int)dst0[0], (int)dst0[1], (int)dst0[2], (int)dst0[3],
                    (double)sc0[0], (double)wb0[0],
                    (double)((post->srcKernelSum != nullptr) ? post->srcKernelSum[0] : 0.f),
                    (int)dstl[0], (double)sclast[0]);
        }
    }
    ++g_dump_call;
}
