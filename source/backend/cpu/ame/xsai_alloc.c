/*
 * xsai_alloc.c - Custom memory allocator for llama.cpp on XSAI RISC-V.
 *
 * Uses a simple free-list with first-fit allocation and block coalescing.
 * All allocations are padded to 64-byte alignment (suitable for AME / tensor
 * operations).
 *
 * Thread-safe: a pthread_mutex_t serialises all alloc/free operations.
 *
 * Build-time defines:
 *   RESERVED_MEMORY_SIZE      - total pool size in bytes (default 1 GiB)
 *   RESERVED_PHYS_BASE_ADDR   - if defined, mmap /dev/mem at this physical
 *                               address instead of anonymous mmap
 *
 * Reserved-region location is resolved at RUNTIME, in this order (first hit wins):
 *   1. environment   XSAI_AME_RESERVED_BASE / XSAI_AME_RESERVED_SIZE (hex or dec)
 *   2. device tree   /proc/device-tree/reserved-memory/direct_map@<addr>/reg
 *   3. compile-time  RESERVED_PHYS_BASE_ADDR (legacy; logged loudly)
 *   4. otherwise     hard failure with an actionable message
 *
 * Why not keep it compile-time only: the same binary must work in both memory
 * layouts this project builds.  appliance images are 8GB RAM / 6GB pool
 * (pool @0x100000000) while ruyi/mega images are 16GB / 4GB pool
 * (pool @0x380000000); a binary pinned to the wrong base maps ordinary kernel
 * RAM as the AMU pool and the first AME store faults (OpenSBI trap, machine
 * dead -- reproduced 2026-09-17, job 279506).
 *
 * Runtime override for host-side correctness tests:
 *   XSAI_ALLOC_MODE=malloc    - use aligned malloc/free; no pool
 *   XSAI_ALLOC_MODE=anon      - use anonymous mmap pool even when
 *                               RESERVED_PHYS_BASE_ADDR is defined
 *   XSAI_ALLOC_POOL_SIZE_MB   - optional anon-host-test pool size override
 */

#include "xsai_alloc.h"

#include <sys/mman.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifndef RESERVED_MEMORY_SIZE
#define RESERVED_MEMORY_SIZE (1024L * 1024L * 1024L) /* 1 GiB */
#warning "RESERVED_MEMORY_SIZE not defined, using 1 GiB"
#endif

/* ------------------------------------------------------------------ */
/* Internal data structures                                             */
/* ------------------------------------------------------------------ */

#define ALIGN 64

/* Each free block has an intrusive header at its start. */
typedef struct free_block {
    size_t size;               /* usable bytes following this header */
    struct free_block *next;
    char _pad[ALIGN - sizeof(size_t) - sizeof(void *)];
} free_block_t;

/* Each allocated block has a small header just before the returned pointer. */
typedef struct alloc_header {
    size_t size;               /* usable bytes following this header */
    char _pad[ALIGN - sizeof(size_t)];
} alloc_header_t;

#define ALLOC_HDR_SIZE  ((size_t)sizeof(alloc_header_t))  /* == ALIGN */
#define FREE_HDR_SIZE   ((size_t)sizeof(free_block_t))    /* == ALIGN */

static void          *pool_base  = NULL;
static size_t         pool_size  = 0;
static free_block_t  *free_list  = NULL;   /* sorted by address */
/* 1 when the pool is backed by physically-contiguous memory (huge pages or
 * RESERVED_PHYS_BASE_ADDR).  The AMU may then use "single-TLB-base + PA
 * offset" addressing: PA = TLB(pool_base) + (ptr - pool_base). */
static int            pool_phys_contiguous = 0;
static int            alloc_malloc_mode = 0;
static int            alloc_host_test_mode = 0;

static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;

/* Diagnostics: number of active allocations, total bytes allocated. */
static size_t xsai_alloc_count       = 0;  /* currently live allocations */
static size_t xsai_alloc_bytes       = 0;  /* currently live bytes (incl. headers) */
static size_t xsai_alloc_total_count = 0;  /* total alloc calls ever */
static size_t xsai_alloc_peak_bytes  = 0;  /* high-water mark of live bytes */

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static int xsai_env_is_one_of(const char * value, const char * a, const char * b, const char * c) {
    return value && (strcmp(value, a) == 0 || strcmp(value, b) == 0 || strcmp(value, c) == 0);
}

static void *xsai_alloc_aligned_block(size_t size, size_t *need_out) {
    if (size > (size_t)-1 - ALLOC_HDR_SIZE) {
        fprintf(stderr, "[xsai_alloc] fatal: allocation size overflow (requested %zu bytes); aborting\n", size);
        fflush(stderr);
        abort();
    }

    const size_t need = align_up(size + ALLOC_HDR_SIZE, ALIGN);
    void *raw = NULL;
    if (posix_memalign(&raw, ALIGN, need) != 0 || raw == NULL) {
        fprintf(stderr, "[xsai_alloc] fatal: malloc failed (requested %zu bytes, need=%zu bytes); aborting\n",
                size, need);
        fflush(stderr);
        abort();
    }

    alloc_header_t *hdr = (alloc_header_t *) raw;
    hdr->size = need - ALLOC_HDR_SIZE;
    if (need_out) {
        *need_out = need;
    }
    return (char *) hdr + ALLOC_HDR_SIZE;
}

static size_t xsai_alloc_pool_size_from_env(size_t default_size) {
    const char *value = getenv("XSAI_ALLOC_POOL_SIZE_MB");
    if (value == NULL || value[0] == '\0') {
        return default_size;
    }

    char *end = NULL;
    const unsigned long long mib = strtoull(value, &end, 0);
    if (end == value || mib == 0 || mib > ((unsigned long long) SIZE_MAX >> 20)) {
        fprintf(stderr,
                "[xsai_alloc] warning: ignoring invalid XSAI_ALLOC_POOL_SIZE_MB=%s\n",
                value);
        return default_size;
    }

    return (size_t) mib << 20;
}

/* ------------------------------------------------------------------ */
/* Reserved region resolution (env -> device tree -> legacy define)     */
/* ------------------------------------------------------------------ */

/* Parse "0x380000000" / "4294967296" / "4g" / "512m"; returns 0 on success. */
static int xsai_parse_size(const char *text, uint64_t *out) {
    if (text == NULL || text[0] == '\0') {
        return -1;
    }
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 0);
    if (end == text || value == 0) {
        return -1;
    }
    if (*end == 'g' || *end == 'G') {
        value <<= 30;
    } else if (*end == 'm' || *end == 'M') {
        value <<= 20;
    } else if (*end != '\0') {
        return -1;
    }
    *out = (uint64_t) value;
    return 0;
}

/* /proc/device-tree/reserved-memory/direct_map@380000000/{reg}
 *   reg = <base_hi base_lo size_hi size_lo>  (#address-cells=#size-cells=2)
 *      or <base size>                        (both cells = 1)
 * Returns 0 when a direct_map node was found and parsed. */
static int xsai_resolve_from_dtb(uint64_t *base, uint64_t *size) {
    const char *dir_path = "/proc/device-tree/reserved-memory";
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        return -1;
    }
    int found = -1;
    struct dirent *ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        const char *name = ent->d_name;
        if (strncmp(name, "direct_map@", 11) != 0) {
            continue;
        }
        char *end = NULL;
        unsigned long long node_base = strtoull(name + 11, &end, 16);
        if (end == name + 11 || node_base == 0) {
            continue;
        }
        *base = (uint64_t) node_base;
        found = 0;
        char reg_path[512];
        snprintf(reg_path, sizeof(reg_path), "%s/%s/reg", dir_path, name);
        FILE *f = fopen(reg_path, "rb");
        if (f != NULL) {
            uint8_t raw[16];
            size_t got = fread(raw, 1, sizeof(raw), f);
            fclose(f);
            uint64_t parsed = 0;
            if (got == sizeof(raw)) {
                /* reg = <base_hi base_lo size_hi size_lo> (#cells = 2) */
                uint64_t hi = 0, lo = 0;
                for (int i = 8; i < 12; ++i) {
                    hi = (hi << 8) | raw[i];
                }
                for (int i = 12; i < 16; ++i) {
                    lo = (lo << 8) | raw[i];
                }
                parsed = (hi << 32) | lo;
            } else if (got == 8) {
                /* reg = <base size> (#cells = 1) */
                for (int i = 4; i < 8; ++i) {
                    parsed = (parsed << 8) | raw[i];
                }
            }
            if (parsed != 0) {
                *size = parsed;
            }
        }
        break;
    }
    closedir(dir);
    return found;
}

/* Fills *base / *size and sets *source to "env" / "dtb" / "compile-time".
 * Returns 0 on success, -1 when nothing could be resolved. */
static int xsai_resolve_reserved_region(uint64_t *base, uint64_t *size, const char **source) {
    uint64_t value = 0;
    if (xsai_parse_size(getenv("XSAI_AME_RESERVED_BASE"), &value) == 0) {
        *base = value;
        if (xsai_parse_size(getenv("XSAI_AME_RESERVED_SIZE"), &value) == 0) {
            *size = value;
        }
        *source = "env";
        return 0;
    }
    uint64_t dtb_size = *size;
    if (xsai_resolve_from_dtb(base, &dtb_size) == 0) {
        *size = dtb_size;
        *source = "dtb";
        return 0;
    }
#ifdef RESERVED_PHYS_BASE_ADDR
    *base = (uint64_t)(RESERVED_PHYS_BASE_ADDR);
    *source = "compile-time(legacy)";
    fprintf(stderr, "[xsai_alloc] WARNING: falling back to the compile-time "
                    "RESERVED_PHYS_BASE_ADDR; set XSAI_AME_RESERVED_BASE or provide a "
                    "device tree (the value may not match this machine)\n");
    return 0;
#else
    (void) base; (void) size; (void) source;
    return -1;
#endif
}

/* Must be called with pool_lock held. */
static int pool_init(void) {
    pool_size = (size_t)RESERVED_MEMORY_SIZE;
    const char *mode = getenv("XSAI_ALLOC_MODE");

    if (xsai_env_is_one_of(mode, "malloc", "libc", "host")) {
        alloc_malloc_mode = 1;
        alloc_host_test_mode = 1;
        pool_phys_contiguous = 0;
        fprintf(stderr, "[xsai_alloc] ACTIVE mode=malloc-host-test  size=%zu MiB\n", pool_size >> 20);
        return 0;
    }

    if (xsai_env_is_one_of(mode, "anon", "anonymous", "qemu")) {
        alloc_host_test_mode = 1;
        pool_size = xsai_alloc_pool_size_from_env(512ULL << 20);
        fprintf(stderr,
                "[xsai_alloc] ACTIVE mode=anon-host-test  size=%zu MiB"
                "  (non-physical, qemu-user only)\n",
                pool_size >> 20);
        pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pool_base != MAP_FAILED) {
            pool_phys_contiguous = 0;
            goto init_pool_metadata;
        }
        fprintf(stderr, "[xsai_alloc] anon-host-test mmap failed\n");
        pool_base = NULL;
        return -1;
    }

    /* Map the reserved physically-contiguous region via /dev/mem.  The whole
     * pool is one contiguous PA range, so pool_phys_contiguous=1 and the AMU
     * may use: PA = resolved_base + (ptr - pool_base). */
    uint64_t reserved_base = 0, reserved_size = pool_size;
    const char *reserved_source = NULL;
    if (xsai_resolve_reserved_region(&reserved_base, &reserved_size, &reserved_source) == 0) {
        pool_size = xsai_alloc_pool_size_from_env((size_t) reserved_size);
        fprintf(stderr, "[xsai_alloc] ACTIVE mode=phys  phys_base=0x%llx  size=%zu MiB  (source=%s)\n",
                (unsigned long long) reserved_base, pool_size >> 20, reserved_source);
        int fd = open("/dev/mem", O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "[xsai_alloc] open /dev/mem failed (need root?)\n");
            return -1;
        }
        pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, (off_t) reserved_base);
        close(fd);
        if (pool_base != MAP_FAILED)
            pool_phys_contiguous = 1;
    } else {
    /* Try to obtain a physically-contiguous pool via huge pages.
     *
     * Huge-page PTEs cover the whole page in a single TLB entry, so every
     * allocation within a single huge page is also physically contiguous by
     * definition.  This is the software contract required for the AMU's
     * "single-TLB-base + PA-offset" addressing mode:
     *   PA_row = TLB(base_VA) + row * stride
     *
     * Prerequisite (RISC-V Linux):
     *   echo N > /proc/sys/vm/nr_hugepages   (N >= pool_size / 2MiB)
     * or reserve at boot with hugepages=N.
     */
    pool_base = MAP_FAILED;

#if defined(__linux__) && defined(MAP_HUGETLB)
    /* Try default huge-page size (typically 2 MiB on RISC-V Sv39). */
    pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                     -1, 0);
    if (pool_base != MAP_FAILED) {
        pool_phys_contiguous = 1;
        fprintf(stderr, "[xsai_alloc] ACTIVE mode=anon-hugepage  size=%zu MiB"
                        "  pool_base=%p  (phys-contiguous)\n",
                pool_size >> 20, pool_base);
    }
#endif /* __linux__ && MAP_HUGETLB */

    if (pool_base == MAP_FAILED) {
        /* Hugepages unavailable: fall back to regular anonymous pages.
         * Physical contiguity is NOT guaranteed across 4 KiB boundaries.
         * The AMU must NOT use single-TLB-base mode in this configuration. */
        fprintf(stderr, "[xsai_alloc] WARNING: hugepages unavailable -- "
                        "physical contiguity NOT guaranteed.\n"
                        "  AMU single-TLB-base mode will corrupt data when a "
                        "tile crosses a physical page boundary.\n"
                        "  Set vm.nr_hugepages >= %zu or define "
                        "RESERVED_PHYS_BASE_ADDR.\n",
                pool_size >> 21 /* pool_size / 2MiB */);
        pool_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (pool_base != MAP_FAILED) {
            mlock(pool_base, pool_size); /* pin pages; NOT a contiguity guarantee */
        }
        pool_phys_contiguous = 0;
    }
    fprintf(stderr, "[xsai_alloc] pool_base = %p\n", pool_base);
    }

    if (pool_base == MAP_FAILED) {
        fprintf(stderr, "[xsai_alloc] mmap failed\n");
        pool_base = NULL;
        return -1;
    }

init_pool_metadata:
    /* Single free block covering the whole pool. */
    free_list        = (free_block_t *)pool_base;
    free_list->size  = pool_size - FREE_HDR_SIZE;
    free_list->next  = NULL;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void *xsai_malloc(size_t size) {
    if (size == 0) return NULL;

    pthread_mutex_lock(&pool_lock);

    if (pool_base == NULL && !alloc_malloc_mode) {
        if (pool_init() != 0) {
            pthread_mutex_unlock(&pool_lock);
            fprintf(stderr, "[xsai_alloc] fatal: failed to initialize allocator pool; aborting\n");
            fflush(stderr);
            abort();
        }
    }

    if (alloc_malloc_mode) {
        size_t need = 0;
        void *ptr = xsai_alloc_aligned_block(size, &need);
        xsai_alloc_count++;
        xsai_alloc_total_count++;
        xsai_alloc_bytes += need;
        if (xsai_alloc_bytes > xsai_alloc_peak_bytes) {
            xsai_alloc_peak_bytes = xsai_alloc_bytes;
        }
        pthread_mutex_unlock(&pool_lock);
        return ptr;
    }

    if (size > (size_t)-1 - ALLOC_HDR_SIZE) {
        pthread_mutex_unlock(&pool_lock);
        fprintf(stderr, "[xsai_alloc] fatal: allocation size overflow (requested %zu bytes); aborting\n", size);
        fflush(stderr);
        abort();
    }

    /* Reserve space for the alloc header and align. */
    size_t need = align_up(size + ALLOC_HDR_SIZE, ALIGN);

    free_block_t *prev = NULL;
    free_block_t *cur  = free_list;
    while (cur) {
        size_t avail = cur->size + FREE_HDR_SIZE;
        if (avail >= need) {
            size_t leftover = avail - need;
            if (leftover >= FREE_HDR_SIZE + ALIGN) {
                /* Split: create a new free block for the leftover. */
                free_block_t *split = (free_block_t *)((char *)cur + need);
                split->size = leftover - FREE_HDR_SIZE;
                split->next = cur->next;
                if (prev) prev->next = split;
                else       free_list = split;
            } else {
                /* Use the whole block (leftover too small to split). */
                if (prev) prev->next = cur->next;
                else       free_list = cur->next;
            }

            alloc_header_t *hdr = (alloc_header_t *)cur;
            hdr->size = need - ALLOC_HDR_SIZE;

            xsai_alloc_count++;
            xsai_alloc_total_count++;
            xsai_alloc_bytes += need;
            if (xsai_alloc_bytes > xsai_alloc_peak_bytes)
                xsai_alloc_peak_bytes = xsai_alloc_bytes;

            pthread_mutex_unlock(&pool_lock);
            return (char *)hdr + ALLOC_HDR_SIZE;
        }
        prev = cur;
        cur  = cur->next;
    }

    fprintf(stderr,
            "[xsai_alloc] fatal: out of memory (requested %zu bytes, need=%zu bytes, active=%zu, used=%zu MiB, capacity=%zu MiB); aborting\n",
            size, need, xsai_alloc_count, xsai_alloc_bytes >> 20, pool_size >> 20);
    pthread_mutex_unlock(&pool_lock);
    fflush(stderr);
    abort();
}

void xsai_free(void *ptr) {
    if (!ptr) return;

    pthread_mutex_lock(&pool_lock);

    if (alloc_malloc_mode) {
        alloc_header_t *hdr = (alloc_header_t *)((char *)ptr - ALLOC_HDR_SIZE);
        const size_t need = hdr->size + ALLOC_HDR_SIZE;
        xsai_alloc_count--;
        xsai_alloc_bytes -= need;
        pthread_mutex_unlock(&pool_lock);
        free(hdr);
        return;
    }

    if (!pool_base ||
        (uintptr_t)ptr < (uintptr_t)pool_base ||
        (uintptr_t)ptr >= (uintptr_t)pool_base + pool_size) {
        fprintf(stderr, "[xsai_alloc] xsai_free(%p): pointer NOT in pool "
                "[%p, %p) -- called free() instead\n",
                ptr, pool_base, (char *)pool_base + pool_size);
        pthread_mutex_unlock(&pool_lock);
        free(ptr); /* fall back to libc for non-pool pointers */
        return;
    }

    alloc_header_t *hdr = (alloc_header_t *)((char *)ptr - ALLOC_HDR_SIZE);
    free_block_t   *blk = (free_block_t *)hdr;
    blk->size = hdr->size + ALLOC_HDR_SIZE - FREE_HDR_SIZE;

    xsai_alloc_count--;
    xsai_alloc_bytes -= (blk->size + FREE_HDR_SIZE);

    /* Insert into free list sorted by address. */
    free_block_t *prev = NULL;
    free_block_t *cur  = free_list;
    while (cur && (char *)cur < (char *)blk) {
        prev = cur;
        cur  = cur->next;
    }
    blk->next = cur;
    if (prev) prev->next = blk;
    else       free_list = blk;

    /* Coalesce with next block. */
    if (blk->next &&
        (char *)blk + FREE_HDR_SIZE + blk->size == (char *)blk->next) {
        blk->size += FREE_HDR_SIZE + blk->next->size;
        blk->next  = blk->next->next;
    }

    /* Coalesce with previous block. */
    if (prev &&
        (char *)prev + FREE_HDR_SIZE + prev->size == (char *)blk) {
        prev->size += FREE_HDR_SIZE + blk->size;
        prev->next  = blk->next;
    }

    pthread_mutex_unlock(&pool_lock);
}

int xsai_in_pool(const void *ptr) {
    return pool_base != NULL
        && (uintptr_t)ptr >= (uintptr_t)pool_base
        && (uintptr_t)ptr <  (uintptr_t)pool_base + pool_size;
}

/* Returns 1 if the pool is backed by physically-contiguous memory.
 *
 * When this returns 1 the AMU may use "single-TLB-base + PA-offset" mode:
 *   PA(ptr) = TLB(pool_base) + (ptr - pool_base)
 * i.e. one TLB lookup for pool_base, then all intra-pool addresses are
 * resolved by pure PA arithmetic without further page-table walks.
 *
 * This is valid because every VA offset within the pool maps to the
 * same PA offset from the pool's physical base — guaranteed by the
 * huge-page or /dev/mem mapping that backs the pool.
 *
 * Returns 0 if the pool uses ordinary 4 KiB pages: physical contiguity
 * is not guaranteed, and the AMU must perform a TLB lookup whenever it
 * crosses a virtual page boundary. */
int xsai_pool_phys_contiguous(void) {
    return pool_phys_contiguous;
}

int xsai_alloc_host_test_mode(void) {
    return alloc_host_test_mode;
}

void xsai_alloc_print_stats(void) {
    pthread_mutex_lock(&pool_lock);
    if (pool_base == NULL && !alloc_malloc_mode) {
        pthread_mutex_unlock(&pool_lock);
        return; /* allocator was never activated */
    }

    size_t used_bytes = xsai_alloc_bytes;
    size_t live_count = xsai_alloc_count;
    size_t total      = xsai_alloc_total_count;
    size_t peak       = xsai_alloc_peak_bytes;
    size_t cap        = alloc_malloc_mode ? peak : pool_size;

    /* Walk free list to compute fragmentation / free space. */
    size_t free_bytes  = 0;
    size_t free_blocks = 0;
    if (!alloc_malloc_mode) {
        free_block_t *cur  = free_list;
        while (cur) {
            free_bytes  += cur->size + FREE_HDR_SIZE;
            free_blocks++;
            cur = cur->next;
        }
    }

    pthread_mutex_unlock(&pool_lock);

    fprintf(stderr, "\n");
    fprintf(stderr, "======= xsai_alloc statistics =================================\n");
    fprintf(stderr, "  pool capacity     : %zu MiB  (%zu bytes)\n",
            cap >> 20, cap);
    fprintf(stderr, "  total alloc calls : %zu\n", total);
    fprintf(stderr, "  live allocations  : %zu\n", live_count);
    fprintf(stderr, "  live bytes        : %.3f MiB  (%zu bytes)\n",
            (double)used_bytes / (1024.0 * 1024.0), used_bytes);
    fprintf(stderr, "  peak usage        : %.3f MiB  (%zu bytes)\n",
            (double)peak / (1024.0 * 1024.0), peak);
    fprintf(stderr, "  peak / capacity   : %.1f%%\n",
            cap > 0 ? (double)peak / (double)cap * 100.0 : 0.0);
    fprintf(stderr, "  free blocks       : %zu  (%.3f MiB free)\n",
            free_blocks, (double)free_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "\n");
}
