/* Checksums of the Kimi K3.1 expert files (step 153).
 *
 *   k3_expert_sums make  <table> <dir> [<dir_b>]   CRC32C of the two halves of every expert's raw-v1 image, read through the
 *                                                   engine's own reader (a recoded file is expanded first); <dir_b> adds the
 *                                                   experts that are not in <dir>
 *   k3_expert_sums check <table> <dir>             every L*-E*.bin of <dir>, read the same way, against the table
 *
 * The table lets the engine check each read from a home that cannot be trusted (DS4_K3_EXPERT_DIR_D_SUMS), and it is the
 * independent check of a store converted by k31_scale_recode.py: make the table before, check after.
 * Build from the repository root:  cc -O2 -pthread -I. -o k3_expert_sums tools/kimi-k3/k3_expert_sums.c ds4_kimi_k3.c
 * K3_SUMS_THREADS sets the reader threads (16). */
#define _GNU_SOURCE
#include "ds4_kimi_k3.h"
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_BYTES 17547264ull
#define HALF_BYTES (FILE_BYTES / 2u)
#define N_KEYS (128u * 1024u)

typedef struct { uint32_t layer, expert; int tier; } item;
static item *g_items; static uint32_t g_n_items;
static uint32_t *g_sums;        /* per key: two CRCs */
static uint8_t *g_has;          /* per key: in the table */
static atomic_uint g_next, g_bad, g_missing, g_err;
static int g_check;

static uint32_t list_dir(const char *dir, int tier, uint8_t *seen) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "cannot list %s: %s\n", dir, strerror(errno)); exit(2); }
    struct dirent *de; uint32_t n = 0;
    while ((de = readdir(d)) != NULL) {
        unsigned l = 0, e = 0; char tail[8] = "";
        if (sscanf(de->d_name, "L%u-E%u.bi%1[n]%1s", &l, &e, tail, tail + 2) != 3 || l >= 128u || e >= 1024u) continue;
        const uint32_t key = (l << 10) | e;
        if (seen[key]) continue;
        seen[key] = 1;
        g_items[g_n_items++] = (item){l, e, tier}; n++;
    }
    closedir(d);
    return n;
}
static void *worker(void *arg) {
    (void)arg;
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, FILE_BYTES) != 0) { atomic_fetch_add(&g_err, 1); return NULL; }
    for (;;) {
        const uint32_t i = atomic_fetch_add(&g_next, 1);
        if (i >= g_n_items) break;
        const item it = g_items[i]; int used = -1;
        if (!ds4_k3_expert_source_read_range_tier(it.layer, it.expert, buf, 0, FILE_BYTES, 1, it.tier, &used) || used != it.tier) {
            fprintf(stderr, "L%u-E%u: read failed (%s)\n", it.layer, it.expert, used != it.tier && used >= 0 ? "served by another directory" : strerror(errno));
            atomic_fetch_add(&g_err, 1); continue;
        }
        const uint32_t a = ds4_k3_crc32c(buf, HALF_BYTES), b = ds4_k3_crc32c((const uint8_t *)buf + HALF_BYTES, FILE_BYTES - HALF_BYTES);
        const uint32_t key = (it.layer << 10) | it.expert;
        if (!g_check) { g_sums[2u * key] = a; g_sums[2u * key + 1u] = b; g_has[key] = 1; continue; }
        if (!g_has[key]) { atomic_fetch_add(&g_missing, 1); continue; }
        if (g_sums[2u * key] != a || g_sums[2u * key + 1u] != b) {
            if (atomic_fetch_add(&g_bad, 1) < 20u) fprintf(stderr, "L%u-E%u: does NOT match the table\n", it.layer, it.expert);
        }
    }
    free(buf);
    return NULL;
}
int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[1], "make") != 0 && strcmp(argv[1], "check") != 0)) {
        fprintf(stderr, "usage: %s make <table> <dir> [<dir_b>] | check <table> <dir>\n", argv[0]);
        return 2;
    }
    g_check = strcmp(argv[1], "check") == 0;
    const char *table = argv[2], *dir = argv[3], *dir_b = !g_check && argc > 4 ? argv[4] : NULL;
    g_items = calloc(N_KEYS, sizeof(item)); g_sums = calloc((size_t)N_KEYS * 2u, sizeof(uint32_t)); g_has = calloc(N_KEYS, 1);
    uint8_t *seen = calloc(N_KEYS, 1);
    if (!g_items || !g_sums || !g_has || !seen) return 2;
    if (g_check) {
        uint64_t fb = 0, half = 0;
        const int n = ds4_k3_expert_sums_read(table, g_sums, g_has, N_KEYS, &fb, &half);
        if (n < 0 || fb != FILE_BYTES || half != HALF_BYTES) { fprintf(stderr, "%s: not a checksum table for this geometry\n", table); return 2; }
        fprintf(stderr, "%s: %d experts\n", table, n);
    }
    ds4_k3_expert_source_configure(dir, dir_b, FILE_BYTES);
    const uint32_t na = list_dir(dir, 0, seen), nb = dir_b ? list_dir(dir_b, 1, seen) : 0u;
    fprintf(stderr, "%u expert files in %s%s", na, dir, dir_b ? "" : "\n");
    if (dir_b) fprintf(stderr, ", %u more in %s\n", nb, dir_b);
    int n_threads = getenv("K3_SUMS_THREADS") ? atoi(getenv("K3_SUMS_THREADS")) : 16;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 64) n_threads = 64;
    pthread_t th[64];
    for (int t = 0; t < n_threads; t++) pthread_create(&th[t], NULL, worker, NULL);
    for (int t = 0; t < n_threads; t++) pthread_join(th[t], NULL);
    uint64_t v2 = 0, ns = 0; ds4_k3_expert_source_v2_stats(&v2, &ns);
    if (atomic_load(&g_err)) { fprintf(stderr, "%u files could not be read: nothing written\n", atomic_load(&g_err)); return 1; }
    if (g_check) {
        printf("%u files of %s read through the engine's reader (%llu of them recoded): %u match the table, %u do NOT, %u are not in the table\n",
               g_n_items, dir, (unsigned long long)v2, g_n_items - atomic_load(&g_bad) - atomic_load(&g_missing), atomic_load(&g_bad), atomic_load(&g_missing));
        return atomic_load(&g_bad) ? 1 : 0;
    }
    if (!ds4_k3_expert_sums_write(table, g_sums, g_has, N_KEYS, FILE_BYTES, HALF_BYTES)) { fprintf(stderr, "cannot write %s: %s\n", table, strerror(errno)); return 1; }
    printf("%s: checksums of %u experts (%llu of the files recoded)\n", table, g_n_items, (unsigned long long)v2);
    return 0;
}
