/* The K3.1 vision tower in the engine (17 Nov 2026): ds4_k31_vision_gpu.cuh behind a C entry point. Built without
 * --use_fast_math (see the Makefile): its float32 arithmetic must stay that of tools/kimi-k3/k31_vision_ref.py.
 *
 * VRAM: the tower's 0.9 GB of BF16 weights stay on the GPU after the first image. Its scratch (86 KB per 14x14 patch, plus at
 * most 512 MB of attention scores: about 1.4 GB for a 1920x1080 photo, 6 GB at the 65,536-patch limit) is freed after every
 * image, so decode gets it back. When the free VRAM is short, the expert cache lends its grown region, as it does for a long
 * prompt pass, and grows again when generation starts. */
#include "ds4_k31_vision_gpu.cuh"

extern "C" uint64_t ds4_gpu_k3_cache_lend(void);

static void k31v_free_scratch(k31v_model &M) {
    if (M.stream) cudaStreamSynchronize(M.stream);
    cudaFree(M.x); cudaFree(M.y); cudaFree(M.qkv); cudaFree(M.o); cudaFree(M.h); cudaFree(M.cs); cudaFree(M.split); cudaFree(M.s);
    M.x = M.y = M.qkv = M.o = M.h = M.cs = M.s = nullptr;
    M.split = nullptr;
    M.cap_n = M.cap_s = 0;
}

/* bytes k31v_reserve and the attention-score buffer of k31v_encode allocate for n patches */
static uint64_t k31v_scratch_bytes(uint64_t n) {
    const uint64_t per_patch = 4u * (K31V_MERGED + K31V_HID + 3u * K31V_QKV + (K31V_OUT / 4u > K31V_QKV ? K31V_OUT / 4u : K31V_QKV) +
                                     K31V_MLP + K31V_HD) + 2u * 3u * K31V_MLP;
    uint64_t qc = ((uint64_t)512 << 20) / ((uint64_t)K31V_HEADS * n * 4u);
    if (qc < 1) qc = 1;
    if (qc > n) qc = n;
    return n * per_patch + (uint64_t)K31V_HEADS * qc * n * 4u;
}

extern "C" int ds4_gpu_k31_vision_encode(const char *dir, const float *patches, uint32_t gh, uint32_t gw, float *out) {
    k31v_model &M = g_k31v;
    const uint64_t weights = M.ready ? 0 : (uint64_t)960 << 20;   /* 887 MB of weights, cuBLAS workspace */
    const uint64_t need = weights + k31v_scratch_bytes((uint64_t)gh * gw) + ((uint64_t)256 << 20);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (free_b < need && ds4_gpu_k3_cache_lend() > 0 && cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (free_b < need) {
        fprintf(stderr, "ds4: K3.1 vision: a %ux%u-patch image needs %.1f GB of VRAM, %.1f GB is free\n", gw, gh, (double)need / 1e9,
                (double)free_b / 1e9);
        return 0;
    }
    if (!k31v_load(dir)) return 0;
    const int ok = k31v_encode(patches, gh, gw, out);
    k31v_free_scratch(M);
    cudaGetLastError();
    return ok;
}
