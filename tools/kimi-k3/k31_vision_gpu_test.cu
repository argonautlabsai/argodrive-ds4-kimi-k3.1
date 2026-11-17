/* Checks ds4_k31_vision_gpu.cuh against k31_vision_ref.py (17 Nov 2026): runs the GPU encoder on the reference's
 * patches (PREFIX.patches.f32 from k31_vision_ref.py) and writes the embeddings as raw float32 for comparison.
 *   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o k31_vision_gpu_test \
 *        tools/kimi-k3/k31_vision_gpu_test.cu -lcublas
 *   k31_vision_gpu_test SHARD_DIR PATCHES.f32 GRID_H GRID_W OUT.f32 */
#include "../../ds4_k31_vision_gpu.cuh"
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s SHARD_DIR PATCHES.f32 GRID_H GRID_W OUT.f32\n", argv[0]); return 2; }
    const uint32_t gh = (uint32_t)atoi(argv[3]), gw = (uint32_t)atoi(argv[4]), n = gh * gw, T = n / 4u;
    std::vector<float> patches((size_t)n * K31V_PIN), out((size_t)T * K31V_OUT);
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(patches.data(), 4, patches.size(), f) != patches.size()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    fclose(f);
    auto t0 = std::chrono::steady_clock::now();
    if (!k31v_load(argv[1])) return 1;
    auto t1 = std::chrono::steady_clock::now();
    if (!k31v_encode(patches.data(), gh, gw, out.data())) { fprintf(stderr, "encode failed\n"); return 1; }
    auto t2 = std::chrono::steady_clock::now();
    if (!k31v_encode(patches.data(), gh, gw, out.data())) { fprintf(stderr, "encode failed\n"); return 1; }
    auto t3 = std::chrono::steady_clock::now();
    f = fopen(argv[5], "wb");
    if (!f || fwrite(out.data(), 4, out.size(), f) != out.size()) { fprintf(stderr, "cannot write %s\n", argv[5]); return 1; }
    fclose(f);
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    printf("grid %ux%u: %u patches -> %u image tokens; load %.0f ms, first encode %.1f ms, second %.1f ms\n", gh, gw, n, T,
           ms(t0, t1), ms(t1, t2), ms(t2, t3));
    return 0;
}
