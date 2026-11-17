/* Kimi K3.1 vision tower + projector on CUDA (17 Nov 2026).
 *
 * MoonViT3d as served by vLLM (kimi_k25_vit.py) and SGLang (kimi_k3_vl.py) for one image (t = 1): 14x14 patches
 * through a bias-free linear, plus the [64, 64, 1024] position table resized bilinearly to the patch grid; 27 blocks of
 * RMSNorm (eps 2^-7, torch's default for BF16 activations) -> wqkv (12 heads x 128) -> 2D rotary on interleaved pairs
 * (x and y frequencies alternating, theta 10000) -> full attention -> wo, and RMSNorm -> fc0 -> gelu (tanh) -> fc1;
 * final RMSNorm; 2x2 spatial merge (sd2_tpool) to 4096 per image token; projector proj.0 -> gelu (erf) -> proj.2 ->
 * RMSNorm (eps 1e-5) to the language model's 7168. tools/kimi-k3/k31_vision_ref.py is the same computation in PyTorch.
 *
 * Weights stay BF16 as published (0.9 GB of VRAM). Every linear runs as three BF16 GEMMs over a three-plane split of the
 * float32 activations (hi + mid + lo), so the products are those of float32 activations with the exact BF16 weights;
 * attention runs in float32 GEMMs. Self-contained (CUDA, cuBLAS), so tools/kimi-k3/k31_vision_gpu_test.cu can check it
 * against the reference on its own; ds4_cuda.cu includes it for the K3 server. */
#ifndef DS4_K31_VISION_GPU_CUH
#define DS4_K31_VISION_GPU_CUH
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define K31V_HID 1024u
#define K31V_HEADS 12u
#define K31V_HD 128u
#define K31V_QKV 1536u
#define K31V_MLP 4096u
#define K31V_LAYERS 27u
#define K31V_PIN 588u          /* 3 x 14 x 14 */
#define K31V_MERGED 4096u      /* 2 x 2 x 1024 */
#define K31V_OUT 7168u
#define K31V_POS 64u

struct k31v_layer {
    float *norm0 = nullptr, *norm1 = nullptr;
    __nv_bfloat16 *wqkv = nullptr, *wo = nullptr, *fc0 = nullptr, *fc1 = nullptr;
};
struct k31v_model {
    bool ready = false;
    __nv_bfloat16 *patch_w = nullptr, *proj0 = nullptr, *proj2 = nullptr;
    float *final_norm = nullptr, *post_norm = nullptr;
    std::vector<float> pos;            /* [64][64][1024], host: resized per image */
    k31v_layer layer[K31V_LAYERS];
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    /* scratch, grown on demand */
    size_t cap_n = 0, cap_s = 0;
    float *x = nullptr, *y = nullptr, *qkv = nullptr, *o = nullptr, *h = nullptr, *s = nullptr, *cs = nullptr;
    __nv_bfloat16 *split = nullptr;
};
static k31v_model g_k31v;

#define K31V_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "ds4: K3.1 vision: %s: %s\n", #call, cudaGetErrorString(e_)); return 0; } } while (0)
#define K31V_CB(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "ds4: K3.1 vision: %s: cuBLAS status %d\n", #call, (int)s_); return 0; } } while (0)

/* ---- weights: the vision_tower.* and mm_projector.* tensors of the checkpoint's safetensors shards ---- */
static bool k31v_read_tensor(const std::vector<std::string> &files, const char *name, std::vector<uint16_t> &out, size_t want) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":", name);
    for (const std::string &fn : files) {
        FILE *f = fopen(fn.c_str(), "rb");
        if (!f) continue;
        uint64_t n = 0;
        if (fread(&n, 8, 1, f) != 1 || n > (64u << 20)) { fclose(f); continue; }
        std::string hdr(n, '\0');
        if (fread(&hdr[0], 1, n, f) != n) { fclose(f); continue; }
        const size_t at = hdr.find(key);
        if (at == std::string::npos) { fclose(f); continue; }
        const size_t end = hdr.find('}', at);
        const std::string obj = hdr.substr(at, end - at);
        const size_t dt = obj.find("\"dtype\":\""), off = obj.find("\"data_offsets\":[");
        unsigned long long a = 0, b = 0;
        if (dt == std::string::npos || obj.compare(dt + 9, 4, "BF16") != 0 || off == std::string::npos ||
            sscanf(obj.c_str() + off + 16, "%llu,%llu", &a, &b) != 2 || b - a != want * 2u) {
            fprintf(stderr, "ds4: K3.1 vision: %s is not a BF16 tensor of %zu values in %s\n", name, want, fn.c_str());
            fclose(f);
            return false;
        }
        out.resize(want);
        const bool ok = fseek(f, (long)(8 + n + a), SEEK_SET) == 0 && fread(out.data(), 2, want, f) == want;
        fclose(f);
        if (!ok) fprintf(stderr, "ds4: K3.1 vision: reading %s failed\n", name);
        return ok;
    }
    fprintf(stderr, "ds4: K3.1 vision: %s not found in the shards\n", name);
    return false;
}

static float k31v_bf16(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

static bool k31v_upload_bf16(const std::vector<std::string> &files, const char *name, size_t n, __nv_bfloat16 **dst) {
    std::vector<uint16_t> h;
    if (!k31v_read_tensor(files, name, h, n)) return false;
    if (cudaMalloc(dst, n * 2) != cudaSuccess || cudaMemcpy(*dst, h.data(), n * 2, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    return true;
}
static bool k31v_upload_f32(const std::vector<std::string> &files, const char *name, size_t n, float **dst) {
    std::vector<uint16_t> h;
    if (!k31v_read_tensor(files, name, h, n)) return false;
    std::vector<float> f(n);
    for (size_t i = 0; i < n; i++) f[i] = k31v_bf16(h[i]);
    if (cudaMalloc(dst, n * 4) != cudaSuccess || cudaMemcpy(*dst, f.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    return true;
}

/* dir holds the shards (model-00095/00096-of-000096.safetensors for K3.1 Pro) */
static int k31v_load(const char *dir) {
    k31v_model &m = g_k31v;
    if (m.ready) return 1;
    std::vector<std::string> files;
    for (int i = 1; i <= 96; i++) {
        char p[4200];
        snprintf(p, sizeof(p), "%s/model-%05d-of-000096.safetensors", dir, i);
        FILE *f = fopen(p, "rb");
        if (f) { fclose(f); files.push_back(p); }
    }
    if (files.empty()) { fprintf(stderr, "ds4: K3.1 vision: no checkpoint shards in %s\n", dir); return 0; }
    bool ok = k31v_upload_bf16(files, "vision_tower.patch_embed.proj.weight", (size_t)K31V_HID * K31V_PIN, &m.patch_w) &&
              k31v_upload_bf16(files, "mm_projector.proj.0.weight", (size_t)K31V_MERGED * K31V_MERGED, &m.proj0) &&
              k31v_upload_bf16(files, "mm_projector.proj.2.weight", (size_t)K31V_OUT * K31V_MERGED, &m.proj2) &&
              k31v_upload_f32(files, "vision_tower.encoder.final_layernorm.weight", K31V_HID, &m.final_norm) &&
              k31v_upload_f32(files, "mm_projector.post_norm.weight", K31V_OUT, &m.post_norm);
    {
        std::vector<uint16_t> h;
        ok = ok && k31v_read_tensor(files, "vision_tower.patch_embed.pos_emb.weight", h, (size_t)K31V_POS * K31V_POS * K31V_HID);
        if (ok) { m.pos.resize(h.size()); for (size_t i = 0; i < h.size(); i++) m.pos[i] = k31v_bf16(h[i]); }
    }
    for (uint32_t l = 0; ok && l < K31V_LAYERS; l++) {
        char n[160];
        k31v_layer &L = m.layer[l];
#define K31V_NAME(s) (snprintf(n, sizeof(n), "vision_tower.encoder.blocks.%u." s, l), n)
        ok = k31v_upload_f32(files, K31V_NAME("norm0.weight"), K31V_HID, &L.norm0) &&
             k31v_upload_f32(files, K31V_NAME("norm1.weight"), K31V_HID, &L.norm1) &&
             k31v_upload_bf16(files, K31V_NAME("wqkv.weight"), (size_t)3u * K31V_QKV * K31V_HID, &L.wqkv) &&
             k31v_upload_bf16(files, K31V_NAME("wo.weight"), (size_t)K31V_HID * K31V_QKV, &L.wo) &&
             k31v_upload_bf16(files, K31V_NAME("mlp.fc0.weight"), (size_t)K31V_MLP * K31V_HID, &L.fc0) &&
             k31v_upload_bf16(files, K31V_NAME("mlp.fc1.weight"), (size_t)K31V_HID * K31V_MLP, &L.fc1);
#undef K31V_NAME
    }
    if (!ok) return 0;
    if (!m.cublas && cublasCreate(&m.cublas) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "ds4: K3.1 vision: cuBLAS init failed\n"); return 0; }
    if (!m.stream && cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking) != cudaSuccess) return 0;
    cublasSetStream(m.cublas, m.stream);
    cublasSetMathMode(m.cublas, CUBLAS_DEFAULT_MATH);   /* float32 GEMMs stay float32 (no TF32) */
    m.ready = true;
    fprintf(stderr, "ds4: K3.1 vision tower loaded from %s (27 blocks, 0.9 GB of BF16 weights)\n", dir);
    return 1;
}

/* ---- kernels ---- */
__global__ static void k31v_split3_kernel(__nv_bfloat16 *dst, const float *src, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = src[i];
    const __nv_bfloat16 hi = __float2bfloat16_rn(v);
    const float r1 = __fsub_rn(v, __bfloat162float(hi));
    const __nv_bfloat16 mid = __float2bfloat16_rn(r1);
    const __nv_bfloat16 lo = __float2bfloat16_rn(__fsub_rn(r1, __bfloat162float(mid)));
    dst[i] = hi; dst[n + i] = mid; dst[2 * n + i] = lo;
}
/* one block per row: y = x * rsqrt(mean(x^2) + eps) * w */
__global__ static void k31v_rmsnorm_kernel(float *y, const float *x, const float *w, uint32_t dim, float eps) {
    const float *xr = x + (size_t)blockIdx.x * dim;
    float *yr = y + (size_t)blockIdx.x * dim;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0u) red[0] = t;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/* rotary on q and k of qkv [n][3][12][128]: pair (2p, 2p+1) of every head turns by angle cs[pos][p] (cos, sin) */
__global__ static void k31v_rope_kernel(float *qkv, const float *cs, uint32_t n) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* (row, q/k, head, pair) */
    const size_t per_row = 2u * K31V_HEADS * (K31V_HD / 2u);
    if (t >= (size_t)n * per_row) return;
    const size_t row = t / per_row, r = t % per_row, which = r / (K31V_HEADS * (K31V_HD / 2u));
    const size_t head = (r / (K31V_HD / 2u)) % K31V_HEADS, p = r % (K31V_HD / 2u);
    float *v = qkv + row * (3u * K31V_QKV) + which * K31V_QKV + head * K31V_HD + 2u * p;
    const float c = cs[(row * (K31V_HD / 2u) + p) * 2u], s = cs[(row * (K31V_HD / 2u) + p) * 2u + 1u];
    const float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
}
/* softmax over each row of length len */
__global__ static void k31v_softmax_kernel(float *s, uint32_t len) {
    float *r = s + (size_t)blockIdx.x * len;
    __shared__ float red[32];
    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) mx = fmaxf(mx, r[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
        if (threadIdx.x == 0u) red[0] = t;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) { const float e = expf(r[i] - mx); r[i] = e; sum += e; }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0u) red[0] = t;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) r[i] *= inv;
}
__global__ static void k31v_gelu_tanh_kernel(float *x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
__global__ static void k31v_gelu_erf_kernel(float *x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}
__global__ static void k31v_add_kernel(float *x, const float *y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
/* 2x2 merge: m[ty*(gw/2)+tx][(kh*2+kw)*1024 + c] = x[(2ty+kh)*gw + 2tx+kw][c] */
__global__ static void k31v_merge_kernel(float *m, const float *x, uint32_t gh, uint32_t gw) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tw = gw / 2u;
    const size_t total = (size_t)(gh / 2u) * tw * K31V_MERGED;
    if (i >= total) return;
    const size_t t = i / K31V_MERGED, r = i % K31V_MERGED, k = r / K31V_HID, c = r % K31V_HID;
    const size_t ty = t / tw, tx = t % tw, kh = k / 2u, kw = k % 2u;
    m[i] = x[((2u * ty + kh) * gw + 2u * tx + kw) * K31V_HID + c];
}

static unsigned k31v_blocks(size_t n) { return (unsigned)((n + 255u) / 256u); }

/* y[n x m] = x[n x k] . W[m x k]^T with W in BF16 and x float32 (three-plane split) */
static int k31v_linear(k31v_model &M, float *y, const float *x, const __nv_bfloat16 *W, uint32_t n, uint32_t m, uint32_t k) {
    const size_t nk = (size_t)n * k;
    k31v_split3_kernel<<<k31v_blocks(nk), 256, 0, M.stream>>>(M.split, x, nk);
    const float one = 1.0f, zero = 0.0f;
    for (int t = 0; t < 3; t++)
        K31V_CB(cublasGemmEx(M.cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)m, (int)n, (int)k, &one, W, CUDA_R_16BF, (int)k,
                             M.split + (size_t)t * nk, CUDA_R_16BF, (int)k, t ? &one : &zero, y, CUDA_R_32F, (int)m,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return 1;
}

static int k31v_reserve(k31v_model &M, size_t n) {
    if (n <= M.cap_n) return 1;
    cudaStreamSynchronize(M.stream);
    cudaFree(M.x); cudaFree(M.y); cudaFree(M.qkv); cudaFree(M.o); cudaFree(M.h); cudaFree(M.cs); cudaFree(M.split);
    M.x = M.y = M.qkv = M.o = M.h = M.cs = nullptr; M.split = nullptr; M.cap_n = 0;
    /* x [n][4096]: stream, then projector hidden; y [n][1024]; o [n][1792]: attention out, then the projector's
       [n/4][7168]; h [n][4096]: patches, MLP hidden, merged tokens, the final embeddings */
    K31V_CK(cudaMalloc(&M.x, n * K31V_MERGED * 4));
    K31V_CK(cudaMalloc(&M.y, n * K31V_HID * 4));
    K31V_CK(cudaMalloc(&M.qkv, n * 3u * K31V_QKV * 4));
    K31V_CK(cudaMalloc(&M.o, n * (K31V_OUT / 4u > K31V_QKV ? K31V_OUT / 4u : K31V_QKV) * 4));
    K31V_CK(cudaMalloc(&M.h, n * K31V_MLP * 4));
    K31V_CK(cudaMalloc(&M.cs, n * K31V_HD * 4));
    K31V_CK(cudaMalloc(&M.split, 3u * n * K31V_MLP * 2));
    M.cap_n = n;
    return 1;
}

/* torch.nn.functional.interpolate(mode="bilinear", align_corners=False) of the [64][64][1024] table to [gh][gw], as
   PyTorch's float kernel computes it */
static void k31v_pos_resize(const std::vector<float> &src, uint32_t gh, uint32_t gw, std::vector<float> &dst) {
    dst.resize((size_t)gh * gw * K31V_HID);
    if (gh == K31V_POS && gw == K31V_POS) { memcpy(dst.data(), src.data(), dst.size() * 4); return; }
    const float sh = (float)K31V_POS / (float)gh, sw = (float)K31V_POS / (float)gw;
    for (uint32_t oy = 0; oy < gh; oy++) {
        float fy = sh * ((float)oy + 0.5f) - 0.5f;
        if (fy < 0.0f) fy = 0.0f;
        const uint32_t y0 = (uint32_t)fy, y1 = y0 < K31V_POS - 1u ? y0 + 1u : y0;
        const float ly1 = fy - (float)y0, ly0 = 1.0f - ly1;
        for (uint32_t ox = 0; ox < gw; ox++) {
            float fx = sw * ((float)ox + 0.5f) - 0.5f;
            if (fx < 0.0f) fx = 0.0f;
            const uint32_t x0 = (uint32_t)fx, x1 = x0 < K31V_POS - 1u ? x0 + 1u : x0;
            const float lx1 = fx - (float)x0, lx0 = 1.0f - lx1;
            const float *a = &src[((size_t)y0 * K31V_POS + x0) * K31V_HID], *b = &src[((size_t)y0 * K31V_POS + x1) * K31V_HID];
            const float *c = &src[((size_t)y1 * K31V_POS + x0) * K31V_HID], *d = &src[((size_t)y1 * K31V_POS + x1) * K31V_HID];
            float *o = &dst[((size_t)oy * gw + ox) * K31V_HID];
            for (uint32_t i = 0; i < K31V_HID; i++) o[i] = ly0 * (lx0 * a[i] + lx1 * b[i]) + ly1 * (lx0 * c[i] + lx1 * d[i]);
        }
    }
}

/* patches: host [gh*gw][588] (channel, row, column of each 14x14 patch); out: host [(gh/2)*(gw/2)][7168] */
static int k31v_encode(const float *patches, uint32_t gh, uint32_t gw, float *out) {
    k31v_model &M = g_k31v;
    if (!M.ready || gh == 0 || gw == 0 || gh % 2u || gw % 2u || gh > 512u || gw > 512u) return 0;
    const uint32_t n = gh * gw, T = n / 4u;
    if (!k31v_reserve(M, n)) return 0;
    /* inputs: patches and the resized position table to the device, rotary angles from double precision */
    std::vector<float> pos, cs((size_t)n * K31V_HD);
    k31v_pos_resize(M.pos, gh, gw, pos);
    for (uint32_t i = 0; i < n; i++) {
        const double px = (double)(i % gw), py = (double)(i / gw);
        for (uint32_t p = 0; p < K31V_HD / 2u; p++) {
            const double freq = 1.0 / pow(10000.0, (double)(4u * (p / 2u)) / (double)K31V_HD);
            const double ang = (p % 2u ? py : px) * freq;
            cs[((size_t)i * (K31V_HD / 2u) + p) * 2u] = (float)cos(ang);
            cs[((size_t)i * (K31V_HD / 2u) + p) * 2u + 1u] = (float)sin(ang);
        }
    }
    K31V_CK(cudaMemcpyAsync(M.h, patches, (size_t)n * K31V_PIN * 4, cudaMemcpyHostToDevice, M.stream));
    K31V_CK(cudaMemcpyAsync(M.y, pos.data(), (size_t)n * K31V_HID * 4, cudaMemcpyHostToDevice, M.stream));
    K31V_CK(cudaMemcpyAsync(M.cs, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice, M.stream));
    if (!k31v_linear(M, M.x, M.h, M.patch_w, n, K31V_HID, K31V_PIN)) return 0;
    const size_t nh = (size_t)n * K31V_HID;
    k31v_add_kernel<<<k31v_blocks(nh), 256, 0, M.stream>>>(M.x, M.y, nh);
    /* attention scores in chunks of query rows: 12 x qc x n floats within 512 MB */
    size_t qc = ((size_t)512 << 20) / ((size_t)K31V_HEADS * n * 4u);
    if (qc < 1) qc = 1;
    if (qc > n) qc = n;
    if (M.cap_s < (size_t)K31V_HEADS * qc * n) {
        cudaStreamSynchronize(M.stream);
        cudaFree(M.s);
        K31V_CK(cudaMalloc(&M.s, (size_t)K31V_HEADS * qc * n * 4));
        M.cap_s = (size_t)K31V_HEADS * qc * n;
    }
    const float scale = 1.0f / sqrtf((float)K31V_HD), one = 1.0f, zero = 0.0f;
    for (uint32_t l = 0; l < K31V_LAYERS; l++) {
        const k31v_layer &L = M.layer[l];
        k31v_rmsnorm_kernel<<<n, 256, 0, M.stream>>>(M.y, M.x, L.norm0, K31V_HID, 1.0f / 128.0f);
        if (!k31v_linear(M, M.qkv, M.y, L.wqkv, n, 3u * K31V_QKV, K31V_HID)) return 0;
        const size_t nr = (size_t)n * 2u * K31V_HEADS * (K31V_HD / 2u);
        k31v_rope_kernel<<<k31v_blocks(nr), 256, 0, M.stream>>>(M.qkv, M.cs, n);
        for (uint32_t q0 = 0; q0 < n; q0 += (uint32_t)qc) {
            const uint32_t nq = n - q0 < qc ? n - q0 : (uint32_t)qc;
            const float *Q = M.qkv + (size_t)q0 * 3u * K31V_QKV, *K = M.qkv + K31V_QKV, *V = M.qkv + 2u * K31V_QKV;
            /* S_h[i][j] = scale * q_i . k_j, stored row i over keys j: (keys x queries) column-major */
            K31V_CB(cublasSgemmStridedBatched(M.cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)nq, (int)K31V_HD, &scale,
                                              K, (int)(3u * K31V_QKV), (long long)K31V_HD, Q, (int)(3u * K31V_QKV), (long long)K31V_HD,
                                              &zero, M.s, (int)n, (long long)nq * n, (int)K31V_HEADS));
            k31v_softmax_kernel<<<(unsigned)(K31V_HEADS * nq), 256, 0, M.stream>>>(M.s, n);
            /* O_h[i] = sum_j P_h[i][j] v_j into o [n][12 x 128] */
            K31V_CB(cublasSgemmStridedBatched(M.cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)K31V_HD, (int)nq, (int)n, &one,
                                              V, (int)(3u * K31V_QKV), (long long)K31V_HD, M.s, (int)n, (long long)nq * n,
                                              &zero, M.o + (size_t)q0 * K31V_QKV, (int)K31V_QKV, (long long)K31V_HD, (int)K31V_HEADS));
        }
        if (!k31v_linear(M, M.y, M.o, L.wo, n, K31V_HID, K31V_QKV)) return 0;
        k31v_add_kernel<<<k31v_blocks(nh), 256, 0, M.stream>>>(M.x, M.y, nh);
        k31v_rmsnorm_kernel<<<n, 256, 0, M.stream>>>(M.y, M.x, L.norm1, K31V_HID, 1.0f / 128.0f);
        if (!k31v_linear(M, M.h, M.y, L.fc0, n, K31V_MLP, K31V_HID)) return 0;
        k31v_gelu_tanh_kernel<<<k31v_blocks((size_t)n * K31V_MLP), 256, 0, M.stream>>>(M.h, (size_t)n * K31V_MLP);
        if (!k31v_linear(M, M.y, M.h, L.fc1, n, K31V_HID, K31V_MLP)) return 0;
        k31v_add_kernel<<<k31v_blocks(nh), 256, 0, M.stream>>>(M.x, M.y, nh);
    }
    k31v_rmsnorm_kernel<<<n, 256, 0, M.stream>>>(M.y, M.x, M.final_norm, K31V_HID, 1.0f / 128.0f);
    /* merge into h [T][4096], projector: x [T][4096] -> gelu -> y [T][7168] -> RMSNorm into h */
    const size_t tm = (size_t)T * K31V_MERGED;
    k31v_merge_kernel<<<k31v_blocks(tm), 256, 0, M.stream>>>(M.h, M.y, gh, gw);
    if (!k31v_linear(M, M.x, M.h, M.proj0, T, K31V_MERGED, K31V_MERGED)) return 0;
    k31v_gelu_erf_kernel<<<k31v_blocks(tm), 256, 0, M.stream>>>(M.x, tm);
    if (!k31v_linear(M, M.o, M.x, M.proj2, T, K31V_OUT, K31V_MERGED)) return 0;
    k31v_rmsnorm_kernel<<<T, 256, 0, M.stream>>>(M.h, M.o, M.post_norm, K31V_OUT, 1e-5f);
    K31V_CK(cudaMemcpyAsync(out, M.h, (size_t)T * K31V_OUT * 4, cudaMemcpyDeviceToHost, M.stream));
    K31V_CK(cudaStreamSynchronize(M.stream));
    K31V_CK(cudaGetLastError());
    return 1;
}
#endif
