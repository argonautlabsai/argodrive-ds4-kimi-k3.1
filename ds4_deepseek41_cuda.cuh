/* V4.1-specific operations. Included after the shared CUDA kernels. */
#include "ds4_deepseek41_gpu.h"

#ifdef __HIP_PLATFORM_AMD__
static cudaStream_t cuda_decode_stream(void) { return nullptr; }
#endif

__device__ static float dsv41_broadcast(float value) {
#ifdef __HIP_PLATFORM_AMD__
    return __shfl(value, 0, 32);
#else
    return __shfl_sync(0xffffffffu, value, 0);
#endif
}

static bool dsv41_has_floats(const ds4_gpu_tensor *t, uint64_t count) {
    return t && t->ptr && count <= t->bytes / sizeof(float);
}

__device__ static float dsv41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

#ifndef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!map || !width || !rows || !dsv41_has_floats(out, count) ||
        !dsv41_has_floats(x, count) || offset > size ||
        (uint64_t)width * 4u > size - offset) return 0;
    const float *w = (const float *)cuda_resolve_weight_ptr(map, offset,
        (uint64_t)width * 4u, ds4_tensor_device_idx(out), "V4.1 norm");
    if (!w) return 0;
    cuda_rms_norm_weight_launch<true>((float *)out->ptr, (const float *)x->ptr, w, width, rows, eps,
        cuda_decode_stream());   /* K1 */
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 norm");
}

extern "C" int ds4_gpu_dsv41_hc_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows,
        uint32_t weight_stride) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || (weight_stride != 4 && weight_stride != 24) ||
        (count + 255u) / 256u > INT32_MAX || !dsv41_has_floats(out, count) ||
        !dsv41_has_floats(x, count * 4u) ||
        !dsv41_has_floats(weights, (uint64_t)(rows - 1u) * weight_stride + 4u)) return 0;
    hc_weighted_sum_kernel<true><<<(count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, (const float *)weights->ptr,
        width, 4u, rows, weight_stride);
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 HC sum");
}

extern "C" int ds4_gpu_dsv41_hc_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *block,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split,
        uint32_t width, uint32_t rows) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || count > (uint64_t)INT32_MAX * 64u ||
        !dsv41_has_floats(out, count * 4u) || !dsv41_has_floats(residual, count * 4u) ||
        !dsv41_has_floats(block, count) || !dsv41_has_floats(split, (uint64_t)rows * 24u)) return 0;
    const float *sp = (const float *)split->ptr;
    hc_expand_kernel<true><<<(count * 4u + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)block->ptr, NULL, NULL,
        (const float *)residual->ptr, sp + 4u, sp + 8u,
        width, 4u, rows, 24u, 24u, 0, 0);
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 HC expansion");
}
#endif

__device__ static float dsv41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) +
        ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__global__ static void dsv41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = dsv41_bf16(x[i]);
}

__global__ static void dsv41_quantize_kernel(float *x, uint32_t width,
                                            uint32_t rows, uint32_t mode) {
    const uint32_t block = mode == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t group = (uint64_t)blockIdx.x * 8u + threadIdx.x / 32u;
    if (group >= (uint64_t)(width / block) * rows) return;
    const uint64_t i = group * block + lane;
    const float value = lane < block ? dsv41_bf16(x[i]) : 0.0f;
    const float peak = dsv41_broadcast(warp_max_f32(fabsf(value)));
    float scale, quantized;
    if (mode == DS4_V41_FP8_E8M0) {
        scale = dsv41_pow2_ceil(__fmul_rn(fmaxf(peak, 1.0e-4f), 1.0f / 448.0f));
        quantized = dsv4_e4m3fn_dequant_dev(__fdiv_rn(fabsf(value), scale));
    } else {
        scale = mode == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(__fdiv_rn(fmaxf(peak, 0.01171875f), 6.0f)) :
            dsv41_pow2_ceil(__fmul_rn(fmaxf(peak, 0x1.8p-124f), 1.0f / 6.0f));
        quantized = dsv4_e2m1fn_dequant_dev(__fdiv_rn(fabsf(value), scale));
    }
    if (lane < block) x[i] = dsv41_bf16(__fmul_rn(copysignf(quantized, value), scale));
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width,
                                       uint32_t rows, ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) || !dsv41_has_floats(x, count) ||
        (count + 255u) / 256u > INT32_MAX) return 0;
    if (format == DS4_V41_BF16)
        dsv41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, count);
    else
        dsv41_quantize_kernel<<<(unsigned)((count / block + 7u) / 8u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, width, rows, format);
    return cuda_ok(cudaGetLastError(), "V4.1 activation rounding");
}

#ifndef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_shared_join(void) {
    if (!g_dsv41_shared.pending) return 1;
    g_dsv41_shared.pending = false;
    const bool ok = cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(),
        g_dsv41_shared.done, 0), "shared expert join");
    if (!ok && !g_decode_graph_capturing)
        (void)cudaStreamSynchronize(g_dsv41_shared.stream);
    return ok;
}

__global__ static void dsv41_shared_gate_up_kernel(
        float *gate, float *up, float *mid,
        const unsigned char *wg, const unsigned char *wu,
        const int8_t *xq, const float *xs,
        uint32_t blocks, uint32_t hidden, float clamp, int dp4a, const int *enabled) {
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= hidden) return;
    if (enabled && !*enabled) {
        if (lane == 0) mid[row] = 0.0f;
        return;
    }
    float g = 0, u = 0;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint64_t off = ((uint64_t)row * blocks + b) * 34u;
        const int8_t *xb = xq + b * 32u;
        const int dg = dot_i8_block((const int8_t *)(wg + off + 2u), xb, 32, dp4a);
        const int du = dot_i8_block((const int8_t *)(wu + off + 2u), xb, 32, dp4a);
        g += __half2float(*(const __half *)(wg + off)) * xs[b] * (float)dg;
        u += __half2float(*(const __half *)(wu + off)) * xs[b] * (float)du;
    }
    g = dsv41_bf16(warp_sum_f32(g));
    u = dsv41_bf16(warp_sum_f32(u));
    if (lane == 0) {
        gate[row] = g;
        up[row] = u;
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fminf(fmaxf(u, -clamp), clamp);
        }
        mid[row] = dsv41_bf16((g / (1.0f + expf(-g))) * u);
    }
}

__global__ static void dsv41_shared_down_kernel(float *out, const unsigned char *w,
        const int8_t *xq, const float *xs, uint32_t blocks, uint32_t width, int dp4a,
        const int *enabled) {
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= width) return;
    if (enabled && !*enabled) {
        if (lane == 0) out[row] = 0.0f;
        return;
    }
    float sum = 0;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint64_t off = ((uint64_t)row * blocks + b) * 34u;
        const int dot = dot_i8_block((const int8_t *)(w + off + 2u), xq + b * 32u, 32, dp4a);
        sum += __half2float(*(const __half *)(w + off)) * xs[b] * (float)dot;
    }
    sum = warp_sum_f32(sum);
    if (lane == 0) out[row] = dsv41_bf16(sum);
}

static bool dsv41_shared_fused(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t gate_off, uint64_t up_off,
        uint64_t down_off, uint32_t width, uint32_t hidden, float clamp,
        const int *enabled) {
    const uint64_t gu_bytes = (uint64_t)hidden * (width / 32u) * 34u;
    const uint64_t down_bytes = (uint64_t)width * (hidden / 32u) * 34u;
    if (!map || gate_off > size || gu_bytes > size - gate_off ||
        up_off > size || gu_bytes > size - up_off ||
        down_off > size || down_bytes > size - down_off) return false;
    const auto *wg = (const unsigned char *)cuda_resolve_weight_ptr(map, gate_off, gu_bytes, 0, "shared gate");
    const auto *wu = (const unsigned char *)cuda_resolve_weight_ptr(map, up_off, gu_bytes, 0, "shared up");
    const auto *wd = (const unsigned char *)cuda_resolve_weight_ptr(map, down_off, down_bytes, 0, "shared down");
    if (!wg || !wu || !wd) return false;
    auto *xq = (int8_t *)g_dsv41_shared.scratch;
    auto *xs = (float *)((char *)g_dsv41_shared.scratch + std::max(width, hidden));
    const cudaStream_t stream = cuda_decode_stream();
    const int dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<width / 32u, 32, 0, stream>>>(
        xq, xs, (const float *)x->ptr, width, width / 32u);
    dsv41_shared_gate_up_kernel<<<(hidden + 7u) / 8u, 256, 0, stream>>>(
        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
        wg, wu, xq, xs, width / 32u, hidden, clamp, dp4a, enabled);
    quantize_q8_0_f32_kernel<<<hidden / 32u, 32, 0, stream>>>(
        xq, xs, (const float *)mid->ptr, hidden, hidden / 32u);
    dsv41_shared_down_kernel<<<(width + 7u) / 8u, 256, 0, stream>>>(
        (float *)out->ptr, wd, xq, xs, hidden / 32u, width, dp4a, enabled);
    return cuda_ok(cudaGetLastError(), "V4.1 fused shared expert");
}

__device__ static uint32_t dsv41_shared_owner(const int32_t *selected,
        uint32_t experts, uint32_t active, uint32_t tie) {
    uint32_t left = 0;
    for (uint32_t i = 0; i < active; i++) left += (uint32_t)selected[i] < experts / 2u;
    return left * 2u == active ? tie : left * 2u < active ? 0u : 1u;
}

__global__ static void dsv41_shared_owner_kernel(int *enabled, const int32_t *selected,
        uint32_t experts, uint32_t active, uint32_t rank, uint32_t tie) {
    *enabled = rank == dsv41_shared_owner(selected, experts, active, tie);
}

__global__ static void dsv41_shared_mask_kernel(float *shared, const int32_t *selected,
        uint32_t width, uint32_t experts, uint32_t active, uint32_t rank, uint32_t tie) {
    const uint32_t row = blockIdx.x;
    __shared__ bool enabled;
    if (!threadIdx.x) enabled = rank == dsv41_shared_owner(selected + row * active, experts, active, tie);
    __syncthreads();
    if (!enabled) for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        shared[(uint64_t)row * width + i] = 0.0f;
}

extern "C" int ds4_gpu_dsv41_shared_mask(ds4_gpu_tensor *shared, const ds4_gpu_tensor *selected,
        uint32_t width, uint32_t rows, uint32_t experts, uint32_t active,
        uint32_t rank, uint32_t tie_owner) {
    if (!width || !rows || rows > 8u || !experts || experts % 2u ||
        !active || active > experts || rank > 1u || tie_owner > 1u ||
        !dsv41_has_floats(shared, (uint64_t)width * rows) ||
        !dsv41_has_floats(selected, (uint64_t)active * rows)) return 0;
    dsv41_shared_mask_kernel<<<rows, 256, 0, cuda_decode_stream()>>>(
        (float *)shared->ptr, (const int32_t *)selected->ptr,
        width, experts, active, rank, tie_owner);
    return cuda_ok(cudaGetLastError(), "V4.1 batched shared owner");
}

extern "C" int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp,
        const ds4_v41_shared_owner *owner) {
    /* DS4_CUDA_SHARED_ASYNC=1 lets discrete GPUs use this path too: with SSD
     * streaming the shared expert then runs while the host loads routed experts. */
    static int async_env = -1;
    if (async_env < 0) { const char *e = getenv("DS4_CUDA_SHARED_ASYNC"); async_env = e && strcmp(e, "0") != 0; }
    if (!ds4_gpu_device_is_spark() && !async_env) return 0;
    if (g_dsv41_shared.active || g_dsv41_shared.pending || !width || !hidden ||
        !dsv41_has_floats(x, width) || !dsv41_has_floats(out, width) ||
        !dsv41_has_floats(gate, hidden) || !dsv41_has_floats(up, hidden) ||
        !dsv41_has_floats(mid, hidden)) return -1;
    const uint64_t blocks = ((uint64_t)std::max(width, hidden) + 31u) / 32u;
    if (owner && (!owner->experts || owner->experts % 2u || !owner->active ||
        owner->active > owner->experts || owner->rank > 1u || owner->tie_owner > 1u ||
        !dsv41_has_floats(owner->selected, owner->active))) return -1;
    if (blocks * 36u > CUDA_DSV41_SHARED_SCRATCH - sizeof(int) ||
        !cuda_dsv41_shared_prepare()) return 0;
    const cudaStream_t main = cuda_decode_stream();
    const bool fused = width % 32u == 0 && hidden % 32u == 0 &&
        !getenv("DS4_CUDA_DISABLE_V41_SHARED_FUSION");
    if (owner && !fused) return 0;
    int *enabled = owner ? (int *)((char *)g_dsv41_shared.scratch +
                                  CUDA_DSV41_SHARED_SCRATCH - sizeof(int)) : NULL;
    /* Decide before the owned-expert filter mutates selected on the main
     * stream. The ready event publishes this decision to the side stream. */
    if (owner) {
        dsv41_shared_owner_kernel<<<1, 1, 0, main>>>(enabled,
            (const int32_t *)owner->selected->ptr, owner->experts, owner->active,
            owner->rank, owner->tie_owner);
        if (!cuda_ok(cudaGetLastError(), "V4.1 shared expert owner")) return -1;
    }
    if (!cuda_ok(cudaEventRecord(g_dsv41_shared.ready, main), "shared expert ready") ||
        !cuda_ok(cudaStreamWaitEvent(g_dsv41_shared.stream, g_dsv41_shared.ready, 0),
                 "shared expert input wait")) return -1;
    g_dsv41_shared.active = true;
    const bool ok = fused ? dsv41_shared_fused(out, gate, up, mid, x, model_map,
        model_size, gate_offset, up_offset, down_offset, width, hidden, clamp, enabled) :
        ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                   gate_offset, width, hidden, x, 1) &&
        ds4_gpu_dsv41_quantize(gate, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                   up_offset, width, hidden, x, 1) &&
        ds4_gpu_dsv41_quantize(up, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_swiglu_tensor(mid, gate, up, hidden, clamp, 1.0f) &&
        ds4_gpu_dsv41_quantize(mid, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                   down_offset, hidden, width, mid, 1) &&
        ds4_gpu_dsv41_quantize(out, width, 1, DS4_V41_BF16);
    g_dsv41_shared.active = false;
    if (!cuda_ok(cudaEventRecord(g_dsv41_shared.done, g_dsv41_shared.stream),
                 "shared expert done")) {
        if (!g_decode_graph_capturing) (void)cudaStreamSynchronize(g_dsv41_shared.stream);
        return -1;
    }
    g_dsv41_shared.pending = true;
    if (!ok) {
        (void)ds4_gpu_dsv41_shared_join();
        return -1;
    }
    return 1;
}

#endif

struct dsv41_rope_args {
    uint32_t width, heads, rows, start, stride, inverse;
    float frequencies[32];
};

__global__ static void dsv41_rope_kernel(float *x, dsv41_rope_args a) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t head = (uint64_t)blockIdx.x * 8u + threadIdx.x / 32u;
    if (head >= (uint64_t)a.heads * a.rows) return;
    const uint32_t row = head / a.heads;
    const float theta = __fmul_rn(float(a.start + row * a.stride), a.frequencies[lane]);
    /* Avoid fast-math range reduction at long absolute positions. */
    const float c = (float)cos((double)theta);
    const float s = (a.inverse ? -1.0f : 1.0f) * (float)sin((double)theta);
    const uint64_t i = head * a.width + a.width - 64u + lane * 2u;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(__fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s)));
    x[i + 1u] = dsv41_bf16(__fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c)));
}

/* K1: dsv41_rope_kernel with W heads per block: the same per-lane arithmetic; with one head per block the double
 * cos/sin of eight heads no longer queue on one SM (64 heads 4.0 -> 2.7 us). */
template <uint32_t W>
__global__ static void dsv41_rope_w_kernel(float *x, dsv41_rope_args a) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t head = (uint64_t)blockIdx.x * W + threadIdx.x / 32u;
    if (head >= (uint64_t)a.heads * a.rows) return;
    const uint32_t row = head / a.heads;
    const float theta = __fmul_rn(float(a.start + row * a.stride), a.frequencies[lane]);
    /* Avoid fast-math range reduction at long absolute positions. */
    const float c = (float)cos((double)theta);
    const float s = (a.inverse ? -1.0f : 1.0f) * (float)sin((double)theta);
    const uint64_t i = head * a.width + a.width - 64u + lane * 2u;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(__fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s)));
    x[i + 1u] = dsv41_bf16(__fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c)));
}

/* K4: a decode rope (one row) takes its cos/sin from a table made once per position, card and kind (compressed or
 * not): dsv41_rope_table_kernel does the per-lane double cos/sin of dsv41_rope_kernel for row 0, and the rotation
 * that reads it is unchanged, so the result is the same. */
__global__ static void dsv41_rope_table_kernel(float2 *tab, dsv41_rope_args a) {
    const uint32_t lane = threadIdx.x & 31u;
    const float theta = __fmul_rn(float(a.start), a.frequencies[lane]);
    /* Avoid fast-math range reduction at long absolute positions. */
    tab[lane] = make_float2((float)cos((double)theta), (float)sin((double)theta));
}

template <uint32_t W>
__global__ static void dsv41_rope_tab_kernel(float *x, dsv41_rope_args a, const float2 *tab) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t head = (uint64_t)blockIdx.x * W + threadIdx.x / 32u;
    if (head >= (uint64_t)a.heads) return;
    const float2 cs = tab[lane];
    const float c = cs.x;
    const float s = (a.inverse ? -1.0f : 1.0f) * cs.y;
    const uint64_t i = head * a.width + a.width - 64u + lane * 2u;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(__fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s)));
    x[i + 1u] = dsv41_bf16(__fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c)));
}

struct dsv41_rope_tab_cache {
    float2 *tab;        /* [2][32]: plain, compressed */
    uint32_t pos[2];
    int valid[2];
};
static dsv41_rope_tab_cache g_dsv41_rope_tab[DS4_MAX_GPUS];

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width,
                                          uint32_t heads, uint32_t rows, uint32_t start,
                                          uint32_t stride, bool compressed, bool inverse) {
    if (width < 64u || !heads || !rows || rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT64_MAX / width ||
        !dsv41_has_floats(x, (uint64_t)width * heads * rows) ||
        ((uint64_t)heads * rows + 7u) / 8u > INT32_MAX) return 0;
    struct frequencies {
        float value[2][32];
        frequencies() {
            for (int kind = 0; kind < 2; kind++) {
                const float base = kind ? 160000.0f : 10000.0f;
                const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
                const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
                for (int i = 0; i < 32; i++) {
                    float f = 1.0f / powf(base, (float)i / 32.0f);
                    if (kind) {
                        const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                        const float smooth = 1.0f - ramp;
                        f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
                    }
                    value[kind][i] = f;
                }
            }
        }
    };
    static const frequencies table;
    dsv41_rope_args args = {width, heads, rows, start, stride, inverse, {0}};
    memcpy(args.frequencies, table.value[compressed ? 1 : 0], sizeof(args.frequencies));
    if (cuda_kfast() && rows == 1u && heads <= 256u && !g_decode_graph_capturing) {   /* K4: the cos/sin table */
        int dev = -1;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < DS4_MAX_GPUS) {
            dsv41_rope_tab_cache &tc = g_dsv41_rope_tab[dev];
            if (!tc.tab && cudaMalloc((void **)&tc.tab, 64u * sizeof(float2)) != cudaSuccess) {
                (void)cudaGetLastError();
                tc.tab = NULL;
            }
            if (tc.tab) {
                const int kind = compressed ? 1 : 0;
                float2 *tab = tc.tab + 32u * (uint32_t)kind;
                if (!tc.valid[kind] || tc.pos[kind] != start) {
                    dsv41_rope_table_kernel<<<1, 32, 0, cuda_decode_stream()>>>(tab, args);
                    tc.valid[kind] = 1;
                    tc.pos[kind] = start;
                }
                dsv41_rope_tab_kernel<1><<<heads, 32, 0, cuda_decode_stream()>>>((float *)x->ptr, args, tab);
                return cuda_ok(cudaGetLastError(), "V4.1 RoPE");
            }
        }
    }
    if (cuda_kfast() && (uint64_t)heads * rows <= 256u)   /* K1 */
        dsv41_rope_w_kernel<1><<<(unsigned)((uint64_t)heads * rows), 32, 0, cuda_decode_stream()>>>((float *)x->ptr, args);
    else
    dsv41_rope_kernel<<<(unsigned)(((uint64_t)heads * rows + 7u) / 8u), 256, 0, cuda_decode_stream()>>>(
        (float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                   uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1u, compressed, inverse);
}

__global__ static void dsv41_engram_kernel(float *residual, const float *kv,
                                           const float *qw, const float *kw,
                                           const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t row = blockIdx.x, head = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    if (mask && !mask[row]) return;
    const uint64_t offset = ((uint64_t)row * 4u + head) * width;
    const uint64_t key = ((uint64_t)row * 5u + head) * width;
    const uint64_t value = ((uint64_t)row * 5u + 4u) * width;
    float h2 = 0, k2 = 0, dot = 0;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i], k = dsv41_bf16(kv[key + i]);
        const uint32_t wi = head * width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (qw[wi] * kw[wi]) * k;
    }
    h2 = dsv41_broadcast(warp_sum_f32(h2));
    k2 = dsv41_broadcast(warp_sum_f32(k2));
    dot = dsv41_broadcast(warp_sum_f32(dot)) *
        rsqrtf(h2 / width + eps) * rsqrtf(k2 / width + eps) * rsqrtf(float(width));
    const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    for (uint32_t i = lane; i < width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] + gate * dsv41_bf16(kv[value + i]));
}

/* K1: dsv41_engram_kernel with C iterations of each lane's loads in flight at once: the same per-lane order, the
 * same expressions and warp trees (fuzzed: 0 of 4.1M values differ), 52 -> 18 us per Engram layer. */
template <uint32_t C>
__global__ static void dsv41_engram_batch_kernel(float *residual, const float *kv,
                                                 const float *qw, const float *kw,
                                                 const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t row = blockIdx.x, head = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    if (mask && !mask[row]) return;
    const uint64_t offset = ((uint64_t)row * 4u + head) * width;
    const uint64_t key = ((uint64_t)row * 5u + head) * width;
    const uint64_t value = ((uint64_t)row * 5u + 4u) * width;
    float h2 = 0, k2 = 0, dot = 0;
    uint32_t i = lane;
    for (; i + 32u * (C - 1u) < width; i += 32u * C) {
        float hv[C], kvv[C], qv[C], kwv[C];
#pragma unroll
        for (uint32_t c = 0; c < C; c++) {
            const uint32_t ii = i + 32u * c;
            hv[c] = residual[offset + ii];
            kvv[c] = kv[key + ii];
            qv[c] = qw[head * width + ii];
            kwv[c] = kw[head * width + ii];
        }
#pragma unroll
        for (uint32_t c = 0; c < C; c++) {
            const float h = hv[c], k = dsv41_bf16(kvv[c]);
            h2 += h * h;
            k2 += k * k;
            dot += h * (qv[c] * kwv[c]) * k;
        }
    }
    for (; i < width; i += 32u) {
        const float h = residual[offset + i], k = dsv41_bf16(kv[key + i]);
        const uint32_t wi = head * width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (qw[wi] * kw[wi]) * k;
    }
    h2 = dsv41_broadcast(warp_sum_f32(h2));
    k2 = dsv41_broadcast(warp_sum_f32(k2));
    dot = dsv41_broadcast(warp_sum_f32(dot)) *
        rsqrtf(h2 / width + eps) * rsqrtf(k2 / width + eps) * rsqrtf(float(width));
    const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    i = lane;
    for (; i + 32u * (C - 1u) < width; i += 32u * C) {
        float rv[C], vv[C];
#pragma unroll
        for (uint32_t c = 0; c < C; c++) {
            rv[c] = residual[offset + i + 32u * c];
            vv[c] = kv[value + i + 32u * c];
        }
#pragma unroll
        for (uint32_t c = 0; c < C; c++)
            residual[offset + i + 32u * c] = dsv41_bf16(rv[c] + gate * dsv41_bf16(vv[c]));
    }
    for (; i < width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] + gate * dsv41_bf16(kv[value + i]));
}

/* K11: dsv41_engram_kernel's arithmetic on one block of 256 threads per (row, stream): the stream's residual, key and
 * qw*kw products land in shared memory in one round trip, warp 0 runs each lane's chains over i = lane + 32k in order
 * (the same expressions), and every thread applies the gate to its own elements. The product qw*kw is the kernel's own
 * first multiply, made once per element. */
__global__ static void __launch_bounds__(256) dsv41_engram_k11_kernel(float * __restrict__ residual,
        const float * __restrict__ kv, const float * __restrict__ qw, const float * __restrict__ kw,
        const uint8_t * __restrict__ mask, uint32_t width, float eps) {
    extern __shared__ float sm[];
    const uint32_t row = blockIdx.x >> 2u, head = blockIdx.x & 3u;
    if (mask && !mask[row]) return;
    float *sh = sm, *sk = sm + width, *sp = sm + 2u * width;
    __shared__ float gate_s;
    const uint64_t offset = ((uint64_t)row * 4u + head) * width;
    const uint64_t key = ((uint64_t)row * 5u + head) * width;
    const uint64_t value = ((uint64_t)row * 5u + 4u) * width;
    for (uint32_t i = threadIdx.x; i < width; i += 256u) {
        const uint32_t wi = head * width + i;
        sh[i] = residual[offset + i];
        sk[i] = kv[key + i];
        sp[i] = qw[wi] * kw[wi];
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t lane = threadIdx.x;
        float h2 = 0, k2 = 0, dot = 0;
        for (uint32_t i = lane; i < width; i += 32u) {
            const float hh = sh[i], k = dsv41_bf16(sk[i]);
            h2 += hh * hh;
            k2 += k * k;
            dot += hh * sp[i] * k;
        }
        h2 = dsv41_broadcast(warp_sum_f32(h2));
        k2 = dsv41_broadcast(warp_sum_f32(k2));
        dot = dsv41_broadcast(warp_sum_f32(dot)) *
            rsqrtf(h2 / width + eps) * rsqrtf(k2 / width + eps) * rsqrtf(float(width));
        const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
        if (lane == 0) gate_s = gate;
    }
    __syncthreads();
    const float gate = gate_s;
    for (uint32_t i = threadIdx.x; i < width; i += 256u)
        residual[offset + i] = dsv41_bf16(sh[i] + gate * dsv41_bf16(kv[value + i]));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                         const ds4_gpu_tensor *qw, const ds4_gpu_tensor *kw,
                                         const ds4_gpu_tensor *mask, uint32_t width,
                                         uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || rows > INT32_MAX || !isfinite(eps) || eps <= 0 ||
        count > UINT64_MAX / 5u || !dsv41_has_floats(residual, count * 4u) ||
        !dsv41_has_floats(kv, count * 5u) || !dsv41_has_floats(qw, (uint64_t)width * 4u) ||
        !dsv41_has_floats(kw, (uint64_t)width * 4u) || (mask && mask->bytes < rows)) return 0;
    static int k11 = -1;
    if (k11 < 0) { const char *e = getenv("DS4_CUDA_K11"); k11 = !(e && !strcmp(e, "0")); }
    const size_t k11_shm = (size_t)3u * width * sizeof(float);
    if (k11 && cuda_kfast() && rows <= 16u && k11_shm <= 96u * 1024u) {   /* K11 */
        static int ready[DS4_MAX_GPUS];
        int dev = -1;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < DS4_MAX_GPUS) {
            if (!ready[dev])
                ready[dev] = cudaFuncSetAttribute(dsv41_engram_k11_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  96 * 1024) == cudaSuccess ? 1 : -1;
            if (ready[dev] == 1) {
                dsv41_engram_k11_kernel<<<rows * 4u, 256, k11_shm, cuda_decode_stream()>>>((float *)residual->ptr,
                    (const float *)kv->ptr, (const float *)qw->ptr, (const float *)kw->ptr,
                    mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
                return cuda_ok(cudaGetLastError(), "V4.1 Engram gate (K11)");
            }
        }
        (void)cudaGetLastError();
    }
    if (cuda_kfast())   /* K1 */
        dsv41_engram_batch_kernel<16><<<rows, 128, 0, cuda_decode_stream()>>>((float *)residual->ptr,
            (const float *)kv->ptr, (const float *)qw->ptr, (const float *)kw->ptr,
            mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    else
    dsv41_engram_kernel<<<rows, 128, 0, cuda_decode_stream()>>>((float *)residual->ptr,
        (const float *)kv->ptr, (const float *)qw->ptr, (const float *)kw->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void dsv41_pool_kernel(float *out, const float *kv, const float *scores,
                                         const float *prev_kv, const float *prev_scores,
                                         uint32_t width, uint32_t pairs, uint32_t tail) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)width * pairs) return;
    const uint32_t col = i % width;
    const int64_t row = (int64_t)(i / width) * 2 - tail;
    const uint64_t b = (uint64_t)(row + 1) * width + col;
    const float ka = row < 0 ? prev_kv[col] : kv[(uint64_t)row * width + col];
    const float sa = row < 0 ? prev_scores[col] : scores[(uint64_t)row * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak), eb = expf(sb - peak);
    out[i] = dsv41_bf16(__fdiv_rn(__fadd_rn(__fmul_rn(ka, ea), __fmul_rn(kv[b], eb)), ea + eb));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores, ds4_gpu_tensor *prev_kv,
                                   ds4_gpu_tensor *prev_scores, uint32_t width,
                                   uint32_t rows, uint32_t start) {
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || rows > UINT32_MAX - start ||
        !dsv41_has_floats(kv, count) || !dsv41_has_floats(scores, count) ||
        !dsv41_has_floats(prev_kv, width) || !dsv41_has_floats(prev_scores, width) ||
        (pairs && !dsv41_has_floats(out, (uint64_t)width * pairs)) ||
        ((uint64_t)width * pairs + 255u) / 256u > INT32_MAX) return 0;
    if (pairs) {
        dsv41_pool_kernel<<<(unsigned)(((uint64_t)width * pairs + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)prev_kv->ptr, (const float *)prev_scores->ptr, width, pairs, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 pair pooling")) return 0;
    }
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    const uint64_t bytes = (uint64_t)width * sizeof(float);
    return last_even < start ||
        (ds4_gpu_tensor_copy(prev_kv, 0, kv, (last_even - start) * bytes, bytes) &&
         ds4_gpu_tensor_copy(prev_scores, 0, scores, (last_even - start) * bytes, bytes));
}

__global__ static void dsv41_candidates_kernel(float *out, const float *scores,
                                               const float *mask, uint32_t width,
                                               uint32_t rows, uint32_t start, uint32_t ratio,
                                               uint32_t mask_stride) {
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)out_width * rows) return;
    const uint32_t row = i / out_width, col = i % out_width;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (mask) {
        out[i] = col < visible && mask[(uint64_t)row * mask_stride + col / 8u] == 0.0f ? scores[i] : -INFINITY;
    } else {
        float best = -INFINITY;
        for (uint32_t j = col * 8u; j < min(visible, (col + 1u) * 8u); j++)
            best = fmaxf(best, scores[(uint64_t)row * width + j]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[i] = best;
    }
}

static int dsv41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                            const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
                            uint32_t start, uint32_t ratio, uint32_t mask_stride) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint64_t count = (uint64_t)(mask ? width : blocks) * rows;
    if (!mask_stride) mask_stride = blocks;
    if (mask_stride < blocks ||
        !dsv41_has_floats(scores, (uint64_t)width * rows) || !dsv41_has_floats(out, count) ||
        (mask && !dsv41_has_floats(mask, (uint64_t)mask_stride * (rows - 1u) + blocks)) ||
        (count + 255u) / 256u > INT32_MAX) return 0;
    dsv41_candidates_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)scores->ptr, mask ? (const float *)mask->ptr : NULL,
        width, rows, start, ratio, mask_stride);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse candidates");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return dsv41_candidates(out, scores, NULL, width, rows, start, ratio, 0);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio,
                                              uint32_t mask_stride) {
    return mask && dsv41_candidates(scores, scores, mask, width, rows, start, ratio, mask_stride);
}

extern "C" int ds4_gpu_dsv41_candidate_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *blocks,
                                                  uint32_t width, uint32_t rows, uint32_t start) {
    if (!rows || start > UINT32_MAX - 8u || rows > UINT32_MAX - 8u - start ||
        (start + rows + 7u) / 8u > width || (start + 8u) / 8u <= 2048u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 2048u) ||
        !dsv41_has_floats(blocks, (uint64_t)rows * width)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 8u) / 8u;
        ds4_gpu_tensor src = *blocks, dst = *selected;
        src.ptr = (float *)src.ptr + (uint64_t)row * width;
        src.bytes = (uint64_t)visible * sizeof(float);
        dst.ptr = (uint32_t *)dst.ptr + (uint64_t)row * 2048u;
        dst.bytes = 2048u * sizeof(uint32_t);
        if (!ds4_gpu_indexer_topk_tensor(&dst, &src, visible, 1u, 2048u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_candidate_mask_batch(ds4_gpu_tensor *mask, const ds4_gpu_tensor *selected,
                                                  uint32_t width, uint32_t rows, uint32_t mask_stride) {
    if (!rows || mask_stride < width ||
        !dsv41_has_floats(selected, (uint64_t)rows * 2048u) ||
        !dsv41_has_floats(mask, (uint64_t)mask_stride * (rows - 1u) + width)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor src = *selected, dst = *mask;
        src.ptr = (uint32_t *)src.ptr + (uint64_t)row * 2048u;
        src.bytes = 2048u * sizeof(uint32_t);
        dst.ptr = (float *)dst.ptr + (uint64_t)row * mask_stride;
        dst.bytes = (uint64_t)width * sizeof(float);
        if (!ds4_gpu_dsv4_topk_mask_tensor(&dst, &src, width, 1u, 2048u)) return 0;
    }
    return 1;
}

__global__ static void dsv41_carry_kernel(uint32_t *packed, float *plain,
                                          uint32_t width, uint32_t rows,
                                          uint32_t words, uint32_t format, bool pack) {
    const uint32_t lanes = ((width + 31u) / 32u) * 32u;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = i / lanes, col = i % lanes;
    if (row >= rows) return;
    if (format == DS4_V41_CARRY_BF16) {
        if (col >= width) return;
        uint16_t *p = (uint16_t *)(packed + (uint64_t)row * words);
        if (pack) p[col] = __float_as_uint(plain[(uint64_t)row * width + col]) >> 16u;
        else plain[(uint64_t)row * width + col] = __uint_as_float((uint32_t)p[col] << 16u);
    } else if (pack) {
        const bool allowed = col < width && plain[(uint64_t)row * width + col] == 0.0f;
#ifdef __HIP_PLATFORM_AMD__
        const uint32_t bits = (uint32_t)(__ballot(allowed) >> ((threadIdx.x % warpSize) / 32u * 32u));
#else
        const uint32_t bits = __ballot_sync(0xffffffffu, allowed);
#endif
        if (!(col & 31u)) packed[(uint64_t)row * words + col / 32u] = bits;
    } else if (col < width) {
        plain[(uint64_t)row * width + col] = packed[(uint64_t)row * words + col / 32u] &
            (1u << (col & 31u)) ? 0.0f : -INFINITY;
    }
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t offset,
                                        ds4_gpu_tensor *plain, uint32_t width,
                                        uint32_t rows, uint32_t format, bool pack) {
    if (!width || width > UINT32_MAX - 31u || !rows || rows > UINT32_MAX - offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1u) / 2u : (width + 31u) / 32u;
    const uint64_t count = (uint64_t)((width + 31u) / 32u) * 32u * rows;
    if (!dsv41_has_floats(packed, (uint64_t)(offset + rows) * words) ||
        !dsv41_has_floats(plain, (uint64_t)rows * width) || (count + 255u) / 256u > INT32_MAX) return 0;
    dsv41_carry_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (uint32_t *)packed->ptr + (uint64_t)offset * words, (float *)plain->ptr,
        width, rows, words, format, pack);
    return cuda_ok(cudaGetLastError(), "V4.1 compact carry");
}

__global__ static void dsv41_gather_kernel(float *out, const float *source,
                                          const int32_t *ids, uint32_t source_rows) {
    const int32_t id = ids[blockIdx.x];
    for (uint32_t col = threadIdx.x; col < 512u; col += blockDim.x)
        out[(uint64_t)blockIdx.x * 512u + col] = id >= 0 && (uint32_t)id < source_rows ?
            source[(uint64_t)id * 512u + col] : NAN;
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u || selected_rows > source_rows ||
        !dsv41_has_floats(source, (uint64_t)source_rows * 512u) ||
        !dsv41_has_floats(out, (uint64_t)selected_rows * 512u) || !dsv41_has_floats(ids, selected_rows)) return 0;
    dsv41_gather_kernel<<<selected_rows, 256, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)source->ptr, (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

/* A warp reuses each query across adjacent keys. Each key keeps its own
 * accumulator and the same 128-element reduction tree. Grouped launches
 * cover a single fully visible row, so only the final group can be short. */
template<int keys_per_warp, bool masked>
__global__ static void dsv41_index_scores_kernel(float *scores, const float *q,
        const float *weights, const float *keys, const float *block_mask, uint32_t width,
        uint32_t start, uint32_t ratio) {
    const uint32_t first = (blockIdx.x * 8u + threadIdx.x / 32u) * keys_per_warp;
    const uint32_t row = blockIdx.y, lane = threadIdx.x & 31u;
    if (first >= width) return;
    scores += (uint64_t)row * width + first;
    if (first >= (start + row + 1u) / ratio || (masked && block_mask[first / 8u] != 0.0f)) {
        if (!lane) {
            #pragma unroll
            for (int i = 0; i < keys_per_warp; i++)
                if (first + i < width) scores[i] = -INFINITY;
        }
        return;
    }
    float4 k[keys_per_warp];
    float total[keys_per_warp] = {};
    #pragma unroll
    for (int i = 0; i < keys_per_warp; i++) {
        const uint32_t key = keys_per_warp == 1 ? first : min(first + i, width - 1u);
        const float *v = keys + (uint64_t)key * 128u + lane;
        k[i] = make_float4(v[0], v[32], v[64], v[96]);
    }
    for (uint32_t h = 0; h < 32u; h++) {
        const float *x = q + ((uint64_t)row * 32u + h) * 128u + lane;
        const float q0 = x[0], q1 = x[32], q2 = x[64], q3 = x[96];
        const float weight = weights[(uint64_t)row * 32u + h];
        #pragma unroll
        for (int i = 0; i < keys_per_warp; i++) {
            const float a = __fadd_rn(__fmul_rn(q0, k[i].x), __fmul_rn(q2, k[i].z));
            const float b = __fadd_rn(__fmul_rn(q1, k[i].y), __fmul_rn(q3, k[i].w));
            const float dot = warp_sum_f32(__fadd_rn(a, b));
            total[i] += fmaxf(dot, 0.0f) * weight;
        }
    }
    if (!lane) {
        #pragma unroll
        for (int i = 0; i < keys_per_warp; i++)
            if (first + i < width) scores[i] = total[i] * (1.0f / 64.0f);
    }
}

template<int keys_per_warp, bool masked>
static void dsv41_launch_index_scores(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys, const ds4_gpu_tensor *block_mask,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    const uint32_t keys_per_block = 8u * keys_per_warp;
    dsv41_index_scores_kernel<keys_per_warp, masked>
        <<<dim3((width + keys_per_block - 1u) / keys_per_block, rows), 256, 0, cuda_decode_stream()>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)keys->ptr, masked ? (const float *)block_mask->ptr : NULL, width, start, ratio);
}

static bool dsv41_group_index_keys(uint32_t width) {
#ifdef __HIP_PLATFORM_AMD__
    (void)width;
    return false;
#else
    return width >= 8192u && ds4_gpu_device_is_spark() && !getenv("DS4_CUDA_DISABLE_V41_GROUP_INDEX_KEYS");
#endif
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                                   const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                                   const ds4_gpu_tensor *keys, uint32_t source_rows,
                                                   uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || source_rows > INT32_MAX || !rows ||
        rows > 65535u || rows > UINT32_MAX - start || (start + rows) / ratio > source_rows ||
        !dsv41_has_floats(scores, (uint64_t)source_rows * rows) ||
        !dsv41_has_floats(q, (uint64_t)rows * 32u * 128u) ||
        !dsv41_has_floats(weights, (uint64_t)rows * 32u) ||
        !dsv41_has_floats(keys, (uint64_t)source_rows * 128u)) return 0;
    /* FP4 with power-of-two scales can exceed FP16's exponent range.
     * Do not pass these already-quantized values through the GLM FP16 tile. */
    if (rows == 1 && (start + 1u) / ratio == source_rows && dsv41_group_index_keys(source_rows))
        dsv41_launch_index_scores<8, false>(scores, q, weights, keys, NULL, source_rows, rows, start, ratio);
    else
        dsv41_launch_index_scores<1, false>(scores, q, weights, keys, NULL, source_rows, rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 index scores");
}

/* The rejection sentinel and original key positions match candidate_filter. */
extern "C" int ds4_gpu_dsv41_indexer_scores_masked_one(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
        const ds4_gpu_tensor *block_mask, uint32_t width) {
    if (!width || width > INT32_MAX) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    if (!dsv41_has_floats(scores, width) || !dsv41_has_floats(q, 32u * 128u) ||
        !dsv41_has_floats(weights, 32u) || !dsv41_has_floats(keys, (uint64_t)width * 128u) ||
        !dsv41_has_floats(block_mask, blocks)) return 0;
    if (dsv41_group_index_keys(width))
        dsv41_launch_index_scores<8, true>(scores, q, weights, keys, block_mask, width, 1, width - 1u, 1u);
    else
        dsv41_launch_index_scores<1, true>(scores, q, weights, keys, block_mask, width, 1, width - 1u, 1u);
    return cuda_ok(cudaGetLastError(), "V4.1 masked index scores");
}

#ifndef __HIP_PLATFORM_AMD__
/* K3 (21 Nov 2026): fewer launches per decode layer, each fused kernel the arithmetic of the launches it replaces
 * (fuzzed against them: 0 of 9.7M values differ). DS4_CUDA_KFAST=0 restores the old launches. */
/* K3 split + weighted sum + norm: warps 0-7 make x = bf16(sum_h res_h * pre_h) and its BF16 norm (the arithmetic of
 * hc_weighted_sum_kernel<true> and rms_norm_weight_kernel<true>, the block tree on named barrier 1); warp 8 runs the
 * Sinkhorn split (hc4_split_one, as hc_split_sinkhorn_kernel's thread 0), which the sum does not read. */
__device__ __forceinline__ static float block_sum_f32_256_bar1(float v, float *partial) {
    const uint32_t t = threadIdx.x;
    partial[t] = v;
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (t < 32u) {
        v = ((partial[t] + partial[t + 128u]) +
             (partial[t + 64u] + partial[t + 192u])) +
            ((partial[t + 32u] + partial[t + 160u]) +
             (partial[t + 96u] + partial[t + 224u]));
        for (uint32_t s = 16u; s; s >>= 1u)
            v += __shfl_down_sync(0xffffffffu, v, s);
        if (t == 0u) partial[0] = v;
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    return partial[0];
}
/* K7: hc4_split_one on lanes 0-15 of a warp, lane = r * 4 + col of the comb matrix (lanes 0-7 also make the pre and
 * post weights). Every value is the operation hc4_split_one does on the same operands in its order: a row's or
 * column's four terms are gathered by shuffles and added in hc4_split_one's order, and the four lanes of that row
 * (column) divide by the same sum. Lanes 16-31 return at once. */
__device__ __forceinline__ static float hc4_gather_sum(float s, float c, uint32_t l0, uint32_t step) {
    const unsigned mask = 0x0000ffffu;
    const float v0 = __shfl_sync(mask, c, l0), v1 = __shfl_sync(mask, c, l0 + step),
                v2 = __shfl_sync(mask, c, l0 + 2u * step), v3 = __shfl_sync(mask, c, l0 + 3u * step);
    s += v0;
    s += v1;
    s += v2;
    s += v3;
    return s;
}
__device__ __forceinline__ static void hc4_split_lanes16(float *out, const float *mix, const float *scale,
        const float *base, uint32_t sinkhorn_iters, float epsv) {
    const uint32_t lane = threadIdx.x & 31u;
    if (lane >= 16u) return;
    const unsigned mask = 0x0000ffffu;
    const uint32_t row0 = lane & ~3u, col0 = lane & 3u;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (lane < 4u) {
        float z = mix[lane] * pre_scale + base[lane];
        out[lane] = 1.0f / (1.0f + expf(-z)) + epsv;
    } else if (lane < 8u) {
        float z = mix[lane] * post_scale + base[lane];
        out[lane] = 2.0f / (1.0f + expf(-z));
    }
    float c = mix[8 + lane] * comb_scale + base[8 + lane];
    {
        const float v0 = __shfl_sync(mask, c, row0), v1 = __shfl_sync(mask, c, row0 + 1u),
                    v2 = __shfl_sync(mask, c, row0 + 2u), v3 = __shfl_sync(mask, c, row0 + 3u);
        float m = -INFINITY;
        m = fmaxf(m, v0);
        m = fmaxf(m, v1);
        m = fmaxf(m, v2);
        m = fmaxf(m, v3);
        c = expf(c - m);
        const float s = hc4_gather_sum(0.0f, c, row0, 1u);
        c = c / s + epsv;
    }
    {
        const float s = hc4_gather_sum(epsv, c, col0, 4u);
        c /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        const float sr = hc4_gather_sum(epsv, c, row0, 1u);
        c /= sr;
        const float sc = hc4_gather_sum(epsv, c, col0, 4u);
        c /= sc;
    }
    out[8 + lane] = c;
}
static int dsv41_k7_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DS4_CUDA_K7"); on = !(e && !strcmp(e, "0")); }
    return on;
}

template <uint32_t PER>
__global__ static void __launch_bounds__(288) dsv41_hc_input_kernel(float *x_out, float *norm_out, float *split,
        const float *mix, const float *residual_hc, const float *pre, const float *scale, const float *base,
        const float *norm_w, uint32_t n_embd, uint32_t sinkhorn_iters, float hc_eps, float rms_eps, int k7) {
    if (threadIdx.x >= 256u) {
        if (k7) hc4_split_lanes16(split, mix, scale, base, sinkhorn_iters, hc_eps);
        else if (threadIdx.x == 256u) hc4_split_one(split, mix, scale, base, sinkhorn_iters, hc_eps);
        return;
    }
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    float v[PER], wv[PER];
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t i = threadIdx.x + j * 256u;
        float acc = 0.0f;   /* hc_weighted_sum_kernel's chain: FFMA(res_h, w_h, acc) from +0 */
        acc = __fmaf_rn(residual_hc[i], w0, acc);
        acc = __fmaf_rn(residual_hc[(uint64_t)n_embd + i], w1, acc);
        acc = __fmaf_rn(residual_hc[2ull * n_embd + i], w2, acc);
        acc = __fmaf_rn(residual_hc[3ull * n_embd + i], w3, acc);
        v[j] = dsv41_bf16(acc);
        x_out[i] = v[j];
    }
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) wv[j] = norm_w[threadIdx.x + j * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) sum += v[j] * v[j];
    __shared__ float partial[256];
    sum = block_sum_f32_256_bar1(sum, partial);
    const float scl = rsqrtf(sum / (float)n_embd + rms_eps);
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) norm_out[threadIdx.x + j * 256u] = dsv41_bf16(v[j] * scl * wv[j]);
}

/* K8: the kernel above on 8 + 1 blocks (see ds4_gpu_dsv41_hc_input). */
struct dsv41_k8_sync { unsigned arrive, gen; float scale, pad; float partial[256]; };
static dsv41_k8_sync *g_dsv41_k8_sync[DS4_MAX_GPUS];
template <uint32_t PER, uint32_t G>
__global__ static void __launch_bounds__(256) dsv41_hc_input_k8_kernel(float * __restrict__ x_out,
        float * __restrict__ norm_out, float * __restrict__ split, const float * __restrict__ mix,
        const float * __restrict__ residual_hc, const float * __restrict__ pre, const float * __restrict__ scale,
        const float * __restrict__ base, const float * __restrict__ norm_w, uint32_t n_embd, uint32_t sinkhorn_iters,
        float hc_eps, float rms_eps, dsv41_k8_sync * __restrict__ sy) {
    if (blockIdx.x == G) {   /* the Sinkhorn split, which the rest does not read */
        if (threadIdx.x < 32u) hc4_split_lanes16(split, mix, scale, base, sinkhorn_iters, hc_eps);
        return;
    }
    constexpr uint32_t T = 256u / G, ROWS = 256u / T;
    __shared__ float s[PER][T];
    __shared__ unsigned my_gen, last_s;
    __shared__ float scale_s;
    const uint32_t lane = threadIdx.x % T, r = threadIdx.x / T, t0 = blockIdx.x * T;
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    for (uint32_t j = r; j < PER; j += ROWS) {   /* the one-block kernel's element (t0 + lane, j) */
        const uint32_t i = t0 + lane + 256u * j;
        float acc = 0.0f;
        acc = __fmaf_rn(residual_hc[i], w0, acc);
        acc = __fmaf_rn(residual_hc[(uint64_t)n_embd + i], w1, acc);
        acc = __fmaf_rn(residual_hc[2ull * n_embd + i], w2, acc);
        acc = __fmaf_rn(residual_hc[3ull * n_embd + i], w3, acc);
        const float v = dsv41_bf16(acc);
        x_out[i] = v;
        s[j][lane] = v;
    }
    __syncthreads();
    if (threadIdx.x < T) {   /* that thread's sum of squares, in its order */
        float sum = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < PER; j++) sum += s[j][threadIdx.x] * s[j][threadIdx.x];
        sy->partial[t0 + threadIdx.x] = sum;
    }
    if (threadIdx.x == 0) my_gen = *(volatile unsigned *)&sy->gen;
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) last_s = atomicAdd(&sy->arrive, 1u) == G - 1u;
    __syncthreads();
    if (last_s) {   /* block_sum_f32_256_bar1's tree over the 256 sums */
        __shared__ float partial[256];
        partial[threadIdx.x] = *(volatile float *)&sy->partial[threadIdx.x];
        __syncthreads();
        if (threadIdx.x < 32u) {
            const uint32_t t = threadIdx.x;
            float v = ((partial[t] + partial[t + 128u]) + (partial[t + 64u] + partial[t + 192u])) +
                      ((partial[t + 32u] + partial[t + 160u]) + (partial[t + 96u] + partial[t + 224u]));
            for (uint32_t s2 = 16u; s2; s2 >>= 1u) v += __shfl_down_sync(0xffffffffu, v, s2);
            if (t == 0u) partial[0] = v;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const float scl = rsqrtf(partial[0] / (float)n_embd + rms_eps);
            sy->scale = scl;
            sy->arrive = 0u;
            __threadfence();
            atomicAdd(&sy->gen, 1u);
            scale_s = scl;
        }
    } else if (threadIdx.x == 0) {
        while (*(volatile unsigned *)&sy->gen == my_gen) { }
        __threadfence();
        scale_s = *(volatile float *)&sy->scale;
    }
    __syncthreads();
    const float scl = scale_s;
    for (uint32_t j = r; j < PER; j += ROWS) {
        const uint32_t i = t0 + lane + 256u * j;
        norm_out[i] = dsv41_bf16(s[j][lane] * scl * norm_w[i]);
    }
}
static int dsv41_k8_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DS4_CUDA_K8"); on = !(e && !strcmp(e, "0")); }
    return on;
}
/* The device's K8 state; made outside a capture (the first, eager pass of every decode island). */
static dsv41_k8_sync *dsv41_k8_state(void) {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= DS4_MAX_GPUS) { (void)cudaGetLastError(); return NULL; }
    if (g_dsv41_k8_sync[dev]) return g_dsv41_k8_sync[dev];
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cuda_decode_stream(), &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        return NULL;
    }
    dsv41_k8_sync *p = NULL;
    if (cudaMalloc((void **)&p, sizeof(*p)) != cudaSuccess || cudaMemset(p, 0, sizeof(*p)) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        (void)cudaGetLastError();
        if (p) (void)cudaFree(p);
        return NULL;
    }
    g_dsv41_k8_sync[dev] = p;
    return p;
}

/* K3 add + BF16 + HC expansion + BF16 (add_kernel, dsv41_bf16_kernel, hc_expand_kernel<false>, dsv41_bf16_kernel)
 * in one launch, one thread per dimension for the four streams; the rounded sum lands in block_out as before.
 * hc_expand_kernel's instructions: FMUL(block, post), then FFMA(comb, res, acc) per source stream. One token. */
__global__ static void dsv41_expand_add_bf16_kernel(float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *post, const float *comb, uint32_t n_embd) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float block_v = dsv41_bf16(block_out[d] + block_add[d]);
    block_out[d] = block_v;
    float res[4];
#pragma unroll
    for (uint32_t src_hc = 0; src_hc < 4u; src_hc++) res[src_hc] = residual_hc[(uint64_t)src_hc * n_embd + d];
#pragma unroll
    for (uint32_t dst_hc = 0; dst_hc < 4u; dst_hc++) {
        float acc = __fmul_rn(block_v, post[dst_hc]);
#pragma unroll
        for (uint32_t src_hc = 0; src_hc < 4u; src_hc++) acc = __fmaf_rn(comb[dst_hc + src_hc * 4u], res[src_hc], acc);
        out_hc[(uint64_t)dst_hc * n_embd + d] = dsv41_bf16(acc);
    }
}
/* K3 the q_a and kv projections of one input: matmul_q8_0_preq_warp8_bf16_kernel's arithmetic for the rows of
 * both weights in one launch (rows_a of wa, then rows_b of wb), on one quantization of x. */
__global__ static void matmul_q8_0_pair_warp8_bf16_kernel(float *out_a, float *out_b, const unsigned char *wa,
        const unsigned char *wb, const int8_t *xq, const float *xscale, uint64_t in_dim, uint32_t rows_a,
        uint32_t rows_b, uint64_t blocks, int use_dp4a) {
    const uint64_t r = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (r >= (uint64_t)rows_a + rows_b) return;
    const bool second = r >= rows_a;
    const uint64_t row = second ? r - rows_a : r;
    const unsigned char *wr = (second ? wb : wa) + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) (second ? out_b : out_a)[row] = dsv41_bf16(acc);
}
/* K3 two BF16 weighted norms in one launch (block 0: PA * 256 wide, block 1: PB * 256 wide), each
 * rms_norm_weight_reg_kernel<true>'s arithmetic. */
template <uint32_t PER>
__device__ __forceinline__ static void rms_norm_weight_bf16_row(float *orow, const float *xr, const float *w, uint32_t n, float eps) {
    float v[PER], wv[PER];
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) v[j] = xr[threadIdx.x + j * 256u];
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) wv[j] = w[threadIdx.x + j * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) sum += v[j] * v[j];
    __shared__ float partial[256];
    sum = block_sum_f32_256(sum, partial);
    const float scale = rsqrtf(sum / (float)n + eps);
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) orow[threadIdx.x + j * 256u] = dsv41_bf16(v[j] * scale * wv[j]);
}
template <uint32_t PA, uint32_t PB>
__global__ static void __launch_bounds__(256) rms_norm_weight_bf16_pair_kernel(float *out_a, const float *x_a, const float *w_a,
        uint32_t n_a, float *out_b, const float *x_b, const float *w_b, uint32_t n_b, float eps) {
    if (blockIdx.x == 0) rms_norm_weight_bf16_row<PA>(out_a, x_a, w_a, n_a, eps);
    else rms_norm_weight_bf16_row<PB>(out_b, x_b, w_b, n_b, eps);
}
/* K3 BF16 rounding in place + Q8_0 quantization of the rounded values (dsv41_bf16_kernel then
 * quantize_q8_0_f32_kernel), one 32-thread block per Q8_0 block. */
__global__ static void dsv41_bf16_quantize_q8_0_kernel(int8_t *xq, float *xscale, float *x, uint64_t in_dim, uint64_t blocks) {
    uint64_t b = blockIdx.x;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    float *xr = x + i0;
    float xv = 0.0f;
    if (threadIdx.x < bn) { xv = dsv41_bf16(xr[threadIdx.x]); xr[threadIdx.x] = xv; }
    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xv);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[b] = d;
    int8_t *dst = xq + b * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xv * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}


/* The single-row Q8_0 projections these replace take the prequant + warp path only when no derived (repacked)
 * weight can serve them and the K3 f32-activation mode is off; elsewhere the entry points decline. */
static bool dsv41_k3_q8_ok(void) {
    return cuda_kfast() && !g_dsv41_shared.active && !g_k3_f32_act && g_derived_ranges.empty();
}

static void *dsv41_k3_prequant(int tier, uint64_t blocks, int8_t **xq, float **xs) {
    const uint64_t scale_offset = (blocks * 32u + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc_on(tier, scale_offset + blocks * sizeof(float), "q8_0 prequant");
    if (!tmp) return NULL;
    *xq = (int8_t *)tmp;
    *xs = (float *)((char *)tmp + scale_offset);
    return tmp;
}

extern "C" int ds4_gpu_dsv41_shared_swiglu(ds4_gpu_tensor *mid, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint32_t in_dim,
        uint32_t out_dim, const ds4_gpu_tensor *x, float clamp, uint32_t rows) {
    if (!dsv41_k3_q8_ok() || rows != 1u || !model_map || !in_dim || !out_dim || (in_dim & 31u) ||
        getenv("DS4_CUDA_DISABLE_V41_SHARED_FUSION") || !dsv41_has_floats(x, in_dim) ||
        !dsv41_has_floats(gate, out_dim) || !dsv41_has_floats(up, out_dim) || !dsv41_has_floats(mid, out_dim)) return 0;
    const uint64_t blocks = in_dim / 32u, bytes = (uint64_t)out_dim * blocks * 34u;
    if (gate_offset > model_size || bytes > model_size - gate_offset ||
        up_offset > model_size || bytes > model_size - up_offset) return 0;
    const int tier = ds4_tensor_device_idx(mid);
    const auto *wg = (const unsigned char *)cuda_resolve_weight_ptr(model_map, gate_offset, bytes, tier, "q8_0");
    const auto *wu = (const unsigned char *)cuda_resolve_weight_ptr(model_map, up_offset, bytes, tier, "q8_0");
    int8_t *xq; float *xs;
    if (!wg || !wu || !dsv41_k3_prequant(tier, blocks, &xq, &xs)) return 0;
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, cuda_decode_stream()>>>(xq, xs, (const float *)x->ptr,
                                                                                in_dim, blocks);
    dsv41_shared_gate_up_kernel<<<(out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, wg, wu, xq, xs, (uint32_t)blocks, out_dim, clamp,
        cuda_q8_use_dp4a(), NULL);
    return cuda_ok(cudaGetLastError(), "V4.1 shared gate/up/SwiGLU");
}

extern "C" int ds4_gpu_dsv41_project_pair_q8(ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
        const void *model_map, uint64_t model_size, uint64_t offset_a, uint64_t offset_b, uint32_t in_dim,
        uint32_t out_a_dim, uint32_t out_b_dim, const ds4_gpu_tensor *x) {
    if (!dsv41_k3_q8_ok() || !model_map || !in_dim || (in_dim & 31u) || !out_a_dim || !out_b_dim ||
        !dsv41_has_floats(x, in_dim) || !dsv41_has_floats(out_a, out_a_dim) || !dsv41_has_floats(out_b, out_b_dim)) return 0;
    const int tier = ds4_tensor_device_idx(out_a);
    if (ds4_tensor_device_idx(out_b) != tier) return 0;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t bytes_a = (uint64_t)out_a_dim * blocks * 34u, bytes_b = (uint64_t)out_b_dim * blocks * 34u;
    if (offset_a > model_size || bytes_a > model_size - offset_a ||
        offset_b > model_size || bytes_b > model_size - offset_b) return 0;
    const auto *wa = (const unsigned char *)cuda_resolve_weight_ptr(model_map, offset_a, bytes_a, tier, "q8_0");
    const auto *wb = (const unsigned char *)cuda_resolve_weight_ptr(model_map, offset_b, bytes_b, tier, "q8_0");
    int8_t *xq; float *xs;
    if (!wa || !wb || !dsv41_k3_prequant(tier, blocks, &xq, &xs)) return 0;
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, cuda_decode_stream()>>>(xq, xs, (const float *)x->ptr,
                                                                                in_dim, blocks);
    matmul_q8_0_pair_warp8_bf16_kernel<<<(out_a_dim + out_b_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
        (float *)out_a->ptr, (float *)out_b->ptr, wa, wb, xq, xs, in_dim, out_a_dim, out_b_dim, blocks,
        cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(), "V4.1 projection pair");
}

extern "C" int ds4_gpu_dsv41_norm_pair_rows(ds4_gpu_tensor *out0, const ds4_gpu_tensor *x0, uint64_t weight0_offset,
        uint32_t n0, ds4_gpu_tensor *out1, const ds4_gpu_tensor *x1, uint64_t weight1_offset, uint32_t n1,
        const void *model_map, uint64_t model_size, float eps, uint32_t rows) {
    if (!cuda_kfast() || rows != 1u || n0 != 1280u || n1 != 512u || !model_map ||
        getenv("DS4_CUDA_DISABLE_V41_POINTWISE_FUSION") ||
        !dsv41_has_floats(out0, n0) || !dsv41_has_floats(x0, n0) || !dsv41_has_floats(out1, n1) ||
        !dsv41_has_floats(x1, n1) || weight0_offset > model_size || (uint64_t)n0 * 4u > model_size - weight0_offset ||
        weight1_offset > model_size || (uint64_t)n1 * 4u > model_size - weight1_offset) return 0;
    const float *w0 = (const float *)cuda_resolve_weight_ptr(model_map, weight0_offset, (uint64_t)n0 * 4u,
                                                             ds4_tensor_device_idx(out0), "V4.1 norm");
    const float *w1 = (const float *)cuda_resolve_weight_ptr(model_map, weight1_offset, (uint64_t)n1 * 4u,
                                                             ds4_tensor_device_idx(out1), "V4.1 norm");
    if (!w0 || !w1) return 0;
    rms_norm_weight_bf16_pair_kernel<5, 2><<<2, 256, 0, cuda_decode_stream()>>>(
        (float *)out0->ptr, (const float *)x0->ptr, w0, n0, (float *)out1->ptr, (const float *)x1->ptr, w1, n1, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 norm pair");
}

extern "C" int ds4_gpu_dsv41_hc_input(ds4_gpu_tensor *x, ds4_gpu_tensor *norm, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual, const ds4_gpu_tensor *pre,
        const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset,
        uint64_t norm_offset, uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float hc_eps, float norm_eps) {
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (!cuda_kfast() || n_hc != 4u || n_embd != 5120u || !model_map || !mix || !split ||
        getenv("DS4_CUDA_DISABLE_V41_POINTWISE_FUSION") ||
        mix->bytes / mix_bytes != 1u || split->bytes / mix_bytes < 1u ||
        !dsv41_has_floats(x, n_embd) || !dsv41_has_floats(norm, n_embd) || !dsv41_has_floats(residual, 4u * n_embd) ||
        !dsv41_has_floats(pre, 4u) || scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        norm_offset > model_size || (uint64_t)n_embd * 4u > model_size - norm_offset) return 0;
    const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset, 3ull * sizeof(float),
                                                                ds4_tensor_device_idx(split), "hc_scale");
    const float *base = (const float *)cuda_resolve_weight_ptr(model_map, base_offset, mix_bytes,
                                                               ds4_tensor_device_idx(split), "hc_base");
    const float *nw = (const float *)cuda_resolve_weight_ptr(model_map, norm_offset, (uint64_t)n_embd * 4u,
                                                             ds4_tensor_device_idx(norm), "V4.1 norm");
    if (!scale || !base || !nw) return 0;
    dsv41_k8_sync *k8 = dsv41_k8_on() && dsv41_k7_on() ? dsv41_k8_state() : NULL;
    if (k8) {   /* K8 */
        dsv41_hc_input_k8_kernel<20, 8><<<9, 256, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)norm->ptr,
            (float *)split->ptr, (const float *)mix->ptr, (const float *)residual->ptr, (const float *)pre->ptr,
            scale, base, nw, n_embd, sinkhorn_iters, hc_eps, norm_eps, k8);
        return cuda_ok(cudaGetLastError(), "V4.1 HC input (K8)");
    }
    dsv41_hc_input_kernel<20><<<1, 288, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)norm->ptr,
        (float *)split->ptr, (const float *)mix->ptr, (const float *)residual->ptr, (const float *)pre->ptr,
        scale, base, nw, n_embd, sinkhorn_iters, hc_eps, norm_eps, dsv41_k7_on());
    return cuda_ok(cudaGetLastError(), "V4.1 HC input");
}
#endif

extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }
extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                          const ds4_gpu_tensor *, uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                   const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                   const ds4_gpu_tensor *, uint32_t, uint32_t,
                                                   uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
#ifdef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_project_pair_q8(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t,
                                             uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
                                             const ds4_gpu_tensor *) { return 0; }
#endif
#ifdef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_shared_swiglu(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *,
                                           uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                                           const ds4_gpu_tensor *, float, uint32_t) { return 0; }
#endif
extern "C" int ds4_gpu_dsv41_matmul_expand_rows(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t,
                                                uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t,
                                                uint32_t) { return 0; }
#ifdef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_norm_pair_rows(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t,
                                            ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t,
                                            const void *, uint64_t, float, uint32_t) { return 0; }
#endif
extern "C" int ds4_gpu_dsv41_rope_quantize(const ds4_gpu_tensor *, ds4_gpu_tensor *, uint64_t, uint32_t,
                                           uint32_t, bool, ds4_v41_activation_format) { return 0; }
extern "C" int ds4_gpu_dsv41_project_f16_bf16(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t,
                                              uint32_t, const ds4_gpu_tensor *) { return 0; }
extern "C" int ds4_gpu_dsv41_attention_decode(ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
                                              const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
                                              uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                              ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_project_q(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                       const ds4_gpu_tensor *, uint32_t, bool) { return 0; }
extern "C" int ds4_gpu_dsv41_attention_low(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                           uint32_t, ds4_gpu_tensor *, uint32_t, bool) { return 0; }
extern "C" int ds4_gpu_dsv41_hc_block_input_rows(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
                                                 ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                 const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                                                 uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_hc_project(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
                                        uint32_t, uint32_t, float) { return 0; }
extern "C" int ds4_gpu_dsv41_router_rows(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                         const void *, uint64_t, uint64_t, uint32_t, uint32_t, float,
                                         uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_matmul_f32_rows(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                             const ds4_gpu_tensor *, uint32_t) { return 0; }
#ifdef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_hc_input(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                      const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t,
                                      uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float,
                                      float) { return 0; }
#endif

extern "C" int ds4_gpu_dsv41_indexer_topk_tensor(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                 uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    return ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k);
}

#ifdef __HIP_PLATFORM_AMD__
__global__ static void dsv41_causal_scores_kernel(float *out, const float *scores,
        uint32_t width, uint32_t stride, uint32_t start, uint32_t ratio) {
    const uint32_t row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) out[(uint64_t)row * width + col] =
        col < (start + row + 1u) / ratio ? scores[(uint64_t)row * stride + col] : -INFINITY;
}
#endif

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                uint32_t width, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        (start + 1u) / ratio <= 512u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 512u) ||
        !dsv41_has_floats(scores, (uint64_t)rows * width)) return 0;
#ifdef __HIP_PLATFORM_AMD__
    if (rows > 1u) {
        if (rows > 65535u) return 0;
        /* The generic sorter already batches rows. Pack their causal prefixes
         * without changing the caller's scores or using its sort scratch. */
        const uint32_t packed_width = (start + rows) / ratio;
        ds4_gpu_tensor *causal = ds4_gpu_tensor_alloc((uint64_t)rows * packed_width * sizeof(float));
        if (!causal) return 0;
        dsv41_causal_scores_kernel<<<dim3((packed_width + 255u) / 256u, rows), 256>>>(
            (float *)causal->ptr, (const float *)scores->ptr, packed_width, width, start, ratio);
        const bool ok = cuda_ok(cudaGetLastError(), "V4.1 causal scores") &&
            ds4_gpu_indexer_topk_tensor(selected, causal, packed_width, rows, 512u);
        ds4_gpu_tensor_free(causal);
        return ok;
    }
#endif
    uint32_t row = 0;
#ifndef __HIP_PLATFORM_AMD__
    /* Batch independent causal rows, splitting only at a sort-width boundary. */
    while (row < rows && (start + row + 1u) / ratio <= 4096u) {
        const uint32_t visible = (start + row + 1u) / ratio;
        const uint32_t sort = visible <= 1024u ? 1024u : visible <= 2048u ? 2048u : 4096u;
        const uint32_t available = (sort + 1u) * ratio - 1u - (start + row);
        const uint32_t count = rows - row < available ? rows - row : available;
        uint32_t *dst = (uint32_t *)selected->ptr + (uint64_t)row * 512u;
        const float *src = (const float *)scores->ptr + (uint64_t)row * width;
        if (sort == 1024u)
            indexer_topk_pow2_kernel<1024, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        else if (sort == 2048u)
            indexer_topk_pow2_kernel<2048, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        else
            indexer_topk_pow2_kernel<4096, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal topk batch")) return 0;
        row += count;
    }
    if (rows - row >= 2u && getenv("DS4_CUDA_NO_TOPK_STREAM") == NULL) {
        indexer_topk_stream512_kernel<true><<<rows - row, 512>>>(
            (uint32_t *)selected->ptr + (uint64_t)row * 512u,
            (const float *)scores->ptr + (uint64_t)row * width,
            width, rows - row, 512u, start + row, ratio);
        return cuda_ok(cudaGetLastError(), "V4.1 wide causal topk batch");
    }
#endif
    for (; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        ds4_gpu_tensor src = *scores, dst = *selected;
        src.ptr = (float *)src.ptr + (uint64_t)row * width;
        src.bytes = (uint64_t)visible * sizeof(float);
        dst.ptr = (uint32_t *)dst.ptr + (uint64_t)row * 512u;
        dst.bytes = 512u * sizeof(uint32_t);
        if (!ds4_gpu_indexer_topk_tensor(&dst, &src, visible, 1u, 512u)) return 0;
    }
    return 1;
}

__global__ static void dsv41_indexer_all_kernel(int32_t *ids, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = i / 512u, col = i % 512u;
    if (row >= rows) return;
    const uint32_t visible = (start + row + 1u) / ratio;
    if (col < min(visible, 512u)) ids[i] = (int32_t)col;
}

extern "C" int ds4_gpu_dsv41_indexer_all_batch(ds4_gpu_tensor *selected, uint32_t rows,
                                               uint32_t start, uint32_t ratio) {
    if (!ratio || !rows || rows > UINT32_MAX - start || (start + rows) / ratio > 512u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 512u)) return 0;
    dsv41_indexer_all_kernel<<<(unsigned)(((uint64_t)rows * 512u + 255u) / 256u), 256, 0,
        cuda_decode_stream()>>>((int32_t *)selected->ptr, rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 indexer select all");
}

/* F16-G: groups of rows per cuBLAS GEMM where bitwise equal to one GEMM per row (checked per card and shape) */
__global__ static void f16g_diff_kernel(const float *a, const float *b, uint64_t n, unsigned int *diff) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(a[i]) != __float_as_uint(b[i])) atomicAdd(diff, 1u);
}

static int f16g_group(void) {
    static int g = -1;
    if (g < 0) { const char *e = getenv("DS4_F16_ROW_GROUP"); g = e && *e ? atoi(e) : 1; if (g < 1) g = 1; if (g > 64) g = 64; }
    return g;
}

static int f16g_gemm_h(cublasHandle_t h, const __half *w, const __half *x, float *y, uint32_t width, uint32_t outputs,
                       uint32_t n) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)outputs, (int)n,
        (int)width, &alpha, w, CUDA_R_16F, (int)width, x, CUDA_R_16F, (int)width, &beta, y, CUDA_R_32F, (int)outputs,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS;
}

/* ---- F16-P: one-row GEMMs issued by launcher threads ----------------------------------------------------------- */
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
struct f16p_job { const __half *w; const __half *x; float *y; uint32_t width, outputs, r0, r1; };
struct f16p_pool {
    int dev = -1, n = 0;
    std::vector<cudaStream_t> st;
    std::vector<cublasHandle_t> h;
    std::vector<cudaEvent_t> done;
    cudaEvent_t start = NULL;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<f16p_job> jobs;
    uint64_t gen = 0;
    std::atomic<int> left{0}, fail{0};
};
static f16p_pool *g_f16p[DS4_MAX_GPUS];

static int f16p_launchers(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("DS4_F16_LAUNCHERS"); n = e && *e ? atoi(e) : 4; if (n < 1) n = 1; if (n > 16) n = 16; }
    return n;
}

static void f16p_worker(f16p_pool *p, int k) {
    (void)cudaSetDevice(p->dev);
    uint64_t seen = 0;
    for (;;) {
        f16p_job j;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv.wait(lk, [&] { return p->gen != seen; });
            seen = p->gen;
            j = p->jobs[(size_t)k];
        }
        bool ok = cudaStreamWaitEvent(p->st[(size_t)k], p->start, 0) == cudaSuccess;
        for (uint32_t r = j.r0; ok && r < j.r1; r++)
            ok = f16g_gemm_h(p->h[(size_t)k], j.w, j.x + (uint64_t)r * j.width, j.y + (uint64_t)r * j.outputs, j.width,
                             j.outputs, 1) != 0;
        ok = ok && cudaEventRecord(p->done[(size_t)k], p->st[(size_t)k]) == cudaSuccess;
        if (!ok) p->fail.store(1);
        p->left.fetch_sub(1, std::memory_order_acq_rel);
    }
}

static f16p_pool *f16p_ensure(int tier) {
    if (tier < 0 || tier >= DS4_MAX_GPUS) return NULL;
    if (g_f16p[tier]) return g_f16p[tier]->n ? g_f16p[tier] : NULL;
    f16p_pool *p = new f16p_pool();   /* never freed: its threads wait on it for ever */
    g_f16p[tier] = p;
    const int n = f16p_launchers();
    if (n < 2) return NULL;
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
    p->dev = g_gpu[tier].device_id;
    if (cudaSetDevice(p->dev) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    bool ok = cublasGetMathMode(cuda_cublas_main_for_tier(tier), &math) == CUBLAS_STATUS_SUCCESS &&
        cudaEventCreateWithFlags(&p->start, cudaEventDisableTiming) == cudaSuccess;
    for (int k = 0; ok && k < n; k++) {
        cudaStream_t st = NULL; cublasHandle_t h = NULL; cudaEvent_t ev = NULL;
        ok = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess &&
            cublasCreate(&h) == CUBLAS_STATUS_SUCCESS && cublasSetStream(h, st) == CUBLAS_STATUS_SUCCESS &&
            cublasSetMathMode(h, math) == CUBLAS_STATUS_SUCCESS;
        p->st.push_back(st); p->h.push_back(h); p->done.push_back(ev);
    }
    (void)cudaSetDevice(cur);
    if (!ok) { (void)cudaGetLastError(); fprintf(stderr, "ds4: F16-P: launcher streams could not be set up on card %d\n", tier); return NULL; }
    p->jobs.resize((size_t)n);
    for (int k = 0; k < n; k++) std::thread(f16p_worker, p, k).detach();
    p->n = n;
    return p;
}

/* rows [0, rows) on the launcher threads; the calling (legacy) stream waits for them */
static int f16p_issue(f16p_pool *p, const __half *w, const __half *x, float *y, uint32_t width, uint32_t outputs,
                      uint32_t rows) {
    if (!cuda_ok(cudaEventRecord(p->start, cuda_decode_stream()), "F16-P start")) return 0;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        for (int k = 0; k < p->n; k++) {
            const uint32_t r0 = (uint32_t)((uint64_t)rows * (uint32_t)k / (uint32_t)p->n);
            const uint32_t r1 = (uint32_t)((uint64_t)rows * (uint32_t)(k + 1) / (uint32_t)p->n);
            p->jobs[(size_t)k] = {w, x, y, width, outputs, r0, r1};
        }
        p->fail.store(0);
        p->left.store(p->n, std::memory_order_release);
        p->gen++;
    }
    p->cv.notify_all();
    while (p->left.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    if (p->fail.load()) { fprintf(stderr, "ds4: F16-P: a launcher failed\n"); return 0; }
    for (int k = 0; k < p->n; k++)
        if (!cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(), p->done[(size_t)k], 0), "F16-P join")) return 0;
    return 1;
}

static int f16g_gemm(int tier, const __half *w, const __half *x, float *y, uint32_t width, uint32_t outputs, uint32_t n) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(tier), CUBLAS_OP_T, CUBLAS_OP_N, (int)outputs, (int)n,
        (int)width, &alpha, w, CUDA_R_16F, (int)width, x, CUDA_R_16F, (int)width, &beta, y, CUDA_R_32F, (int)outputs,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "F16 row group");
}

/* rows [0, rows) of y = w^T x, x already in F16: groups where checked equal, else one GEMM per row */
static int f16g_rows(int tier, const __half *w, const __half *x, float *y, uint32_t width, uint32_t outputs,
                     uint32_t rows) {
    static std::map<std::tuple<int, uint32_t, uint32_t, uint32_t>, int> verdict;
    static float *check[DS4_MAX_GPUS];
    static uint64_t check_cap[DS4_MAX_GPUS];
    static unsigned int *diff[DS4_MAX_GPUS];
    /* F16-P */
    static std::map<std::tuple<int, uint32_t, uint32_t>, int> par_verdict;
    f16p_pool *pool = rows >= 32u && !g_dsv41_shared.active && !g_decode_graph_capturing && g_tp2g_capturing < 0 ?
        f16p_ensure(tier) : NULL;
    if (pool) {
        const auto key = std::make_tuple(tier, width, outputs);
        const auto it = par_verdict.find(key);
        if (it != par_verdict.end() && it->second == 1) return f16p_issue(pool, w, x, y, width, outputs, rows);
        if (it == par_verdict.end()) {   /* first sight: rows by the launchers, the first 32 again on the main handle */
            const uint64_t need = (uint64_t)32u * outputs;
            if (check_cap[tier] < need) {
                if (check[tier]) (void)cudaFree(check[tier]);
                check[tier] = NULL; check_cap[tier] = 0;
                if (!cuda_ok(cudaMalloc((void **)&check[tier], (size_t)need * sizeof(float)), "F16-P check")) return 0;
                check_cap[tier] = need;
            }
            if (!diff[tier] && !cuda_ok(cudaMalloc((void **)&diff[tier], sizeof(unsigned int)), "F16-P diff")) return 0;
            if (!f16p_issue(pool, w, x, y, width, outputs, rows)) return 0;
            for (uint32_t i = 0; i < 32u; i++)
                if (!f16g_gemm(tier, w, x + (uint64_t)i * width, check[tier] + (uint64_t)i * outputs, width, outputs, 1)) return 0;
            unsigned int h_diff = 1u;
            if (!cuda_ok(cudaMemsetAsync(diff[tier], 0, sizeof(unsigned int), cuda_decode_stream()), "F16-P diff clear")) return 0;
            f16g_diff_kernel<<<(unsigned)((need + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(check[tier], y, need, diff[tier]);
            if (!cuda_ok(cudaGetLastError(), "F16-P compare") ||
                !cuda_ok(cudaMemcpy(&h_diff, diff[tier], sizeof(unsigned int), cudaMemcpyDeviceToHost), "F16-P verdict")) return 0;
            const int v = h_diff == 0u;
            par_verdict[key] = v;
            fprintf(stderr, "ds4: F16-P on card %d: %u x %u by %d launcher threads: %s\n", tier, width, outputs, pool->n,
                    v ? "bitwise equal to the main handle, used" : "DIFFERS from the main handle, kept on it");
            if (v) return 1;
            /* the launchers' rows are not used: all rows again on the main handle below */
        }
    }
    const uint32_t G = (uint32_t)f16g_group();
    for (uint32_t r0 = 0; r0 < rows; ) {
        const uint32_t n = rows - r0 < G ? rows - r0 : G;
        float *yr = y + (uint64_t)r0 * outputs;
        const __half *xr = x + (uint64_t)r0 * width;
        int v = n > 1u ? -1 : 0;
        if (n > 1u) {
            const auto it = verdict.find(std::make_tuple(tier, width, outputs, n));
            if (it != verdict.end()) v = it->second;
        }
        if (v == 1) {
            if (!f16g_gemm(tier, w, xr, yr, width, outputs, n)) return 0;
        } else if (v == 0) {
            for (uint32_t i = 0; i < n; i++)
                if (!f16g_gemm(tier, w, xr + (uint64_t)i * width, yr + (uint64_t)i * outputs, width, outputs, 1)) return 0;
        } else {   /* first sight of this shape on this card: both ways, compared */
            const uint64_t need = (uint64_t)n * outputs;
            if (check_cap[tier] < need) {
                if (check[tier]) (void)cudaFree(check[tier]);
                check[tier] = NULL; check_cap[tier] = 0;
                if (!cuda_ok(cudaMalloc((void **)&check[tier], (size_t)need * sizeof(float)), "F16 row group check")) return 0;
                check_cap[tier] = need;
            }
            if (!diff[tier] && !cuda_ok(cudaMalloc((void **)&diff[tier], sizeof(unsigned int)), "F16 row group diff")) return 0;
            for (uint32_t i = 0; i < n; i++)
                if (!f16g_gemm(tier, w, xr + (uint64_t)i * width, check[tier] + (uint64_t)i * outputs, width, outputs, 1)) return 0;
            if (!f16g_gemm(tier, w, xr, yr, width, outputs, n)) return 0;
            unsigned int h_diff = 1u;
            if (!cuda_ok(cudaMemsetAsync(diff[tier], 0, sizeof(unsigned int), cuda_decode_stream()), "F16 row group diff clear")) return 0;
            f16g_diff_kernel<<<(unsigned)((need + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(check[tier], yr, need, diff[tier]);
            if (!cuda_ok(cudaGetLastError(), "F16 row group compare") ||
                !cuda_ok(cudaMemcpy(&h_diff, diff[tier], sizeof(unsigned int), cudaMemcpyDeviceToHost), "F16 row group verdict")) return 0;
            v = h_diff == 0u ? 1 : 0;
            verdict[std::make_tuple(tier, width, outputs, n)] = v;
            fprintf(stderr, "ds4: F16 row groups on card %d: %u x %u, %u rows per GEMM: %s\n", tier, width, outputs, n,
                    v ? "bitwise equal to one row per GEMM, used" : "DIFFERS from one row per GEMM, kept row by row");
            if (!v && !cuda_ok(cudaMemcpyAsync(yr, check[tier], (size_t)need * sizeof(float), cudaMemcpyDeviceToDevice,
                                               cuda_decode_stream()), "F16 row group keep rows")) return 0;
        }
        r0 += n;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map,
                                             uint64_t model_size, uint64_t weight_offset,
                                             uint32_t width, uint32_t outputs, uint32_t rows,
                                             const ds4_gpu_tensor *in) {
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !dsv41_has_floats(in, (uint64_t)width * rows) ||
        !dsv41_has_floats(out, (uint64_t)outputs * rows)) return 0;
#ifndef __HIP_PLATFORM_AMD__
    /* F16-G: the cuBLAS one-row path of ds4_gpu_matmul_f16_tensor (the default for these shapes), grouped */
    if (rows > 1u && (f16g_group() > 1 || f16p_launchers() > 1) && g_cublas_ready && !g_quality_mode &&
        !getenv("DS4_CUDA_SERIAL_F16_MATMUL") && !getenv("DS4_CUDA_NO_F16_CUBLAS_ONE") &&
        !getenv("DS4_CUDA_SERIAL_ROUTER") && !getenv("DS4_CUDA_F16_SMALL_OUT") &&
        !(width == 4096u && outputs == 256u) &&
        width <= INT32_MAX && outputs <= INT32_MAX && weight_offset <= model_size &&
        outputs <= (model_size - weight_offset) / sizeof(__half) / width) {
        const uint64_t bytes = (uint64_t)width * outputs * sizeof(__half);
        const int tier = ds4_tensor_device_idx(out);
        if (ds4_tensor_device_idx(in) == tier) {
            const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, bytes, tier, "V4.1 F16 row groups");
            const uint64_t count = (uint64_t)rows * width;
            __half *x = (__half *)cuda_tmp_alloc_on(tier, count * sizeof(__half), "F16 row group inputs");
            if (w && x) {
                f32_to_f16_kernel<<<(count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
                    x, (const float *)in->ptr, count);
                if (!cuda_ok(cudaGetLastError(), "F16 row group convert")) return 0;
                return f16g_rows(tier, (const __half *)w, x, (float *)out->ptr, width, outputs, rows);
            }
        }
    }
#endif
#ifndef __HIP_PLATFORM_AMD__
    if (rows > 1u && outputs <= 128u && g_cublas_ready && !g_quality_mode &&
        !getenv("DS4_CUDA_SERIAL_F16_MATMUL") && !getenv("DS4_CUDA_NO_F16_CUBLAS_ONE") &&
        !getenv("DS4_CUDA_SERIAL_ROUTER") && !getenv("DS4_CUDA_F16_SMALL_OUT")) {
        if (width > INT32_MAX || outputs > INT32_MAX || weight_offset > model_size ||
            outputs > (model_size - weight_offset) / sizeof(__half) / width) return 0;
        const uint64_t bytes = (uint64_t)width * outputs * sizeof(__half);
        const int tier = ds4_tensor_device_idx(out);
        if (ds4_tensor_device_idx(in) != tier) return 0;
        const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, bytes, tier,
                                               "V4.1 F16 vector batch");
        const uint64_t count = (uint64_t)rows * width;
        __half *x = (__half *)cuda_tmp_alloc_on(tier, count * sizeof(__half), "F16 vector batch inputs");
        if (!w || !x) return 0;
        f32_to_f16_kernel<<<(count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
            x, (const float *)in->ptr, count);
        if (!cuda_ok(cudaGetLastError(), "F16 vector batch convert")) return 0;
        const float alpha = 1.0f, beta = 0.0f;
        /* Small output projections benefit from one conversion launch. Larger
         * GEMVs are faster when input conversion is interleaved with each row.
         * Keep the one-row cuBLAS reduction in both cases. */
        for (uint32_t row = 0; row < rows; row++) {
            float *y = (float *)out->ptr + (uint64_t)row * outputs;
            const __half *xr = x + (uint64_t)row * width;
            const cublasStatus_t status = cublasGemmEx(cuda_cublas_for_tier(tier),
                    CUBLAS_OP_T, CUBLAS_OP_N, outputs, 1, width, &alpha,
                    w, CUDA_R_16F, width, xr, CUDA_R_16F, width,
                    &beta, y, CUDA_R_32F, outputs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(status, "F16 vector batch")) return 0;
        }
        return 1;
    }
#endif
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = *in, y = *out;
        x.ptr = (float *)x.ptr + (uint64_t)row * width;
        x.bytes = (uint64_t)width * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)row * outputs;
        y.bytes = (uint64_t)outputs * sizeof(float);
        if (!ds4_gpu_matmul_f16_tensor(&y, model_map, model_size, weight_offset,
                                       width, outputs, &x, 1u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                                                    const void *model_map, uint64_t model_size,
                                                    uint64_t a, uint64_t b,
                                                    const ds4_gpu_tensor *heads, uint32_t rows) {
    if (!rows || !dsv41_has_floats(out, (uint64_t)rows * 5120u) ||
        !dsv41_has_floats(low, (uint64_t)rows * 8192u) ||
        !dsv41_has_floats(heads, (uint64_t)rows * 32768u)) return 0;
#ifdef __HIP_PLATFORM_AMD__
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)row * 32768u;
        x.bytes = 32768u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)row * 8192u;
        y.bytes = 8192u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_tensor(&y, model_map, model_size,
                a, 4096u, 1024u, 8u, &x)) return 0;
    }
#else
    /* Quantization launches one grid-y entry per row and head group. */
    for (uint32_t first = 0; first < rows; ) {
        const uint32_t count = min(rows - first, 65535u / 8u);
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)first * 32768u;
        x.bytes = (uint64_t)count * 32768u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)first * 8192u;
        y.bytes = (uint64_t)count * 8192u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_rows_exact_tensor(&y, model_map, model_size,
                a, 4096u, 1024u, 8u, 0u, 8u, &x, count)) return 0;
        first += count;
    }
#endif
    return ds4_gpu_dsv41_quantize(low, 8192u, rows, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model_map, model_size,
            b, 8192u, 5120u, low, rows);
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t a, uint64_t b,
        const ds4_gpu_tensor *heads, uint32_t rows, uint32_t rank) {
    const uint64_t shard_bytes = UINT64_C(4) * 1024 * (4096 / 32 * 34);
    if (rank > 1 || !rows || a > model_size || 2u * shard_bytes > model_size - a ||
        !dsv41_has_floats(out, (uint64_t)rows * 5120u) ||
        !dsv41_has_floats(low, (uint64_t)rows * 4096u) ||
        !dsv41_has_floats(heads, (uint64_t)rows * 16384u)) return 0;
    for (uint32_t first = 0; first < rows;) {
        const uint32_t count = min(rows - first, 65535u / 4u);
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)first * 16384u;
        x.bytes = (uint64_t)count * 16384u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)first * 4096u;
        y.bytes = (uint64_t)count * 4096u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_rows_exact_tensor(&y, model_map, model_size,
                a + rank * shard_bytes, 4096u, 1024u, 4u, 0u, 4u, &x, count)) return 0;
        first += count;
    }
    return ds4_gpu_dsv41_quantize(low, 4096u, rows, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_kslice_rows_tensor(out, model_map, model_size,
            b, 8192u, 5120u, rank * 4096u, 4096u, low, rows);
}

/* Metal folds these bf16 roundings into the producing kernels; here the
 * standalone rounding pass follows (or precedes) the unchanged operator. */
static uint32_t ds4_gpu_dsv41_rows_of(const ds4_gpu_tensor *t, uint32_t width) {
    const uint64_t bytes = ds4_gpu_tensor_bytes(t);
    const uint64_t rows = width ? bytes / ((uint64_t)width * sizeof(float)) : 0;
    return rows ? (uint32_t)rows : 1u;
}

extern "C" int ds4_gpu_rms_norm_weight_bf16_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    return ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, weight_offset, n, eps) &&
           ds4_gpu_dsv41_quantize(out, n, 1, DS4_V41_BF16);
}

extern "C" int ds4_gpu_matmul_q8_0_bf16_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (n_tok == 1 && cuda_kfast()) {   /* K1: the single-row Q8 kernel rounds at its store when that path runs */
        g_q8_round_request = true;
        g_q8_round_done = false;
        const int ok = ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
        g_q8_round_request = false;
        if (!ok) return 0;
        if (g_q8_round_done) return 1;
        return ds4_gpu_dsv41_quantize(out, (uint32_t)out_dim, 1, DS4_V41_BF16);
    }
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok) &&
           ds4_gpu_dsv41_quantize(out, (uint32_t)out_dim, (uint32_t)n_tok, DS4_V41_BF16);
}

extern "C" int ds4_gpu_matmul_q8_0_bf16io_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, ds4_gpu_tensor *x, uint64_t n_tok) {
#ifndef __HIP_PLATFORM_AMD__
    if (n_tok == 1u && cuda_kfast() && in_dim && (in_dim & 31u) == 0u && !g_dsv41_shared.active &&
        dsv41_has_floats(x, in_dim)) {   /* K3: x rounded and quantized in one launch */
        int8_t *xq; float *xs;
        void *tmp = dsv41_k3_prequant(ds4_tensor_device_idx(out), in_dim / 32u, &xq, &xs);
        if (tmp) {
            dsv41_bf16_quantize_q8_0_kernel<<<(unsigned)(in_dim / 32u), 32, 0, cuda_decode_stream()>>>(
                xq, xs, (float *)x->ptr, in_dim, in_dim / 32u);
            if (!cuda_ok(cudaGetLastError(), "V4.1 BF16 + Q8_0 input")) return 0;
            g_q8_preq_x = x->ptr;
            g_q8_preq_tmp = tmp;
            const int ok = ds4_gpu_matmul_q8_0_bf16_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, 1);
            g_q8_preq_x = NULL;
            g_q8_preq_tmp = NULL;
            return ok;
        }
    }
#endif
    return ds4_gpu_dsv41_quantize(x, (uint32_t)in_dim, (uint32_t)n_tok, DS4_V41_BF16) &&
           ds4_gpu_matmul_q8_0_bf16_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_hc_weighted_sum_bf16_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights,
        uint32_t n_embd, uint32_t n_hc) {
    return ds4_gpu_hc_weighted_sum_tensor(out, residual_hc, weights, n_embd, n_hc) &&
           ds4_gpu_dsv41_quantize(out, n_embd, ds4_gpu_dsv41_rows_of(out, n_embd), DS4_V41_BF16);
}

extern "C" int ds4_gpu_hc_expand_split_bf16_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc) &&
           ds4_gpu_dsv41_quantize(out_hc, n_hc * n_embd, ds4_gpu_dsv41_rows_of(out_hc, n_hc * n_embd), DS4_V41_BF16);
}

/* block_out is scratch owned by the caller: the sum lands in it before the expansion. */
extern "C" int ds4_gpu_hc_expand_split_add_bf16_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    ds4_gpu_tensor *block = (ds4_gpu_tensor *)block_out;
    const uint32_t rows = ds4_gpu_dsv41_rows_of(block_out, n_embd);
#ifndef __HIP_PLATFORM_AMD__
    if (cuda_kfast() && rows == 1u && n_hc == 4u && out_hc && block_add && residual_hc && split &&
        out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)) == 1u &&
        dsv41_has_floats(block_add, n_embd) && dsv41_has_floats(residual_hc, 4u * n_embd) &&
        dsv41_has_floats(split, 24u)) {   /* K3: one launch */
        const float *sp = (const float *)split->ptr;
        dsv41_expand_add_bf16_kernel<<<(n_embd + 255u) / 256u, 256, 0, cuda_decode_stream()>>>((float *)out_hc->ptr,
            (float *)block->ptr, (const float *)block_add->ptr, (const float *)residual_hc->ptr, sp + 4u, sp + 8u, n_embd);
        return cuda_ok(cudaGetLastError(), "V4.1 add + HC expansion");
    }
#endif
    return ds4_gpu_add_tensor(block, block_out, block_add, (uint64_t)n_embd * rows) &&
           ds4_gpu_dsv41_quantize(block, n_embd, rows, DS4_V41_BF16) &&
           ds4_gpu_hc_expand_split_bf16_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_dsv41_rope_bf16(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                       uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_quantize(x, width * heads, rows, DS4_V41_BF16) &&
           ds4_gpu_dsv41_rope(x, width, heads, rows, start, compressed, inverse);
}
