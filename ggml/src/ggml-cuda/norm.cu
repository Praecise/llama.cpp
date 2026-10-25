#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

// Group norm runs in three kernels so that a large group is spread over many
// blocks: partial statistics per chunk of a group (Welford, so the variance
// does not suffer the cancellation of a single sum of squares), a combine per
// group, and one elementwise pass that normalises and, when fused, applies the
// per-channel scale and shift and a SiLU. That is three passes over the data
// where the unfused group norm, scale, shift and SiLU take eleven.

struct group_norm_stat {
    float n;
    float mean;
    float m2;
};

static __device__ __forceinline__ group_norm_stat group_norm_merge(const group_norm_stat a, const group_norm_stat b) {
    const float n = a.n + b.n;
    if (n == 0.0f) {
        return a;
    }
    const float delta = b.mean - a.mean;
    const float mean  = a.mean + delta * (b.n / n);
    const float m2    = a.m2 + b.m2 + delta * delta * (a.n * b.n / n);
    return { n, mean, m2 };
}

static __device__ __forceinline__ group_norm_stat group_norm_warp_merge(group_norm_stat s) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        group_norm_stat o;
        o.n    = __shfl_xor_sync(0xffffffff, s.n, offset);
        o.mean = __shfl_xor_sync(0xffffffff, s.mean, offset);
        o.m2   = __shfl_xor_sync(0xffffffff, s.m2, offset);
        s      = group_norm_merge(s, o);
    }
    return s;
}

// One block per (group, chunk): Welford statistics of the chunk.
static __global__ void group_norm_partial_f32(const float * x, group_norm_stat * partial, const int group_size,
                                              const int chunk, const int nchunks, const int64_t ne_elements) {
    const int     group = blockIdx.x / nchunks;
    const int     c     = blockIdx.x % nchunks;
    const int64_t base  = (int64_t) group * group_size;
    const int     begin = c * chunk;
    // The last group is shorter when the channels do not divide evenly.
    const int     end   = (int) min((int64_t) min(begin + chunk, group_size), ne_elements - base);

    ggml_cuda_pdl_sync();
    group_norm_stat s = { 0.0f, 0.0f, 0.0f };
    for (int j = begin + threadIdx.x; j < end; j += blockDim.x) {
        const float v = x[base + j];
        s.n += 1.0f;
        const float d = v - s.mean;
        s.mean += d / s.n;
        s.m2 += d * (v - s.mean);
    }
    s = group_norm_warp_merge(s);

    __shared__ group_norm_stat warp_stats[32];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        warp_stats[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const int nwarps = blockDim.x / WARP_SIZE;
        s = lane < nwarps ? warp_stats[lane] : group_norm_stat{ 0.0f, 0.0f, 0.0f };
        s = group_norm_warp_merge(s);
        if (lane == 0) {
            partial[blockIdx.x] = s;
        }
    }
}

// One warp per group: combine the chunks into mean and 1/std.
static __global__ void group_norm_combine_f32(const group_norm_stat * partial, float2 * mean_rstd, const int nchunks,
                                              const int ngroups, const float eps) {
    const int group = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane  = threadIdx.x % WARP_SIZE;
    if (group >= ngroups) {
        return;
    }
    group_norm_stat s = { 0.0f, 0.0f, 0.0f };
    for (int c = lane; c < nchunks; c += WARP_SIZE) {
        s = group_norm_merge(s, partial[group * nchunks + c]);
    }
    s = group_norm_warp_merge(s);
    if (lane == 0) {
        mean_rstd[group] = make_float2(s.mean, rsqrtf(s.m2 / s.n + eps));
    }
}

// Elementwise: normalise, then (optionally) scale and shift per channel and
// apply SiLU.
template <bool affine, bool silu>
static __global__ void group_norm_apply_f32(const float * x, float * dst, const float2 * mean_rstd, const float * scale,
                                            const float * shift, const int64_t n, const int group_size, const int plane,
                                            const int channels) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float2 mr = mean_rstd[i / group_size];
    float        v  = (x[i] - mr.x) * mr.y;
    if constexpr (affine) {
        const int c = (int) ((i / plane) % channels);
        v           = v * scale[c] + shift[c];
    }
    if constexpr (silu) {
        v = v / (1.0f + expf(-v));
    }
    dst[i] = v;
}

// centered: subtract the row mean first, i.e. layer norm rather than RMS norm.
template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool centered = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");
    static_assert(!centered || do_multiply, "the centered variant is only fused with a multiply");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp  = 0.0f;
                float  tsum = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                        if constexpr (centered) {
                            tsum += xs[k].x + xs[k].y + xs[k].z + xs[k].w;
                        }
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                if constexpr (centered) {
                    __syncthreads();
                    tsum = block_reduce<block_reduce_method::SUM, block_size>(tsum, s_sum);
                }
                const float row_mean = centered ? tsum / ncols : 0.0f;
                const float scale    = rsqrtf(tmp / ncols - row_mean * row_mean + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*(xs[k].x - row_mean), scale*(xs[k].y - row_mean),
                                               scale*(xs[k].z - row_mean), scale*(xs[k].w - row_mean));
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                    }
                }
            } else {
                float tmp  = 0.0f; // partial sum for thread in warp
                float tsum = 0.0f;

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                    if constexpr (centered) {
                        tsum += xi;
                    }
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                if constexpr (centered) {
                    __syncthreads();
                    tsum = block_reduce<block_reduce_method::SUM, block_size>(tsum, s_sum);
                }

                const float row_mean = centered ? tsum / ncols : 0.0f;
                const float mean     = tmp / ncols - row_mean * row_mean;
                const float scale    = rsqrtf(mean + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        dstc[col]         = scale * (xc[col] - row_mean) * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        dstc[col]         = scale * (xc[col] - row_mean) * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        dstc[col] = scale_out * (scale * xc[col]);
                    } else {
                        dstc[col] = scale * xc[col];
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

// Elements of one group each statistics block reads.
static constexpr int GROUP_NORM_CHUNK = 16384;

static void group_norm_f32_cuda(ggml_backend_cuda_context & ctx, const float * x, float * dst, const int num_groups,
                                const float eps, const int group_size, const int64_t ne_elements, const float * scale,
                                const float * shift, const bool silu, const int plane, const int channels) {
    cudaStream_t stream  = ctx.stream();
    const int    nchunks = (group_size + GROUP_NORM_CHUNK - 1) / GROUP_NORM_CHUNK;

    ggml_cuda_pool_alloc<group_norm_stat> partial(ctx.pool(), (size_t) num_groups * nchunks);
    ggml_cuda_pool_alloc<float2>          mean_rstd(ctx.pool(), num_groups);

    group_norm_partial_f32<<<num_groups * nchunks, 256, 0, stream>>>(x, partial.get(), group_size, GROUP_NORM_CHUNK,
                                                                   nchunks, ne_elements);
    const int warps_per_block = 8;
    group_norm_combine_f32<<<(num_groups + warps_per_block - 1) / warps_per_block, warps_per_block * WARP_SIZE, 0,
                             stream>>>(partial.get(), mean_rstd.get(), nchunks, num_groups, eps);

    const int      threads = 256;
    const unsigned blocks  = (unsigned) ((ne_elements + threads - 1) / threads);
    const bool     affine  = scale != nullptr;
    if (affine && silu) {
        group_norm_apply_f32<true, true><<<blocks, threads, 0, stream>>>(x, dst, mean_rstd.get(), scale, shift,
                                                                        ne_elements, group_size, plane, channels);
    } else if (affine) {
        group_norm_apply_f32<true, false><<<blocks, threads, 0, stream>>>(x, dst, mean_rstd.get(), scale, shift,
                                                                         ne_elements, group_size, plane, channels);
    } else {
        group_norm_apply_f32<false, false><<<blocks, threads, 0, stream>>>(x, dst, mean_rstd.get(), nullptr, nullptr,
                                                                          ne_elements, group_size, plane, channels);
    }
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale, centered>, launch_params, \
            x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            add, add_stride_row, add_stride_channel, add_stride_sample, \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out); \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale, bool centered = false>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");
    static_assert(!centered || do_multiply, "the centered variant is only fused with a multiply");

    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }

    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  const bool     centered) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    GGML_ASSERT(mul_t != nullptr || !centered);
    if (mul_t == nullptr) {
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        if (centered) {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false, /*centered*/true>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
        } else {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
        }
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (centered) {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false, /*centered*/true>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                add_t, add_stride_row, add_stride_channel, add_stride_sample,
                add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
        } else {
            rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
                blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
                mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                add_t, add_stride_row, add_stride_channel, add_stride_sample,
                add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    GGML_UNUSED(stream);
    group_norm_f32_cuda(ctx, src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), nullptr,
                        nullptr, false, 0, 1);
}

void ggml_cuda_op_group_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * gn, ggml_tensor * mul,
                                   ggml_tensor * add, ggml_tensor * silu) {
    const ggml_tensor * src0 = gn->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0));

    const int num_groups = gn->op_params[0];
    float     eps;
    memcpy(&eps, gn->op_params + 1, sizeof(float));

    const ggml_tensor * scale = mul->src[0] == gn ? mul->src[1] : mul->src[0];
    const ggml_tensor * shift = add->src[0] == mul ? add->src[1] : add->src[0];
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && shift->type == GGML_TYPE_F32);

    ggml_tensor * out        = silu != nullptr ? silu : add;
    const int     group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(ctx, (const float *) src0->data, (float *) out->data, num_groups * src0->ne[3], eps,
                        group_size, ggml_nelements(src0), (const float *) scale->data, (const float *) shift->data,
                        silu != nullptr, (int) (src0->ne[0] * src0->ne[1]), (int) src0->ne[2]);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream, dst->op == GGML_OP_NORM);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream, dst->op == GGML_OP_NORM);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
