// test-conv-1d-f32.cpp: an F32 kernel convolves in F32 on every backend.
//
// ggml_conv_1d and ggml_conv_1d_dw unfold the input with im2col and multiply the
// columns by the kernel. The columns used to be F16 for an F32 kernel, which
// rounds every input to F16 and, on CUDA, makes the product an F16 cuBLAS
// product accumulated in F16. The test runs both ops with F32 kernels on each
// backend (asking for full FP32 products where a backend offers it) and compares
// them with a double precision reference at a bound F16 rounding cannot meet.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// One geometry: kernel size, input and output channels (OC = 0: depthwise), input length, stride, padding, dilation
struct conv_case {
    int64_t K;
    int64_t IC;
    int64_t OC;
    int64_t T;
    int     s0;
    int     p0;
    int     d0;
};

static const conv_case CASES[] = {
    { 7, 96,  96, 211, 1, 0, 1 },  // dilated residual unit of a speech decoder, dilations 1, 3, 9
    { 7, 96,  96, 211, 1, 0, 3 },
    { 7, 96,  96, 211, 1, 0, 9 },
    { 1, 96,  96, 211, 1, 0, 1 },  // pointwise
    { 3, 64, 128, 157, 1, 0, 1 },  // channel expansion
    { 4, 32,  16,  97, 2, 1, 1 },  // strided, padded
    { 7, 64,   0, 211, 1, 0, 1 },  // depthwise
    { 5, 33,   0,  97, 2, 2, 2 },  // depthwise, strided, padded, dilated
};

// Deterministic LCG mapped to [-1, 1]
static uint64_t g_rng = 0x2545f491ULL;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

// y[o][t] = sum_ic sum_k w[o][ic][k] * x[ic][t*s0 + k*d0 - p0], zero outside the input
static std::vector<double> reference(const conv_case & c, const std::vector<float> & w, const std::vector<float> & x, int64_t OL) {
    const bool    dw = c.OC == 0;
    const int64_t OC = dw ? c.IC : c.OC;
    std::vector<double> y(OL * OC, 0.0);
    for (int64_t o = 0; o < OC; o++) {
        for (int64_t t = 0; t < OL; t++) {
            double acc = 0.0;
            for (int64_t ic = dw ? o : 0; ic < (dw ? o + 1 : c.IC); ic++) {
                for (int64_t k = 0; k < c.K; k++) {
                    const int64_t ti = t * c.s0 + k * c.d0 - c.p0;
                    if (ti < 0 || ti >= c.T) {
                        continue;
                    }
                    const int64_t wi = dw ? k + o * c.K : k + ic * c.K + o * c.K * c.IC;
                    acc += (double) w[wi] * (double) x[ti + ic * c.T];
                }
            }
            y[t + o * OL] = acc;
        }
    }
    return y;
}

// NMSE of one case on one backend, negative on a shape mismatch
static double run_case(ggml_backend_t backend, const conv_case & c) {
    const bool    dw = c.OC == 0;
    const int64_t OL = (c.T + 2 * c.p0 - c.d0 * (c.K - 1) - 1) / c.s0 + 1;
    const int64_t OC = dw ? c.IC : c.OC;

    std::vector<float> w(c.K * (dw ? 1 : c.IC) * OC);
    std::vector<float> x(c.T * c.IC);
    for (float & v : w) {
        v = frand();
    }
    for (float & v : x) {
        v = frand();
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, dw ? 1 : c.IC, OC);
    struct ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.T, c.IC);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(tw, w.data(), 0, ggml_nbytes(tw));
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));

    struct ggml_tensor * ty = dw ? ggml_conv_1d_dw(ctx, tw, tx, c.s0, c.p0, c.d0)
                                 : ggml_conv_1d   (ctx, tw, tx, c.s0, c.p0, c.d0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ty);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    double nmse = -1.0;
    if (ty->type == GGML_TYPE_F32 && ty->ne[0] == OL && ty->ne[1] == OC && ggml_nelements(ty) == OL * OC) {
        std::vector<float> y(OL * OC);
        ggml_backend_tensor_get(ty, y.data(), 0, ggml_nbytes(ty));
        const std::vector<double> ref = reference(c, w, x, OL);
        double num = 0.0;
        double den = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            num += (y[i] - ref[i]) * (y[i] - ref[i]);
            den += ref[i] * ref[i];
        }
        nmse = num / (den + 1e-30);
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return nmse;
}

int main(void) {
    int fails = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        // a backend that rounds F32 products to a narrower format by default (TF32) is asked for full FP32;
        // one that offers no such switch is only checked when it is the CPU
        auto set_fp32_gemm = (void (*)(ggml_backend_t, bool))
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_fp32_gemm");
        if (set_fp32_gemm) {
            set_fp32_gemm(backend, true);
        } else if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            printf("%s: skipped, no full FP32 product switch\n", ggml_backend_dev_name(dev));
            ggml_backend_free(backend);
            continue;
        }

        for (const conv_case & c : CASES) {
            const double nmse = run_case(backend, c);
            // F16 columns land near 1e-7; F32 columns with F32 accumulation near 1e-14
            const bool ok = nmse >= 0.0 && nmse <= 1e-10;
            if (!ok) {
                fails++;
            }
            printf("%s: conv_1d%s K=%d IC=%d OC=%d T=%d s0=%d p0=%d d0=%d: nmse %.2e %s\n",
                ggml_backend_dev_name(dev), c.OC == 0 ? "_dw" : "", (int) c.K, (int) c.IC, (int) c.OC, (int) c.T,
                c.s0, c.p0, c.d0, nmse, ok ? "OK" : "FAIL");
        }
        ggml_backend_free(backend);
    }

    printf(fails == 0 ? "all F32 conv_1d checks passed\n" : "%d F32 conv_1d checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
