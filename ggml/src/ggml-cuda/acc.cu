#include "acc.cuh"

// dst = src0, then src1 added into the view of dst at `offset` with strides s1, s2, s3 (in elements).
// Each src1 element names its own dst element, so any strides a view can carry are handled, in the order
// the CPU backend uses; a view's elements are distinct, so no two threads add into one element.
static __global__ void acc_copy_f32(const float * x, float * dst, const int64_t ne) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ne) {
        dst[i] = x[i];
    }
}

static __global__ void acc_add_f32(const float * y, float * dst,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset) {
    const int64_t j = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= ne10*ne11*ne12*ne13) {
        return;
    }
    const int64_t i10 = j % ne10;
    const int64_t i11 = (j / ne10) % ne11;
    const int64_t i12 = (j / (ne10*ne11)) % ne12;
    const int64_t i13 = j / (ne10*ne11*ne12);
    dst[offset + i13*s3 + i12*s2 + i11*s1 + i10] += y[j];
}

static void acc_f32_cuda(const float * x, const float * y, float * dst, const int64_t n_elements,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset, cudaStream_t stream) {
    if (x != dst) {
        const int64_t blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
        acc_copy_f32<<<blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(x, dst, n_elements);
    }
    const int64_t n1 = ne10*ne11*ne12*ne13;
    const int64_t blocks = (n1 + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    if (blocks > 0) {
        acc_add_f32<<<blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(y, dst, ne10, ne11, ne12, ne13, s1, s2, s3, offset);
    }
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / sizeof(float);
    const int64_t s2     = dst->op_params[1] / sizeof(float);
    const int64_t s3     = dst->op_params[2] / sizeof(float);
    const int64_t offset = dst->op_params[3] / sizeof(float);

    acc_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], s1, s2, s3, offset, stream);
}
