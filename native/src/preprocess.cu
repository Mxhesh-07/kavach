#include <cuda_runtime.h>
#include <cstdint>

__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src_bgr,
    float* __restrict__ dst_planar,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale_x, float scale_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    int sx = min(__float2int_rz(x * scale_x), src_w - 1);
    int sy = min(__float2int_rz(y * scale_y), src_h - 1);

    int src_idx = (sy * src_w + sx) * 3;
    int plane_size = dst_w * dst_h;
    int dst_idx = y * dst_w + x;

    // RGB layout conversion + 1/255.0f normalization
    dst_planar[0 * plane_size + dst_idx] = src_bgr[src_idx + 2] * (1.0f / 255.0f);
    dst_planar[1 * plane_size + dst_idx] = src_bgr[src_idx + 1] * (1.0f / 255.0f);
    dst_planar[2 * plane_size + dst_idx] = src_bgr[src_idx + 0] * (1.0f / 255.0f);
}

void launch_preprocess_cuda(
    const uint8_t* src, float* dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    float sx = (float)src_w / dst_w;
    float sy = (float)src_h / dst_h;
    preprocess_kernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h, sx, sy);
}
