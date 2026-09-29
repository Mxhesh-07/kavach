#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

struct CandidateBox {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// YOLO11 / YOLOv8 output tensor format: [1, 84, 8400]
// 84 rows: 4 bbox coordinates (cx, cy, w, h) + 80 class probabilities
__global__ void decode_yolo_boxes_kernel(
    const float* __restrict__ d_output, // shape [84, 8400]
    CandidateBox* __restrict__ d_candidates,
    int* __restrict__ d_candidate_count,
    int max_candidates,
    float conf_threshold,
    int orig_w,
    int orig_h,
    int input_w,
    int input_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int NUM_ANCHORS = 8400;
    const int NUM_CLASSES = 80;

    if (idx >= NUM_ANCHORS) return;

    // Output is stored in row-major order: [84, 8400]
    // d_output[row * 8400 + idx]
    float cx = d_output[0 * NUM_ANCHORS + idx];
    float cy = d_output[1 * NUM_ANCHORS + idx];
    float w  = d_output[2 * NUM_ANCHORS + idx];
    float h  = d_output[3 * NUM_ANCHORS + idx];

    // Find class with highest confidence
    float max_score = -1.0f;
    int max_class_id = -1;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        float score = d_output[(4 + c) * NUM_ANCHORS + idx];
        if (score > max_score) {
            max_score = score;
            max_class_id = c;
        }
    }

    if (max_score >= conf_threshold) {
        int pos = atomicAdd(d_candidate_count, 1);
        if (pos < max_candidates) {
            float scale_x = (float)orig_w / input_w;
            float scale_y = (float)orig_h / input_h;

            float x1 = (cx - w * 0.5f) * scale_x;
            float y1 = (cy - h * 0.5f) * scale_y;
            float x2 = (cx + w * 0.5f) * scale_x;
            float y2 = (cy + h * 0.5f) * scale_y;

            d_candidates[pos].x1 = fmaxf(0.0f, x1);
            d_candidates[pos].y1 = fmaxf(0.0f, y1);
            d_candidates[pos].x2 = fminf((float)orig_w, x2);
            d_candidates[pos].y2 = fminf((float)orig_h, y2);
            d_candidates[pos].score = max_score;
            d_candidates[pos].class_id = max_class_id;
        }
    }
}

__device__ inline float box_iou(const CandidateBox& a, const CandidateBox& b) {
    float left = fmaxf(a.x1, b.x1);
    float top = fmaxf(a.y1, b.y1);
    float right = fminf(a.x2, b.x2);
    float bottom = fminf(a.y2, b.y2);

    float width = fmaxf(0.0f, right - left);
    float height = fmaxf(0.0f, bottom - top);
    float intersection = width * height;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

__global__ void nms_mask_kernel(
    const CandidateBox* __restrict__ d_candidates,
    uint64_t* __restrict__ d_mask,
    int count,
    float iou_threshold
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= count || col >= count) return;

    if (row < col) {
        if (d_candidates[row].class_id == d_candidates[col].class_id) {
            float iou = box_iou(d_candidates[row], d_candidates[col]);
            if (iou > iou_threshold) {
                atomicOr((unsigned long long*)&d_mask[col / 64], 1ULL << (col % 64));
            }
        }
    }
}

void launch_postprocess_cuda(
    const float* d_output,
    CandidateBox* d_candidates,
    int* d_candidate_count,
    int max_candidates,
    float conf_threshold,
    float iou_threshold,
    int orig_w,
    int orig_h,
    int input_w,
    int input_h,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_candidate_count, 0, sizeof(int), stream);

    dim3 block(256);
    dim3 grid((8400 + block.x - 1) / block.x);

    decode_yolo_boxes_kernel<<<grid, block, 0, stream>>>(
        d_output,
        d_candidates,
        d_candidate_count,
        max_candidates,
        conf_threshold,
        orig_w,
        orig_h,
        input_w,
        input_h
    );
}
