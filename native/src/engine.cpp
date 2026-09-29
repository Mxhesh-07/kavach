#include "kavach_native/engine.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace kavach_native {

TensorRTEngine::TensorRTEngine() = default;

TensorRTEngine::~TensorRTEngine() {
    shutdown();
}

bool TensorRTEngine::initialize(const std::string& engine_path, int max_batch_size) {
    if (initialized_) return true;
    max_batch_size_ = max_batch_size;

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[TensorRTEngine] Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(get_trt_logger()));
    if (!runtime_) return false;

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) return false;

    context_ = engine_->createExecutionContext();
    if (!context_) return false;

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&start_event_));
    CHECK_CUDA(cudaEventCreate(&stop_event_));

    // Calculate memory allocations
    input_size_bytes_ = max_batch_size_ * 3 * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * sizeof(float);
    output_size_bytes_ = max_batch_size_ * 84 * 8400 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&gpu_input_buffer_, input_size_bytes_));
    CHECK_CUDA(cudaMalloc(&gpu_output_buffer_, output_size_bytes_));
    CHECK_CUDA(cudaHostAlloc(&pinned_input_buffer_, input_size_bytes_, cudaHostAllocDefault | cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc(&pinned_output_buffer_, output_size_bytes_, cudaHostAllocDefault | cudaHostAllocPortable));

    // Set tensor addresses (TensorRT 8.6+ & 10.x compatibility)
    context_->setTensorAddress(input_tensor_name_.c_str(), gpu_input_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), gpu_output_buffer_);

    initialized_ = true;
    return true;
}

void TensorRTEngine::shutdown() {
    if (!initialized_) return;

    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
    if (start_event_) { cudaEventDestroy(start_event_); start_event_ = 0; }
    if (stop_event_) { cudaEventDestroy(stop_event_); stop_event_ = 0; }

    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_) { engine_->destroy(); engine_ = nullptr; }
    runtime_.reset();

    if (gpu_input_buffer_) { cudaFree(gpu_input_buffer_); gpu_input_buffer_ = nullptr; }
    if (gpu_output_buffer_) { cudaFree(gpu_output_buffer_); gpu_output_buffer_ = nullptr; }
    if (pinned_input_buffer_) { cudaFreeHost(pinned_input_buffer_); pinned_input_buffer_ = nullptr; }
    if (pinned_output_buffer_) { cudaFreeHost(pinned_output_buffer_); pinned_output_buffer_ = nullptr; }

    initialized_ = false;
}

bool TensorRTEngine::infer_batch(
    const std::vector<const uint8_t*>& frames,
    int orig_w,
    int orig_h,
    std::vector<FrameResult>& results
) {
    if (!initialized_ || frames.empty()) return false;

    int batch_size = std::min(static_cast<int>(frames.size()), max_batch_size_);
    results.resize(batch_size);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Preprocessing: Convert RGB uint8 to planar float32
    float* host_input = static_cast<float*>(pinned_input_buffer_);
    size_t plane_size = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
    float scale_x = static_cast<float>(orig_w) / MODEL_INPUT_WIDTH;
    float scale_y = static_cast<float>(orig_h) / MODEL_INPUT_HEIGHT;

    for (int b = 0; b < batch_size; ++b) {
        float* batch_r = host_input + (b * 3 + 0) * plane_size;
        float* batch_g = host_input + (b * 3 + 1) * plane_size;
        float* batch_b = host_input + (b * 3 + 2) * plane_size;
        const uint8_t* img = frames[b];

        for (int y = 0; y < MODEL_INPUT_HEIGHT; ++y) {
            int sy = std::min(static_cast<int>(y * scale_y), orig_h - 1);
            for (int x = 0; x < MODEL_INPUT_WIDTH; ++x) {
                int sx = std::min(static_cast<int>(x * scale_x), orig_w - 1);
                int s_idx = (sy * orig_w + sx) * 3;
                int d_idx = y * MODEL_INPUT_WIDTH + x;

                batch_r[d_idx] = img[s_idx + 0] / 255.0f;
                batch_g[d_idx] = img[s_idx + 1] / 255.0f;
                batch_b[d_idx] = img[s_idx + 2] / 255.0f;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t prep_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Async Host-to-Device Copy
    size_t cur_input_bytes = batch_size * 3 * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH * sizeof(float);
    size_t cur_output_bytes = batch_size * 84 * 8400 * sizeof(float);

    CHECK_CUDA(cudaMemcpyAsync(gpu_input_buffer_, pinned_input_buffer_, cur_input_bytes, cudaMemcpyHostToDevice, stream_));

    // Enqueue Inference on non-blocking stream
    cudaEventRecord(start_event_, stream_);
    context_->enqueueV3(stream_);
    cudaEventRecord(stop_event_, stream_);

    // Async Device-to-Host Copy
    CHECK_CUDA(cudaMemcpyAsync(pinned_output_buffer_, gpu_output_buffer_, cur_output_bytes, cudaMemcpyDeviceToHost, stream_));

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    float infer_ms_cuda = 0.0f;
    cudaEventElapsedTime(&infer_ms_cuda, start_event_, stop_event_);
    int64_t infer_ns = static_cast<int64_t>(infer_ms_cuda * 1e6f);

    auto t2 = std::chrono::high_resolution_clock::now();

    // Postprocessing: Parse YOLO outputs per batch entry
    float* host_output = static_cast<float*>(pinned_output_buffer_);
    int num_anchors = 8400;

    for (int b = 0; b < batch_size; ++b) {
        float* batch_out = host_output + b * (84 * 8400);
        FrameResult& res = results[b];
        res.preprocess_ns = prep_ns;
        res.inference_ns = infer_ns;

        std::vector<Detection> candidates;
        for (int i = 0; i < num_anchors; ++i) {
            float max_score = 0.0f;
            int best_cls = -1;

            for (int c = 0; c < NUM_CLASSES; ++c) {
                float score = batch_out[(4 + c) * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    best_cls = c;
                }
            }

            if (max_score >= 0.25f) {
                float cx = batch_out[0 * num_anchors + i];
                float cy = batch_out[1 * num_anchors + i];
                float w = batch_out[2 * num_anchors + i];
                float h = batch_out[3 * num_anchors + i];

                Detection det;
                det.x1 = std::max(0.0f, (cx - w * 0.5f) * scale_x);
                det.y1 = std::max(0.0f, (cy - h * 0.5f) * scale_y);
                det.x2 = std::min(static_cast<float>(orig_w), (cx + w * 0.5f) * scale_x);
                det.y2 = std::min(static_cast<float>(orig_h), (cy + h * 0.5f) * scale_y);
                det.confidence = max_score;
                det.class_id = best_cls;
                det.compute_foot_point();
                candidates.push_back(det);
            }
        }

        // NMS
        std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

        int det_count = 0;
        std::vector<bool> suppressed(candidates.size(), false);
        for (size_t i = 0; i < candidates.size() && det_count < MAX_DETECTIONS_PER_FRAME; ++i) {
            if (suppressed[i]) continue;
            res.detections[det_count++] = candidates[i];

            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id) {
                    float x1 = std::max(candidates[i].x1, candidates[j].x1);
                    float y1 = std::max(candidates[i].y1, candidates[j].y1);
                    float x2 = std::min(candidates[i].x2, candidates[j].x2);
                    float y2 = std::min(candidates[i].y2, candidates[j].y2);
                    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
                    float a_area = (candidates[i].x2 - candidates[i].x1) * (candidates[i].y2 - candidates[i].y1);
                    float b_area = (candidates[j].x2 - candidates[j].x1) * (candidates[j].y2 - candidates[j].y1);
                    float iou = inter / (a_area + b_area - inter + 1e-6f);
                    if (iou > 0.45f) suppressed[j] = true;
                }
            }
        }
        res.num_detections = det_count;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    int64_t post_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    for (int b = 0; b < batch_size; ++b) {
        results[b].postprocess_ns = post_ns;
    }

    return true;
}

} // namespace kavach_native
