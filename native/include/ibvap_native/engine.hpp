#pragma once

#include "cuda_check.hpp"
#include "frame.hpp"
#include "detection.hpp"
#include "result.hpp"
#include "tensorrt_logger.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
#include <string>
#include <vector>

namespace ibvap_native {

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    bool initialize(const std::string& engine_path, int max_batch_size = 4);
    void shutdown();

    bool infer_batch(
        const std::vector<const uint8_t*>& frames,
        int orig_w,
        int orig_h,
        std::vector<FrameResult>& results
    );

    int get_max_batch_size() const noexcept { return max_batch_size_; }
    bool is_initialized() const noexcept { return initialized_; }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    cudaStream_t stream_{0};
    cudaEvent_t start_event_{0};
    cudaEvent_t stop_event_{0};

    void* gpu_input_buffer_{nullptr};
    void* gpu_output_buffer_{nullptr};
    void* pinned_input_buffer_{nullptr};
    void* pinned_output_buffer_{nullptr};

    size_t input_size_bytes_{0};
    size_t output_size_bytes_{0};
    int max_batch_size_{4};
    bool initialized_{false};

    std::string input_tensor_name_{"images"};
    std::string output_tensor_name_{"output0"};
};

} // namespace ibvap_native
