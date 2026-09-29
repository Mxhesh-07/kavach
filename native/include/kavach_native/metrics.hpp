#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <mutex>

namespace kavach_native {

struct StagePercentiles {
    float min_ms{0.0f};
    float mean_ms{0.0f};
    float p50_ms{0.0f};
    float p90_ms{0.0f};
    float p95_ms{0.0f};
    float p99_ms{0.0f};
    float max_ms{0.0f};
    size_t count{0};
};

class NativeMetrics {
public:
    static NativeMetrics& instance() {
        static NativeMetrics inst;
        return inst;
    }

    void record_inference(float ms);
    void record_pipeline(float ms);
    StagePercentiles get_inference_percentiles();
    StagePercentiles get_pipeline_percentiles();

private:
    std::mutex mtx_;
    std::vector<float> infer_samples_;
    std::vector<float> pipe_samples_;
};

} // namespace kavach_native
