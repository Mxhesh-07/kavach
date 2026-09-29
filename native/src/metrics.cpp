#include "kavach_native/metrics.hpp"
#include <numeric>
#include <cmath>

namespace kavach_native {

void NativeMetrics::record_inference(float ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    infer_samples_.push_back(ms);
    if (infer_samples_.size() > 10000) infer_samples_.erase(infer_samples_.begin());
}

void NativeMetrics::record_pipeline(float ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    pipe_samples_.push_back(ms);
    if (pipe_samples_.size() > 10000) pipe_samples_.erase(pipe_samples_.begin());
}

static StagePercentiles compute(std::vector<float> samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    auto p = [&](float pct) {
        size_t idx = std::min(n - 1, std::max(size_t(0), size_t(std::ceil(pct / 100.0f * n)) - 1));
        return samples[idx];
    };
    float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
    return {samples.front(), sum / n, p(50.0f), p(90.0f), p(95.0f), p(99.0f), samples.back(), n};
}

StagePercentiles NativeMetrics::get_inference_percentiles() {
    std::lock_guard<std::mutex> lock(mtx_);
    return compute(infer_samples_);
}

StagePercentiles NativeMetrics::get_pipeline_percentiles() {
    std::lock_guard<std::mutex> lock(mtx_);
    return compute(pipe_samples_);
}

} // namespace kavach_native
