#pragma once

#include "detection.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ibvap_native {

struct FrameResult {
    std::string camera_id;
    uint64_t sequence_number{0};
    int64_t capture_timestamp_ns{0};
    int64_t preprocess_ns{0};
    int64_t inference_ns{0};
    int64_t postprocess_ns{0};
    int64_t tracking_ns{0};
    
    int num_detections{0};
    Detection detections[MAX_DETECTIONS_PER_FRAME];
    
    float preprocess_ms() const noexcept { return static_cast<float>(preprocess_ns) / 1e6f; }
    float inference_ms() const noexcept { return static_cast<float>(inference_ns) / 1e6f; }
    float postprocess_ms() const noexcept { return static_cast<float>(postprocess_ns) / 1e6f; }
    float tracking_ms() const noexcept { return static_cast<float>(tracking_ns) / 1e6f; }
    float total_ms() const noexcept {
        return static_cast<float>(preprocess_ns + inference_ns + postprocess_ns + tracking_ns) / 1e6f;
    }
};

} // namespace ibvap_native
