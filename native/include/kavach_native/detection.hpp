#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kavach_native {

constexpr int MAX_DETECTIONS_PER_FRAME = 128;
constexpr int NUM_CLASSES = 80;

struct Detection {
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
    float confidence{0.0f};
    int class_id{0};
    int track_id{-1};
    float foot_x{0.0f};
    float foot_y{0.0f};
    int age{1};

    void compute_foot_point() noexcept {
        foot_x = (x1 + x2) * 0.5f;
        foot_y = y2;
    }
};

} // namespace kavach_native
