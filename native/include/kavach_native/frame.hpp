#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <chrono>

namespace kavach_native {

constexpr int FRAME_MAX_WIDTH = 1920;
constexpr int FRAME_MAX_HEIGHT = 1080;
constexpr int MODEL_INPUT_WIDTH = 640;
constexpr int MODEL_INPUT_HEIGHT = 384;

struct RawFrame {
    uint8_t* data{nullptr};
    int width{0};
    int height{0};
    int channels{3};
    size_t size_bytes{0};
    uint64_t sequence{0};
    std::string camera_id;
    int64_t capture_timestamp_ns{0};
    bool is_gpu_resident{false};
};

} // namespace kavach_native
