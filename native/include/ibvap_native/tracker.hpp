#pragma once

#include "detection.hpp"
#include <vector>
#include <unordered_map>
#include <chrono>

namespace ibvap_native {

struct Track {
    int track_id{0};
    Detection det;
    int age{1};
    int hits{1};
    int time_since_update{0};
    float vx{0.0f};
    float vy{0.0f};
};

class NativeTracker {
public:
    NativeTracker(float max_cosine_distance = 0.2f, int max_age = 30, int n_init = 3);
    ~NativeTracker() = default;

    void update(std::vector<Detection>& detections);
    void reset();

private:
    int next_id_{1};
    int max_age_;
    int n_init_;
    std::unordered_map<int, Track> tracks_;
};

} // namespace ibvap_native
