#include "ibvap_native/tracker.hpp"
#include <cmath>

namespace ibvap_native {

NativeTracker::NativeTracker(float max_cosine_distance, int max_age, int n_init)
    : max_age_(max_age), n_init_(n_init) {}

void NativeTracker::update(std::vector<Detection>& detections) {
    for (auto& det : detections) {
        float best_dist = 100.0f;
        int best_id = -1;

        for (auto& [id, track] : tracks_) {
            float dx = det.foot_x - track.det.foot_x;
            float dy = det.foot_y - track.det.foot_y;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < best_dist && det.class_id == track.det.class_id) {
                best_dist = dist;
                best_id = id;
            }
        }

        if (best_id != -1 && best_dist < 60.0f) {
            det.track_id = best_id;
            Track& t = tracks_[best_id];
            t.det = det;
            t.age++;
            t.hits++;
            t.time_since_update = 0;
            det.age = t.age;
        } else {
            det.track_id = next_id_++;
            Track t;
            t.track_id = det.track_id;
            t.det = det;
            t.age = 1;
            t.hits = 1;
            t.time_since_update = 0;
            tracks_[det.track_id] = t;
        }
    }

    // Prune stale tracks
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        it->second.time_since_update++;
        if (it->second.time_since_update > max_age_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

void NativeTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

} // namespace ibvap_native
