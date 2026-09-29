#include "kavach_native/scheduler.hpp"
#include <chrono>

namespace kavach_native {

NativeScheduler::NativeScheduler(std::shared_ptr<TensorRTEngine> engine, int max_batch, float max_wait_ms)
    : engine_(engine), max_batch_(max_batch), max_wait_ms_(max_wait_ms) {}

NativeScheduler::~NativeScheduler() {
    stop();
}

void NativeScheduler::start() {
    if (running_.load()) return;
    running_.store(true);
    worker_thread_ = std::thread(&NativeScheduler::worker_loop, this);
}

void NativeScheduler::stop() {
    if (!running_.load()) return;
    running_.store(false);
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

bool NativeScheduler::push_frame(const RawFrame& frame) {
    if (!running_.load()) return false;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    pending_frames_[frame.camera_id] = frame;
    if (std::find(camera_order_.begin(), camera_order_.end(), frame.camera_id) == camera_order_.end()) {
        camera_order_.push_back(frame.camera_id);
    }
    cv_.notify_one();
    return true;
}

std::optional<FrameResult> NativeScheduler::poll_result(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto it = result_queues_.find(camera_id);
    if (it != result_queues_.end()) {
        return it->second.try_pop();
    }
    return std::nullopt;
}

void NativeScheduler::worker_loop() {
    while (running_.load()) {
        std::vector<RawFrame> batch;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() {
                return !running_.load() || !pending_frames_.empty();
            });

            if (!running_.load()) break;

            auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(static_cast<int64_t>(max_wait_ms_ * 1000));
            while (pending_frames_.size() < static_cast<size_t>(max_batch_) && std::chrono::steady_clock::now() < deadline) {
                cv_.wait_until(lock, deadline);
                if (!running_.load()) break;
            }

            if (!running_.load()) break;

            for (const auto& cam_id : camera_order_) {
                auto it = pending_frames_.find(cam_id);
                if (it != pending_frames_.end()) {
                    batch.push_back(it->second);
                    pending_frames_.erase(it);
                    if (batch.size() >= static_cast<size_t>(max_batch_)) break;
                }
            }
        }

        if (batch.empty()) continue;

        std::vector<const uint8_t*> ptrs;
        ptrs.reserve(batch.size());
        for (const auto& f : batch) ptrs.push_back(f.data);

        std::vector<FrameResult> results;
        if (engine_ && engine_->infer_batch(ptrs, batch[0].width, batch[0].height, results)) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (size_t i = 0; i < results.size(); ++i) {
                results[i].camera_id = batch[i].camera_id;
                results[i].sequence_number = batch[i].sequence;
                results[i].capture_timestamp_ns = batch[i].capture_timestamp_ns;
                result_queues_[batch[i].camera_id].try_push_overwrite(results[i]);
            }
        }
    }
}

} // namespace kavach_native
