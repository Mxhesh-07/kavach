#pragma once

#include "engine.hpp"
#include "bounded_spsc_queue.hpp"
#include "frame.hpp"
#include "result.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <string>

namespace ibvap_native {

class NativeScheduler {
public:
    NativeScheduler(std::shared_ptr<TensorRTEngine> engine, int max_batch = 4, float max_wait_ms = 3.0f);
    ~NativeScheduler();

    void start();
    void stop();

    bool push_frame(const RawFrame& frame);
    std::optional<FrameResult> poll_result(const std::string& camera_id);

private:
    void worker_loop();

    std::shared_ptr<TensorRTEngine> engine_;
    int max_batch_;
    float max_wait_ms_;

    std::atomic<bool> running_{false};
    std::thread worker_thread_;

    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, RawFrame> pending_frames_;
    std::vector<std::string> camera_order_;

    std::unordered_map<std::string, BoundedSPSCQueue<FrameResult, 64>> result_queues_;
};

} // namespace ibvap_native
