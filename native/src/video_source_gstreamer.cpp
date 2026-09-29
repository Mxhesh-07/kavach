#include "kavach_native/video_source.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace kavach_native {

class GStreamerVideoSource : public NativeVideoSource {
public:
    GStreamerVideoSource() = default;
    ~GStreamerVideoSource() override { stop(); }

    bool open(const std::string& uri) override {
        uri_ = uri;
        return true;
    }

    void start_capture(std::function<void(const RawFrame&)> on_frame) override {
        if (running_.load()) return;
        running_.store(true);
        capture_thread_ = std::thread([this, on_frame]() {
            uint64_t frame_id = 0;
            const int width = 1920;
            const int height = 1080;
            std::vector<uint8_t> dummy_frame(width * height * 3, 0);

            while (running_.load()) {
                auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();

                RawFrame frame{};
                frame.data = dummy_frame.data();
                frame.width = width;
                frame.height = height;
                frame.channels = 3;
                frame.step = width * 3;
                frame.frame_id = ++frame_id;
                frame.timestamp_ns = now_ns;
                frame.camera_id = 0;

                if (on_frame) {
                    on_frame(frame);
                }

                // Simulate 30 FPS pacing
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        });
    }

    void stop() override {
        if (running_.load()) {
            running_.store(false);
            if (capture_thread_.joinable()) {
                capture_thread_.join();
            }
        }
    }

    bool is_running() const override {
        return running_.load();
    }

private:
    std::string uri_;
    std::atomic<bool> running_{false};
    std::thread capture_thread_;
};

} // namespace kavach_native
