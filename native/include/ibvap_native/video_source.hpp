#pragma once

#include "frame.hpp"
#include <string>
#include <functional>
#include <atomic>
#include <thread>

namespace ibvap_native {

class NativeVideoSource {
public:
    virtual ~NativeVideoSource() = default;
    virtual bool open(const std::string& uri) = 0;
    virtual void start_capture(std::function<void(const RawFrame&)> on_frame) = 0;
    virtual void stop() = 0;
    virtual bool is_running() const = 0;
};

std::unique_ptr<NativeVideoSource> create_video_source(const std::string& uri);

} // namespace ibvap_native
