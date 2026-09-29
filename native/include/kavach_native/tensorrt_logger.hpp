#pragma once

#include <NvInfer.h>
#include <iostream>

namespace kavach_native {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

inline TRTLogger& get_trt_logger() {
    static TRTLogger logger;
    return logger;
}

} // namespace kavach_native
