#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(err) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)
