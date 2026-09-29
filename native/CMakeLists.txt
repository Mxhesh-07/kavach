cmake_minimum_required(VERSION 3.20)
project(kavach_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

set(SOURCES
    src/engine.cpp
    src/scheduler.cpp
    src/tracker.cpp
    src/metrics.cpp
    src/video_source_gstreamer.cpp
    src/video_source_ffmpeg.cpp
    src/preprocess.cu
    src/postprocess.cu
    src/bindings.cpp
)

pybind11_add_module(kavach_native ${SOURCES})

target_link_libraries(kavach_native PRIVATE
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
)
