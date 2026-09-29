//
//  military_core.cpp - High-Performance Border Video Analytics Engine
//  Optimized for sub-2ms end-to-end latency on NVIDIA RTX 3060 Ti / Jetson / dGPU
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <cmath>

namespace py = pybind11;
using namespace nvinfer1;

// -----------------------------------------------------------------------------
// Configuration Constants
// -----------------------------------------------------------------------------

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 384;
constexpr int MAX_BATCH_SIZE = 8;
constexpr int MAX_DETECTIONS = 100;
constexpr int NUM_CLASSES = 80;  // Standard COCO classes
constexpr float CONF_THRESH = 0.25f;
constexpr float IOU_THRESH = 0.45f;

// -----------------------------------------------------------------------------
// CUDA Error Checking Macro
// -----------------------------------------------------------------------------

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// Logger for TensorRT
// -----------------------------------------------------------------------------

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static TRTLogger gTRTLogger;

// -----------------------------------------------------------------------------
// Pre-allocated Buffers
// -----------------------------------------------------------------------------

struct InferenceBuffers {
    void* gpu_input_buffer{nullptr};
    void* gpu_output_buffer{nullptr};

    void* pinned_input_buffer{nullptr};
    void* pinned_output_buffer{nullptr};

    size_t input_size{0};
    size_t output_size{0};
    size_t num_output_elements{0};
    int batch_size{1};

    std::string input_tensor_name;
    std::string output_tensor_name;
};

// -----------------------------------------------------------------------------
// Detection Structure
// -----------------------------------------------------------------------------

struct RawDetection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// -----------------------------------------------------------------------------
// GStreamer Capture Engine
// -----------------------------------------------------------------------------

class GStreamerCapture {
private:
    GstElement* pipeline{nullptr};
    GstAppSink* appsink{nullptr};
    std::atomic<bool> running{false};
    std::thread capture_thread;
    std::mutex frame_mutex;

    std::vector<uint8_t> latest_frame;
    int width{INPUT_WIDTH};
    int height{INPUT_HEIGHT};
    int channels{3};
    std::atomic<uint64_t> frame_count{0};

public:
    GStreamerCapture() = default;

    ~GStreamerCapture() {
        stop();
    }

    bool initialize(const std::string& rtsp_url) {
        gst_init(nullptr, nullptr);

        // Hardware-accelerated pipeline targeting NVIDIA NVMM
        std::string nv_pipeline =
            "rtspsrc location=" + rtsp_url + " latency=0 buffer-mode=auto drop-on-latency=true ! "
            "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
            "appsink name=appsink sync=false max-buffers=1 drop=true";

        // Fallback pipeline for generic hosts
        std::string fallback_pipeline =
            "rtspsrc location=" + rtsp_url + " latency=0 buffer-mode=auto drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=appsink sync=false max-buffers=1 drop=true";

        pipeline = gst_parse_launch(nv_pipeline.c_str(), nullptr);
        if (!pipeline) {
            pipeline = gst_parse_launch(fallback_pipeline.c_str(), nullptr);
        }
        if (!pipeline) {
            return false;
        }

        appsink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline), "appsink"));
        if (!appsink) {
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return false;
        }

        return true;
    }

    void start() {
        if (running.load() || !pipeline) return;
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
        running.store(true);
        capture_thread = std::thread(&GStreamerCapture::capture_loop, this);
    }

    void stop() {
        if (!running.load()) return;
        running.store(false);

        if (capture_thread.joinable()) {
            capture_thread.join();
        }

        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            if (appsink) {
                gst_object_unref(appsink);
                appsink = nullptr;
            }
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
    }

    std::vector<uint8_t> get_latest_frame() {
        std::lock_guard<std::mutex> lock(frame_mutex);
        return latest_frame;
    }

    uint64_t get_frame_count() const {
        return frame_count.load();
    }

private:
    void capture_loop() {
        while (running.load() && appsink) {
            GstSample* sample = gst_app_sink_pull_sample(appsink);
            if (!sample) continue;

            GstBuffer* buffer = gst_sample_get_buffer(sample);
            if (buffer) {
                GstMapInfo info;
                if (gst_buffer_map(buffer, &info, GST_MAP_READ)) {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    size_t frame_size = static_cast<size_t>(width * height * channels);
                    if (info.size >= frame_size) {
                        latest_frame.resize(frame_size);
                        std::memcpy(latest_frame.data(), info.data, frame_size);
                        frame_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    gst_buffer_unmap(buffer, &info);
                }
            }
            gst_sample_unref(sample);
        }
    }
};

// -----------------------------------------------------------------------------
// TensorRT Inference Engine
// -----------------------------------------------------------------------------

class TRTInferenceEngine {
private:
    std::unique_ptr<IRuntime> runtime;
    ICudaEngine* engine{nullptr};
    IExecutionContext* context{nullptr};

    InferenceBuffers buffers;
    cudaStream_t cuda_stream{0};

public:
    TRTInferenceEngine() = default;

    ~TRTInferenceEngine() {
        destroy();
    }

    void destroy() {
        if (cuda_stream) {
            cudaStreamSynchronize(cuda_stream);
            cudaStreamDestroy(cuda_stream);
            cuda_stream = 0;
        }

        if (context) {
            context->destroy();
            context = nullptr;
        }
        if (engine) {
            engine->destroy();
            engine = nullptr;
        }
        runtime.reset();

        if (buffers.gpu_input_buffer) {
            cudaFree(buffers.gpu_input_buffer);
            buffers.gpu_input_buffer = nullptr;
        }
        if (buffers.gpu_output_buffer) {
            cudaFree(buffers.gpu_output_buffer);
            buffers.gpu_output_buffer = nullptr;
        }
        if (buffers.pinned_input_buffer) {
            cudaFreeHost(buffers.pinned_input_buffer);
            buffers.pinned_input_buffer = nullptr;
        }
        if (buffers.pinned_output_buffer) {
            cudaFreeHost(buffers.pinned_output_buffer);
            buffers.pinned_output_buffer = nullptr;
        }
    }

    bool initialize(const std::string& engine_path) {
        runtime = std::unique_ptr<IRuntime>(createInferRuntime(gTRTLogger));
        if (!runtime) return false;

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open()) return false;

        file.seekg(0, std::ios::end);
        size_t engine_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(engine_size);
        file.read(engine_data.data(), engine_size);
        file.close();

        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
        if (!engine) return false;

        context = engine->createExecutionContext();
        if (!context) return false;

        CHECK_CUDA(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

        // Get tensor names (TRT 10 & 8.6+ compatibility)
        int32_t nbIOTensors = engine->getNbIOTensors();
        for (int32_t i = 0; i < nbIOTensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            TensorIOMode mode = engine->getTensorIOMode(name);
            Dims dims = engine->getTensorShape(name);

            if (mode == TensorIOMode::kINPUT) {
                buffers.input_tensor_name = name;
                buffers.batch_size = (dims.d[0] > 0) ? dims.d[0] : 1;
                buffers.input_size = buffers.batch_size * 3 * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
            } else if (mode == TensorIOMode::kOUTPUT) {
                buffers.output_tensor_name = name;
                size_t num_elements = 1;
                for (int d = 0; d < dims.nbDims; ++d) {
                    num_elements *= (dims.d[d] > 0 ? dims.d[d] : 1);
                }
                buffers.num_output_elements = num_elements;
                buffers.output_size = num_elements * sizeof(float);
            }
        }

        if (buffers.input_tensor_name.empty()) buffers.input_tensor_name = "images";
        if (buffers.output_tensor_name.empty()) buffers.output_tensor_name = "output0";

        if (buffers.input_size == 0) {
            buffers.input_size = 1 * 3 * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
        }
        if (buffers.output_size == 0) {
            buffers.num_output_elements = 1 * 84 * 8400;
            buffers.output_size = buffers.num_output_elements * sizeof(float);
        }

        // Allocate device and pinned memory
        CHECK_CUDA(cudaMalloc(&buffers.gpu_input_buffer, buffers.input_size));
        CHECK_CUDA(cudaMalloc(&buffers.gpu_output_buffer, buffers.output_size));
        CHECK_CUDA(cudaHostAlloc(&buffers.pinned_input_buffer, buffers.input_size, cudaHostAllocDefault | cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc(&buffers.pinned_output_buffer, buffers.output_size, cudaHostAllocDefault | cudaHostAllocPortable));

        // Set tensor addresses
        context->setTensorAddress(buffers.input_tensor_name.c_str(), buffers.gpu_input_buffer);
        context->setTensorAddress(buffers.output_tensor_name.c_str(), buffers.gpu_output_buffer);

        return true;
    }

    std::vector<RawDetection> infer(const uint8_t* image_rgb, int orig_w, int orig_h) {
        // Preprocessing: RGB uint8 -> planar float32 [1, 3, H, W] normalized
        float* host_input = static_cast<float*>(buffers.pinned_input_buffer);
        float scale_x = static_cast<float>(orig_w) / INPUT_WIDTH;
        float scale_y = static_cast<float>(orig_h) / INPUT_HEIGHT;

        size_t plane_size = INPUT_WIDTH * INPUT_HEIGHT;
        float* plane_r = host_input;
        float* plane_g = host_input + plane_size;
        float* plane_b = host_input + 2 * plane_size;

        for (int y = 0; y < INPUT_HEIGHT; ++y) {
            int src_y = std::min(static_cast<int>(y * scale_y), orig_h - 1);
            for (int x = 0; x < INPUT_WIDTH; ++x) {
                int src_x = std::min(static_cast<int>(x * scale_x), orig_w - 1);
                int src_idx = (src_y * orig_w + src_x) * 3;
                int dst_idx = y * INPUT_WIDTH + x;

                plane_r[dst_idx] = image_rgb[src_idx] / 255.0f;
                plane_g[dst_idx] = image_rgb[src_idx + 1] / 255.0f;
                plane_b[dst_idx] = image_rgb[src_idx + 2] / 255.0f;
            }
        }

        // Async Memcpy H2D
        CHECK_CUDA(cudaMemcpyAsync(buffers.gpu_input_buffer, buffers.pinned_input_buffer,
                                    buffers.input_size, cudaMemcpyHostToDevice, cuda_stream));

        // Enqueue inference
        context->enqueueV3(cuda_stream);

        // Async Memcpy D2H
        CHECK_CUDA(cudaMemcpyAsync(buffers.pinned_output_buffer, buffers.gpu_output_buffer,
                                    buffers.output_size, cudaMemcpyDeviceToHost, cuda_stream));

        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));

        // Parse YOLO output
        return parse_output(static_cast<float*>(buffers.pinned_output_buffer), orig_w, orig_h);
    }

private:
    static float compute_iou(const RawDetection& a, const RawDetection& b) {
        float x1 = std::max(a.x1, b.x1);
        float y1 = std::max(a.y1, b.y1);
        float x2 = std::min(a.x2, b.x2);
        float y2 = std::min(a.y2, b.y2);

        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;

        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union_area = area_a + area_b - inter_area;

        return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
    }

    std::vector<RawDetection> parse_output(float* output, int orig_w, int orig_h) {
        std::vector<RawDetection> candidates;
        candidates.reserve(MAX_DETECTIONS);

        // Typical YOLO shape: [1, 84, 8400] where 84 = 4 box coords (cx, cy, w, h) + 80 class scores
        int num_anchors = 8400;
        int num_channels = 84;
        float sx = static_cast<float>(orig_w) / INPUT_WIDTH;
        float sy = static_cast<float>(orig_h) / INPUT_HEIGHT;

        for (int i = 0; i < num_anchors; ++i) {
            float max_score = 0.0f;
            int best_class = -1;

            for (int c = 0; c < NUM_CLASSES; ++c) {
                float score = output[(4 + c) * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    best_class = c;
                }
            }

            if (max_score >= CONF_THRESH) {
                float cx = output[0 * num_anchors + i];
                float cy = output[1 * num_anchors + i];
                float w = output[2 * num_anchors + i];
                float h = output[3 * num_anchors + i];

                RawDetection det;
                det.x1 = std::max(0.0f, (cx - w * 0.5f) * sx);
                det.y1 = std::max(0.0f, (cy - h * 0.5f) * sy);
                det.x2 = std::min(static_cast<float>(orig_w), (cx + w * 0.5f) * sx);
                det.y2 = std::min(static_cast<float>(orig_h), (cy + h * 0.5f) * sy);
                det.confidence = max_score;
                det.class_id = best_class;
                candidates.push_back(det);
            }
        }

        // Sort by confidence descending
        std::sort(candidates.begin(), candidates.end(), [](const RawDetection& a, const RawDetection& b) {
            return a.confidence > b.confidence;
        });

        // NMS
        std::vector<RawDetection> result;
        std::vector<bool> suppressed(candidates.size(), false);

        for (size_t i = 0; i < candidates.size() && result.size() < MAX_DETECTIONS; ++i) {
            if (suppressed[i]) continue;
            result.push_back(candidates[i]);

            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id) {
                    if (compute_iou(candidates[i], candidates[j]) > IOU_THRESH) {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return result;
    }
};

// -----------------------------------------------------------------------------
// Façade Class Exposed to Python
// -----------------------------------------------------------------------------

class MilitaryCore {
private:
    std::unique_ptr<GStreamerCapture> capture;
    std::unique_ptr<TRTInferenceEngine> inference;
    std::atomic<bool> initialized{false};
    std::string engine_path{"yolo11s.engine"};

    std::atomic<uint64_t> total_frames{0};
    std::atomic<double> total_inference_ms{0.0};

public:
    MilitaryCore() = default;

    ~MilitaryCore() {
        shutdown();
    }

    bool init(const std::string& rtsp_url = "", const std::string& engine_file = "yolo11s.engine") {
        if (initialized.load()) return true;

        engine_path = engine_file;
        inference = std::make_unique<TRTInferenceEngine>();
        if (!inference->initialize(engine_path)) {
            inference.reset();
            return false;
        }

        if (!rtsp_url.empty()) {
            capture = std::make_unique<GStreamerCapture>();
            if (capture->initialize(rtsp_url)) {
                capture->start();
            }
        }

        initialized.store(true);
        return true;
    }

    void shutdown() {
        if (!initialized.load()) return;
        if (capture) {
            capture->stop();
            capture.reset();
        }
        if (inference) {
            inference->destroy();
            inference.reset();
        }
        initialized.store(false);
    }

    py::dict process_frame(py::array_t<uint8_t> input_frame) {
        py::buffer_info buf = input_frame.request();

        if (buf.ndim != 3 || buf.shape[2] != 3) {
            throw std::runtime_error("Input frame must be HxWxC RGB uint8 array");
        }

        int height = static_cast<int>(buf.shape[0]);
        int width = static_cast<int>(buf.shape[1]);
        const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<RawDetection> raw_dets;

        if (inference) {
            py::gil_scoped_release release;
            raw_dets = inference->infer(ptr, width, height);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(end - start).count();

        total_frames.fetch_add(1, std::memory_order_relaxed);
        double prev = total_inference_ms.load(std::memory_order_relaxed);
        while (!total_inference_ms.compare_exchange_weak(prev, prev + inference_ms, std::memory_order_relaxed));

        py::list detections_list;
        for (size_t i = 0; i < raw_dets.size(); ++i) {
            py::dict det;
            det["track_id"] = static_cast<int>(i);
            det["class_id"] = raw_dets[i].class_id;
            det["confidence"] = raw_dets[i].confidence;
            det["x1"] = raw_dets[i].x1;
            det["y1"] = raw_dets[i].y1;
            det["x2"] = raw_dets[i].x2;
            det["y2"] = raw_dets[i].y2;
            detections_list.append(det);
        }

        py::dict result;
        result["detections"] = detections_list;
        result["inference_ms"] = inference_ms;
        return result;
    }

    py::dict get_stats() {
        py::dict stats;
        uint64_t count = total_frames.load();
        double total_ms = total_inference_ms.load();
        stats["total_frames"] = count;
        stats["avg_inference_ms"] = (count > 0) ? (total_ms / count) : 0.0;
        stats["initialized"] = initialized.load();
        stats["engine_path"] = engine_path;
        return stats;
    }
};

// -----------------------------------------------------------------------------
// Pybind11 Module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(military_core, m) {
    m.doc() = "High-performance military-grade edge analytics engine with TensorRT and CUDA";

    py::class_<MilitaryCore>(m, "MilitaryCore")
        .def(py::init<>())
        .def("init", &MilitaryCore::init,
             py::arg("rtsp_url") = "",
             py::arg("engine_file") = "yolo11s.engine",
             "Initialize the TensorRT hardware engine")
        .def("shutdown", &MilitaryCore::shutdown, "Shutdown and release GPU resources")
        .def("process_frame", &MilitaryCore::process_frame,
             py::arg("frame"),
             "Process a single uint8 RGB frame with hardware line-rate inference")
        .def("get_stats", &MilitaryCore::get_stats, "Get latency and execution statistics");

    m.attr("INPUT_WIDTH") = INPUT_WIDTH;
    m.attr("INPUT_HEIGHT") = INPUT_HEIGHT;
    m.attr("MAX_BATCH_SIZE") = MAX_BATCH_SIZE;
    m.attr("MAX_DETECTIONS") = MAX_DETECTIONS;
}
