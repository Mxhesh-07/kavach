#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ibvap_native/engine.hpp"
#include "ibvap_native/scheduler.hpp"
#include "ibvap_native/tracker.hpp"
#include "ibvap_native/metrics.hpp"

namespace py = pybind11;
using namespace ibvap_native;

PYBIND11_MODULE(ibvap_native, m) {
    m.doc() = "IBVAP Native C++20 / TensorRT / CUDA Engine";

    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("x1", &Detection::x1)
        .def_readwrite("y1", &Detection::y1)
        .def_readwrite("x2", &Detection::x2)
        .def_readwrite("y2", &Detection::y2)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("track_id", &Detection::track_id)
        .def_readwrite("foot_x", &Detection::foot_x)
        .def_readwrite("foot_y", &Detection::foot_y)
        .def_readwrite("age", &Detection::age);

    py::class_<FrameResult>(m, "FrameResult")
        .def(py::init<>())
        .def_readwrite("camera_id", &FrameResult::camera_id)
        .def_readwrite("sequence_number", &FrameResult::sequence_number)
        .def_property_readonly("preprocess_ms", &FrameResult::preprocess_ms)
        .def_property_readonly("inference_ms", &FrameResult::inference_ms)
        .def_property_readonly("postprocess_ms", &FrameResult::postprocess_ms)
        .def_property_readonly("total_ms", &FrameResult::total_ms)
        .def("get_detections", [](const FrameResult& res) {
            py::list list;
            for (int i = 0; i < res.num_detections; ++i) {
                list.append(res.detections[i]);
            }
            return list;
        });

    py::class_<TensorRTEngine, std::shared_ptr<TensorRTEngine>>(m, "TensorRTEngine")
        .def(py::init<>())
        .def("initialize", &TensorRTEngine::initialize, py::arg("engine_path"), py::arg("max_batch_size") = 4)
        .def("shutdown", &TensorRTEngine::shutdown)
        .def("is_initialized", &TensorRTEngine::is_initialized)
        .def("get_max_batch_size", &TensorRTEngine::get_max_batch_size);

    py::class_<NativeScheduler>(m, "NativeScheduler")
        .def(py::init<std::shared_ptr<TensorRTEngine>, int, float>(),
             py::arg("engine"), py::arg("max_batch") = 4, py::arg("max_wait_ms") = 3.0f)
        .def("start", &NativeScheduler::start)
        .def("stop", &NativeScheduler::stop);

    py::class_<NativeTracker>(m, "NativeTracker")
        .def(py::init<float, int, int>(),
             py::arg("max_cosine_distance") = 0.2f, py::arg("max_age") = 30, py::arg("n_init") = 3)
        .def("update", &NativeTracker::update)
        .def("reset", &NativeTracker::reset);
}
