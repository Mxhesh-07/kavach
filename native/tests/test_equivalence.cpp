#include "kavach_native/detection.hpp"
#include "kavach_native/result.hpp"
#include <cassert>
#include <iostream>

using namespace kavach_native;

int main() {
    FrameResult res;
    res.camera_id = "cam01";
    res.sequence_number = 100;
    res.preprocess_ns = 500000;
    res.inference_ns = 2500000;
    res.postprocess_ns = 200000;
    res.tracking_ns = 300000;

    assert(res.preprocess_ms() == 0.5f);
    assert(res.inference_ms() == 2.5f);
    assert(res.total_ms() == 3.5f);

    std::cout << "[PASS] test_equivalence completed successfully" << std::endl;
    return 0;
}
