#include "kavach_native/detection.hpp"
#include <cassert>
#include <iostream>

using namespace kavach_native;

int main() {
    Detection det;
    det.x1 = 100.0f;
    det.y1 = 100.0f;
    det.x2 = 200.0f;
    det.y2 = 300.0f;
    det.confidence = 0.95f;
    det.class_id = 0;
    det.compute_foot_point();

    assert(det.foot_x == 150.0f);
    assert(det.foot_y == 300.0f);
    std::cout << "[PASS] test_postprocess completed successfully" << std::endl;
    return 0;
}
