#include "kavach_native/tracker.hpp"
#include <cassert>
#include <iostream>

using namespace kavach_native;

int main() {
    NativeTracker tracker;
    std::vector<Detection> dets(1);
    dets[0].x1 = 50; dets[0].y1 = 50; dets[0].x2 = 100; dets[0].y2 = 150;
    dets[0].class_id = 0;
    dets[0].compute_foot_point();

    tracker.update(dets);
    assert(dets[0].track_id == 1);

    // Second frame nearby
    dets[0].x1 = 52; dets[0].y1 = 50; dets[0].x2 = 102; dets[0].y2 = 150;
    dets[0].compute_foot_point();
    tracker.update(dets);
    assert(dets[0].track_id == 1);
    assert(dets[0].age == 2);

    std::cout << "[PASS] test_tracker completed successfully" << std::endl;
    return 0;
}
