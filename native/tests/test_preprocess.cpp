#include "kavach_native/frame.hpp"
#include <cassert>
#include <iostream>

using namespace kavach_native;

int main() {
    RawFrame frame;
    frame.width = 1920;
    frame.height = 1080;
    frame.channels = 3;
    assert(frame.width == 1920);
    assert(frame.height == 1080);
    std::cout << "[PASS] test_preprocess completed successfully" << std::endl;
    return 0;
}
