#include "ibvap_native/scheduler.hpp"
#include <cassert>
#include <iostream>

using namespace ibvap_native;

int main() {
    auto engine = std::make_shared<TensorRTEngine>();
    NativeScheduler scheduler(engine, 4, 3.0f);
    scheduler.start();
    scheduler.stop();
    std::cout << "[PASS] test_scheduler completed successfully" << std::endl;
    return 0;
}
