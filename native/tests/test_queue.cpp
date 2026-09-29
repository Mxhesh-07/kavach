#include "ibvap_native/bounded_spsc_queue.hpp"
#include <cassert>
#include <iostream>

using namespace ibvap_native;

void test_queue_push_pop() {
    BoundedSPSCQueue<int, 4> q;
    assert(q.empty());
    assert(q.try_push(10));
    assert(q.try_push(20));
    assert(q.size() == 2);

    auto item1 = q.try_pop();
    assert(item1.has_value() && item1.value() == 10);
    auto item2 = q.try_pop();
    assert(item2.has_value() && item2.value() == 20);
    assert(q.empty());
}

void test_queue_overwrite() {
    BoundedSPSCQueue<int, 4> q;
    for (int i = 0; i < 10; ++i) {
        q.try_push_overwrite(i);
    }
    assert(q.size() == 4);
    auto val = q.try_pop();
    assert(val.has_value() && val.value() == 6);
}

int main() {
    test_queue_push_pop();
    test_queue_overwrite();
    std::cout << "[PASS] test_queue completed successfully" << std::endl;
    return 0;
}
