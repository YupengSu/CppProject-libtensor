#include <ctime>

#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // TensorImpl t1 = tensor({5, 6, 8}, dt::float32);
    TensorImpl t1 = ones({1});
    TensorImpl t2 = ones({2,3,4});
    // TensorImpl t3 = cat({t1, t2}, 1);
    TensorImpl t3 = t1;
    cout << t3 << endl;
    cout << tile(t3, {5}) << endl;

    return 0;
}
