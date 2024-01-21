#include <ctime>

#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // TensorImpl t1 = tensor({5, 6, 8}, dt::float32);
    TensorImpl t1 = ones({1});
    TensorImpl t2 = rand({4,4,4});
    // TensorImpl t3 = cat({t1, t2}, 1);
    TensorImpl t3 = t1;
    cout << t3 << endl;
    
    clock_t start, end_time;
    start = clock();
    cout << t2.device << endl;
    cout << t2.mean(1) << endl;
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    t2 = t2.cuda();
    start = clock();
    cout << t2.device << endl;
    cout << t2.mean(1) << endl;
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    return 0;
}
