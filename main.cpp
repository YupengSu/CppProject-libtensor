#include <ctime>

#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    Tensor t1 = tensor({5, 6, 8}, dt::float32);
    Tensor t2 = tensor({5, 6, 7}, dt::float32);
    t1 = t1.to(dev::cuda);
    t2 = t2.to(dev::cuda);
    cerr << "Done To Cuda" << endl;
    cout << t1 << endl;
    Tensor t3 = mul(t1, t2);
    // t3.info("T3");
    cout << t3 << endl;
    cout << ts::min(t3, 0) << endl;

    Tensor t4(ts::rand({2}));
    // t4.info("T4");
    cout << t4 << endl;
    cout << ts::min(t4, 0) << endl;

    Tensor t5 (rand({5,5}));
    cout << ts::min(t5, 1) << endl;
    cout << t2.min(0) << endl;
    cout << (t1 >= t2);
    return 0;
}
