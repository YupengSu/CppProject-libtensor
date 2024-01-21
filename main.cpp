#include <ctime>

#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // TensorImpl t1 = tensor({5, 6, 8}, dt::float32);
    TensorImpl t1 = ones({4});
    TensorImpl t2 = rand({4});
    // TensorImpl t3 = cat({t1, t2}, 1);

    vector<TensorImpl> t_list = {t1, t2};
    
    clock_t start, end_time;
    start = clock();
    cout << t1.device << endl;
    cout << ts::min(t1,0) << endl;
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    t1 = t1.cuda();
    start = clock();
    cout << t1.device << endl;
    cout << ts::min(t1,0) << endl;
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    return 0;
}
