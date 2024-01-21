#include <ctime>

#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // TensorImpl t1 = tensor({5, 6, 8}, dt::float32);
    TensorImpl t1 = ones({256,256});
    TensorImpl t2 = ones({256,256});
    // TensorImpl t3 = cat({t1, t2}, 1);

    vector<TensorImpl> t_list = {t1, t2};
    
    clock_t start, end_time;
    start = clock();
    cout << t1.device << endl;
    ts::matrix_multiply(t1, t2);
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    t1 = t1.cuda();
    t2 = t2.cuda();
    start = clock();
    cout << t1.device << endl;
    ts::matrix_multiply(t1, t2);
    end_time = clock();
    cout << "time: " << (double)(end_time - start) / CLOCKS_PER_SEC << endl;

    return 0;
}
