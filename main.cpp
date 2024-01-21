#include <ctime>

#include "test_cases.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    test_case::einsum();
    // TensorImpl t1 = ones({64,64});
    // t1 = t1.cuda();
    // t1 = sum(t1, 0);
    // cout << t1 << endl;
}
