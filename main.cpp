#include <ctime>

#include "test_cases.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    TensorImpl t1 = rand({3, 4}, dt::int8);
    TensorImpl t2 = einsum("ij->ji", {t1});

    cout << t1 << endl;
    cout << t2 << endl;
}
