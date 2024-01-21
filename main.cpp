#include <ctime>

#include "test_cases.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    TensorImpl t1 = rand({3}, dt::int8);
    cout << t1 << endl;
    TensorImpl t2 = t1.unsqueeze(1);
    cout << t2 << endl;


}
