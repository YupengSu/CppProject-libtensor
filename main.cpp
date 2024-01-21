#include <ctime>

#include "test_cases.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // test_case::specify_init();
    TensorImpl t1 = rand({2,3,4});
    TensorImpl t2 = t1.unsqueeze(1);
    cout << t1 << endl;
    cout << t2 << endl;
    t2.info();
    cout <<(t1 == t2[0]) << endl;
    return 0;
}
