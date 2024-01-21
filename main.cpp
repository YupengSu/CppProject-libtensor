#include <ctime>

#include "test_cases.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
<<<<<<< HEAD
    // test_case::specify_init();
    TensorImpl t1 = rand({2,3,4});
    TensorImpl t2 = t1.unsqueeze(0);
    cout << t1 << endl;
    cout << t2 << endl;
    t2.info();
    cout <<(t1 == t2[0]) << endl;
    return 0;
=======
    test_case::mutating();

>>>>>>> 83e16ccc2250742f9eca23f3c3a39f952ed9d0fd
}
