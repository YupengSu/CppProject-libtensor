#include <iostream>
#include "serial_tensor.hpp"

using namespace ts;

int main() {

    Tensor t1 = tensor({
                {0.1,1.2},
                {2.2,3.1},
                {4.9,5.2},
        
        }, dt::float32);
        
    cout << t1 << endl;
    Tensor t2 = tensor({
                {0.2,1.3},
                {2.2,3.1},
                {4.9,5.2},
        }, dt::float32);
    cout << t2 << endl;
    cout.setf(ios::boolalpha);
    cout << eq(t1,t2) << endl;
}