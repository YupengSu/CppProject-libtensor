#include <iostream>
#include "serial_tensor.hpp"

using namespace ts;

int main() {
    

    Tensor t1 = rand({5,6}, dt::bool8);
    cout << t1.type() << endl;
    cout << t1.size() << endl;
    cout << t1.data_ptr() << endl;
     (t1[{0,1}] = 0);
    cout << t1;

}