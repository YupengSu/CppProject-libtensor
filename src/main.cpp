#include <iostream>
#include "config.hpp"
#include "serial_tensor.hpp"

// #include "tensor.hpp"


using namespace ts;

int main() {
    

    Tensor t1 = rand({5,6}, dt::int8);
    cout << t1.type() << endl;
    cout << t1.size() << endl;
    cout << t1.data_ptr() << endl;
    cout << t1;

}