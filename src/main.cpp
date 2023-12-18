#include <iostream>
#include <vector>
#include "config.hpp"
#include "serial_tensor.hpp"

// #include "tensor.hpp"


using namespace ts;
//
int main() {
    
    // Tensor t1 = (rand({5, 2}));
    Tensor t1 = tensor({1,2,3,4}, ts::float32);
    cout << t1 << endl << endl;

    t1.view( {2,5} )[{0,1}] = 1;
    cout << t1;
}