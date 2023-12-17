#include <algorithm>
#include <ios>
#include <iostream>
#include <vector>

// #include "tensor.hpp"
#include "base_tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    
    Tensor t1 = (rand({5, 2}));
    cout << t1 << endl<<endl;

    t1.view( {2,5} )[{0,1}]=1;
    cout << t1;
}