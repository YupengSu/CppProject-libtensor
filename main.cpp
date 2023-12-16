#include <algorithm>
#include <ios>
#include <iostream>
#include <vector>

// #include "tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    
    Tensor t1 = rand({3,2});
    Tensor t2 = rand({3,2});
    cout << t1 << endl;
    cout << t2 << endl;
    
    cout << cat({t1,t2}, 1) << endl;

    Tensor t3 = tile(t1, {2,2});
    cout << t3 << endl;

//     cout << t.shape << endl;
//     cout << t(1,{0,2}) << endl;
//     cout << t.slice(1,1) << endl;
}
