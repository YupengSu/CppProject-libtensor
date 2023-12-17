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
    
    Tensor t1 (rand({2, 2}));
    cout << t1 << endl;
    // cout << t1({1,0}) << endl;
}
