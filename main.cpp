#include <algorithm>
#include <ios>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

// #include "tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    
    // Tensor t1 = rand({3,2});
    // Tensor t2 = rand({3,2});

    // Tensor t1 = rand({2,2});
    // Tensor t2 = rand({2,2});
    // cout << t1 << endl;
    // cout << t2 << endl;
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
    // cout << cat({t1,t2}, 1) << endl;

    // Tensor t3 = tile(t1, {2,2});
    // cout << t3 << endl;
    
    // Tensor t3 = ts::add(t1,t2);
    // cout << t3 << endl;
    // Tensor t4 = t1.add(t2);
    // cout << t4 << endl;
    // Tensor t5 = t1 + t2;
    // cout << t5 << endl;
    // Tensor t6 = ts::add(t1, 1);
    // cout << t6 << endl;
    
    // Tensor t7 = t1.div(2);
    // cout << t7 << endl;

    // Tensor t8 = log(t1);
    // cout << t8 << endl;

    // Tensor t8 = ts::sum(t1, 0);
    // cout << t8 << endl;
    // Tensor t9 = ts::sum(t1, 1);
    // cout << t9 << endl;

    // Tensor t10 = t1.sum(0);
    // cout << t10 << endl;
    // Tensor t11 = t1.sum(1);
    // cout << t11 << endl;

//     cout << t.shape << endl;
//     cout << t(1,{0,2}) << endl;
//     cout << t.slice(1,1) << endl;
}
