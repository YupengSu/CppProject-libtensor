#include <algorithm>
#include <ios>
#include <vector>

// #include "tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    Tensor t = rand({3, 5});
    cout << t << endl << endl;
    cout << t.slice(0, 1) << endl << endl;
    cout << t.shape << endl;
    cout << t(1,{0,2}) << endl;
    cout << t.slice(1,1) << endl;
}
