#include <algorithm>
#include <ios>
#include <vector>

// #include "tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    Tensor t = rand({3, 5, 6});
    cout << t.shape << endl ;
    cout << t << endl << endl;
    Tensor b =  t.slice(2, 0);

    cout << b.shape <<  endl;
    for (int i : b.stride) {
        cout << i << " ";
    }
    cout << endl;
    cout << b(0) << endl;
    cout << b(0)[{4}] << endl;

//     cout << t.shape << endl;
//     cout << t(1,{0,2}) << endl;
//     cout << t.slice(1,1) << endl;
}
