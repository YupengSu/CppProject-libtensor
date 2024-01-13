#include <iostream>

#include "serial_tensor.hpp"

using namespace ts;

int main() {
    Tensor t1 = tensor(
        {
            {0.1, 1.2},
            {2.2, 3.1},
            {4.9, 5.2},

        },
        dt::float32);

    cout << t1 << endl;
    Tensor t2 = tensor(
        {
            {0.2, 1.3},
            {2.2, 3.1},
            {4.9, 5.2},
        },
        dt::float32);
    cout << t2 << endl;

    cout << t1.device << endl;
    Tensor t3 = add(t1, t2);
    cout << t3 << endl;

    t1 = t1.to(dev::cuda);
    t2 = t2.to(dev::cuda);

    cout << t1.device << endl;
    Tensor t4 = add(t1, t2);
    cout << t4 << endl;
}