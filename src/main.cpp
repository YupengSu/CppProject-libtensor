#include <iostream>
#include "serial_tensor.hpp"

using namespace ts;

int main() {
    

    Tensor t1 = tensor({
        {
            {
                {1,2},
                {3,4}
            },
            {   
                {5,6},
                {7,8}
            },
            {
                {9,10},
                {11,12}
            }
        }
        }, dt::int32);
    cout << t1;

}