#include <algorithm>
#include <ios>
#include <vector>

#include "tensor.hpp"

using namespace ts;

int main() {
    // Tensor a = tensor({{{2.1, 2, 3}, {2, 3, 4}, {1, 2, 3}},
    //                               {{2, 2, 3}, {2, 3, 4}, {1, 2, 3}}});
    // cout << a << endl;

    Tensor<> b = Tensor<>({{{2.1, 2, 3}, {2, 3, 4}, {1, 2, 3}},
                                     {{2, 2, 3}, {2, 3, 4}, {1, 2, 3}}}, int8);
    cout << b << endl;
}