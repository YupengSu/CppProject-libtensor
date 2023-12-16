#include <algorithm>
#include <ios>
#include <vector>

// #include "tensor.hpp"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace ts;
//
int main() {
    Tensor a = tensor({{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}});
    cout << "+++++++++++++++++++++ Tensor A +++++++++++++++++++++" << endl;
    cout << "a ndim " << a.ndim << endl;
    cout << "a offset " << a.data.offset() << endl;
    cout << "a shape " << a.shape << endl;
    cout << a << endl;
    cout << endl;


    cout << "+++++++++++++++++++++ Slice +++++++++++++++++++++" << endl;
    Tensor sl(a.slice(0));
    cout << "slice ndim " << sl.ndim << endl;
    cout << "slice offset " << sl.data.offset() << endl;
    cout << "slice shape " << sl.shape << endl;
    cout << sl << endl;


    cout << "+++++++++++++++++++++ Tensor rd +++++++++++++++++++++" << endl;
    Tensor rd = rand({2, 3, 4});
    cout << "rd ndim " << rd.ndim << endl;
    cout << "rd offset " << rd.data.offset() << endl;
    cout << "rd shape " << rd.shape << endl;
    cout << rd << endl;

    cout << "+++++++++++++++++++++ Tensor zr +++++++++++++++++++++" << endl;
    Tensor zr = zeros({2, 3, 4});
    cout << "zr ndim " << zr.ndim << endl;
    cout << "zr offset " << zr.data.offset() << endl;
    cout << "zr shape " << zr.shape << endl;
    cout << zr << endl;

    cout << "+++++++++++++++++++++ Tensor on +++++++++++++++++++++" << endl;
    Tensor on = ones({2, 3, 4});
    cout << "on ndim " << on.ndim << endl;
    cout << "on offset " << on.data.offset() << endl;
    cout << "on shape " << on.shape << endl;
    cout << on << endl;


    cout << "+++++++++++++++++++++ Tensor fl +++++++++++++++++++++" << endl;
    Tensor fl = full({1, 2, 3, 4}, 1.14514);
    cout << "fl ndim " << fl.ndim << endl;
    cout << "fl offset " << fl.data.offset() << endl;
    cout << "fl shape " << fl.shape << endl;
    cout << fl << endl;

    cout << "+++++++++++++++++++++ Tensor ey +++++++++++++++++++++" << endl;
    Tensor ey = eye(2);
    cout << "ey ndim " << ey.ndim << endl;
    cout << "ey offset " << ey.data.offset() << endl;
    cout << "ey shape " << ey.shape << endl;
    cout << ey << endl;
}
