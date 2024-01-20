#include "test_cases.hpp"

namespace ts {

void basic() {
    TensorImpl a = tensor({{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}});
    cout << "+++++++++++++++++++++ Tensor A +++++++++++++++++++++" << endl;
    cout << "a ndim " << a.ndim << endl;
    cout << "a offset " << a.data.offset() << endl;
    cout << "a shape " << a.shape << endl;
    cout << a << endl;
    cout << endl;

    cout << "+++++++++++++++++++++ Slice +++++++++++++++++++++" << endl;
    TensorImpl sl(a.slice(0));
    cout << "slice ndim " << sl.ndim << endl;
    cout << "slice offset " << sl.data.offset() << endl;
    cout << "slice shape " << sl.shape << endl;
    cout << sl << endl;

    cout << "+++++++++++++++++++++ Tensor rand +++++++++++++++++++++" << endl;
    TensorImpl rd = rand({2, 3, 4});
    cout << "rand ndim " << rd.ndim << endl;
    cout << "rand offset " << rd.data.offset() << endl;
    cout << "rand shape " << rd.shape << endl;
    cout << rd << endl;

    cout << "+++++++++++++++++++++ Tensor zeros +++++++++++++++++++++" << endl;
    TensorImpl zr = zeros({2, 3, 4});
    cout << "zeros ndim " << zr.ndim << endl;
    cout << "zeros offset " << zr.data.offset() << endl;
    cout << "zeros shape " << zr.shape << endl;
    cout << zr << endl;

    cout << "+++++++++++++++++++++ Tensor ones +++++++++++++++++++++" << endl;
    TensorImpl on = ones({2, 3, 4});
    cout << "ones ndim " << on.ndim << endl;
    cout << "ones offset " << on.data.offset() << endl;
    cout << "ones shape " << on.shape << endl;
    cout << on << endl;

    cout << "+++++++++++++++++++++ Tensor full +++++++++++++++++++++" << endl;
    TensorImpl fl = full({1, 2, 3, 4}, 1.14514);
    cout << "fl ndim " << fl.ndim << endl;
    cout << "fl offset " << fl.data.offset() << endl;
    cout << "fl shape " << fl.shape << endl;
    cout << fl << endl;

    cout << "+++++++++++++++++++++ Tensor eye +++++++++++++++++++++" << endl;
    TensorImpl ey = eye(2);
    cout << "eye ndim " << ey.ndim << endl;
    cout << "eye offset " << ey.data.offset() << endl;
    cout << "eye shape " << ey.shape << endl;
    cout << ey << endl;
}
}  // namespace ts
