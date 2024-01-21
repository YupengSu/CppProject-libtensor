#include "test_cases.hpp"

#include "config.hpp"
#include "data_type.cuh"
#include "serial_tensor.hpp"

using namespace ts;
namespace test_case {
const dev target_device = dev::cuda;

void specify_init() {
    TensorImpl t1 = tensor({2, 3, 4}).to(target_device);
    TensorImpl t2 =
        tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}});

    TensorImpl t3 =
        tensor({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, dt::int32);

    TensorImpl t4 = rand({2, 3, 4}, dt::int32);

    t1.info("Tensor 1");
    cout << t1 << endl;

    t2.info("Tensor 2");
    cout << t2 << endl;

    t3.info("Tensor 3");
    cout << t3 << endl;
}
void indexing() {
    if (target_device == dev::cuda) {
        cout << "Indexing test case is not supported on CUDA device" << endl;
    }
    TensorImpl t1 = rand({2, 3, 4}, dt::int8);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "t1[1,0,2]: " << t1.locate({1, 0, 2}) << endl;

    data_t& val = t1.locate({1, 0, 2});
    cout << "Modifying t1[1,0,2] to 0" << endl;
    val = 0;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after Indexing and modification");
}

void slicing() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "t1[0]" << endl;
    cout << t1[0] << endl << endl;

    cout << "t1[0][0]" << endl;
    cout << t1[0][0] << endl << endl;

    cout << "t1[0][0][0]" << endl;
    cout << t1[0][0][0] << endl << endl;

    TensorImpl t2 = t1[0];
    t2.info("Tensor 2");
    cout << "Modifying t2[0][0] to 0" << endl;
    t2[0][0] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after slicing and mutation");
}

void mutating() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "t1[0][0] = 0" << endl;
    t1[0][0] = 0;
    cout << "t1:\n" << t1 << endl << endl;

    cout << "t1[1][1] = {5, 6,7, 8}" << endl;
    t1[1][1] = {5, 6, 7, 8};
    cout << "t1:\n" << t1 << endl << endl;

    t1.info("Tensor 1 after mutation");
}

void transpose() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "Transpose t1 to t2 with transposed dims {0, 1}" << endl;
    TensorImpl t2 = transpose(t1, 0, 1);
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    cout << endl
         << "==========================Two dim "
            "example:=========================="
         << endl;
    t1 = rand({3, 4}, dt::int8).to(target_device);
    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "Transpose t1 to t2 with transposed dims {0, 1}" << endl;
    t2 = t1.transpose(1, 0);
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after transpose and mutation");
}

void permute() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "Permute t1 to t2 with dims {1, 2, 0}" << endl;
    TensorImpl t2 = permute(t1, {1, 2, 0});
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    cout << endl
         << "==========================Two dim "
            "example:========================== "
         << endl;

    t1 = rand({3, 4}, dt::int8).to(target_device);
    t1.info("Tensor 1");
    cout << t1 << endl << endl;
    cout << "Transpose t1 to t2 with transposed dims {0, 1}" << endl;
    t2 = t1.permute({1, 0});
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after permute and mutation");
}

void view() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    cout << "View t1 to t2 with shape {3, 8}" << endl;
    TensorImpl t2 = view(t1, {3, 8});
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    cout << endl
         << "==========================Two dim "
            "example:========================== "
         << endl;

    t1 = rand({3, 4}, dt::int8).to(target_device);
    t1.info("Tensor 1");
    cout << t1 << endl << endl;
    cout << "View t1 to t2 with shape {4, 3}" << endl;
    t2 = t1.view({4, 3});
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[2][1] to 0" << endl;
    t2[2][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after view and mutation");
}

void squeeze() {
    TensorImpl t1 = rand({3, 1, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;
    cout << "Squeeze t1 to t2" << endl;

    TensorImpl t2 = t1.squeeze();
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

}

void unsqueeze() {
    TensorImpl t1 = rand({3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;
    cout << "Unsqueeze t1 to t2" << endl;

    TensorImpl t2 = t1.unsqueeze(1);
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

}
void einsum() {
    // TODO: implement einsum test
}
}  // namespace test_case
