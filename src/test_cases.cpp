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

    clock_t start, end;
    // 1) Extracting elements along a diagonal
    TensorImpl t1 = rand({4,4});
    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    vector<TensorImpl> tensors = {t1};
    t1 = ts::einsum("ii->i", tensors);
    cout << "einsum(\"ii->i\", tensors):" << endl;
    cout << t1 << endl << endl;

    // 2) Computing a matrix transpose
    TensorImpl t2 = rand({4,4});
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    tensors = {t2};
    t2 = ts::einsum("ij->ji", tensors);
    cout << "einsum(\"ij->ji\", tensors):" << endl;
    cout << t2 << endl << endl;

    // 3) Permuting array elements 
    // FIXME: not working
    TensorImpl t3 = rand({2,2,2});
    t3.info("Tensor 3");
    cout << t3 << endl << endl;

    tensors = {t3};
    t3 = ts::einsum("kij->kji", tensors);
    cout << "einsum(\"kij->kji\", tensors):" << endl;
    cout << t3 << endl << endl;

    // 4) Reduce sum
    TensorImpl t4 = ones({256, 256});
    t4.info("Tensor 4");
    start = clock();
    tensors = {t4};
    t4 = ts::einsum("ij->", tensors);
    end = clock();
    cout << "einsum(\"ij->\", tensors):" << endl;
    cout << t4 << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << endl << endl;

    t4 = ones({256, 256}).cuda();
    t4.info("Tensor 4");
    start = clock();
    tensors = {t4};
    t4 = ts::einsum("ij->", tensors);
    end = clock();
    cout << "einsum(\"ij->\", tensors):" << endl;
    cout << t4 << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << endl << endl;

    // 5) Sum along dimension
    TensorImpl t5 = ones({64, 64});
    t5.info("Tensor 5");
    start = clock();
    tensors = {t5};
    t5 = ts::einsum("ij->j", tensors);
    end = clock();
    cout << "einsum(\"ij->j\", tensors):" << endl;
    cout << t5 << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << endl << endl;

    t5 = ones({64, 64}).cuda();
    t5.info("Tensor 5");
    start = clock();
    tensors = {t5};
    t5 = ts::einsum("ij->j", tensors);
    end = clock();
    cout << "einsum(\"ij->j\", tensors):" << endl;
    cout << t5 << endl;
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << endl << endl;

    // 6) Matrix vector multiplication
    TensorImpl t6 = rand({16, 16});
    TensorImpl t7 = rand({16});
}

void serialization() {
    TensorImpl t1 = rand({2, 3, 4}, dt::int8).to(target_device);

    t1.info("Tensor 1");
    cout << t1 << endl << endl;

    string path = "t1.bin";
    cout << "Serializing t1 to " << path << endl;
    t1.save(path);

    cout << "Loading t2 from " << path << endl;
    TensorImpl t2 = TensorImpl::load(path);
    t2.info("Tensor 2");
    cout << t2 << endl << endl;

    cout << "Modifying t2[1][1] to 0" << endl;
    t2[1][1] = 0;
    cout << "t2:\n" << t2 << endl;
    cout << "t1:\n" << t1 << endl;

    t1.info("Tensor 1 after Saving");
    t2.info("Tensor 2 after Loading ");
}
}  // namespace test_case
