#pragma once

#include "serial_tensor.hpp"



/*
Show Cases for Tensor
1. Implementation of Tensor class
2. Implementation of data_t
3. Implementation of Size
4. Implementation of Storage
5. Implementation of exceptions
6. Implementation of datatype
7. Implementation of device
8. Show below test cases
*/ 





namespace test_case {

void multitype();

void specify_init();

void indexing();

void slicing();

void mutating();

void transpose();

void permute();

void view();

void squeeze();

void unsqueeze();

void einsum();

void serialization();

void cuda_acc();

}