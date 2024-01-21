#pragma once

#include "serial_tensor.hpp"

namespace test_case {
void specify_init();

void indexing();

void slicing();

void mutating();

void transpose();

void permute();

void view();

void einsum();

void serialization();

void cuda_acc();

}