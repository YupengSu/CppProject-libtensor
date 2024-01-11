#pragma once
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include "base_tensor.hpp"
#include "size.hpp"
#include "storage.hpp"
#include "config.hpp"
#include "exception.hpp"

using namespace std;

namespace ts
{
       using data_t = float;
       class Tensor
       {
       public:
              Tensor operator+(const Tensor &other);
              Tensor operator-(const Tensor &other);
              Tensor operator*(const Tensor &other);
              Tensor operator/(const Tensor &other);

              Tensor add(const Tensor &other);
              Tensor sub(const Tensor &other);
              Tensor mul(const Tensor &other);
              Tensor div(const Tensor &other);

              Tensor add(data_t other);
              Tensor sub(data_t other);
              Tensor mul(data_t other);
              Tensor div(data_t other);

              Tensor sum(int dim);
              Tensor mean(int dim);
              Tensor max(int dim);
              Tensor min(int dim);

              Tensor eq(const Tensor &other);
              Tensor ne(const Tensor &other);
              Tensor gt(const Tensor &other);
              Tensor ge(const Tensor &other);
              Tensor lt(const Tensor &other);
              Tensor le(const Tensor &other);

              Tensor operator==(const Tensor &other);
              Tensor operator!=(const Tensor &other);
              Tensor operator>(const Tensor &other);
              Tensor operator>=(const Tensor &other);
              Tensor operator<(const Tensor &other);
              Tensor operator<=(const Tensor &other);
            
       };

       Tensor add(const Tensor t1, data_t t2);
       Tensor sub(const Tensor t1, data_t t2);
       Tensor mul(const Tensor t1, data_t t2);
       Tensor div(const Tensor t1, data_t t2);

       Tensor add(const Tensor t1, const Tensor t2);
       Tensor sub(const Tensor t1, const Tensor t2);
       Tensor mul(const Tensor t1, const Tensor t2);
       Tensor div(const Tensor t1, const Tensor t2);

       Tensor log(const Tensor t);

       Tensor sum(const Tensor t, int dim);
       Tensor mean(const Tensor t, int dim);
       Tensor max(const Tensor t, int dim);
       Tensor min(const Tensor t, int dim);

       // comparison
       Tensor eq(const Tensor t1, const Tensor t2);
       Tensor ne(const Tensor t1, const Tensor t2);
       Tensor gt(const Tensor t1, const Tensor t2);
       Tensor ge(const Tensor t1, const Tensor t2);
       Tensor lt(const Tensor t1, const Tensor t2);
       Tensor le(const Tensor t1, const Tensor t2);

       //other
       Tensor einsum(string eq, vector<Tensor> tensors);
} // namespace ts
