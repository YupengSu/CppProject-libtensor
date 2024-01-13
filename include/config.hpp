#pragma once
#include <cstdint>
#include <iostream>
#include <ostream>

#define DEFAULT_DTYPE dt::float32
#define DEFAULT_DEVICE dev::cpu

using namespace std;
namespace ts {
enum class dt {
    int8,
    float32,
    float64,
    int32,
    bool8

};
enum class dev { cpu, cuda };

ostream &operator<<(ostream &os, const dt dtype);

ostream &operator<<(ostream &os, const dev device);


}  // namespace ts
