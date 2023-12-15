#include <initializer_list>
#include <iostream>
using namespace std;

class ShapeElem {
   public:
    ShapeElem* next;
    int len;

    ShapeElem(int _len, ShapeElem* _next) : next(_next), len(_len) {}

    void print_shape() {
        if (next != nullptr) {
            cout << " " << len;
            next->print_shape();
        } else {
            cout << " " << len << "\n";
        }
    }

    int array_len() {
        if (next != nullptr) {
            return len * next->array_len();
        } else {
            return len;
        }
    }
};

template <class value_type>
class ArrayInit {
   public:
    void* data = nullptr;
    size_t len;
    bool is_final;

    ArrayInit(std::initializer_list<value_type> init)
        : data((void*)init.begin()), len(init.size()), is_final(true) {
            cout << "build scaler" << endl;
        }

    ArrayInit(std::initializer_list<ArrayInit<value_type>> init)
        : data((void*)init.begin()), len(init.size()), is_final(false) {
            cout << "build array" << endl;

        }

    ShapeElem* shape() {
        if (is_final) {
            ShapeElem* out = new ShapeElem(len, nullptr);
        } else {
            ArrayInit<value_type>* first = (ArrayInit<value_type>*)data;
            ShapeElem* out = new ShapeElem(len, first->shape());
        }
    }
    void assign(value_type** pointer) {
        if (is_final) {
            for (size_t k = 0; k < len; k++) {
                (*pointer)[k] = (((value_type*)data)[k]);
            }
            (*pointer) = (*pointer) + len;
        } else {
            ArrayInit<value_type>* data_array = (ArrayInit<value_type>*)data;
            for (int k = 0; k < len; k++) {
                data_array[k].assign(pointer);
            }
        }
    }
};

int main() {
    auto x = ArrayInit<int>({{1, 2, 3}, {92, 1, 3}});
    auto shape = x.shape();
    shape->print_shape();
    int* data = new int[shape->array_len()];
    int* running_pointer = data;
    x.assign(&running_pointer);
    for (int i = 0; i < shape->array_len(); i++) {
        cout << " " << data[i];
    }
    cout << "\n";
}