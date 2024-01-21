#include "test_cases.hpp"
#include "config.hpp"
#include "serial_tensor.hpp"

using namespace ts;
namespace test_case {
    void specify_init() {
        TensorImpl t1 = tensor({2, 3, 4});
        TensorImpl t2 = tensor( {
            {       { 1, 2}
                    , { 3, 4}},
            {       { 5, 6},
                    { 7, 8}},
            {       { 9,10},
                    {11,12}}  
        });

        TensorImpl t3 = tensor( 
            {       { 1, 2, 3, 4 }, 
                        { 5, 6, 7, 8 }, 
                        { 9,10,11,12 } }
        , dt::int32);

        TensorImpl t4 = rand({2,3,4}, dt::int32);

        t1.info("Tensor 1");
        cout << t1 << endl;

        t2.info("Tensor 2");
        cout << t2 << endl;

        t3.info("Tensor 3");
        cout << t3 << endl;
    }
}  // namespace ts
