#include <ctime>
#include <iostream>

#include "config.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start, end_time;
int main() {
    // Tensor t1 = rand({{256,256,256}},
    //     dt::float32);

    // Tensor t2 = rand({{256,256,256}},
    //     dt::float32);
    double endtime;
    // cout << "Tensro Size: " << t1.shape << endl;

    // cout << "=====================" << "Testing on " << t1.device <<
    // "=====================" << endl;

    // Tensor t3 = t1;
    // start=clock();
    // for (int i = 0; i < 10; i++) {
    //     t3 = t3 + t2;
    // }
    // end_time=clock();
    //  endtime=(double)(end_time-start)/CLOCKS_PER_SEC;
    // cout<<"Total time:"<<endtime*1000<<"ms"<<endl;

    // t1 = t1.to(dev::cuda);
    // t2 = t2.to(dev::cuda);

    Tensor t1 = tensor({1,2,3}, dt::int8);
    Tensor t2 = tensor({4, 5,6}, dt::int8);


    Tensor t4 = einsum("i,i->i", {t1, t2});
    cout << t4 << endl;



    end_time = clock();
    endtime = (double)(end_time - start) / CLOCKS_PER_SEC;
    cout << "Total time:" << endtime * 1000 << "ms" << endl;
}