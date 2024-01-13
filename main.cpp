#include <iostream>
#include<ctime>
#include "serial_tensor.hpp"

using namespace ts;

clock_t start,end_time;
int main() {
    Tensor t1 = rand({128,64,64},
        dt::float32);

    Tensor t2 = rand({128,64,64},
        dt::float32);
    cout << t1.device << endl;

    start=clock();
    Tensor t3 = add(t1, t2);
    end_time=clock();
	double endtime=(double)(end_time-start)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<endtime*1000<<"ms"<<endl;

    t1 = t1.to(dev::cuda);
    t2 = t2.to(dev::cuda);

    cout << t1.device << endl;
    start=clock();
    Tensor t4 = add(t1, t2);
    end_time=clock();
    endtime=(double)(end_time-start)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<endtime*1000<<"ms"<<endl;
}