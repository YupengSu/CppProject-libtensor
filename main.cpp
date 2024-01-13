#include <iostream>
#include <ctime>
#include "config.hpp"
#include "serial_tensor.hpp"

using namespace ts;
using namespace std;

clock_t start,end_time;
int main() {
    Tensor t1 = rand({256,256},
        dt::float64);

    Tensor t2 = rand({256,256},
        dt::float64);

    cout << "Tensro Size: " << t1.shape << endl;

    cout << "=====================" << "Testing on " << t1.device << "=====================" << endl;

    start=clock();
    Tensor t3 = t1 +t2;
    end_time=clock();
	double endtime=(double)(end_time-start)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<endtime*1000<<"ms"<<endl;

    t1 = t1.to(dev::cuda);
    t2 = t2.to(dev::cuda);

    cout << "=====================" << "Testing on " << t1.device << "=====================" << endl;
    start=clock();
    Tensor t4 = t1 + t2;
    end_time=clock();
    endtime=(double)(end_time-start)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<endtime*1000<<"ms"<<endl;
    // cout << "Tensor3: \n" << t3 << endl;
    // cout << "Tensor4: \n" << t4 << endl;
    // t4 = t4.to(dev::cpu);

}