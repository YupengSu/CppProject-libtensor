#include <cstddef>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>

#include "serial_tensor.hpp"
#include "exception.hpp"
#include "storage.hpp"

namespace ts
{

    Tensor::Tensor(Size shape) : data(shape.size())
    {
        this->ndim = shape.ndim;          // number of dimensions
        this->shape = shape;              // shape of tensor
        this->stride.reserve(shape.ndim); // stride of tensor
        // reserve() is used to allocate memory for a vector
        this->offset = 0;
    }

    bool CHECK_SAME_SHAPE(Tensor t1, Tensor t2, string msg)
    {
        if (t1.shape != t2.shape)
        {
            throw runtime_error(msg);
        }
        return true;
    }

    //////////////add operators

    Tensor add(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] + t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor add(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] + t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::add(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::add(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other;
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator+(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    //////////////////sub operators
    Tensor sub(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] - t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor sub(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] - t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::sub(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator-(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::sub(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other;
        }
        return Tensor(data, this->shape.shape);
    }

    ////////////////mul operators

    Tensor mul(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] * t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor mul(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] * t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::operator*(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::mul(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::mul(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other;
        }
        return Tensor(data, this->shape.shape);
    }

    ////////////////////////div operators

    Tensor div(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] / t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor div(const Tensor t1, data_t t2)
    {
        if (t2 == 0)
            throw runtime_error("Division by zero");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] / t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::div(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (other.data[i] == 0)
            {
                throw runtime_error("Division by zero");
            }
            data[i] = this->data[i] / other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator/(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (other.data[i] == 0)
            {
                throw runtime_error("Division by zero");
            }
            data[i] = this->data[i] / other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::div(data_t other)
    {
        if (other == 0)
            throw runtime_error("Division by zero");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] / other;
        }
        return Tensor(data, this->shape.shape);
    }

    ///////////log operators

    Tensor log(const Tensor t)
    {
        vector<data_t> data(t.data.size);
        int size = t.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = std::log(t.data[i]);
        }
        return Tensor(data, t.shape.shape);
    }

    ///////////////find sum 0 means row wise and 1 means column wise

    //sum
    Tensor sum(const Tensor t,int dim){
        if(dim==0){
            vector<data_t> data(t.shape[0]);
            for(int i=0;i<t.shape[0];i++){
                data[i]=0;
                for(int j=0;j<t.shape[1];j++){
                    data[i]+=t.data[i*t.shape[1]+j];
                }
            }
            return Tensor(data,{t.shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(t.shape[1]);
            for(int i=0;i<t.shape[1];i++){
                data[i]=0;
                for(int j=0;j<t.shape[0];j++){
                    data[i]+=t.data[j*t.shape[1]+i];
                }
            }
            return Tensor(data,{t.shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::sum(int dim){
        if(dim==0){
            vector<data_t> data(this->shape[0]);
            for(int i=0;i<this->shape[0];i++){
                data[i]=0;
                for(int j=0;j<this->shape[1];j++){
                    data[i]+=this->data[i*this->shape[1]+j];
                }
            }
            return Tensor(data,{this->shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(this->shape[1]);
            for(int i=0;i<this->shape[1];i++){
                data[i]=0;
                for(int j=0;j<this->shape[0];j++){
                    data[i]+=this->data[j*this->shape[1]+i];
                }
            }
            return Tensor(data,{this->shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    //mean
    Tensor mean(const Tensor t,int dim){
        if(dim==0){
            vector<data_t> data(t.shape[0]);
            for(int i=0;i<t.shape[0];i++){
                data[i]=0;
                for(int j=0;j<t.shape[1];j++){
                    data[i]+=t.data[i*t.shape[1]+j];
                }
                data[i]/=t.shape[1];
            }
            return Tensor(data,{t.shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(t.shape[1]);
            for(int i=0;i<t.shape[1];i++){
                data[i]=0;
                for(int j=0;j<t.shape[0];j++){
                    data[i]+=t.data[j*t.shape[1]+i];
                }
                data[i]/=t.shape[0];
            }
            return Tensor(data,{t.shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::mean(int dim){
        if(dim==0){
            vector<data_t> data(this->shape[0]);
            for(int i=0;i<this->shape[0];i++){
                data[i]=0;
                for(int j=0;j<this->shape[1];j++){
                    data[i]+=this->data[i*this->shape[1]+j];
                }
                data[i]/=this->shape[1];
            }
            return Tensor(data,{this->shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(this->shape[1]);
            for(int i=0;i<this->shape[1];i++){
                data[i]=0;
                for(int j=0;j<this->shape[0];j++){
                    data[i]+=this->data[j*this->shape[1]+i];
                }
                data[i]/=this->shape[0];
            }
            return Tensor(data,{this->shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    //max
    Tensor max(Tensor t,int dim){
        if(dim==0){
            vector<data_t> data(t.shape[0]);
            for(int i=0;i<t.shape[0];i++){
                data[i]=t.data[i*t.shape[1]];
                for(int j=0;j<t.shape[1];j++){
                    if(data[i]<t.data[i*t.shape[1]+j]){
                        data[i]=t.data[i*t.shape[1]+j];
                    }
                }
            }
            return Tensor(data,{t.shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(t.shape[1]);
            for(int i=0;i<t.shape[1];i++){
                data[i]=t.data[i];
                for(int j=0;j<t.shape[0];j++){
                    if(data[i]<t.data[j*t.shape[1]+i]){
                        data[i]=t.data[j*t.shape[1]+i];
                    }
                }
            }
            return Tensor(data,{t.shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::max(int dim){
        if(dim==0){
            vector<data_t> data(this->shape[0]);
            for(int i=0;i<this->shape[0];i++){
                data[i]=this->data[i*this->shape[1]];
                for(int j=0;j<this->shape[1];j++){
                    if(data[i]<this->data[i*this->shape[1]+j]){
                        data[i]=this->data[i*this->shape[1]+j];
                    }
                }
            }
            return Tensor(data,{this->shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(this->shape[1]);
            for(int i=0;i<this->shape[1];i++){
                data[i]=this->data[i];
                for(int j=0;j<this->shape[0];j++){
                    if(data[i]<this->data[j*this->shape[1]+i]){
                        data[i]=this->data[j*this->shape[1]+i];
                    }
                }
            }
            return Tensor(data,{this->shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    //min
    Tensor min(Tensor t,int dim){
        if(dim==0){
            vector<data_t> data(t.shape[0]);
            for(int i=0;i<t.shape[0];i++){
                data[i]=t.data[i*t.shape[1]];
                for(int j=0;j<t.shape[1];j++){
                    if(data[i]>t.data[i*t.shape[1]+j]){
                        data[i]=t.data[i*t.shape[1]+j];
                    }
                }
            }
            return Tensor(data,{t.shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(t.shape[1]);
            for(int i=0;i<t.shape[1];i++){
                data[i]=t.data[i];
                for(int j=0;j<t.shape[0];j++){
                    if(data[i]>t.data[j*t.shape[1]+i]){
                        data[i]=t.data[j*t.shape[1]+i];
                    }
                }
            }
            return Tensor(data,{t.shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::min(int dim){
        if(dim==0){
            vector<data_t> data(this->shape[0]);
            for(int i=0;i<this->shape[0];i++){
                data[i]=this->data[i*this->shape[1]];
                for(int j=0;j<this->shape[1];j++){
                    if(data[i]>this->data[i*this->shape[1]+j]){
                        data[i]=this->data[i*this->shape[1]+j];
                    }
                }
            }
            return Tensor(data,{this->shape[0]});
        }
        else if(dim==1){
            vector<data_t> data(this->shape[1]);
            for(int i=0;i<this->shape[1];i++){
                data[i]=this->data[i];
                for(int j=0;j<this->shape[0];j++){
                    if(data[i]>this->data[j*this->shape[1]+i]){
                        data[i]=this->data[j*this->shape[1]+i];
                    }
                }
            }
            return Tensor(data,{this->shape[1]});
        }
        else{
            throw runtime_error("Dimension out of range");
        }
    }

    ///////////////comparison
    Tensor eq(const Tensor t1,const Tensor t2){
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if(t1.data[i]==t2.data[i]){
                data[i]=1;
            }
            else{
                data[i]=0;
            }
        }
        return Tensor(data, t1.shape.shape);
    }
// template<typename T>
// Tensor<bool> eq(const Tensor<T>& t1, const Tensor<T>& t2) {
//     if (t1.getShape() != t2.getShape()) {
//         throw std::invalid_argument("Tensor shapes do not match.");
//     }

//     std::vector<bool> resultData(t1.getTotalSize(), false);
//     for (int i = 0; i < t1.getTotalSize(); ++i) {
//         resultData[i] = (t1[i] == t2[i]);
//     }

//     return Tensor<bool>(resultData, t1.getShape());
// }

    //////////////other
    Tensor einsum(string eq,vector<Tensor> tensors){
        if(eq == "i,i->"){
            //dot production
            int result = 0;
            for(size_t i = 0;i<tensors.size();++i){
                
            }
        }
    }



}