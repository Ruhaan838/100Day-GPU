#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <cuda_runtime.h>

#define N 1 //scalar only 

using namespace std;

struct Tensor {
    float* data;
    float* grad;
    bool requires_grad;
    vector<Tensor*> parents;
    function<void()> backward_fn;

    Tensor(float value, bool requires_grad_ = false) : requires_grad(requires_grad_){
        cudaMallocManaged((void**)&data, sizeof(float));
        *data = value;

        if(requires_grad){
            cudaMallocManaged((void**)&grad, sizeof(float));
            *grad = 0.0f;
        } else {
            grad = nullptr;
        }
    }

    ~Tensor(){
        cudaFree(data);
        if (grad) cudaFree(grad);
    }

    void backward(){
        if(!requires_grad) return;
        if(*grad == 0.0f) *grad = 1.0f;
        if(backward_fn) backward_fn();
        for (Tensor* parent: parents){
            parent->backward();
        }
    }

    float value() const {return *data;}
    float show_grad() const {return requires_grad ? *grad : 0.0f;}
};

#endif