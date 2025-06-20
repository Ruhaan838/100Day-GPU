#include "ops.h"

Tensor* add(Tensor* a, Tensor* b){
    Tensor* ans = new Tensor(0.0f, a->requires_grad || b->requires_grad);
    add_kernel<<<1, 1>>>(a -> data, b->data, ans->data, N); 
    cudaDeviceSynchronize();

    if(ans->requires_grad){
        ans->parents = {a, b};
        ans->backward_fn = [=]() {
            if (a->requires_grad) *(a->grad) += *(ans->grad);
            if (b->requires_grad) *(b->grad) += *(ans->grad);
        };
    }

    return ans;
}

Tensor* mul(Tensor* a, Tensor* b){
    Tensor* ans = new Tensor(0.0f, a->requires_grad || b->requires_grad);
    mul_kernel<<<1, 1>>>(a -> data, b->data, ans->data, N); 
    cudaDeviceSynchronize();

    if(ans->requires_grad){
        ans->parents = {a, b};
        ans->backward_fn = [=]() {
            if (a->requires_grad) *(a->grad) += *(b->grad) * (*(ans->grad));
            if (b->requires_grad) *(b->grad) += *(b->grad) * (*(ans->grad));
        };
    }

    return ans;
}