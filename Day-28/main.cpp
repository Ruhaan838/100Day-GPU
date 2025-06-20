#include "ops.h"
#include "tensor.h"
#include <iostream>

using namespace std;

int main(){
    Tensor* a = new Tensor(2.0f, true);
    Tensor* b = new Tensor(3.0f, true);

    Tensor* c = add(a, b);
    Tensor* d = mul(c, b);

    d->backward();
    
    cout << "a = " << a->value() << ", grad = " << a->show_grad() << '\n';
    cout << "b = " << b->value() << ", grad = " << b->show_grad() << '\n';
    cout << "d = " << d->value() << '\n';

    delete a;
    delete b;
    delete c;
    delete d;

    return 0; 
}
