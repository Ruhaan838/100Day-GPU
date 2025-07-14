#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_forward(
    const float *x, const float *h,
    const float *Wz, const float *Wr, const float *Wh,
    const float *Uz, const float *Ur, const float *Uh,
    const float *bz, const float *br, const float *bh,
    float *h_new,
    int input_size, int hidden_size
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;

    float z = bz[i];
    float r = br[i];
    float h_hat = bh[i];

    for(int j = 0; j < input_size; j++){
        z += Wz[i * hidden_size + j] * x[j];
        r += Wr[i * hidden_size + j] * x[j];
        h_hat += Wh[i * input_size + j] * x[j];
    }

    for(int j = 0; j < hidden_size; j++){
        z += Uz[i * hidden_size + j] * h[j];
        r += Ur[i * hidden_size + j] * h[j];
    }
    z = sigmoid(z);
    r = sigmoid(r);

    for(int j = 0; j < hidden_size; j++){
        h_hat += Uh[i * hidden_size + j] * h[j] * r;
    }
    h_hat = tanhf(h_hat);

    h_new[i] = (1.0f - z) * h[i] + z * h_hat;
}

int main(){
    const int input_size = 4;
    const int hidden_size = 3;

    float x[input_size] = {1, 2, 3, 4};
    float h[hidden_size] = {0.5, -0.5, 0.2};

    float Wz[hidden_size * input_size];
    float Wr[hidden_size * input_size];
    float Wh[hidden_size * input_size];

    float Uz[hidden_size * hidden_size];
    float Ur[hidden_size * hidden_size];
    float Uh[hidden_size * hidden_size];

    float bz[hidden_size];
    float br[hidden_size];
    float bh[hidden_size];

    for (int i = 0; i < hidden_size * input_size; i++) {
        Wz[i] = 0.1f; Wr[i] = 0.1f; Wh[i] = 0.1f;
    }
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        Uz[i] = 0.1f; Ur[i] = 0.1f; Uh[i] = 0.1f;
    }
    for (int i = 0; i < hidden_size; i++) {
        bz[i] = 0.1f; br[i] = 0.1f; bh[i] = 0.1f;
    }

    float *dx, *dh, *dWz, *dWr, *dWh, *dUz, *dUr, *dUh, *dbz, *dbr, *dbh, *dh_new;

    size_t in_size = input_size * sizeof(float);
    size_t hidden_size1 = hidden_size * sizeof(float);
    size_t hidden_size2 = input_size * hidden_size * sizeof(float);
    size_t hidden_size3 = hidden_size * hidden_size * sizeof(float);

    cudaMalloc(&dx, in_size);
    cudaMalloc(&dh, hidden_size1);
    cudaMalloc(&dWz, hidden_size2);
    cudaMalloc(&dWr, hidden_size2);
    cudaMalloc(&dWh, hidden_size2);
    cudaMalloc(&dUz, hidden_size3);
    cudaMalloc(&dUr, hidden_size3);
    cudaMalloc(&dUh, hidden_size3);
    cudaMalloc(&dbz, hidden_size1);
    cudaMalloc(&dbr, hidden_size1);
    cudaMalloc(&dbh, hidden_size1);
    cudaMalloc(&dh_new, hidden_size1);

    cudaMemcpy(dx, x, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, h, hidden_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dWz, Wz, hidden_size2, cudaMemcpyHostToDevice);
    cudaMemcpy(dWr, Wr, hidden_size2, cudaMemcpyHostToDevice);
    cudaMemcpy(dWh, Wh, hidden_size2, cudaMemcpyHostToDevice);
    cudaMemcpy(dUz, Uz, hidden_size3, cudaMemcpyHostToDevice);
    cudaMemcpy(dUr, Ur, hidden_size3, cudaMemcpyHostToDevice);
    cudaMemcpy(dUh, Uh, hidden_size3, cudaMemcpyHostToDevice);
    cudaMemcpy(dbz, bz, hidden_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dbr, br, hidden_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dbh, bh, hidden_size1, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (hidden_size + threads - 1) / threads;
    gru_forward<<<blocks, threads>>>(dx, dh, dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dh_new, input_size, hidden_size);
    cudaDeviceSynchronize();

    float h_new[hidden_size];
    cudaMemcpy(h_new, dh_new, hidden_size1, cudaMemcpyDeviceToHost);

    printf("h_new: ");
    for (int i = 0; i < hidden_size; i++) {
        printf("%f ", h_new[i]);
    }
    printf("\n");

    cudaFree(dx); cudaFree(dh); cudaFree(dWz); cudaFree(dWr); cudaFree(dWh);
    cudaFree(dUz); cudaFree(dUr); cudaFree(dUh);
    cudaFree(dbz); cudaFree(dbr); cudaFree(dbh); cudaFree(dh_new);

    return 0;
}