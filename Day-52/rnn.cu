#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void rnn_forward(
    const float *x, const float *h,
    const float *Wx, const float *Wh,
    const float *bx, const float *bh,
    float *h_new,
    int input_size, int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;

    float sum = bh[i];
    for (int j = 0; j < input_size; j++) {
        sum += Wx[i * input_size + j] * x[j];
    }
    for (int j = 0; j < hidden_size; j++) {
        sum += Wh[i * hidden_size + j] * h[j];
    }
    h_new[i] = tanhf(sum);
}

int main() {
    const int input_size = 4;
    const int hidden_size = 3;

    float x[input_size] = {1, 2, 3, 4};
    float h[hidden_size] = {0.5, -0.5, 0.2};

    float Wx[hidden_size * input_size];
    float Wh[hidden_size * hidden_size];
    float bx[hidden_size];
    float bh[hidden_size];

    for (int i = 0; i < hidden_size * input_size; i++) Wx[i] = 0.1f;
    for (int i = 0; i < hidden_size * hidden_size; i++) Wh[i] = 0.1f;
    for (int i = 0; i < hidden_size; i++) { bx[i] = 0.1f; bh[i] = 0.1f; }

    float *dx, *dh, *dWx, *dWh, *dbx, *dbh, *dh_new;
    size_t in_size = input_size * sizeof(float);
    size_t hidden_size1 = hidden_size * sizeof(float);
    size_t hidden_size2 = input_size * hidden_size * sizeof(float);
    size_t hidden_size3 = hidden_size * hidden_size * sizeof(float);

    cudaMalloc(&dx, in_size);
    cudaMalloc(&dh, hidden_size1);
    cudaMalloc(&dWx, hidden_size2);
    cudaMalloc(&dWh, hidden_size3);
    cudaMalloc(&dbx, hidden_size1);
    cudaMalloc(&dbh, hidden_size1);
    cudaMalloc(&dh_new, hidden_size1);

    cudaMemcpy(dx, x, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dh, h, hidden_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dWx, Wx, hidden_size2, cudaMemcpyHostToDevice);
    cudaMemcpy(dWh, Wh, hidden_size3, cudaMemcpyHostToDevice);
    cudaMemcpy(dbx, bx, hidden_size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dbh, bh, hidden_size1, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (hidden_size + threads - 1) / threads;
    rnn_forward<<<blocks, threads>>>(dx, dh, dWx, dWh, dbx, dbh, dh_new, input_size, hidden_size);
    cudaDeviceSynchronize();

    float h_new[hidden_size];
    cudaMemcpy(h_new, dh_new, hidden_size1, cudaMemcpyDeviceToHost);

    printf("h_new: ");
    for (int i = 0; i < hidden_size; i++) {
        printf("%f ", h_new[i]);
    }
    printf("\n");

    cudaFree(dx); cudaFree(dh); cudaFree(dWx); cudaFree(dWh);
    cudaFree(dbx); cudaFree(dbh); cudaFree(dh_new);
    return 0;
}
