#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

const int BATCH_SIZE = 1024;
const int DIM = 128;
const float MARGIN = 1.0f;

using namespace std;

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << '\n';
        exit(EXIT_FAILURE);
    }
}

__device__ float euxliden_dist(const float* x1, const float* x2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__global__ void contranstive_loss(
    const float* x1, const float* x2, const int* label,
    float* loss, int batch_size, int dim, float margin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float dist = euxliden_dist(&x1[idx * dim], &x2[idx * dim], dim);
    int y = label[idx];

    float temp = fmaxf(0.0f, margin - dist);
    float loss_val = (1 - y) * 0.5f * dist * dist + y * 0.5f * temp * temp;
    loss[idx] = loss_val;
}

void init_data(float** x1, float** x2, int** labels, int batch_size, int dim) {
    *x1 = new float[batch_size * dim];
    *x2 = new float[batch_size * dim];
    *labels = new int[batch_size];

    for (int i = 0; i < batch_size * dim; i++) {
        (*x1)[i] = static_cast<float>(rand()) / RAND_MAX;
        (*x2)[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < batch_size; i++) {
        (*labels)[i] = rand() % 2;
    }
}

int main() {
    float *x1, *x2, *loss;
    int *labels;

    float *d_x1, *d_x2, *d_loss;
    int *d_labels;

    loss = new float[BATCH_SIZE];

    init_data(&x1, &x2, &labels, BATCH_SIZE, DIM);

    size_t b_d_size = BATCH_SIZE * DIM * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_x1, b_d_size));
    CUDA_CHECK(cudaMalloc(&d_x2, b_d_size));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x1, x1, b_d_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, x2, b_d_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    contranstive_loss<<<blocks, threads>>>(d_x1, d_x2, d_labels, d_loss, BATCH_SIZE, DIM, MARGIN);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(loss, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Some values of Contrastive Loss:\n";
    for (int i = 0; i < 10; i++) {
        cout << "Loss[" << i << "]: " << loss[i] << '\n';
    }

    delete[] x1;
    delete[] x2;
    delete[] labels;
    delete[] loss;

    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_loss));

    return 0;
}
