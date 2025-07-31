#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

__global__ void assign_and_compute(const float* data_x, const float* data_y, const float* centroid_x, const float* centroid_y,
                                   int* labels, float* sum_x, float* sum_y, int* count, int sample_size, int k) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    float x = data_x[idx];
    float y = data_y[idx];

    float mindist = 1e20f;
    int best = 0;

    for (int i = 0; i < k; i++) {
        float dx = x - centroid_x[i];
        float dy = y - centroid_y[i];
        float dist = dx * dx + dy * dy;
        if (dist < mindist) {
            mindist = dist;
            best = i;
        }
    }

    labels[idx] = best;

    atomicAdd(&sum_x[best], x);
    atomicAdd(&sum_y[best], y);
    atomicAdd(&count[best], 1);
}

__global__ void update_centroids(float* centroid_x, float* centroid_y, const float* sum_x, const float* sum_y,
                                 const int* count, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    if (count[i] > 0) {
        centroid_x[i] = sum_x[i] / count[i];
        centroid_y[i] = sum_y[i] / count[i];
    }
}

int main() {
    srand(time(0)); 

    const int sample_size = 1024;
    const int k = 4;
    const int max_iter = 100;
    const float threshold = 0.0001f;

    float *h_data_x = new float[sample_size];
    float *h_data_y = new float[sample_size];
    float *h_centroid_x = new float[k];
    float *h_centroid_y = new float[k];

    for (int i = 0; i < sample_size; i++) {
        h_data_x[i] = static_cast<float>(rand() % 400) / 100.0f;
        h_data_y[i] = static_cast<float>(rand() % 400) / 100.0f;
    }

    for (int i = 0; i < k; i++) {
        h_centroid_x[i] = static_cast<float>(rand() % 400) / 100.0f;
        h_centroid_y[i] = static_cast<float>(rand() % 400) / 100.0f;
    }

    float *data_x, *data_y, *centroid_x, *centroid_y, *sum_x, *sum_y;
    int *labels, *count;

    cudaMalloc(&data_x, sample_size * sizeof(float));
    cudaMalloc(&data_y, sample_size * sizeof(float));
    cudaMalloc(&centroid_x, k * sizeof(float));
    cudaMalloc(&centroid_y, k * sizeof(float));
    cudaMalloc(&sum_x, k * sizeof(float));
    cudaMalloc(&sum_y, k * sizeof(float));
    cudaMalloc(&count, k * sizeof(int));
    cudaMalloc(&labels, sample_size * sizeof(int));

    cudaMemcpy(data_x, h_data_x, sample_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_y, h_data_y, sample_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(centroid_x, h_centroid_x, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(centroid_y, h_centroid_y, k * sizeof(float), cudaMemcpyHostToDevice);

    float old_x[128], old_y[128], new_x[128], new_y[128];

    int threads_block = 256;
    int blocks_grid = (sample_size + threads_block - 1) / threads_block;
    int blocks_cent = (k + threads_block - 1) / threads_block;

    for (int iter = 0; iter < max_iter; iter++) {
        cudaMemset(sum_x, 0, k * sizeof(float));
        cudaMemset(sum_y, 0, k * sizeof(float));
        cudaMemset(count, 0, k * sizeof(int));

        assign_and_compute<<<blocks_grid, threads_block>>>(data_x, data_y, centroid_x, centroid_y,
                                                           labels, sum_x, sum_y, count, sample_size, k);
        cudaDeviceSynchronize();

        update_centroids<<<blocks_cent, threads_block>>>(centroid_x, centroid_y, sum_x, sum_y, count, k);
        cudaDeviceSynchronize();

        cudaMemcpy(new_x, centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(new_y, centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost);

        bool converged = true;
        for (int i = 0; i < k; i++) {
            float dx = new_x[i] - old_x[i];
            float dy = new_y[i] - old_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist >= threshold) {
                converged = false;
                break;
            }
        }

        if (converged) break;

        // Copy new -> old
        for (int i = 0; i < k; i++) {
            old_x[i] = new_x[i];
            old_y[i] = new_y[i];
        }
    }

    cudaMemcpy(h_centroid_x, centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroid_y, centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Final centroids:\n";
    for (int i = 0; i < k; i++) {
        cout << "(" << h_centroid_x[i] << ", " << h_centroid_y[i] << ")\n";
    }

    cudaFree(data_x);
    cudaFree(data_y);
    cudaFree(centroid_x);
    cudaFree(centroid_y);
    cudaFree(sum_x);
    cudaFree(sum_y);
    cudaFree(count);
    cudaFree(labels);

    // Free host memory
    delete[] h_data_x;
    delete[] h_data_y;
    delete[] h_centroid_x;
    delete[] h_centroid_y;

    return 0;
}
