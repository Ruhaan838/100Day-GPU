#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void LayerNorm(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;

    if (row < rows) {
        extern __shared__ float row_data[];

        int tid = threadIdx.x;
        int stride = blockDim.x;

        // Load data from global to shared memory
        for (int col = tid; col < cols; col += stride) {
            row_data[col] = input[row * cols + col];
        }

        __syncthreads();

        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += row_data[col];
        }
        mean /= cols;

        // Compute variance
        float var = 0.0f;
        for (int col = 0; col < cols; col++) {
            float diff = row_data[col] - mean;
            var += diff * diff;
        }
        var /= cols;

        float eps = 1e-7f;
        float stddev = sqrtf(var + eps);

        // Normalize and store to output
        for (int col = tid; col < cols; col += stride) {
            output[row * cols + col] = (row_data[col] - mean) / stddev;
        }
    }
}

void print_mat(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const int rows = 10, cols = 10;
    size_t size = rows * cols * sizeof(float);

    float* in = new float[rows * cols];
    float* out = new float[rows * cols];

    for (int i = 0; i < rows * cols; ++i) {
        in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_in, *d_out;
    hipMalloc(&d_in, size);
    hipMalloc(&d_out, size);

    hipMemcpy(d_in, in, size, hipMemcpyHostToDevice);

    dim3 grid(rows);                // One block per row
    dim3 block(32);                // Number of threads per block (for cols)
    size_t shared_mem = cols * sizeof(float); // Shared mem per block

    hipLaunchKernelGGL(LayerNorm, grid, block, shared_mem, 0, d_in, d_out, rows, cols);
    hipDeviceSynchronize();

    hipMemcpy(out, d_out, size, hipMemcpyDeviceToHost);

    printf("Input Matrix:\n");
    print_mat(in, rows, cols);

    printf("\nOutput Matrix (LayerNorm):\n");
    print_mat(out, rows, cols);

    hipFree(d_in);
    hipFree(d_out);
    delete[] in;
    delete[] out;

    return 0;
}
