#include <hip/hip_runtime.h>
#include <iostream>

const int LOAD_SIZE = 32;

__global__ void prefix_sum_kernel(float *data, float* out, int N){
    int thred_idx = threadIdx.x;
    int idx = 2 * blockIdx.x * blockDim.x + thred_idx;

    __shared__ float shared_data[LOAD_SIZE];
    if (idx < N)
        shared_data[thred_idx] = data[idx];
    if (idx + blockDim.x < N)
        shared_data[thred_idx + blockDim.x] = data[idx + blockDim.x];
    
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        int j = stride * 2 * (thred_idx + 1) - 1;
        if (j < LOAD_SIZE)
            shared_data[j] += shared_data[j - stride];
    }
    __syncthreads();

    for (int stride = LOAD_SIZE / 4; stride >= 1; stride /= 2){
        __syncthreads();
        int j = stride * 2 * (thred_idx + 1) - 1;
        if (j < LOAD_SIZE - stride)
            shared_data[j + stride] += shared_data[j];
        __syncthreads();
    }

    if (idx < N)
        out[idx] = shared_data[thred_idx];
    if (idx + blockDim.x < N)
        out[idx + blockDim.x] = shared_data[thred_idx + blockDim.x];
    __syncthreads();
}

void check_hip(hipError_t err){
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void print_data(float* data, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const int N = 10;
    float A[N], C[N];

    for (int i = 0; i < N; i++) 
        A[i] = i + 1.0f;
    
    printf("Input data:\n");
    print_data(A, N);

    float *dA, *dC;
    check_hip(hipMalloc((void**)&dA, N * sizeof(float)));
    check_hip(hipMalloc((void**)&dC, N * sizeof(float)));

    check_hip(hipMemcpy(dA, A, N * sizeof(float), hipMemcpyHostToDevice));
    
    dim3 block_size(16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    hipLaunchKernelGGL(prefix_sum_kernel, grid_size, block_size, 0, 0, dA, dC, N);

    check_hip(hipGetLastError());
    check_hip(hipDeviceSynchronize());
    check_hip(hipMemcpy(C, dC, N * sizeof(float), hipMemcpyDeviceToHost));
    printf("Output data:\n");
    print_data(C, N);
    check_hip(hipFree(dA));
    check_hip(hipFree(dC));
    return 0;
}