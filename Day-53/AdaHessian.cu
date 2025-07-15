#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void adaHessianKernel(
    float* theta,
    const float* grad,
    const float* gradPerturbed,
    float* m_moment,
    float* v_moment,
    const float lr,
    const float beta1, 
    const float beta2,
    const float eps,
    const float delta,
    int N
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    float h_diag = (gradPerturbed[idx] - grad[idx]) / delta;
    m_moment[idx] = beta1 * m_moment[idx] + (1.0f - beta1) * grad[idx];
    v_moment[idx] = beta2 * v_moment[idx] + (1.0f - beta2) * powf(h_diag, 2);

    theta[idx] -= lr * m_moment[idx] / (sqrtf(v_moment[idx]) + eps);
}

int main(){
    const int N = 10;
    size_t size = N * sizeof(float);
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-7f;
    const float delta = 1e-4f;

    float theta[N], grad[N], gradPerturbed[N], m_moment[N], v_moment[N];

    for(int i = 0; i < N; i++){
        theta[i] = 1.0f;
        grad[i] = 0.1f;
        gradPerturbed[i] = 0.1f + 0.001f * i;
        m_moment[i] = 0.0f;
        v_moment[i] = 0.0f;
    }

    float *d_theta, *d_grad, *d_gradPerturbed, *d_m_moment, *d_v_moment;
    cudaMalloc((void**)&d_theta, size);
    cudaMalloc((void**)&d_grad, size);
    cudaMalloc((void**)&d_gradPerturbed, size);
    cudaMalloc((void**)&d_m_moment, size);
    cudaMalloc((void**)&d_v_moment, size);

    cudaMemcpy(d_theta, theta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradPerturbed, gradPerturbed, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_moment, m_moment, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_moment, v_moment, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    adaHessianKernel<<<grid_size, block_size>>>(
        d_theta, d_grad, d_gradPerturbed, d_m_moment, d_v_moment,
        lr, beta1, beta2, eps, delta, N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(theta, d_theta, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_moment, d_m_moment, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_moment, d_v_moment, size, cudaMemcpyDeviceToHost);

    printf("Updated theta Values:\n");
    for(int i = 0; i < N; i++){
        printf("%f ", theta[i]);
    }
    printf("\n");

    cudaFree(d_theta);
    cudaFree(d_grad);
    cudaFree(d_gradPerturbed);
    cudaFree(d_m_moment);
    cudaFree(d_v_moment);

    return 0;
}