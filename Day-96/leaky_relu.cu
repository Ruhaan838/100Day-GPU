#include <iostream>
#include <cuda_runtime.h>

__global__ void leaky_relu_vec4_kernel(
    const float* input,
    float alpha,
    float* output,
    size_t total_vec4,
    size_t total_floats
){
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for(size_t v = idx; v < total_vec4; v += stride){
        float4 x = __ldg(&in4[v]);
        x.x = fmaxf(x.x, alpha * x.x);
        x.y = fmaxf(x.y, alpha * x.y);
        x.z = fmaxf(x.z, alpha * x.z);
        x.w = fmaxf(x.w, alpha * x.w);
        out4[v] = x;
    }

    size_t offset = total_vec4 * 4;
    for(size_t i = offset + idx; i < total_floats; i+= stride){
        float v = __ldg(&input[i]);
        output[i] = fmaxf(v, alpha * v);
    }
}

int main() {
    size_t n = 32;
    float alpha = 0.01f;
    float *h_in = new float[n], *h_out = new float[n];
    for (size_t i = 0; i < n; i++) h_in[i] = (i % 2 == 0 ? -i : i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    size_t total_vec4 = n / 4;
    size_t threads = 256;
    size_t blocks = (total_vec4 + threads - 1) / threads;
    leaky_relu_vec4_kernel<<<blocks, threads>>>(d_in, alpha, d_out, total_vec4, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}
