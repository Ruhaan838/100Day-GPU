#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define EXPM1f(x) expm1f(x)
#define EXP2f(x) __exp2f(x)

__global__ __launch_bounds__ (1024, 4)
void elu_fp16(const float* input, float* output, size_t total, float alpha){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec8 = (total / 8) * 8;
    
    for (size_t base = tid * 8; base < vec8; base += stride * 8){
        float4 f0 = __ldg((const float4*)(input + base));
        float4 f1 = __ldg((const float4*)(input + base + 4));

        f0.x = f0.x > 0.0f ? f0.x : alpha * EXPM1f(f0.x);
        f0.y = f0.y > 0.0f ? f0.y : alpha * EXPM1f(f0.y);
        f0.z = f0.z > 0.0f ? f0.z : alpha * EXPM1f(f0.z);
        f0.w = f0.w > 0.0f ? f0.w : alpha * EXPM1f(f0.w);

        f1.x = f1.x > 0.0f ? f1.x : alpha * EXPM1f(f1.x);
        f1.y = f1.y > 0.0f ? f1.y : alpha * EXPM1f(f1.y);
        f1.z = f1.z > 0.0f ? f1.z : alpha * EXPM1f(f1.z);
        f1.w = f1.w > 0.0f ? f1.w : alpha * EXPM1f(f1.w);

        reinterpret_cast<float4*>(output + base)[0] = f0;
        reinterpret_cast<float4*>(output + base)[0] = f1;
    }
    for(size_t i = vec8 + tid; i < total; i += stride){
        float x = __ldg(&input[i]);
        output[i] = x > 0.f ? x : alpha * EXPM1f(x);
    }
}

int main() {
    size_t total = 120;
    float alpha = 1.0f;
    float *d_in, *d_out;
    float *in = new float[total];
    float *out = new float[total];

    for (size_t i = 0; i < total; i++) in[i] = (i * 10);
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_in, in, total * sizeof(float), cudaMemcpyHostToDevice);

    int block = 1024;
    int grid = (total + block - 1) / block;
    elu_fp16<<<grid, block>>>(d_in, d_out, total, alpha);

    cudaMemcpy(out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < total; i++){
        printf("%f ", out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] in;
    delete[] out;
    return 0;
}
