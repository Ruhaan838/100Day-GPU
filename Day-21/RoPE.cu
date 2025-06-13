#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ void apply_rotary_embedding(
    float* query,
    float* key,
    const int head_dim,
    const int position,
    const float base = 1000.0f 
) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(base, (float)(i) / head_dim);
        float theta = position * freq;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float q_real = query[i];
        float q_img = query[i + 1];
        float k_real = key[i];
        float k_img = key[i + 1];

        query[i] = q_real * cos_theta - q_img * sin_theta;
        query[i + 1] = q_real * sin_theta + q_img * cos_theta;

        key[i] = k_real * cos_theta - k_img * sin_theta;
        key[i + 1] = k_real * sin_theta + k_img * cos_theta;
    }
}

__global__ void RoPE_Kernel(
    float* query, // [batch, seq_len, heads, head_dim]
    float* key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len * num_heads;
    if (idx >= total_threads) return;

    int batch_idx = idx / (seq_len * num_heads);
    int seq_idx = (idx / num_heads) % seq_len;
    int head_idx = idx % num_heads;

    int base_idx = batch_idx * (seq_len * num_heads * head_dim) +
                   seq_idx * (num_heads * head_dim) +
                   head_idx * head_dim;

    apply_rotary_embedding(
        &query[base_idx],
        &key[base_idx],
        head_dim,
        seq_idx
    );
}

void apply_rope(
    float* d_queries,
    float* d_keys,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    int total = batch_size * seq_len * num_heads;
    dim3 block_size(256);
    dim3 grid_size((total + block_size.x - 1) / block_size.x);

    RoPE_Kernel<<<grid_size, block_size>>>(
        d_queries,
        d_keys,
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}

void initialize_random_data(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void print_tensor_sample(float* data, int head_dim, const char* name) {
    printf("\n%s Sample:\n", name);
    for (int i = 0; i < head_dim && i < 8; i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");
}

int main() {
    srand(time(NULL));

    const int batch_size = 2;
    const int seq_len = 4;
    const int num_heads = 8;
    const int head_dim = 64;

    const int total_size = batch_size * seq_len * num_heads * head_dim;
    const size_t bytes = total_size * sizeof(float);

    float* h_queries = (float*)malloc(bytes);
    float* h_keys = (float*)malloc(bytes);
    float* h_queries_result = (float*)malloc(bytes);
    float* h_keys_result = (float*)malloc(bytes);

    initialize_random_data(h_queries, total_size);
    initialize_random_data(h_keys, total_size);

    print_tensor_sample(h_queries, head_dim, "Original Query");
    print_tensor_sample(h_keys, head_dim, "Original Key");

    float* d_queries;
    float* d_keys;

    cudaMalloc((void**)&d_queries, bytes);
    cudaMalloc((void**)&d_keys, bytes);

    cudaMemcpy(d_queries, h_queries, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, h_keys, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    apply_rope(d_queries, d_keys, batch_size, seq_len, num_heads, head_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_queries_result, d_queries, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys_result, d_keys, bytes, cudaMemcpyDeviceToHost);

    print_tensor_sample(h_queries_result, head_dim, "RoPE Query");
    print_tensor_sample(h_keys_result, head_dim, "RoPE Key");

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    cudaFree(d_queries);
    cudaFree(d_keys);
    free(h_queries);
    free(h_keys);
    free(h_queries_result);
    free(h_keys_result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
