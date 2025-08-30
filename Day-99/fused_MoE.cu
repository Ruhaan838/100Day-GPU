#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

const int NUM_EXPERTS = 8;
const int INPUT_SIZE = 1024;
const int OUTPUT_SIZE = 512;
const int TOP_K = 2;

void CHECK_CUDA(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CHECK_CUBLAS(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *msg = "UNKNOWN";
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED: msg = "NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED: msg = "ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE: msg = "INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH: msg = "ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR: msg = "MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: msg = "EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR: msg = "INTERNAL_ERROR"; break;
            default: break;
        }
        fprintf(stderr, "CUBLAS error: %d (%s)\n", status, msg);
        exit(EXIT_FAILURE);
    }
}

__global__ void topk_softmax(const float* scores, int* indices, float* probs, int batch_size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    __shared__ float local[NUM_EXPERTS];
    int tid = threadIdx.x;
    if (tid < NUM_EXPERTS) {
        local[tid] = scores[b * NUM_EXPERTS + tid];
    }
    __syncthreads();

    if (tid == 0) {
        float vals[TOP_K];
        int idxs[TOP_K];
        for (int k = 0; k < TOP_K; k++) { vals[k] = -INFINITY; idxs[k] = -1; }

        for (int e = 0; e < NUM_EXPERTS; e++) {
            float v = local[e];
            for (int k = 0; k < TOP_K; k++) {
                if (v > vals[k]) {
                    for (int j = TOP_K - 1; j > k; j--) {
                        vals[j] = vals[j - 1];
                        idxs[j] = idxs[j - 1];
                    }
                    vals[k] = v;
                    idxs[k] = e;
                    break;
                }
            }
        }

        float maxVal = vals[0];
        for (int k = 1; k < TOP_K; k++) maxVal = fmaxf(maxVal, vals[k]);
        float sumExp = 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            vals[k] = expf(vals[k] - maxVal);
            sumExp += vals[k];
        }
        for (int k = 0; k < TOP_K; k++) {
            indices[b * TOP_K + k] = idxs[k];
            probs[b * TOP_K + k] = vals[k] / sumExp;
        }
    }
}

void runMoE(const float* input,
            const float* gating_W,
            const float* expert_W,
            int batch_size,
            float* output) {

    float *d_input = nullptr, *d_gating_W = nullptr, *d_expert_W = nullptr;
    float *d_output = nullptr, *d_scores = nullptr, *d_topk_scores = nullptr;
    int *d_topk_indices = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, (size_t)batch_size * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gating_W, (size_t)NUM_EXPERTS * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_W, (size_t)NUM_EXPERTS * OUTPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, (size_t)batch_size * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, (size_t)batch_size * NUM_EXPERTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_scores, (size_t)batch_size * TOP_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_topk_indices, (size_t)batch_size * TOP_K * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_input, input, (size_t)batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gating_W, gating_W, (size_t)NUM_EXPERTS * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_W, expert_W, (size_t)NUM_EXPERTS * OUTPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, (size_t)batch_size * OUTPUT_SIZE * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,   
    NUM_EXPERTS, batch_size, INPUT_SIZE,
    &alpha,
    d_gating_W, INPUT_SIZE,     
    d_input, INPUT_SIZE,        
    &beta,
    d_scores, NUM_EXPERTS));    

    dim3 grid(batch_size);
    dim3 block(NUM_EXPERTS); 
    topk_softmax<<<grid, block>>>(d_scores, d_topk_indices, d_topk_scores, batch_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    vector<int> indices(batch_size * TOP_K);
    vector<float> probs(batch_size * TOP_K);
    CHECK_CUDA(cudaMemcpy(indices.data(), d_topk_indices, (size_t)batch_size * TOP_K * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(probs.data(), d_topk_scores, (size_t)batch_size * TOP_K * sizeof(float), cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < TOP_K; ++k) {
            int e = indices[b * TOP_K + k];
            if (e < 0 || e >= NUM_EXPERTS) continue;
            float weight = probs[b * TOP_K + k];

            const float* d_expert_ptr = d_expert_W + (size_t)e * OUTPUT_SIZE * INPUT_SIZE;

            float cur_beta = (k == 0 ? 1.0f : 1.0f);
            CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            OUTPUT_SIZE, 1, INPUT_SIZE,
            &weight,
            d_expert_ptr, OUTPUT_SIZE,
            d_input + b * INPUT_SIZE, INPUT_SIZE,
            &cur_beta,
            d_output + b * OUTPUT_SIZE, OUTPUT_SIZE));

        }
    }

    // Copy output back
    CHECK_CUDA(cudaMemcpy(output, d_output, (size_t)batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(d_input);
    cudaFree(d_gating_W);
    cudaFree(d_expert_W);
    cudaFree(d_output);
    cudaFree(d_scores);
    cudaFree(d_topk_scores);
    cudaFree(d_topk_indices);
}

int main() {
    const int batch_size = 32;

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    vector<float> input((size_t)batch_size * INPUT_SIZE);
    vector<float> gating_W((size_t)NUM_EXPERTS * INPUT_SIZE);
    vector<float> expert_W((size_t)NUM_EXPERTS * OUTPUT_SIZE * INPUT_SIZE);
    vector<float> output((size_t)batch_size * OUTPUT_SIZE);

    for (auto &x: input) x = dist(rng);
    for (auto &x: gating_W) x = dist(rng);
    for (auto &x: expert_W) x = dist(rng);

    runMoE(input.data(), gating_W.data(), expert_W.data(), batch_size, output.data());

    printf("Output sample [0-9]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Output[%d]: %f\n", i, output[i]);
    }

    return 0;
}
