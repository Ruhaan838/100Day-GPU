#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void compute_scores(const float* Q, const float* K, float* scores, int N, int d_model, int d_k) {
    int head = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float sum = 0.0f;
        int offset = head * d_k;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[i * d_model + offset + k];
            float k_val = K[j * d_model + offset + k];
            sum += q_val * k_val;
        }
        float scale = 1.0f / sqrtf((float)d_k);
        scores[head * N * N + i * N + j] = sum * scale;
    }
}

__global__ void softmax_kernel(float* scores, int N) {
    int head = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N) {
        int base = head * N * N + row * N;
        float max_val = -1e20f;
        for (int j = 0; j < N; j++) {
            float val = scores[base + j];
            if (val > max_val)
                max_val = val;
        }
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float exp_val = expf(scores[base + j] - max_val);
            scores[base + j] = exp_val;
            sum += exp_val;
        }
        for (int j = 0; j < N; j++)
            scores[base + j] /= sum;
    }
}

__global__ void compute_out_kernel(const float* scores, const float* V, float* out, int N, int d_model, int d_k) {
    int head = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && k < d_k) {
        float sum = 0.0f;
        int offset = head * d_k;
        for (int j = 0; j < N; j++) {
            float attn = scores[head * N * N + i * N + j];
            float v_val = V[j * d_model + offset + k];
            sum += attn * v_val;
        }
        out[i * d_model + offset + k] = sum;
    }
}

int main() {
    const int N = 4;         // sequence length
    const int d_model = 8;   // embedding size
    const int h = 2;         // number of heads
    const int d_k = d_model / h;

    float Q_h[N * d_model];
    float K_h[N * d_model];
    float V_h[N * d_model];

    for (int i = 0; i < N * d_model; i++) {
        Q_h[i] = 1.0f;
        K_h[i] = 2.0f;
        V_h[i] = 3.0f;
    }

    float *Q_d, *K_d, *V_d, *scores_d, *out_d;
    cudaMalloc(&Q_d, N * d_model * sizeof(float));
    cudaMalloc(&K_d, N * d_model * sizeof(float));
    cudaMalloc(&V_d, N * d_model * sizeof(float));
    cudaMalloc(&scores_d, h * N * N * sizeof(float));
    cudaMalloc(&out_d, N * d_model * sizeof(float));

    cudaMemcpy(Q_d, Q_h, N * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, N * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, N * d_model * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim1(16, 16);
    dim3 gridDim1((N + blockDim1.x - 1) / blockDim1.x, (N + blockDim1.y - 1) / blockDim1.y, h);

    compute_scores<<<gridDim1, blockDim1>>>(Q_d, K_d, scores_d, N, d_model, d_k);

    dim3 blockDim2(1, 256);
    dim3 gridDim2(1, (N + blockDim2.y - 1) / blockDim2.y, h);

    softmax_kernel<<<gridDim2, blockDim2>>>(scores_d, N);

    dim3 blockDim3(16, 16);
    dim3 gridDim3((d_k + blockDim3.x) / blockDim3.x, (N + blockDim3.y) / blockDim3.y, h);

    compute_out_kernel<<<gridDim3, blockDim3>>>(scores_d, V_d, out_d, N, d_model, d_k);

    float out_h[N * d_model];
    cudaMemcpy(out_h, out_d, N * d_model * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * d_model; i++) {
        printf("%f ", out_h[i]);
    }
    printf("\n");

    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(scores_d);
    cudaFree(out_d);

    return 0;
}
