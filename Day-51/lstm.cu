#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

const int TIMESTEPS = 5;

__device__ float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_kernel(const float* xt, const float* ht_prev, const float* ct_prev,
    const float* W, const float* U, const float* b, float* ht, float* ct, int hidden_size, int batch_size){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size * batch_size) return;

    int n_idx = idx % hidden_size;

    float input_gate = sigmoid(W[n_idx] * xt[idx] + 
                                U[n_idx] * ht_prev[idx]+
                                b[n_idx]);

    float forget_gate = sigmoid(W[hidden_size + n_idx] * xt[idx] + 
                                U[hidden_size + n_idx] * ht_prev[idx] + 
                                b[hidden_size + n_idx]);

    float output_gate = sigmoid(W[2 * hidden_size + n_idx] * xt[idx] + 
                                U[2 * hidden_size + n_idx] * ht_prev[idx]+
                                b[2 * hidden_size + n_idx]);

    float candidate = tanhf(W[3 * hidden_size + n_idx] * xt[idx]+
                            U[3 * hidden_size + n_idx] * ht_prev[idx]+
                            b[3 * hidden_size + n_idx]);

    ct[idx] = forget_gate * ct_prev[idx] + input_gate * candidate;
    ht[idx] = output_gate * tanhf(ct[idx]);
}

int main() {
    int hidden_size = 128;
    int batch_size = 2;
    int num_ele = hidden_size * batch_size;
    
    size_t num_size = num_ele * sizeof(float);
    size_t hidden = 4 * hidden_size * sizeof(float);

    float *xt, *ht_prev, *ct_prev, *W, *U, *b, *ht, *ct;
    cudaMallocManaged(&xt, num_size);
    cudaMallocManaged(&ht_prev, num_size);
    cudaMallocManaged(&ct_prev, num_size);

    cudaMallocManaged(&W, hidden);
    cudaMallocManaged(&U, hidden);
    cudaMallocManaged(&b, hidden);

    cudaMallocManaged(&ht, num_size);
    cudaMallocManaged(&ct, num_size);

    for(int i = 0; i < num_ele; i++){
        ht_prev[i] = 0.5f;
        ct_prev[i] = 0.5f;
        xt[i] = 1.0f;
    }

    for(int i = 0; i < hidden_size; i++){
        W[i] = 0.5f;
        U[i] = 0.5f;
        b[i] = 0.5f;
        
        W[hidden_size + i] = 0.5f;
        U[hidden_size + i] = 0.5f;
        b[hidden_size + i] = 0.2f;

        W[2 * hidden_size + i] = 0.5f;
        U[2 * hidden_size + i] = 0.5f;
        b[2 * hidden_size + i] = 0.3f;

        W[3 * hidden_size + i] = 0.5f;
        U[3 * hidden_size + i] = 0.5f;
        b[3 * hidden_size + i] = 0.0f;
    }

    int threads = 128;
    int num_blocks = (num_ele + threads - 1) / threads;

    cout << fixed << setprecision(6);
    cout << "LSTM processing over " << TIMESTEPS << " timesetps:\n";

    for(int t = 0; t < TIMESTEPS; t++){
        lstm_kernel<<<num_blocks, threads>>>(xt, ht_prev, ct_prev, W, U, b, ht, ct, hidden_size, batch_size);
        cudaDeviceSynchronize();

        for(int i = 0; i < num_ele; i++){
            ht_prev[i] = ht[i];
            ct_prev[i] = ct[i];
        }
        cout << "at Timestep " << t + 1 << ": ht[0] = " << ht[0] << ", ct[0] = " << ct[0] << '\n';
    }

    cudaFree(xt);
    cudaFree(ht_prev);
    cudaFree(ct_prev);
    cudaFree(W);
    cudaFree(U);
    cudaFree(b);
    cudaFree(ht);
    cudaFree(ct);

    return 0;
}