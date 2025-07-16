#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int HIDDEN_SIZE = 128;
const int INPUT_SIZE = 128;
const int SEQ_LEN = 50;
const int BATCH_SIZE = 32;

using namespace std;

__device__ float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_forward(float* input, float* h_prev, float* c_prev, float* W, float* U,
    float* b, float* h_out, float* c_out){
        int batch_idx = blockIdx.x;
        int neuron_idx = threadIdx.x;

        if(neuron_idx >= HIDDEN_SIZE) return;

        float x_t = input[batch_idx * INPUT_SIZE + neuron_idx];
        float h_prev_t = h_prev[batch_idx * HIDDEN_SIZE + neuron_idx];
        float c_prev_t = c_prev[batch_idx * HIDDEN_SIZE + neuron_idx];

        float it = sigmoid(x_t * W[neuron_idx] + h_prev_t * U[neuron_idx] + b[neuron_idx]);
        float ft = sigmoid(x_t * W[neuron_idx + HIDDEN_SIZE] + h_prev_t * U[neuron_idx + HIDDEN_SIZE] + b[neuron_idx + HIDDEN_SIZE]);
        float ot = sigmoid(x_t * W[neuron_idx + 2 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 2 * HIDDEN_SIZE] + b[neuron_idx + 2 * HIDDEN_SIZE]);
        float gt = sigmoid(x_t * W[neuron_idx + 3 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 3 * HIDDEN_SIZE] + b[neuron_idx + 3 * HIDDEN_SIZE]);

        float ct = ft * c_prev_t + it * gt;
        float ht = ot * tanhf(ct);

        h_out[batch_idx * HIDDEN_SIZE + neuron_idx] = ht;
        c_out[batch_idx * HIDDEN_SIZE + neuron_idx] = ct;
}

void bidirectional_lstm(float* input, float* h_forward, float* c_forward, float* h_backward, float* c_backward, float* W, float *U, float *b, float* output){
    for(int t = 0; t < SEQ_LEN; t++){
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + t * BATCH_SIZE * INPUT_SIZE, h_forward, c_forward, W, U, b, h_forward, c_forward);
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + (SEQ_LEN - 1 - t) * BATCH_SIZE * INPUT_SIZE, h_backward, c_backward, W, U, b, h_backward, c_backward);
    }
    cudaDeviceSynchronize();

    size_t size = BATCH_SIZE * HIDDEN_SIZE * sizeof(float);
    cudaMemcpy(output, h_forward, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + BATCH_SIZE * HIDDEN_SIZE, h_backward, size, cudaMemcpyDeviceToHost);
}

int main(){
    srand(time(NULL));
    size_t in_size = SEQ_LEN * BATCH_SIZE * INPUT_SIZE;
    size_t hb_size = BATCH_SIZE * HIDDEN_SIZE;
    size_t hi_size = HIDDEN_SIZE * INPUT_SIZE;
    size_t hh_size = HIDDEN_SIZE * HIDDEN_SIZE;

    vector<float> h_input(in_size, 1.0f);
    vector<float> h_output(2 * hb_size, 1.0f);
    vector<float> h_W(4 * hi_size, 1.0f);
    vector<float> h_U(4 * hh_size, 1.0f);
    vector<float> h_b(4 * HIDDEN_SIZE, 1.0f);

    for(auto& i : h_W) i = ((float) rand() / RAND_MAX) * 1.0f;
    for(auto& i : h_U) i = ((float) rand() / RAND_MAX) * 1.0f;
    for(auto& i : h_b) i = 0.0f;

    float *d_input, *d_h_forward, *d_c_forward, *d_h_backward, *d_c_backward, *dW, *dU, *db, *d_output;
    cudaMalloc(&d_input, in_size * sizeof(float));
    cudaMalloc(&d_h_forward, hb_size * sizeof(float));
    cudaMalloc(&d_c_forward, hb_size * sizeof(float));
    cudaMalloc(&d_h_backward, hb_size * sizeof(float));
    cudaMalloc(&d_c_backward, hb_size * sizeof(float));

    cudaMalloc(&dW, 4 * hi_size * sizeof(float));
    cudaMalloc(&dU, 4 * hh_size * sizeof(float));
    cudaMalloc(&db, 4 * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, 2 * hb_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, h_W.data(), 4 * hi_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dU, h_U.data(), 4 * hh_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, h_b.data(), 4 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_h_forward, 0, hb_size * sizeof(float));
    cudaMemset(d_c_forward, 0, hb_size * sizeof(float));
    cudaMemset(d_h_backward, 0, hb_size * sizeof(float));
    cudaMemset(d_c_backward, 0, hb_size * sizeof(float));

    bidirectional_lstm(d_input, d_h_forward, d_c_forward, d_h_backward, d_c_backward, dW, dU, db, d_output);

    cudaMemcpy(h_output.data(), d_output, 2 * hb_size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Bidirectional LSTM out:\n";
    for(int i = 0; i < 10; i++) cout << h_output[i] << " ";
    cout << '\n';

    cudaFree(d_input);
    cudaFree(d_h_backward);
    cudaFree(d_h_forward);
    cudaFree(d_c_backward);
    cudaFree(d_c_forward);
    cudaFree(dW);
    cudaFree(dU);
    cudaFree(db);
    cudaFree(d_output);
}
