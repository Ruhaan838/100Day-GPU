#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

using namespace std;

__global__ void swiGLU_Kernel(const float* x, const float* W1, const float* W2, float* out, int batch_size, int hidden_dim, int out_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_dim){
        float sum_W1 = 0.0f;
        float sum_W2 = 0.0f;

        for (int i = 0; i < hidden_dim; i++){
            sum_W1 += x[row * hidden_dim + i] * W1[col + i * out_dim];
            sum_W2 += x[row * hidden_dim + i] * W2[col + i * out_dim];
        }

        float sigmoid_val = 1.0f / (1.0f + expf(-sum_W1));
        float ans = sum_W1 * sigmoid_val * sum_W2;

        out[row * out_dim + col] = ans;
    }
}

void init_value(float* W, const int dim1, const int dim2){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < dim1 * dim2; i++)
        W[i] = dis(gen);
}

void print_data(float* data, const int dim1, const int dim2){
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            cout << data[i * dim1 + j] << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}

void run_swiGLU(const float* x, float* out, const int batch_size, const int hidden_dim, const int out_dim, bool print_w = false){
    
    // W1, W2 initalizer it with random values
    // allocate the memory for x, W1, W2, out in GPU
    // define gird size and block_size

    float *W1 = new float[hidden_dim * out_dim];
    float *W2 = new float[hidden_dim * out_dim];
    init_value(W1, hidden_dim, out_dim);
    init_value(W2, hidden_dim, out_dim);

    float *dx, *dout, *dW1, *dW2;

    cudaMalloc(&dx, batch_size * hidden_dim * sizeof(float));
    cudaMalloc(&dout, batch_size * out_dim * sizeof(float));
    cudaMalloc(&dW1, hidden_dim * out_dim * sizeof(float));
    cudaMalloc(&dW2, hidden_dim * out_dim * sizeof(float));

    cudaMemcpy(dx, x, batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW1, W1, hidden_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, W2, hidden_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, 
                    (out_dim + block_size.y - 1) / block_size.y);
    
    swiGLU_Kernel<<<grid_size, block_size>>>(dx, dW1, dW2, dout, batch_size, hidden_dim, out_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cerr << "CUDA Error at" << __FILE__ << __LINE__ << cudaGetErrorString(err) << '\n';
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(out, dout, batch_size * out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    if (print_w){
        cout << "W1 \n";
        print_data(W1, hidden_dim, out_dim);
        cout << "W2 \n";
        print_data(W2, hidden_dim, out_dim);
    }

    free(W1);
    free(W2);
    cudaFree(dx);
    cudaFree(dW1);
    cudaFree(dW2);
    cudaFree(dout);
}

int main(){
    int batch_size = 32;
    int hidden_dim = 128;
    int out_dim = 64;

    float* x = new float[batch_size * hidden_dim];
    init_value(x, batch_size, hidden_dim);
    float* out = new float[batch_size * out_dim];

    cout << "Data - (X): \n";
    print_data(x, batch_size, hidden_dim);

    run_swiGLU(x, out, batch_size, hidden_dim, out_dim, true);

    cout << "Data - (out): \n";
    print_data(out, batch_size, out_dim);
    
    free(x);
    free(out);

}