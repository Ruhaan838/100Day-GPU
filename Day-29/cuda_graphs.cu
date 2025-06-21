#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;


void cuda_check(cudaError_t err, string msg){
    if (err != cudaSuccess){
        cout << "Cuda Error at "  << __FILE__ << __LINE__ << cudaGetErrorString(err) << " For " << msg << "\n";
        exit(EXIT_FAILURE);
    }
}

const int N = 100000;
const int NUM_ITERATIONS = 10000;
const int BLOCK_SIZE = 256;

__global__ void matrixAdd(const float* a, const float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

__global__ void matrixScale(float* a, float scale, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] = a[idx ] * scale;
}

__global__ void matrixSquare(float* a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] = a[idx] * a[idx];
}

__global__ void matrixOffset(float* a, float offset, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] = a[idx] + offset; 
}

void Result_varification(float* a, float* b, float* c, float* verify, int n){
    for (int i = 0; i < n; i++){
        float temp = a[i] + b[i];
        temp = temp * 2.0f;
        temp = temp * temp;
        verify[i] = temp + 1.0f;
    }

    bool match = true;
    for (int i = 0; i < n; i++){
        if (abs(verify[i] - c[i]) > 1e-5){
            match = false;
            printf("Mismatch at idx %d: Expected %f, got %f\n", i, verify[i], c[i]);
            break;
        }
    }

    if (match)
        printf("All values match not a single problem\n");
}

int main(){
    float *a, *b, *c, *verify;
    float *da, *db, *dc;
    size_t size = N * sizeof(float);

    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    verify = (float*)malloc(size);

    for (int i = 0; i < N; i++){
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    cuda_check(cudaMalloc(&da, size), "allocate da");
    cuda_check(cudaMalloc(&db, size), "allocate db");
    cuda_check(cudaMalloc(&dc, size), "allocate dc");

    cuda_check(cudaMemcpy(da, a, size, cudaMemcpyHostToDevice), "da -> a");
    cuda_check(cudaMemcpy(db, b, size, cudaMemcpyHostToDevice), "db -> b");

    cudaStream_t stream;
    cudaEvent_t start, stop;
    cuda_check(cudaStreamCreate(&stream), "Stream create");
    cuda_check(cudaEventCreate(&start), "Stream start");
    cuda_check(cudaEventCreate(&stop), "Stream stop");


    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < 10; i++){
        matrixAdd<<<grid_size, BLOCK_SIZE, 0, stream>>>(da, db, dc, N);
        matrixScale<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 2.0f, N);
        matrixSquare<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, N);
        matrixOffset<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 1.0f, N);
    }
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < NUM_ITERATIONS; i++){
        matrixAdd<<<grid_size, BLOCK_SIZE, 0, stream>>>(da, db, dc, N);
        matrixScale<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 2.0f, N);
        matrixSquare<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, N);
        matrixOffset<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 1.0f, N);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float mil_s = 0;
    cudaEventElapsedTime(&mil_s, start, stop);
    printf("Without CUDA Graphs: %.3f ms\n", mil_s);

    cuda_check(cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost), "c -> dc");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "stream capture");
    matrixAdd<<<grid_size, BLOCK_SIZE, 0, stream>>>(da, db, dc, N);
    matrixScale<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 2.0f, N);
    matrixSquare<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, N);
    matrixOffset<<<grid_size, BLOCK_SIZE, 0, stream>>>(dc, 1.0f, N);
    cuda_check(cudaStreamEndCapture(stream, &graph), "Stream End capture");
    cuda_check(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "graph instatiante");

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&mil_s, start, stop);
    printf("With CUDA Graphs: %.3f ms\n", mil_s);

    cuda_check(cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost), "c - > dc");
    Result_varification(a, b, c, verify, N);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
    free(verify);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);

    return 0;
}