#include <iostream>

__global__ void MatrixAdd(const float* a, const float* b, float* c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // gives the row idx
    int j = blockIdx.y * blockDim.y + threadIdx.y; // gives the col idx

    if (i >= N || j >= N) return;

    c[i * N + j] = a[i * N + j] + b[i * N + j]; // simple add
}

// lol!!, I am sooo lazy that's why this function here !!! XD
void print_mat(const float* mat, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%.2f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main(){
    const int N = 10;
    float *A, *B, *C;

    A = (float*)malloc(N * N * sizeof(float)); //allocatte the N * N size mat
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    //fill the mat with 1 and 2 and 0.
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    float *da, *db, *dc; // make the vairable for the cuda

    cudaMalloc((void**)&da, N * N * sizeof(float)); // allocate the N * N size mat in cuda
    cudaMalloc((void**)&db, N * N * sizeof(float));
    cudaMalloc((void**)&dc, N * N * sizeof(float));

    cudaMemcpy(da, A, N * N * sizeof(float), cudaMemcpyHostToDevice); // copy the hole mat to cuda
    cudaMemcpy(db, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 16); // NEW: dim3: can stores 3D data.
    dim3 dimGrid(ceil(N / 32.0f), ceil(N / 16.0f)); // make the block size and grid size for 2D!!!

    MatrixAdd<<<dimGrid, dimBlock>>>(da, db, dc, N);
    cudaDeviceSynchronize(); // NEW: it forces the CPU to wait for the GPU to finish its tasks. (basically stops the sysnchronization).

    cudaMemcpy(C, dc, N * N * sizeof(float), cudaMemcpyDeviceToHost); //get the result mat.

    printf("C:\n");
    print_mat(C, N);

    printf("A:\n");
    print_mat(A, N);

    printf("B:\n");
    print_mat(B, N);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

}