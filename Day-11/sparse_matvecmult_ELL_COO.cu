#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void ELL_COO_kernel(const float* a, const float* x, float* data_ell,
                                int* indx_ell, float* data_coo, int* row_coo, int* col_coo, float* output_mat, const int threshold,
                                const int N, const int M, int* global_coo_counter)
    {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N) return;

    int count = 0;
    
    //processing_row
    for (int col=0; col < M; ++col){
        float val = a[row * M + col];
        if(val != 0){
            if (count < threshold){
                //ell formate
                data_ell[count * N + row] = val;
                indx_ell[count * N + row] = col;
                count++;
            } else {
                // coo formate
                int coo_idx = atomicAdd(global_coo_counter, 1);
                data_coo[coo_idx] = val;
                row_coo[coo_idx] = row;
                col_coo[coo_idx] = col;
            }
        }
    }

    //fill unsed ell sloths with zeros
    for (int i = count; i < threshold; ++i){
        data_ell[i * N + row] = 0;
        indx_ell[i * N + row] = -1;
    }

    //do the mat and vec multiplication using ell formate 
    float acc = 0.0f;
    for (int p = 0; p < threshold; ++p){
        int idx = indx_ell[p * N + row];
        if (idx != -1)
            acc += data_ell[p * N + row] * x[idx];
    }

    for (int i = 0; i < *global_coo_counter; ++i){
        if(row_coo[i] == row)
            acc += data_coo[i] * x[col_coo[i]];
    }

    output_mat[row] = acc;
}

int main(){
    const int N = 1000; //row
    const int M = 1000; //col
    const int threshold = 20;

    float* a = new float[N * M];
    //init to zeros
    float* data_ell = new float[N * threshold]();
    float* data_coo = new float[N * M]();
    int* indx_ell = new int[N * threshold]();
    int* row_coo = new int[N * M]();
    int* col_coo = new int[N * M]();

    float* x = new float[M];
    float* output_mat = new float[N];

    int* d_global_coo_count;
    cudaMalloc(&d_global_coo_count, sizeof(int)); CudaError("Failed to allocate global coo counter");
    cudaMemset(d_global_coo_count, 0, sizeof(int)); CudaError("Failed to init the global coo counter with 0");

    for (int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            a[i * M + j] = (i + j) % 3 == 0 ? i + j : 0;
        }
    }

    for (int i = 0; i < M; i++)
        x[i] = 1.0f;

    float *da, *dx, *d_data_ell, *d_data_coo, *d_output_mat;
    int *d_indx_ell, *d_row_coo, *d_col_coo;

    
    cudaMalloc(&da, N * M * sizeof(float)); CudaError("Failed to allocate the memory for da");
    cudaMalloc(&dx, M * sizeof(float)); CudaError("Failed to allocate the memory for dx");
    cudaMalloc(&d_data_ell, N * threshold * sizeof(float)); CudaError("Failed to allocate the memory for data_ell");
    cudaMalloc(&d_data_coo, N * M * sizeof(float)); CudaError("Failed to allocate the memory for data_coo");
    cudaMalloc(&d_indx_ell, N * threshold * sizeof(float)); CudaError("Failed to allocate the memory for d_indx_ell");
    cudaMalloc(&d_row_coo, N * M * sizeof(float)); CudaError("Failed to allocate the memory for d_row_coo");
    cudaMalloc(&d_col_coo, N * M * sizeof(float)); CudaError("Failed to allocate the memory for d_col_coo");
    cudaMalloc(&d_output_mat, N * sizeof(float)); CudaError("Failed to allocate the memory for d_output_mat");

    
    cudaMemcpy(da, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, a, M * sizeof(float), cudaMemcpyHostToDevice);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    dim3 block_size(256);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); CudaError("Failed to start the Cuda Event");
    cudaEventCreate(&stop); CudaError("Failed to stop the Cuda Event");

    cudaEventRecord(start); CudaError("Failed to Recored the start event");

    ELL_COO_kernel<<<num_blocks, block_size>>>(
        da, dx, d_data_ell, d_indx_ell, d_data_coo, d_row_coo,
        d_col_coo, d_output_mat, threshold, N, M, d_global_coo_count
    );

    CudaError("Failed to Launch the ELL and COO kernel");
    cudaDeviceSynchronize(); CudaError("Failed to sync the cuda device");

    cudaEventRecord(stop); CudaError("Failed to Recored the stop event");
    cudaEventSynchronize(stop);CudaError("Failed to sync the stop event");

    float mil_s = 0;
    cudaEventElapsedTime(&mil_s, start, stop); CudaError("Failed to elapsed Time");
    std::cout << "CUDA kernel time: " << mil_s / 1000.0 << " seconds" << std::endl;

    cudaMemcpy(data_ell, d_data_ell, N * threshold * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy data_ell");
    cudaMemcpy(data_coo, d_data_coo, N * M * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy data_coo");
    cudaMemcpy(indx_ell, d_indx_ell, N * threshold * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy data_ell");

    cudaMemcpy(row_coo, d_row_coo, N * M * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy row_coo");
    cudaMemcpy(col_coo, d_col_coo, N * M * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy col_coo");

    cudaMemcpy(output_mat, d_output_mat, N * sizeof(float), cudaMemcpyDeviceToHost); CudaError("Failed to copy output_mat");


    cudaEventDestroy(start); CudaError("Failed to Destroy the start event");
    cudaEventDestroy(stop); CudaError("Failed to Destroy the stop event");

    int global_coo_count;
    cudaMemcpy(&global_coo_count, d_global_coo_count, sizeof(int), cudaMemcpyDeviceToHost); CudaError("Failed to copy global_coo_count");

    for(int i = 0; i < 10; ++i)
        std::cout << "Coo[" << i << "]: val=" << data_coo[i] << ", row = " << row_coo[i] << ", col = " << col_coo[i] << std::endl;
    
    FILE *output_file = fopen("cuda_results.txt", "w");
    if (output_file == nullptr){
        std::cerr << "Failed to open output file! \n";
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; i++)
        fprintf(output_file, "%.10f\n", output_mat[i]);
    
    fclose(output_file); 
    


    cudaFree(da);
    cudaFree(dx);
    cudaFree(d_data_coo);
    cudaFree(d_data_ell);
    cudaFree(d_indx_ell);
    cudaFree(d_row_coo);
    cudaFree(d_col_coo);
    cudaFree(d_output_mat);

    delete[] a;
    delete[] data_ell;
    delete[] data_coo;
    delete[] indx_ell;
    delete[] row_coo;
    delete[] col_coo;
    delete[] x;
    delete[] output_mat;

    return 0;
}