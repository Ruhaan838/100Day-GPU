#include "cuda_helper.cuh"

void printMat(const char* name, const float* mat, int row, int cols){
    cout << name << " (" << row << "x" << cols << "):\n";
    for (int i = 0; i < row; i++){
        for (int j = 0; j < cols; j++){
                cout << mat[i * cols + j] << " ";
        }
        cout << '\n';
    }
}

void init_random_mat(float* mat, int size, float min_val, float max_val){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; i++)
        mat[i] = dis(gen);
    
}

cudaError_t allocate_Device_Mem(float** device_ptr, size_t size){
    return cudaMalloc(device_ptr, size);
}

void free_Device_Mem(float* device_ptr){
    if (device_ptr)
        cudaFree(device_ptr);
}
void checkCudaStatus(cudaError_t status){
    if (status != cudaSuccess){
        cerr << "CUDA error: " << cudaGetErrorString(status) << '\n';
        exit(EXIT_FAILURE);
    }
}
void checkCublasStatus(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS){
        cerr << "CUBLAS Error: " << status << '\n';
        exit(EXIT_FAILURE);
    }
}
