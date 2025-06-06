#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

void printMat(const char* name, const float* mat, int row, int cols);
void init_random_mat(float* mat, int size, float min_val, float max_val);
cudaError_t allocate_Device_Mem(float** device_ptr, size_t size);
void free_Device_Mem(float* device_ptr);
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);