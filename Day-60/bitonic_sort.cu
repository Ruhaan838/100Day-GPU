#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int N = 1024;

__global__ void bitonic_sort_kernel(float* array){
    __shared__ float s_data[N];
    int tid = threadIdx.x;

    if (tid < N){
        s_data[tid] = array[tid];
    }

    __syncthreads();

    for(int k = 2; k <= N; k *= 2){
        for(int j = k / 2; j > 0; j /= 2){
            int ixj = tid ^ j;
            if(ixj > tid){
                if((tid & k) == 0){
                    if (s_data[tid] > s_data[ixj]){
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                } else {
                    if (s_data[tid] < s_data[ixj]){
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if(tid < N){
        array[tid] = s_data[tid];
    }
}

void bitonic_sort(float *array){
    float *d_array;
    size_t size = N * sizeof(float);

    cudaMalloc((void**)&d_array, size);
    cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice);

    bitonic_sort_kernel<<<1, N>>>(d_array);

    cudaDeviceSynchronize();

    cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void print_array(float *array){
    for(int i = 0; i < N; i++){
        printf("%f ", array[i]);
    }
    printf("\n");
}

int main(){
    float *array = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        array[i] = rand() % 1000 + 1; 
    }

    printf("Original array:\n");
    print_array(array);

    bitonic_sort(array);

    printf("Sorted array:\n");
    print_array(array);

    free(array);
    return 0;
}