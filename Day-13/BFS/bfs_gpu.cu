#include "bfs.h"
#include "bfs_kernel.cu"

void bfs_gpu(int source, int num_vertex, int num_edgs, int* h_edgs, int* h_dest, int* h_labels){
    int *d_edgs, *d_dest, *d_labels, *d_done;

    CUDA_ERROR(cudaMalloc((void**)&d_edgs, (num_vertex + 1) * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_dest, num_edgs * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_labels, num_vertex * sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&d_done, sizeof(int)));

    CUDA_ERROR(cudaMemset(d_labels, -1, num_vertex * sizeof(int)));

    CUDA_ERROR(cudaMemcpy(d_edgs, h_edgs, (num_vertex + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_dest, h_dest, num_edgs * sizeof(int), cudaMemcpyHostToDevice));

    int init_level = 0;
    CUDA_ERROR(cudaMemcpy(d_labels + source, &init_level, sizeof(int), cudaMemcpyHostToDevice));

    int level = 0;
    int h_done;
    int thread_block = THREADS_PER_BLOCK;
    int block_per_grid = (num_vertex + thread_block - 1) / thread_block;

    do {
        h_done = 1;
        CUDA_ERROR(cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice));

        bfs_kernel<<<block_per_grid, thread_block>>>(level, num_vertex, d_edgs, d_edgs, d_labels, d_done);
        CUDA_ERROR(cudaDeviceSynchronize());

        CUDA_ERROR(cudaMemcpy(&h_done, d_done, sizeof(int)), cudaMemcpyDeviceToHost);
        level++;
    } while (!h_done && level < num_vertex);

    CUDA_ERROR(cudaMemcpy(h_labels, d_labels, num_vertex * sizeof(int)), cudaMemcpyDeviceToHost);

    CUDA_ERROR(cudaFree(d_edgs));
    CUDA_ERROR(cudaFree(d_dest));
    CUDA_ERROR(cudaFree(d_labels));
    CUDA_ERROR(cudaFree(d_done));
    

}