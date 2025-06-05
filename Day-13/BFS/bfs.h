#ifndef BFS_H
#define BFS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define THREADS_PER_BLOCK 256
#define MAX_FRONTIER_SIZE 100000000
#define AVG_EDES_PER_VERTEX 8
#define NUM_VERTEX 100000000

#define CUDA_ERROR(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (code %d) at %s:%d\n",            \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


void generate_random_graph(int num_vertex, int* num_edgs, int** edgs, int** dest);
void bfs_gpu(int source, int num_vertex, int num_edgs, int* h_edgs, int* h_dest, int* h_labels);
void bfs_cpu(int source, int num_vertex, int num_edgs, int* edgs, int* dest, int* labels);


#endif