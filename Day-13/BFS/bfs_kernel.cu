#include "bfs.h"

__global__ void bfs_kernel(int level, int num_vertex, int* edge, int* lebels, int* done){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < num_vertex && labels[idx] == level){
        for(int edge = edges[idx]; edge < edges[idx + 1]; edge++){
            int neighbor = dest[edge];

            if(atomicCAS(&labels[neighbor], -1, level + 1) == -1)
                atomicExch(done, 0);
        }
    } 
}
