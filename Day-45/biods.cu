#include <cuda_runtime.h>
#include <stdio.h>

__global__ void flockKernel(const float* agents, float* next_agent, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = 4 * i;
    float x = agents[base + 0];
    float y = agents[base + 1];
    float vx = agents[base + 2];
    float vy = agents[base + 3];

    const float r = 5.0f;
    const float r_sq = r * r;
    const float alpha = 0.05f;

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int neighbor_Count = 0;

    for(int j = 0; j < N; j++){
        if (j == i) continue;

        int jbase = 4 * j;
        float xj = agents[jbase + 0];
        float yj = agents[jbase + 1];

        float dx = xj - x;
        float dy = yj - y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < r_sq){
            sum_vx += agents[jbase + 2];
            sum_vy += agents[jbase + 3];
            neighbor_Count++;
        }
    }

    float new_vx = vx;
    float new_vy = vy;
    if (neighbor_Count > 0){
        float avg_vx = sum_vx / neighbor_Count;
        float avg_vy = sum_vy / neighbor_Count;
        // v_new = v + alpha * (avg_v - v)
        new_vx = vx + alpha * (avg_vx - vx);
        new_vy = vy + alpha * (avg_vy - vy);
    }

    float new_x = x + new_vx;
    float new_y = y + new_vy;

    next_agent[base + 0] = new_x;
    next_agent[base + 1] = new_y;
    next_agent[base + 2] = new_vx;
    next_agent[base + 3] = new_vy;
}

int main(){
    int N = 2;
    
    size_t a_size = N * 4 * sizeof(float);

    float agents[] = {0.0, 0.0, 1.0, 0.0, 
                    3.0, 4.0, 0.0, -1.0};

    float *agents_next = (float*)malloc(a_size);

    float *d_agents, *d_agents_next;
    cudaMalloc((void**)&d_agents, a_size);
    cudaMalloc((void**)&d_agents_next, a_size);

    cudaMemcpy(d_agents, agents, a_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    flockKernel<<<grid_size, block_size>>>(d_agents, d_agents_next, N);
    cudaDeviceSynchronize();

    cudaMemcpy(agents_next, d_agents_next, a_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N * 4; i++){
        printf("%f ", agents_next[i]);
    }
    printf("\n");

    cudaFree(d_agents);
    cudaFree(d_agents_next);
}