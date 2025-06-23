#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

__global__ void Game_of_life_kernel(const int *in, int *out, int w, int h){

    extern __shared__ int s_tiled[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int g_x = bx * blockDim.x + tx;
    int g_y = by * blockDim.y + ty;

    int s_x = tx + 1;
    int s_y = ty + 1;
    int s_w = blockDim.x + 2;

    // center
    if (g_x < w && g_y < h)
        s_tiled[s_y * s_w + s_x] = in[g_y * w + g_x];
    else
        s_tiled[s_y * s_w + s_x] = 0;

    // top
    if (ty == 0)
        s_tiled[(s_y - 1) * s_w + s_x] = (g_y > 0 && g_x < w) ? in[(g_y - 1) * w + g_x] : 0;

    // bottom
    if (ty == blockDim.y - 1)
        s_tiled[(s_y + 1) * s_w + s_x] = (g_y + 1 < h && g_x < w) ? in[(g_y + 1) * w + g_x] : 0;

    // left
    if (tx == 0)
        s_tiled[s_y * s_w + s_x - 1] = (g_x > 0 && g_y < h) ? in[g_y * w + g_x - 1] : 0;

    // right
    if (tx == blockDim.x - 1)
        s_tiled[s_y * s_w + s_x + 1] = (g_x + 1 < w && g_y < h) ? in[g_y * w + g_x + 1] : 0;

    // top-left
    if (tx == 0 && ty == 0)
        s_tiled[(s_y - 1) * s_w + s_x - 1] = (g_x > 0 && g_y > 0) ? in[(g_y - 1) * w + g_x - 1] : 0;

    // top-right
    if (tx == blockDim.x - 1 && ty == 0)
        s_tiled[(s_y - 1) * s_w + s_x + 1] = (g_x + 1 < w && g_y > 0) ? in[(g_y - 1) * w + g_x + 1] : 0;

    // bottom-left
    if (tx == 0 && ty == blockDim.y - 1)
        s_tiled[(s_y + 1) * s_w + s_x - 1] = (g_x > 0 && g_y + 1 < h) ? in[(g_y + 1) * w + g_x - 1] : 0;

    // bottom-right
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1)
        s_tiled[(s_y + 1) * s_w + s_x + 1] = (g_x + 1 < w && g_y + 1 < h) ? in[(g_y + 1) * w + g_x + 1] : 0;

    __syncthreads();

    if (g_x < w && g_y < h) {
        int sum = 0;
        sum += s_tiled[(s_y - 1) * s_w + (s_x - 1)];
        sum += s_tiled[(s_y - 1) * s_w + s_x];
        sum += s_tiled[(s_y - 1) * s_w + (s_x + 1)];
        sum += s_tiled[s_y * s_w + (s_x - 1)];
        sum += s_tiled[s_y * s_w + (s_x + 1)];
        sum += s_tiled[(s_y + 1) * s_w + (s_x - 1)];
        sum += s_tiled[(s_y + 1) * s_w + s_x];
        sum += s_tiled[(s_y + 1) * s_w + (s_x + 1)];

        int cell = s_tiled[s_y * s_w + s_x];
        int new_state = 0;
        if (cell == 1 && (sum == 2 || sum == 3)) {
            new_state = 1;
        } else if (cell == 0 && sum == 3) {
            new_state = 1;
        }
        out[g_y * w + g_x] = new_state;
    }
}


int main(){
    const int w = 64;
    const int h = 64;
    const int size = w * h;

    int *grid = (int *)malloc(size * sizeof(int));
    int *out = (int *)malloc(size * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < size; i++)
        grid[i] = rand() % 2; 
    
    int *dgrid; 
    int *dout;
    cudaMalloc((void **)&dgrid, size * sizeof(int));
    cudaMalloc((void **)&dout, size * sizeof(int));

    cudaMemcpy(dgrid, grid, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_size((w + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (h + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    size_t shared_memory_size = (BLOCK_DIM_X + 2) * (BLOCK_DIM_Y + 2) * sizeof(int);

    Game_of_life_kernel<<<grid_size, block_size, shared_memory_size>>>(dgrid, dout, w, h);
    cudaDeviceSynchronize();

    cudaMemcpy(out, dout, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("output \n");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%d ", out[i * w + j]);
        }
        printf("\n");
    }

    free(grid);
    free(out);
    cudaFree(dgrid);
    cudaFree(dout);

    return 0;

}