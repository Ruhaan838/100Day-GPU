#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int NUM_SIMULATIONS = 1024;
const int MAX_DEPTH = 100;

struct GameState{
    int moves[10];
    int num_moves;
    bool is_terminal;
    float reward;

    __device__ GameState next_state(int action){
        GameState new_state = *this;
        new_state.reward += (action % 2 == 0) ? 1.0f : -1.0f;
        new_state.is_terminal = (new_state.reward > 10 || new_state.reward < -10);
        return new_state;
    }

    __device__ int get_random_action(curandState* state){
        if (num_moves == 0) return -1;
        return moves[curand(state) % num_moves];
    }

};

struct Node{
    GameState state;
    int visits;
    float value;
};

__device__ float rollout(GameState state, curandState* rand_state){
    int depth = 0;
    while (!state.is_terminal && depth < MAX_DEPTH){
        int action = state.get_random_action(rand_state);
        if (action == -1) break;
        state = state.next_state(action);
        depth++;
    }
    return state.reward;
}

__global__ void mcts_kernel(Node* nodes, int num_nodes, float* result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_nodes) return;
    curandState rand_state;
    curand_init(idx, 0, 0, &rand_state);

    float total_reward = 0;
    for (int i = 0; i < NUM_SIMULATIONS; i++)
        total_reward += rollout(nodes[idx].state, &rand_state);

    result[idx] = total_reward / NUM_SIMULATIONS;
}

void run_mcts_kernel(Node* nodes, const int num_nodes){
    Node* d_node;
    float* d_result;
    float* result = (float*)malloc(num_nodes * sizeof(float));

    cudaMalloc(&d_node, num_nodes * sizeof(Node));
    cudaMalloc(&d_result, num_nodes * sizeof(float));

    cudaMemcpy(d_node, nodes, num_nodes * sizeof(Node), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);

    mcts_kernel<<<block_size, grid_size>>>(d_node, num_nodes, d_result);

    cudaMemcpy(result, d_result, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_nodes; i++)
        nodes[i].value = result[i];
    
    free(result);
    cudaFree(d_node);
    cudaFree(d_result);
}

int main(){
    Node node;
    node.state.num_moves = 10;
    node.state.is_terminal = false;
    node.visits = 0;
    node.value = 0;

    run_mcts_kernel(&node, 1);

    printf("MCTS result %f\n", node.value);
    return 0;

}