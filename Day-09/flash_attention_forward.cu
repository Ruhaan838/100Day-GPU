#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

using namespace std;

#define SHARED_RAM_SIZE 1024 //this constant is for ensure the we have enough RAM for the QKV.
#define seq_len 2
#define embed_dim 2

constexpr int col_block_size = SHARED_RAM_SIZE / (4 * embed_dim); // this for column that we need to define for the our key and value.
constexpr int row_block_size = SHARED_RAM_SIZE / (4 * embed_dim); // this for row that we need to define for the our query.

// this two variable ensure that we can't fit the all q,k,v all in one 
// by using this tiled we can loop over and ensure which we can and which we can't
// Because we can’t fit all queries and keys into shared memory at once, we split them into tiles of
constexpr int Total_row_tiled = (seq_len + col_block_size - 1) / col_block_size; // ceil(sequence_length / Block_row_size)
constexpr int Total_col_tiled = (seq_len + row_block_size - 1) / row_block_size;

__global__ void FlashAttentionForward(
    const float* query, // q (seq_len, embd_dim)
    const float* key, // k (seq_len, embd_dim)
    const float* value, // v (seq_len, embd_dim)
    float* output, // o (seq_len, embd_dim)
    float *max_value, // max vals
    float* sum_value, // sumed vals
    const float attention_scale){  // 1 / sqrt(dk) 


        int thread_idx = threadIdx.x;

        float attention_score[row_block_size * col_block_size];
        float attention_weigths[row_block_size * col_block_size];

        float query_shared[row_block_size * embed_dim];
        float key_shared[col_block_size * embed_dim];
        float value_shared[col_block_size * embed_dim];

        //loding the key and value data 
        // because we have to load them by colwise
        for (int col = 0; col < Total_col_tiled; ++col){
            if (thread_idx < col_block_size){ // this ensure the thread overloading
                for(int d = 0; d < embed_dim; ++d){
                    size_t t_embd_idx = thread_idx * embed_dim + d; // access the embd_data from global memory to each thread.
                    size_t col_idx = col * col_block_size * embed_dim + t_embd_idx; // help to put the data to shared memory.
                    key_shared[t_embd_idx] = key[col_idx];
                    value_shared[t_embd_idx] = value[col_idx];
                }
            }
        }

        __syncthreads(); //wait until the all data loaded.

        for(int row = 0; row < Total_row_tiled; ++row){
            if (thread_idx < row_block_size){
                //first load query to shared mem.
                for(int d = 0; d < embed_dim; ++d){
                    size_t t_embd_idx = thread_idx * embed_dim + d;
                    size_t row_idx = row * row_block_size * embed_dim + t_embd_idx;
                    query_shared[t_embd_idx] = query[row_idx];
                }
            }

            __syncthreads(); //wait until the all data loaded.

            if(thread_idx < row_block_size){
                float row_max = -1e20; //help to compute the softmax for numralical stabilty.
                for (int k = 0; k < col_block_size; ++k){
                    float score = 0.0f;
                    for (int d = 0; d < embed_dim; ++d)
                        score += query_shared[thread_idx * embed_dim + d] * key_shared[k * embed_dim + d]; // Q @ K.T
                    score *= attention_scale; // Q @ K.T / sqrt(d_k)
                    attention_score[thread_idx * col_block_size + k] = score; //update the attention_score.
                    row_max = fmaxf(row_max, score);
                }

                //compute the attention softmax
                /*
                    softmax(x) = e^x - max(x) / sum(e^x - max(x));
                */
                float softmax_div_sum = 0.0f;
                for(int k = 0; k < col_block_size; ++k){
                    float weight = expf(attention_score[thread_idx * col_block_size + k] - row_max);
                    attention_weigths[thread_idx * col_block_size + k] = weight;
                    softmax_div_sum += weight;
                }


                // this block of the code is caclulate 
                // softmax(Q @ K.T) @ V 
                for (int d = 0; d < embed_dim; ++d){
                    float cache_weight_sum = 0.0f;
                    for (int k = 0; k < col_block_size; ++k)
                        cache_weight_sum += attention_weigths[thread_idx * col_block_size + k] * value_shared[k * embed_dim + d];

                    size_t output_idx = row * row_block_size * embed_dim + thread_idx * embed_dim + d;
                    output[output_idx] = (softmax_div_sum > 0) ? (cache_weight_sum / softmax_div_sum): 0; //avoid zero div.
                }
            }
            __syncthreads();
        }
    }

void print_data(float data[seq_len][embed_dim]) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++)
            printf("%f ", data[i][j]);
        printf("\n");
    }
    printf("\n");
}


void CudaError(const char* msg){
    cudaError_t error = cudaGetLastError(); 
    if (error != cudaSuccess){
        std::cerr << msg << "- Getting CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    float (*query)[embed_dim] = new float[seq_len][embed_dim];
    float (*key)[embed_dim] = new float[seq_len][embed_dim];
    float (*value)[embed_dim] = new float[seq_len][embed_dim];
    
    float (*output)[embed_dim] = new float[seq_len][embed_dim];

    float *sum_value = new float[seq_len]();
    float *max_value = new float[seq_len];

    for (int i = 0; i < seq_len; i++)
        max_value[i] = -1e20;
    
    for(int i = 0; i < seq_len; i++){
        for (int j = 0; j < embed_dim; j++){
            query[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            key[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            value[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;

            output[i][j] = 0.0f;
        }
    }

    float *d_query, *d_key, *d_value, *d_output;
    float *d_max_value, *d_sum_value;

    size_t size_seq = seq_len * sizeof(float);
    size_t size = size_seq * embed_dim;

    cudaMalloc(&d_query, size); CudaError("Error to allocate the d_query");
    cudaMalloc(&d_key, size); CudaError("Error to allocate the d_key");
    cudaMalloc(&d_value, size); CudaError("Error to allocate the d_value");

    cudaMalloc(&d_output, size); CudaError("Error to allocate the d_output");

    cudaMalloc(&d_max_value, size_seq); CudaError("Error to allocate the d_max_value");
    cudaMalloc(&d_sum_value, size_seq); CudaError("Error to allocate the d_sum_value");

    cudaMemcpy(d_query, query, size, cudaMemcpyHostToDevice); CudaError("Error to copy d_query");
    cudaMemcpy(d_key, key, size, cudaMemcpyHostToDevice); CudaError("Error to copy d_key");
    cudaMemcpy(d_value, value, size, cudaMemcpyHostToDevice); CudaError("Error to copy d_value");
    cudaMemcpy(d_output, output, size, cudaMemcpyHostToDevice); CudaError("Error to copy d_output");


    cudaMemcpy(d_max_value, max_value, size_seq, cudaMemcpyHostToDevice); CudaError("Error to copy d_max_value");
    cudaMemcpy(d_sum_value, sum_value, size_seq, cudaMemcpyHostToDevice); CudaError("Error to copy d_sum_value");

    float attention_scale = 1.0f / sqrt(embed_dim);

    dim3 block_dim(row_block_size); //one thread per row in query block
    dim3 block_grid(1); //one block for simplicity.

    FlashAttentionForward<<<block_grid, block_dim>>>(d_query, d_key, d_value, d_output, d_max_value, d_sum_value, attention_scale);
    CudaError("Failed to load the kernel");

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost); CudaError("Error to copy output data");

    cudaMemcpy(max_value, d_max_value, size_seq, cudaMemcpyDeviceToHost); CudaError("Error to copy max_value");
    cudaMemcpy(sum_value, d_sum_value, size_seq, cudaMemcpyDeviceToHost); CudaError("Error to copy sum_value");


    cout << "Query:" << '\n';
    print_data(query);

    cout << "Key:" << '\n';
    print_data(key);

    cout << "Value:" << '\n';
    print_data(value);

    cout << "Output:" << '\n';
    print_data(output);

    //clean-up
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_max_value);
    cudaFree(d_sum_value);

    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;
    delete[] sum_value;
    delete[] max_value;

    return 0;
}