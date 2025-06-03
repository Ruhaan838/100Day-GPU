#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <curand.h>
#include <fstream>

/*There are some bugs in the code that I fixed in Day-11*/
using namespace std;

__global__ void init_neg_inf(float* arr, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        arr[idx] = -INFINITY;
    }
}

__global__ void FlashAttentionForward(
    const float* query_ptr,
    const float* key_ptr,
    const float* value_ptr,
    float *output,

    const int seq_len,
    const int embd_dim,

    const int col_block_size,
    const int row_block_size,

    const int total_row_size,
    const int total_col_size,

    const float attention_scale,

    float* sum_mat,
    float* max_mat
) {
    int thread_x = threadIdx.x;
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;


    // the gridDim.y is num_heads 
    // we perfrom the attention for the (b, s, n_heads, d_k) 
    // where: d_k is embd_dim // n_heads
    int qkv_offset = (block_idx_x * gridDim.y * seq_len * embd_dim) + (block_idx_y * seq_len * embd_dim);
    //offset for (sum/max) mat
    int lm_offset = (block_idx_x * gridDim.y * seq_len) + (block_idx_y * seq_len); 

    //define the dynamic shared RAM for the QKV and attention_score
    extern __shared__ float shared_memory[];
    int tile_size = col_block_size * embd_dim;
    float* query_shared = shared_memory;
    float* key_shared = &shared_memory[tile_size];
    float* value_shared = &shared_memory[tile_size * 2];
    float* attention_shared = &shared_memory[tile_size * 3];

    float eps = 1e-10;

    //load the Key Value to shared memeory
    for (int col = 0; col < total_col_size; col++){
        // loading the key and value
        for (int embd = 0; embd < embd_dim; embd++){
            int temp_shared_offset = (thread_x * embd_dim) + embd;
            int temp_mat_offset = qkv_offset + ((col * col_block_size + thread_x) * embd_dim) + embd;
            key_shared[temp_shared_offset]   = key_ptr[temp_mat_offset];
            value_shared[temp_shared_offset] = value_ptr[temp_mat_offset];
        }

        __syncthreads();

        for (int row = 0; row < total_row_size; row++){
            // loading the query data 
            for (int embd = 0; embd < embd_dim; embd++){
                int query_offset = qkv_offset + ((row * row_block_size + thread_x) * embd_dim) + embd;
                query_shared[(thread_x * embd_dim) + embd] = query_ptr[query_offset];
            }
            
            __syncthreads();

            // this reduce offset access the data from the max and sum mat 
            int reduce_offset = lm_offset + (row_block_size * row) + thread_x;
            float row_max = max_mat[reduce_offset];
            float row_sum = sum_mat[reduce_offset];

            float row_max_new = -1e20; // init the max row with big number
            for (int col_inner = 0; col_inner < col_block_size; col_inner++){
                float sum = 0; // use the cached sum method
                // perfroming the Q @ K.T
                for (int embd = 0; embd < embd_dim; embd++)
                    sum += query_shared[(thread_x * embd_dim) + embd] * key_shared[(col_inner * embd_dim) + embd];
                // pefroming the Q @ K.T / sqrt(d_k)
                sum *= attention_scale;
                // update the attention_score
                attention_shared[(col_block_size * thread_x) + col_inner] = sum;

                if (sum > row_max_new)
                    row_max_new = sum;

            }

            float row_sum_new = 0; // row_sum_new = sum(exp(attention_score - row_max_new))
            for (int col_inner = 0; col_inner < col_block_size; col_inner++){
                int temp_col_offset = (col_block_size * thread_x) + col_inner;
                // exp(attention_score - row_max_new)
                attention_shared[temp_col_offset] = __expf(attention_shared[temp_col_offset] - row_max_new);
                // sum the exp of the attention_score
                row_sum_new += attention_shared[temp_col_offset];
            }
            
            float row_max_f = max(row_max, row_max_new);
            float row_sum_f = (__expf(row_max - row_max_f) * row_sum) + (__expf(row_max_new - row_max_f) * row_sum_new);

            //write the output and sum_mat, and max_mat
            for (int embd = 0; embd < embd_dim; embd++){
                float weight = 0.0f;
                for (int col_inner = 0; col_inner < col_block_size; col_inner++) {
                    int idx_att = (col_block_size * thread_x) + col_inner;
                    int idx_val = (col_inner * embd_dim) + embd;
                    weight += attention_shared[idx_att] * value_shared[idx_val];
                }

                int output_offset = qkv_offset + (tile_size * row) + (thread_x * embd_dim) + embd;
                float old_out = output[output_offset];

                // Numerically‐stable merge of “old” + “new” contributions:
                float coef_old = row_sum * __expf(row_max - row_max_f);      
                float coef_new = __expf(row_max_new - row_max_f);            

                float numer_old = coef_old * old_out;                         
                float numer_new = coef_new * weight;                          

                // output_ij = (1 / new_sum) * (old_sum * e ^ (old_max - new_max) + e ^ attention_score - new_max) * weight
                // this output formula is actually numerically stable that's why it's takes more formula
                output[output_offset] = (numer_old + numer_new) / (eps + row_sum_f);
            }

            int new_reduce_offset = lm_offset + (row_block_size * row) + thread_x;
            max_mat[new_reduce_offset] = row_max_f;
            sum_mat[new_reduce_offset] = row_sum_f;
        }
        __syncthreads();

    }
}

template <typename T>
T* allocate_init_cuda_memory(size_t size, bool init_with_zero = false){
    T* data_ptr;
    cudaMalloc(&data_ptr, size);

    if (init_with_zero)
        cudaMemset(data_ptr, 0, size);
    else {
        curandGenerator_t generator;
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT); 
        curandSetGeneratorOffset(generator, time(0)); 
        curandGenerateUniform(generator, reinterpret_cast<float*>(data_ptr), size / sizeof(T));
        curandDestroyGenerator(generator);
    }

    return data_ptr;
}

template <typename T>
void write_mat_to_file(T* mat, const string& filename, int batch_size, int num_heads, int seq_len, int embd_dim){
    ofstream file(filename);
    if (!file){
        cerr << "Could not open the file!" << endl;
        return;
    }

    for (int b = 0; b < batch_size; ++b){
        for(int h = 0; h < num_heads; ++h){
            for(int s = 0; s < seq_len; ++s){
                for (int j = 0; j < embd_dim; ++j){
                    file << mat[(b * num_heads * seq_len * embd_dim) +
                         (h * seq_len * embd_dim) + 
                         (s * embd_dim) + j];
                    if (j < embd_dim - 1)
                        file << ", ";
                }
                file << endl;
            }
            file << endl;
        }
    }
    file.close();
}

template <typename T>
void print_matrix(T* mat, int batch_size, int num_heads, int seq_len, int embd_dim){
    int size = batch_size * num_heads * seq_len * embd_dim;
    T* host_mat = new T[size];
    cudaMemcpy(host_mat, mat, size * sizeof(T), cudaMemcpyDeviceToHost);

    cout << "Matrix:\n";
    for (int b = 0; b < batch_size; ++b){
        for(int h = 0; h < num_heads; ++h){
            for(int s = 0; s < seq_len; ++s){
                for (int j = 0; j < embd_dim; ++j){
                    cout << host_mat[(b * num_heads * seq_len * embd_dim) +
                         (h * seq_len * embd_dim) + 
                         (s * embd_dim) + j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    delete[] host_mat;
}

int main(){
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 64;
    const int embd_dim = 64;

    const int col_block_size = 32;
    const int row_block_size = 32;

    const int total_col_size = ceil((float)seq_len / col_block_size);
    const int total_row_size = ceil((float)seq_len / row_block_size);
    const float attention_scale = 1.0f / sqrt(embd_dim);

    size_t mat_size = batch_size * num_heads * seq_len * embd_dim * sizeof(float);
    size_t vector_size = batch_size * num_heads * seq_len * sizeof(float);

    float* query_mat = allocate_init_cuda_memory<float>(mat_size);
    float* key_mat = allocate_init_cuda_memory<float>(mat_size);
    float* value_mat = allocate_init_cuda_memory<float>(mat_size);

    float* output = allocate_init_cuda_memory<float>(mat_size, true);

    float* sum_mat = allocate_init_cuda_memory<float>(vector_size, true);
    float* max_mat = allocate_init_cuda_memory<float>(vector_size);

    int num_reduce_elems = batch_size * num_heads * seq_len;
    int threads_per_block = 256;
    int blocks_for_init = (num_reduce_elems + threads_per_block - 1) / threads_per_block;
    init_neg_inf<<<blocks_for_init, threads_per_block>>>(max_mat, num_reduce_elems);
    cudaDeviceSynchronize(); 

    const int shared_mem_size = 4 * col_block_size * embd_dim * sizeof(float);
    int max_shared_mem;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    
    dim3 grid_dim(batch_size, num_heads);
    dim3 block_dim(col_block_size);

    FlashAttentionForward<<<grid_dim, block_dim, shared_mem_size>>>(
        query_mat, key_mat, value_mat, output, seq_len, embd_dim, col_block_size, 
        row_block_size, total_row_size, total_col_size, attention_scale, 
        sum_mat, max_mat
    );

    cudaDeviceSynchronize();
    int size = batch_size * num_heads * seq_len * embd_dim;
    float* query_host = new float[size];
    float* key_host = new float[size];
    float* value_host = new float[size];
    float* output_host = new float[size];

    cudaMemcpy(query_host, query_mat, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(key_host, key_mat, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(value_host, value_mat, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_host, output, mat_size, cudaMemcpyDeviceToHost);

    write_mat_to_file(query_host, "query_out.csv", batch_size, num_heads, seq_len, embd_dim);
    write_mat_to_file(key_host, "key_out.csv", batch_size, num_heads, seq_len, embd_dim);
    write_mat_to_file(value_host, "value_out.csv", batch_size, num_heads, seq_len, embd_dim);
    write_mat_to_file(output_host, "output.csv", batch_size, num_heads, seq_len, embd_dim);

    cout << "Query ";
    print_matrix(query_mat, batch_size, num_heads, seq_len, embd_dim);

    cout << "Key: ";
    print_matrix(key_mat, batch_size, num_heads, seq_len, embd_dim);

    cout << "Value: ";
    print_matrix(value_mat, batch_size, num_heads, seq_len, embd_dim);

    cout << "Output: ";
    print_matrix(output, batch_size, num_heads, seq_len, embd_dim);

    cudaFree(query_mat);
    cudaFree(key_mat);
    cudaFree(value_mat);
    cudaFree(output);
    cudaFree(sum_mat);
    cudaFree(max_mat);

    return 0;

}
