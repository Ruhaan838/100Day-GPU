#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

using namespace std;

#define SHARED_RAM_SIZE 1024
#define seq_len 2
#define embed_dim 2

constexpr int col_block_size = SHARED_RAM_SIZE / (4 * embed_dim);
constexpr int row_block_size = SHARED_RAM_SIZE / (4 * embed_dim);
constexpr int Total_row_tiled = (seq_len + col_block_size - 1) / col_block_size;
constexpr int Total_col_tiled = (seq_len + row_block_size - 1) / row_block_size;

__global__ void FlashAttentionForward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float *max_value,
    float* sum_value,
    const float attention_scale) {

    int thread_idx = threadIdx.x;

    float attention_score[row_block_size * col_block_size];
    float attention_weights[row_block_size * col_block_size];

    float query_shared[row_block_size * embed_dim];
    float key_shared[col_block_size * embed_dim];
    float value_shared[col_block_size * embed_dim];

    for (int col = 0; col < Total_col_tiled; ++col) {
        if (thread_idx < col_block_size) {
            for (int d = 0; d < embed_dim; ++d) {
                size_t t_embd_idx = thread_idx * embed_dim + d;
                size_t col_idx = col * col_block_size * embed_dim + t_embd_idx;
                key_shared[t_embd_idx] = key[col_idx];
                value_shared[t_embd_idx] = value[col_idx];
            }
        }
    }

    __syncthreads();

    for (int row = 0; row < Total_row_tiled; ++row) {
        if (thread_idx < row_block_size) {
            for (int d = 0; d < embed_dim; ++d) {
                size_t t_embd_idx = thread_idx * embed_dim + d;
                size_t row_idx = row * row_block_size * embed_dim + t_embd_idx;
                query_shared[t_embd_idx] = query[row_idx];
            }
        }

        __syncthreads();

        if (thread_idx < row_block_size) {
            float row_max = -1e20f;
            for (int k = 0; k < col_block_size; ++k) {
                float score = 0.0f;
                for (int d = 0; d < embed_dim; ++d)
                    score += query_shared[thread_idx * embed_dim + d] * key_shared[k * embed_dim + d];
                score *= attention_scale;
                attention_score[thread_idx * col_block_size + k] = score;
                row_max = fmaxf(row_max, score);
            }

            float softmax_div_sum = 0.0f;
            for (int k = 0; k < col_block_size; ++k) {
                float weight = expf(attention_score[thread_idx * col_block_size + k] - row_max);
                attention_weights[thread_idx * col_block_size + k] = weight;
                softmax_div_sum += weight;
            }

            for (int d = 0; d < embed_dim; ++d) {
                float cache_weight_sum = 0.0f;
                for (int k = 0; k < col_block_size; ++k)
                    cache_weight_sum += attention_weights[thread_idx * col_block_size + k] * value_shared[k * embed_dim + d];

                size_t output_idx = row * row_block_size * embed_dim + thread_idx * embed_dim + d;
                output[output_idx] = (softmax_div_sum > 0) ? (cache_weight_sum / softmax_div_sum) : 0;
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

void HipError(const char* msg) {
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        std::cerr << msg << " - HIP Error: " << hipGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float (*query)[embed_dim] = new float[seq_len][embed_dim];
    float (*key)[embed_dim] = new float[seq_len][embed_dim];
    float (*value)[embed_dim] = new float[seq_len][embed_dim];
    float (*output)[embed_dim] = new float[seq_len][embed_dim];

    float *sum_value = new float[seq_len]();
    float *max_value = new float[seq_len];

    for (int i = 0; i < seq_len; i++)
        max_value[i] = -1e20f;

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
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

    hipMalloc(&d_query, size); HipError("hipMalloc d_query");
    hipMalloc(&d_key, size); HipError("hipMalloc d_key");
    hipMalloc(&d_value, size); HipError("hipMalloc d_value");
    hipMalloc(&d_output, size); HipError("hipMalloc d_output");
    hipMalloc(&d_max_value, size_seq); HipError("hipMalloc d_max_value");
    hipMalloc(&d_sum_value, size_seq); HipError("hipMalloc d_sum_value");

    hipMemcpy(d_query, query, size, hipMemcpyHostToDevice); HipError("hipMemcpy d_query");
    hipMemcpy(d_key, key, size, hipMemcpyHostToDevice); HipError("hipMemcpy d_key");
    hipMemcpy(d_value, value, size, hipMemcpyHostToDevice); HipError("hipMemcpy d_value");
    hipMemcpy(d_output, output, size, hipMemcpyHostToDevice); HipError("hipMemcpy d_output");
    hipMemcpy(d_max_value, max_value, size_seq, hipMemcpyHostToDevice); HipError("hipMemcpy d_max_value");
    hipMemcpy(d_sum_value, sum_value, size_seq, hipMemcpyHostToDevice); HipError("hipMemcpy d_sum_value");

    float attention_scale = 1.0f / sqrt(embed_dim);

    dim3 block_dim(row_block_size);
    dim3 block_grid(1);

    hipLaunchKernelGGL(FlashAttentionForward, block_grid, block_dim, 0, 0,
                       d_query, d_key, d_value, d_output,
                       d_max_value, d_sum_value, attention_scale);
    HipError("Kernel launch");

    hipMemcpy(output, d_output, size, hipMemcpyDeviceToHost); HipError("hipMemcpy output");
    hipMemcpy(max_value, d_max_value, size_seq, hipMemcpyDeviceToHost); HipError("hipMemcpy max_value");
    hipMemcpy(sum_value, d_sum_value, size_seq, hipMemcpyDeviceToHost); HipError("hipMemcpy sum_value");

    cout << "Query:\n";
    print_data(query);

    cout << "Key:\n";
    print_data(key);

    cout << "Value:\n";
    print_data(value);

    cout << "Output:\n";
    print_data(output);

    hipFree(d_query);
    hipFree(d_key);
    hipFree(d_value);
    hipFree(d_output);
    hipFree(d_max_value);
    hipFree(d_sum_value);

    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;
    delete[] sum_value;
    delete[] max_value;

    return 0;
}
