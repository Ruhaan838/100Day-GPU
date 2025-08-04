#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// #define DO_BACKWARD


extern "C" __global__ void qwen2vl_mrope_kernel(
    float* q,
    float* k,
    const float* cos,
    const float* sin,
    int sl, // seq_len
    int b, //batch_size
    int n_qh, //num_of_Q_heads
    int n_kh, //num_of_K_heads
    int hd, //head dim
    int pad_n_qh, //pad Qheads
    int pad_n_kh, //pad Kheads
    int pad_hd, //pad heads
    int mrope_section_t, //mrope section "t"
    int mrope_section_h, //mrope section "h"
    bool backward_pass // want a do backward pass or not?
) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_row = b * sl;
    if (token_id >= n_row) return;

    float* q_token = q + token_id * n_qh * hd;
    float* k_token = k + token_id * n_qh * hd;

    const int token_offset = token_id * hd;
    const float* t_cos = cos + token_offset;
    const float* h_cos = cos + b * sl * hd + token_offset;
    const float* w_cos = cos + 2 * b * sl * hd + token_offset;
    const float* t_sin = sin + token_offset;
    const float* h_sin = sin + b * sl * hd + token_offset;
    const float* w_sin = sin + 2 * b * sl * hd + token_offset;

    int half_hd = hd / 2;
    int h_end = mrope_section_t + mrope_section_h;

    for(int head = 0; head < n_qh; head++){
        float* q_head_ptr = q_token + head * hd;
        for(int d = 0; d < half_hd; d++){
            float q1 = q_head_ptr[d];
            float q2 = q_head_ptr[d + half_hd];

            float cos_val = 0.0f, sin_val = 0.0f;
            if (d < mrope_section_t){
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end){
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd){
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_q1, new_q2;
            if (!backward_pass){
                new_q1 = q1 * cos_val - q2 * sin_val;
                new_q2 = q2 * cos_val + q1 * sin_val;
            } else {
                new_q1 = q1 * cos_val + q2 * sin_val;
                new_q2 = q2 * cos_val - q1 * sin_val;
            }
            q_head_ptr[d] = new_q1;
            q_head_ptr[d + half_hd] = new_q2;
        }
    }

    for(int head = 0; head < n_kh; head++){
        float* k_head_ptr = k_token + head * hd;
        for(int d = 0; d < half_hd; d++){
            float k1 = k_head_ptr[d];
            float k2 = k_head_ptr[d + half_hd];

            float cos_val = 0.0f, sin_val = 0.0f;
            if (d < mrope_section_t){
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end){
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd){
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_k1, new_k2;
            if(!backward_pass){
                new_k1 = k1 * cos_val - k2 * sin_val;
                new_k2 = k2 * cos_val + k1 * sin_val;
            } else {
                new_k1 = k1 * cos_val + k2 * sin_val;
                new_k2 = k2 * cos_val - k1 * sin_val;
            }
            k_head_ptr[d] = new_k1;
            k_head_ptr[d + half_hd] = new_k2;
        }
    }
}

void qwan2vl_mrope_forward(
    float* dq,
    float* dk,
    const float* d_cos,
    const float* d_sin,
    int b,
    int sl,
    int n_qh,
    int n_kh,
    int hd,
    int mrope_section_t,
    int mrope_section_h
){
    int pad_n_qh = n_qh;
    int pad_n_kh = n_kh;
    int pad_hd = hd;

    int n_row = b * sl;
    int threads = 256;
    int blocks = (n_row + threads - 1) / threads;
    qwen2vl_mrope_kernel<<<blocks, threads>>>(dq, dk, d_cos, d_sin, sl, b, n_qh, n_kh, hd, 
        pad_n_qh, pad_n_kh, pad_hd,
         mrope_section_t, mrope_section_h, false);
    cudaDeviceSynchronize();

}

void qwen2vl_mrope_backward(
    float* dq,
    float* dk,
    const float* d_cos,
    const float* d_sin,
    int b,
    int sl,
    int n_qh,
    int n_kh,
    int hd,
    int mrope_section_t,
    int mrope_section_h
){
    int pad_n_qh = n_qh;
    int pad_n_kh = n_kh;
    int pad_hd = hd;

    int n_row = b * sl;
    int threads = 256;
    int blocks = (n_row + threads - 1) / threads;
    qwen2vl_mrope_kernel<<<blocks, threads>>>(dq, dk, d_cos, d_sin, sl, b, n_qh, n_kh, hd, 
        pad_n_qh, pad_n_kh, pad_hd,
         mrope_section_t, mrope_section_h, false);
    cudaDeviceSynchronize();
}

void print_data(float* data, int n_row, int n_qh, int hd, char ans){
    printf("Transformed %c values:\n", ans);
    for(int i = 0; i < n_row; i++){
        printf("Token %d:\n", i);
        for(int head = 0; head < n_qh; head++){
            printf("\tQ Head %d: ", head);
            for(int d = 0; d < hd; d++){
                int idx = i * n_qh * hd + head * hd + d;
                printf("%0.3f", data[idx]);
            }
            printf("\n");
        }
    }
}

int main(){
    const int b = 2;
    const int sl = 4;
    const int n_qh = 2;
    const int n_kh = 2;
    const int hd = 8;
    const int mrope_section_t = 3;
    const int mrope_section_h = 2;

    int n_row = b * sl;
    size_t size_q = n_row * n_qh * hd * sizeof(float);
    size_t size_k = n_row * n_kh * hd * sizeof(float);
    size_t size_cos = 3 * n_row * hd * sizeof(float);
    size_t size_sin = size_cos;

    float* q = (float*)malloc(size_q);
    float* k = (float*)malloc(size_k);
    float* cos = (float*)malloc(size_cos);
    float* sin = (float*)malloc(size_sin);

    for(size_t i = 0; i < size_q / sizeof(float); i++) {q[i] = i;}
    for(size_t i = 0; i < size_k / sizeof(float); i++) {k[i] = i;}

    for(size_t i = 0; i < size_cos / sizeof(float); i++) {
        cos[i] = cosf(i * 0.01f);
        sin[i] = sinf(i * 0.01f);
    }

    float *dq, *dk, *d_cos, *d_sin;
    cudaMalloc(&dq, size_q);
    cudaMalloc(&dk, size_k);
    cudaMalloc(&d_cos, size_cos);
    cudaMalloc(&d_sin, size_sin);

    cudaMemcpy(dq, q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, size_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos, size_cos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin, size_sin, cudaMemcpyHostToDevice);

    printf("--- Forward Pass ---\n");
    qwan2vl_mrope_forward(dq, dk, d_cos, d_sin, b, sl, n_qh, n_kh, hd, mrope_section_t, mrope_section_h);

    print_data(q, n_row, n_qh, hd, 'Q');
    print_data(k, n_row, n_kh, hd, 'K');


    #ifdef DO_BACKWARD
        printf("--- Backward Pass ---\n");
        qwan2vl_mrope_forward(dq, dk, d_cos, d_sin, b, sl, n_qh, n_kh, hd, mrope_section_t, mrope_section_h);

        print_data(q, n_row, n_qh, hd, 'Q');
        print_data(k, n_row, n_kh, hd, 'K');
    #endif

}