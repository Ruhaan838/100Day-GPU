#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <random>

#define BLOCK_SIZE 256
#define NEG_INF -std::numeric_limits<float>::infinity()

// Add CUDA_CHECK macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Sum over columns per row
template<typename T>
__global__ void sum_rows_kernel(const T* __restrict__ in, T* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;
    T s = 0;
    const T* row_ptr = in + idx * cols;
    for (int j = 0; j < cols; ++j) s += row_ptr[j];
    out[idx] = s;
}

// Forward D: per-token dot(dO, O)
__global__ void forward_D_Kernel(const float* __restrict__ dO,
                                 const float* __restrict__ O,
                                 float* D, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float* dO_ptr = dO + idx * d;
    const float* O_ptr  = O  + idx * d;
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) sum += dO_ptr[i] * O_ptr[i];
    D[idx] = sum;
}

// Forward attention + softmax in one fused 2D grid kernel
dim3 get_block2d() { return dim3(16, 16); }

__global__ void forward_att_softmax_Kernel(
    const float* __restrict__ Qi,
    const float* __restrict__ Kj,
    const float* __restrict__ norm_li,
    float* att, float* soft, float* maxv, float* sumv,
    int Qr, int Kc, int d, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= Qr || col >= Kc) return;

    // compute scaled score
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        acc += Qi[row*d + k] * Kj[col*d + k];
    }
    float val = acc * scale;
    att[row*Kc + col] = val;

    __syncthreads();
    
    // row max (first thread in each row)
    if (threadIdx.x == 0) {
        float m = NEG_INF;
        for (int j = 0; j < Kc; ++j) {
            float v = att[row*Kc + j];
            if (v > m) m = v;
        }
        maxv[row] = m;
    }
    __syncthreads();

    // exp and store (fixed: use maxv[row] instead of norm_li and maxv)
    float e = expf(val - maxv[row]);
    soft[row*Kc + col] = e;

    __syncthreads();
    
    // row sum (first thread)
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int j = 0; j < Kc; ++j) s += soft[row*Kc + j];
        sumv[row] = s;
    }
    __syncthreads();

    // normalize
    if (sumv[row] > 0.0f) {
        soft[row*Kc + col] /= sumv[row];
    }
}

// Backward dV: dV_j = Soft^T * dO_i
__global__ void backward_dv_Kernel(const float* __restrict__ soft,
                                   const float* __restrict__ dOi,
                                   float* __restrict__ dVj,
                                   int Qr, int Kc, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= Kc || col >= d) return;

    float sum = 0.0f;
    for (int i = 0; i < Qr; ++i) {
        sum += soft[i*Kc + row] * dOi[i*d + col];
    }
    dVj[row*d + col] = sum;
}

// Compute dS matrix for backward pass
__global__ void compute_dS_Kernel(const float* __restrict__ soft,
                                  const float* __restrict__ dOi,
                                  const float* __restrict__ Vj,
                                  const float* __restrict__ Di,
                                  float* __restrict__ dSi,
                                  int Qr, int Kc, int d, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= Qr || col >= Kc) return;

    // compute dO_i * V_j
    float dO_V = 0.0f;
    for (int k = 0; k < d; ++k) {
        dO_V += dOi[row*d + k] * Vj[col*d + k];
    }
    
    // dS = P * (dO*V - D)
    float dS_val = soft[row*Kc + col] * (dO_V - Di[row]);
    dSi[row*Kc + col] = dS_val * scale;
}

// Backward dQ: dQ_i = dS * K_j
__global__ void backward_dq_Kernel(const float* __restrict__ dSi,
                                   const float* __restrict__ Kj,
                                   float* __restrict__ dQi,
                                   int Qr, int Kc, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Qr || col >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < Kc; ++k) {
        sum += dSi[row*Kc + k] * Kj[k*d + col];
    }
    dQi[row*d + col] = sum;
}

// Backward dK: dK_j = dS^T * Q_i
__global__ void backward_dk_Kernel(const float* __restrict__ dSi,
                                   const float* __restrict__ Qi,
                                   float* __restrict__ dKj,
                                   int Qr, int Kc, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Kc || col >= d) return;

    float sum = 0.0f;
    for (int i = 0; i < Qr; ++i) {
        sum += dSi[i*Kc + row] * Qi[i*d + col];
    }
    dKj[row*d + col] = sum;
}

// Atomic accumulation into global gradients
template<typename T>
__global__ void atomic_accum(T* __restrict__ out,
                             const T* __restrict__ in,
                             int M, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    atomicAdd(&out[offset + idx], in[idx]);
}

void flashAttention2BackwardPass(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    float* dQ, float* dK, float* dV,
    int N, int d, int Bc, int Br, const float* Lhost) {
    float scale = 1.0f / sqrtf((float)d);

    // allocate D
    float *D;
    CUDA_CHECK(cudaMalloc(&D, N*sizeof(float)));
    int gN = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
    forward_D_Kernel<<<gN, BLOCK_SIZE>>>(dO, O, D, N, d);
    CUDA_CHECK(cudaGetLastError());

    // zero grads
    CUDA_CHECK(cudaMemset(dQ, 0, N*d*sizeof(float)));
    CUDA_CHECK(cudaMemset(dK, 0, N*d*sizeof(float)));
    CUDA_CHECK(cudaMemset(dV, 0, N*d*sizeof(float)));

    // scratch buffers
    float *att, *soft, *maxv, *sumv, *dS_temp, *temp_grad;
    CUDA_CHECK(cudaMalloc(&att, Br*Bc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&soft, Br*Bc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&maxv, Br*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sumv, Br*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS_temp, Br*Bc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp_grad, max(Br*d, Bc*d)*sizeof(float)));

    dim3 b2 = get_block2d();

    for (int j = 0; j < N; j += Bc) {
        int Kc_cur = min(Bc, N - j);
        const float* Kj = K + j*d;
        const float* Vj = V + j*d;
        
        for (int i = 0; i < N; i += Br) {
            int Qr_cur = min(Br, N - i);
            const float* Qi   = Q + i*d;
            const float* dOi  = dO + i*d;
            const float* Li   = Lhost + i;
            const float* Di   = D + i;

            dim3 g2((Kc_cur + b2.x-1)/b2.x,
                    (Qr_cur + b2.y-1)/b2.y);
            
            // forward attention+softmax
            forward_att_softmax_Kernel<<<g2, b2>>>(
                Qi, Kj, Li, att, soft, maxv, sumv,
                Qr_cur, Kc_cur, d, scale);
            CUDA_CHECK(cudaGetLastError());

            // compute dS matrix
            compute_dS_Kernel<<<g2, b2>>>(
                soft, dOi, Vj, Di, dS_temp,
                Qr_cur, Kc_cur, d, scale);
            CUDA_CHECK(cudaGetLastError());

            // backward dV block
            dim3 g_dv((d + b2.x-1)/b2.x,
                      (Kc_cur + b2.y-1)/b2.y);
            backward_dv_Kernel<<<g_dv, b2>>>(
                soft, dOi, temp_grad, Qr_cur, Kc_cur, d);
            CUDA_CHECK(cudaGetLastError());
            // accumulate into global dV
            int Mv = Kc_cur * d;
            atomic_accum<<<(Mv+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                dV, temp_grad, Mv, j*d);
            CUDA_CHECK(cudaGetLastError());

            // backward dQ block
            dim3 g_dq((d + b2.x-1)/b2.x,
                      (Qr_cur + b2.y-1)/b2.y);
            backward_dq_Kernel<<<g_dq, b2>>>(
                dS_temp, Kj, temp_grad,
                Qr_cur, Kc_cur, d);
            CUDA_CHECK(cudaGetLastError());
            int Mq = Qr_cur * d;
            atomic_accum<<<(Mq+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                dQ, temp_grad, Mq, i*d);
            CUDA_CHECK(cudaGetLastError());

            // backward dK block
            dim3 g_dk((d + b2.x-1)/b2.x,
                      (Kc_cur + b2.y-1)/b2.y);
            backward_dk_Kernel<<<g_dk, b2>>>(
                dS_temp, Qi, temp_grad,
                Qr_cur, Kc_cur, d);
            CUDA_CHECK(cudaGetLastError());
            int Mk = Kc_cur * d;
            atomic_accum<<<(Mk+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                dK, temp_grad, Mk, j*d);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaFree(att));
    CUDA_CHECK(cudaFree(soft));
    CUDA_CHECK(cudaFree(maxv));
    CUDA_CHECK(cudaFree(sumv));
    CUDA_CHECK(cudaFree(dS_temp));
    CUDA_CHECK(cudaFree(temp_grad));
    CUDA_CHECK(cudaFree(D));
}

// Test function
void test_flash_attention() {
    int N = 512;    
    int d = 64;     
    int Bc = 128;   
    int Br = 128;   
    
    float *h_Q, *h_K, *h_V, *h_O, *h_dO, *h_L;
    float *h_dQ, *h_dK, *h_dV;
    
    h_Q = (float*)malloc(N*d*sizeof(float));
    h_K = (float*)malloc(N*d*sizeof(float));
    h_V = (float*)malloc(N*d*sizeof(float));
    h_O = (float*)malloc(N*d*sizeof(float));
    h_dO = (float*)malloc(N*d*sizeof(float));
    h_L = (float*)malloc(N*sizeof(float));
    h_dQ = (float*)malloc(N*d*sizeof(float));
    h_dK = (float*)malloc(N*d*sizeof(float));
    h_dV = (float*)malloc(N*d*sizeof(float));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int i = 0; i < N*d; ++i) {
        h_Q[i] = dis(gen);
        h_K[i] = dis(gen);
        h_V[i] = dis(gen);
        h_O[i] = dis(gen);
        h_dO[i] = dis(gen);
    }
    for (int i = 0; i < N; ++i) {
        h_L[i] = dis(gen) + 2.0f; 
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_dO, *d_L;
    float *d_dQ, *d_dK, *d_dV;
    
    CUDA_CHECK(cudaMalloc(&d_Q, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dO, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV, N*d*sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O, h_O, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dO, h_dO, N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L, h_L, N*sizeof(float), cudaMemcpyHostToDevice));
    
    flashAttention2BackwardPass(
        d_Q, d_K, d_V, d_O, d_dO,
        d_dQ, d_dK, d_dV,
        N, d, Bc, Br, d_L);
    
    CUDA_CHECK(cudaMemcpy(h_dQ, d_dQ, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dK, d_dK, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dV, d_dV, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Sample gradients:\n");
    printf("dQ[0:5] = ");
    for (int i = 0; i < 5; ++i) printf("%.4f ", h_dQ[i]);
    printf("\n");
    printf("dK[0:5] = ");
    for (int i = 0; i < 5; ++i) printf("%.4f ", h_dK[i]);
    printf("\n");
    printf("dV[0:5] = ");
    for (int i = 0; i < 5; ++i) printf("%.4f ", h_dV[i]);
    printf("\n");
    
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_dO); free(h_L);
    free(h_dQ); free(h_dK); free(h_dV);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O)); CUDA_CHECK(cudaFree(d_dO)); CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_dQ)); CUDA_CHECK(cudaFree(d_dK)); CUDA_CHECK(cudaFree(d_dV));
    
}

int main() {
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("\n");
    
    test_flash_attention();
    
    return 0;
}