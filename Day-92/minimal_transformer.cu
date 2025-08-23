#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

void CUDA_CHECK(cudaError_t err){
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CUBLAS_CHECK(cublasStatus_t st){
    if (st != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cuBLAS error %s at %s:%d: %d\n", st, __FILE__, __LINE__, (int)st);
        exit(EXIT_FAILURE);
    }
}

//inline relu kernel
__global__ void relu_kernel(float* x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}

// Layernorm (x - mean) / sqrt(var + eps) * gamma + beta //try to make the kernel much like possible to cublas
__global__ void layer_norm_kernel(const float* x_in, const float* gamma, const float* beta, 
    float* x_out, int rows, int dim, float eps){
    int row = blockIdx.x; if (row >= rows) return;
    extern __shared__ float sh[]; //shared memry accross the blockDim

    float local = 0.f;
    for(int c = threadIdx.x; c < dim; c += blockDim.x){
        local += x_in[row * dim + c];
    }

    sh[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (threadIdx.x < s) 
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sh[0] / (float)dim;

    local = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x){
        float d = x_in[row * dim + c] - mean;
        local += d * d;
    }
    sh[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (threadIdx.x < s)
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sh[0] / (float)dim + eps);

    for (int c = threadIdx.x; c < dim; c += blockDim.x){
        float v = (x_in[row * dim + c] - mean) * inv_std;
        float g = gamma ? gamma[c] : 1.0f;
        float be = beta ? beta[c] : 0.0f;
        x_out[row * dim + c] = v * g + be;
    }
}

// this softmax over the last dim for (rows, dim) and it's inplace version
__global__ void softmax_lastdim_kernel(float* x, int rows, int dim){
    int row = blockIdx.x; if (row >= rows) return;
    extern __shared__ float sh[];

    //row wise max
    float max_val = -1e30f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x){
        float v = x[row * dim + c];
        max_val = fmaxf(max_val, v);
    }
    sh[threadIdx.x] = max_val; __syncthreads(); // load the max_values to shared memoery to accross the threads.
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); //find the max accross the rows.
        __syncthreads();
    }
    max_val = sh[0];
    //exp_sum
    float sum = 0.0f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x){
        float e = expf(x[row * dim + c] - max_val); //caclulate the exp accross the rows
        x[row * dim + c] = e;
        sum += e;
    }
    sh[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; // sum them up
        __syncthreads();
    }
    float denom = sh[0];
    for(int c = threadIdx.x; c < dim; c += blockDim.x){
        x[row * dim + c] /= denom; //load the all data to out global memoery
    }
}

// Bias addition kernel
__global__ void bias_add_kernel(float* y, const float* b, int out, int elems){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elems){
        int c = i % out;
        y[i] += b[c];
    }
}

/*
    Row Major Linear:
    y = x @ w^T + b;
    use the cublasSgemm to perfrom the matmul
*/

static void linear_forward_rowmajor_kernel(cublasHandle_t handle, 
    const float* x, //shape (n, in) 
    const float* W, //shape (out, in)
    const float* b, // shape (out) or none
    float* y, //shape (n, out)
    int N, int in, int out    
){
    const float alpha = 1.0f, beta = 0.0f;
    //cublas cublasSgemm API math:
    // C = alpha * op(A) * op(B) + beta * C 
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T, // this two args say that, transpose the matrix or not ..OP_N means no, ..OP_T means yes
            out, N, in, // m= outsize, n= N(innerdim), k= insize
            &alpha, //alpha is sclae for multiplication in our case it's 1.0
            W, out, //A=w, lda=out // A is (out x int) opA=T -> (in x out)
            x, N,  // B=x, ldb=n // B is (N x in), opB=N -> (N x in)
            &beta, 
            y, out //C=y, ldc=out // C is (out x N) col-major => row-major [N x out]
        )
    );
    if (b){ //bias can be nullptr
        int threads = 256;
        int elems = N * out;
        int blocks = (elems + threads - 1) / threads; //ceil(elems)
        bias_add_kernel<<<blocks, threads>>>(y, b, out, elems);
        CUDA_CHECK(cudaGetLastError());
    }
}

// this one just work as .view in pytorch
// like change the shape from [B, H, S, Dh] -> [B, S, D]
__global__ void merge_heads_kernel(const float* src, float* dst, 
    int B, int H, int S, int Dh, int D){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int eles = B * S * D;
    if (i >= eles) return;

    int d = i % D;
    int t = i / D;

    int s = t % S;
    int b = t / S;

    int h = d / Dh;
    int dh = d % Dh;
    int src_idx = ((b * H + h) * S + s) * Dh + dh; // [b, h, s, dh]
    dst[i] = src[src_idx];
}

//residual kernel
__global__ void residual_add_kernel(const float* a, const float* b, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

struct MHAttention{
    int B, S, D, H, Dh; // where D = H * Dh
    float scale; // 1 / sqrt(Dh)

    void forward(cublasHandle_t handle, float* Q, float* K, float *V, 
        float *att_score, float* ctx){
            const float alpha = scale, beta0 = 0.0f;
            long long stride = (long long)S * Dh;
            int batch_count = B * H;
            //strided version of gemm for batch data
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(
                    handle, 
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    S, S, Dh, // row of mat op Q and att_score, // cols of op K attention, // cols of Q and att_score
                    &alpha,
                    Q, S, stride, // Q, leading dim of mat, offset how many jumps that you can take ex: Q[i] -> Q[i+1]
                    K, S, stride, // for K " 
                    &beta0,
                    att_score, S, (long long)(S * S),
                    batch_count
                )
            );

            int rows = B * H * S, dim = S;
            int threads = 256; size_t sh = threads * sizeof(float);
            softmax_lastdim_kernel<<<rows, threads, sh>>>(att_score, rows, dim);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(
                cublasSgemmStridedBatched(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    S, Dh, S,
                    &alpha, 
                    att_score, S, (long long)S*S,
                    V, S, stride,
                    &beta0, //put value to zero we don't want to add anything
                    ctx, S, stride,
                    batch_count
                )
            );
    }
};

void allocate_attn_weights(float*& Wq, float*& bq, float*& Wk, float*& bk, float*& Wv, float*& bv, size_t WD, size_t D){
    CUDA_CHECK(cudaMalloc(&Wq, WD)); CUDA_CHECK(cudaMalloc(&bq, D));
    CUDA_CHECK(cudaMalloc(&Wk, WD)); CUDA_CHECK(cudaMalloc(&bk, D));
    CUDA_CHECK(cudaMalloc(&Wv, WD)); CUDA_CHECK(cudaMalloc(&bv, D));
}

void allocate_FFN_LN_weights(float*& W1, float*& b1, float*& W2, float*& b2, 
    float*& gamma1, float*& gamma2, float*& beta1, float*& beta2,
    size_t WD1, size_t D1, size_t D2, size_t WD2){
    CUDA_CHECK(cudaMalloc(&W1, WD1)); CUDA_CHECK(cudaMalloc(&b1, D1));
    CUDA_CHECK(cudaMalloc(&W2, WD2)); CUDA_CHECK(cudaMalloc(&b2, D2));
    CUDA_CHECK(cudaMalloc(&gamma1, D2)); CUDA_CHECK(cudaMalloc(&beta1, D2));
    CUDA_CHECK(cudaMalloc(&gamma2, D2)); CUDA_CHECK(cudaMalloc(&beta2, D2));
}

void allocate_transformer_buffers(
    float*& x_norm1, float*& x_norm2,
    float*& Q, float*& K, float*& V, float*& att_score, float*& ctx, float*& mh_out,
    float*& res1, float*& ff1, float*& ff2, float*& res2,
    int B, int S, int D, int H, int Dff
){
    size_t BSD = (size_t)B * S * D * sizeof(float);
    size_t BSSS = (size_t)B * H * S * S * sizeof(float);
    size_t BSDff = (size_t)B * S * Dff * sizeof(float);

    CUDA_CHECK(cudaMalloc(&x_norm1, BSD));
    CUDA_CHECK(cudaMalloc(&x_norm2, BSD));
    CUDA_CHECK(cudaMalloc(&Q, BSD));
    CUDA_CHECK(cudaMalloc(&K, BSD));
    CUDA_CHECK(cudaMalloc(&V, BSD));
    CUDA_CHECK(cudaMalloc(&att_score, BSSS));
    CUDA_CHECK(cudaMalloc(&ctx, BSD));
    CUDA_CHECK(cudaMalloc(&mh_out, BSD));
    CUDA_CHECK(cudaMalloc(&res1, BSD));
    CUDA_CHECK(cudaMalloc(&ff1, BSDff));
    CUDA_CHECK(cudaMalloc(&ff2, BSDff));
    CUDA_CHECK(cudaMalloc(&res2, BSD));
}

struct TransformerBlock{
    int B, S, D, H, Dh, Dff;

    float *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo; // attn weights
    float *W1, *b1, *W2, *b2; //FFN weights
    float *gamma1, *beta1, *gamma2, *beta2; // LN scales/bias

    float *x_norm1, *x_norm2;
    float *Q, *K, *V, *att_score, *ctx, *mh_out, *res1, *ff1, *ff2, *res2;

    cublasHandle_t handle;

    TransformerBlock(int B_, int S_, int D_, int H_, int Dff_):B(B_), S(S_), D(D_), H(H_), Dh(D_/H_), Dff(Dff_){
        assert (D % H == 0);
        CUBLAS_CHECK(cublasCreate(&handle));

        size_t WD = (size_t)D*D, WD1=(size_t)Dff*D, WD2=(size_t)D*Dff;
        allocate_attn_weights(Wq, bq, Wk, bk, Wv, bv, WD * sizeof(float), D * sizeof(float));
        CUDA_CHECK(cudaMalloc(&Wo, WD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&bo, D * sizeof(float)));
        allocate_FFN_LN_weights(W1, b1, W2, b2, gamma1, gamma2, beta1, beta2, WD1 * sizeof(float), Dff*sizeof(float), D*sizeof(float), WD2 *sizeof(float));
        allocate_transformer_buffers(x_norm1, x_norm2, Q, K, V, att_score, ctx, mh_out, res1, ff1, ff2, res2, B, S, D, H, Dff);

        size_t max_size = std::max({WD, WD1, WD2, (size_t)Dff, (size_t)D});
        vector<float> tmp; tmp.resize(max_size);
        mt19937 rng(42); normal_distribution<float> N01(0.f, 0.02f);

        auto fill = [&](float* dptr, size_t n){
            tmp.resize(n);
            for(size_t i = 0; i < n; ++i) tmp[i] = N01(rng);
            CUDA_CHECK(cudaMemcpy(dptr, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        };
        fill(Wq, WD); fill(Wk, WD); fill(Wv, WD); fill(Wo, WD); fill(W1, WD1); fill(W2, WD2);
        auto fill_zero = [&](float* dptr, size_t n){
            CUDA_CHECK(cudaMemset(dptr, 0, n * sizeof(float)));
        };
        fill_zero(bq, D); fill_zero(bk, D); fill_zero(bv, D); fill_zero(bo, D); fill_zero(b1, Dff); fill_zero(b2, D);
        tmp.resize(D); for(int i = 0; i < D; i++) tmp[i] = 1.0f; CUDA_CHECK(cudaMemcpy(gamma1, tmp.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gamma2, tmp.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        std::fill(tmp.begin(), tmp.begin() + D, 0.0f); CUDA_CHECK(cudaMemcpy(beta1, tmp.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(beta2, tmp.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~TransformerBlock(){
        cudaFree(Wq); cudaFree(bq); cudaFree(Wk); cudaFree(bk); cudaFree(Wv); cudaFree(bv); cudaFree(Wo); cudaFree(bo);
        cudaFree(W1); cudaFree(b1); cudaFree(W2); cudaFree(b2);
        cudaFree(gamma1); cudaFree(beta1); cudaFree(gamma2); cudaFree(beta2);
        cudaFree(x_norm1); cudaFree(x_norm2);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(att_score); cudaFree(ctx); cudaFree(mh_out); cudaFree(res1); cudaFree(ff1); cudaFree(ff2); cudaFree(res2);
        cublasDestroy(handle);
    }

    void forward(float* x){
        int N = B * S; int threads = 256; size_t sh = threads * sizeof(float);
        // LN1 (pre-attn-norm)
        layer_norm_kernel<<<N, threads, sh>>>(x, gamma1, beta1, x_norm1, N, D, 1e-5f);
        CUDA_CHECK(cudaGetLastError());

        // Q K V = x_norm1 * W + b
        linear_forward_rowmajor_kernel(handle, x_norm1, Wq, bq, Q, N, D, D);
        linear_forward_rowmajor_kernel(handle, x_norm1, Wk, bk, K, N, D, D);
        linear_forward_rowmajor_kernel(handle, x_norm1, Wv, bv, V, N, D, D);
        CUDA_CHECK(cudaGetLastError());

        // Multi-head attention
        MHAttention att{B, S, D, H, D / H, 1.0f / sqrtf((float)Dh)};
        att.forward(handle, Q, K, V, att_score, ctx);
        CUDA_CHECK(cudaGetLastError());

        int total = B * S *D;
        merge_heads_kernel<<<(total+threads-1)/threads, threads>>>(ctx, mh_out, B, H, S, Dh, D);
        CUDA_CHECK(cudaGetLastError());

        //out projection
        linear_forward_rowmajor_kernel(handle, mh_out, Wo, bo, mh_out, N, D, D);
        CUDA_CHECK(cudaGetLastError());

        //residual1 = residual1 = x * mh_out, then RELU
        residual_add_kernel<<<(total+threads-1)/threads, threads>>>(x, mh_out, res1, total);
        CUDA_CHECK(cudaGetLastError());
        relu_kernel<<<(total+threads-1)/threads, threads>>>(res1, total);

        //LN2 (pre-FFN-norm)
        layer_norm_kernel<<<N, threads, sh>>>(res1, gamma2, beta2, x_norm2, N, D, 1e-5f);
        CUDA_CHECK(cudaGetLastError());

        //FFN: RELU(W1*x_norm2 + b1, ) W2 +v2
        linear_forward_rowmajor_kernel(handle, x_norm2, W1, b1, ff1, N, D, Dff);
        relu_kernel<<<(N * Dff + threads - 1) / threads, threads>>>(ff1, N * Dff);
        linear_forward_rowmajor_kernel(handle, ff1, W2, b2, ff2, N, Dff, D);
        CUDA_CHECK(cudaGetLastError());

        //residual2 = res1 + ff2
        residual_add_kernel<<<(total+threads-1)/threads, threads>>>(res1, ff2, res2, total);
        CUDA_CHECK(cudaGetLastError());
        relu_kernel<<<(total+threads-1)/threads, threads>>>(res2, total);

        CUDA_CHECK(cudaMemcpy(x, res2, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }
};

int main(){
    int B=2, S=16, D=64, H=4, Dff=256;

    size_t Nd = B * S * D;
    float* x = nullptr;
    CUDA_CHECK(cudaMalloc(&x, Nd * sizeof(float)));
    vector<float> h_x(Nd);
    mt19937 rng(42); normal_distribution<float> N01(0.f, 1.f);
    for (auto &v : h_x) v = 0.02f * N01(rng);
    CUDA_CHECK(cudaMemcpy(x, h_x.data(), Nd * sizeof(float), cudaMemcpyHostToDevice));

    TransformerBlock transformer{B, S, D, H, Dff};
    transformer.forward(x);

    CUDA_CHECK(cudaMemcpy(h_x.data(), x, Nd * sizeof(float), cudaMemcpyDeviceToHost));
    for (const auto &v : h_x) printf("%f ", v); printf("\n");

    CUDA_CHECK(cudaFree(x));
    return 0;
}