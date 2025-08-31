#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>


// const int MAX_BATCH_SIZE = 32;
// const int MAX_CHANNELS = 4;
// const int MAX_HEIGHT = 1024;
// const int MAX_WIDTH = 1024;
const int MAX_SRAM_SIZE = 32;

template <typename T>
void CHECK_CUDA(T err){
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s code line = %d\n", cudaGetErrorString(err), __LINE__);
        exit(EXIT_FAILURE);
    }
}

struct DiffusionSamplingParams{
    float beta_start;
    float beta_end;
    int num_diff_timesteps;
    int sampling_timesteps;
    bool use_ddim;
    float eta; // DDIM param
};

void preComputeDiffusionParams(const DiffusionSamplingParams* params, float* alphas,
                                float* alphas_cumprod, float* sqrt_one_minus_alphs_cumprod, float* sigmas){

    float* betas = new float[params->num_diff_timesteps];

    //torch.linspace 
    // (start, start + (end - start) / steps - 1, ..., start + (steps - 2) * (end - start) / steps - 1, end) 
    // from torch.linspace docs.
    for(int t = 0; t < params->num_diff_timesteps; t++){
        float beta = params->beta_start + (params->beta_end - params->beta_start) * t / (params->num_diff_timesteps - 1);
        betas[t] = beta;
        alphas[t] = 1.0f - beta;
    }
    //torch.cumprod
    // y_i = x_1 * x_2 * ... * x_i
    alphas_cumprod[0] = alphas[0];
    for (int t = 1; t < params->num_diff_timesteps; t++) {
        alphas_cumprod[t] = alphas_cumprod[t - 1] * alphas[t];
    }

    for (int t = 0; t < params->num_diff_timesteps; t++) {
        sqrt_one_minus_alphs_cumprod[t] = sqrtf(1.0f - alphas_cumprod[t]);
        sigmas[t] = sqrtf((1.0f - alphas_cumprod[t-1]) / (1.0f - alphas_cumprod[t]) * 
                          (1.0f - alphas_cumprod[t] / alphas_cumprod[t-1]));
    }
    delete[] betas;
}

__global__ void initRandomStates(curandState* states, unsigned long seed, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void diffusionSampleStep(
    float* x,
    float* noise_pred,
    float* denoised_output,
    float alpha_t,
    float alpha_prev,
    float sigma_t,
    curandState* random_states,
    int batch_size,
    int channels,
    int height,
    int width,
    bool use_ddim,
    float noise_scale
) {
    const int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    const int pixel_idx = y_idx * width + x_idx;

    __shared__ float s_noise_pred[MAX_SRAM_SIZE][MAX_SRAM_SIZE];

    for(int b = 0; b < batch_size; b++){
        for(int c = 0; c < channels; c++){
            const int idx = ((b * channels + c) * height * width) + pixel_idx;

            if (x_idx < width && y_idx < height){
                float x_t = x[idx];
                float eps = noise_pred[idx];

                if (threadIdx.x < MAX_SRAM_SIZE && threadIdx.y < MAX_SRAM_SIZE) {
                    s_noise_pred[threadIdx.y][threadIdx.x] = eps;
                }

                __syncthreads();

                float x_prev;

                if (use_ddim){
                    float alpha_ratio = alpha_prev / alpha_t;
                    float sigma = sigma_t * noise_scale;

                    float pred_x0 = (x_t - sqrtf(1.0f - alpha_t) * eps) / sqrtf(alpha_t);
                    x_prev = sqrtf(alpha_prev) * pred_x0;

                    if (noise_scale > 0.0f){
                        float z = curand_normal(&random_states[blockIdx.x]);
                        x_prev += sigma * z;
                    }
                } else {
                    float pred_x0 = (x_t - sqrtf(1.0f - alpha_t) * eps) / sqrtf(alpha_t);
                    float mean = (sqrtf(alpha_prev) * (1.0f - alpha_t) * pred_x0 + 
                                    sqrtf(alpha_t) * (1.0f - alpha_prev) * x_t) / (1.0f - alpha_t * alpha_prev);
                    float var = (1.0f - alpha_prev) * (1.0f - alpha_t) * (1.0f - alpha_t * alpha_prev);
                    float z = curand_normal(&random_states[blockIdx.x]);
                    x_prev = mean + sqrtf(var) * z;
                }

                x_prev = fmaxf(-1.0f, fminf(1.0f, x_prev));

                denoised_output[idx] = x_prev;
            }
        }
    }
}

__global__ void predictNoise(
    const float* model_weights,
    const float* noise_input,
    float* noise_pred,
    int batch_size,
    int channels,
    int height,
    int width,
    int timestep
) {
   const int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    const int pixel_idx = y_idx * width + x_idx;

    for(int b = 0; b < batch_size; b++){
        for(int c = 0; c < channels; c++){
            const int idx = ((b * channels + c) * height * width) + pixel_idx;

            noise_pred[idx] = 0.1f * noise_input[idx] * (timestep / 100.0f);
        }
    }
}

extern "C" void sampleDiffusionModel(
    float* init_noise,
    float* final_sample,
    float* model_weights,
    DiffusionSamplingParams* params,
    int batch_size, int channels, int height, int width
){
    float* d_sample, *d_noise_pred, *d_temp_buffer, *d_model_weights;
    float* d_alphs, *d_alphs_cumprod, *d_sqrt_one_minus_alphs_cumprod, *d_sigmas;
    curandState *d_random_states;

    size_t sample_size = batch_size * channels * height * width * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_sample, sample_size));
    CHECK_CUDA(cudaMalloc(&d_temp_buffer, sample_size));
    CHECK_CUDA(cudaMalloc(&d_noise_pred, sample_size));
    CHECK_CUDA(cudaMalloc(&d_random_states, height * width * sizeof(curandState)));

    CHECK_CUDA(cudaMemcpy(d_sample, init_noise, sample_size, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_alphs, params->num_diff_timesteps * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alphs_cumprod, params->num_diff_timesteps * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sqrt_one_minus_alphs_cumprod, params->num_diff_timesteps * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sigmas, params->num_diff_timesteps * sizeof(float)));

    float* alphs = new float[params->num_diff_timesteps];
    float *alphs_cumprod = new float[params->num_diff_timesteps];
    float *sqrt_one_minus_alphs_cumprod = new float[params->num_diff_timesteps];
    float *sigmas = new float[params->num_diff_timesteps];

    preComputeDiffusionParams(params, alphs, alphs_cumprod, sqrt_one_minus_alphs_cumprod, sigmas);

    CHECK_CUDA(cudaMemcpy(d_alphs, alphs, params->num_diff_timesteps * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alphs_cumprod, alphs_cumprod, params->num_diff_timesteps * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sqrt_one_minus_alphs_cumprod, sqrt_one_minus_alphs_cumprod, params->num_diff_timesteps * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigmas, sigmas, params->num_diff_timesteps * sizeof(float), cudaMemcpyHostToDevice));

    size_t model_size = 1 * 1024 * 1024 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_model_weights, model_size));
    CHECK_CUDA(cudaMemcpy(d_model_weights, model_weights, model_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    initRandomStates<<<gridSize, blockSize>>>(d_random_states, time(NULL), width * height);

    int timestep_specing = params->num_diff_timesteps / params->sampling_timesteps;

    for(int step = 0; step < params->sampling_timesteps; step++){
        int t = params->num_diff_timesteps - 1 - step * timestep_specing;
        int t_prev = (step == params->sampling_timesteps - 1) ? 0 : 
                        params->num_diff_timesteps - 1 - (step - 1) * timestep_specing;

        predictNoise<<<gridSize, blockSize>>>(
            d_model_weights,
            d_sample,
            d_noise_pred,
            batch_size,
            channels,
            height,
            width,
            t
        );

        diffusionSampleStep<<<gridSize, blockSize>>>(
            d_sample,
            d_noise_pred,
            d_temp_buffer,
            alphs_cumprod[t],
            alphs_cumprod[t_prev],
            sigmas[t],
            d_random_states,
            batch_size,
            channels,
            height,
            width,
            params->use_ddim,
            0.8f
        );

        float* temp = d_sample;
        d_sample = d_temp_buffer;
        d_temp_buffer = temp;
    }

    CHECK_CUDA(cudaMemcpy(final_sample, d_sample, sample_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_sample));
    CHECK_CUDA(cudaFree(d_temp_buffer));
    CHECK_CUDA(cudaFree(d_noise_pred));
    CHECK_CUDA(cudaFree(d_model_weights));
    CHECK_CUDA(cudaFree(d_alphs));
    CHECK_CUDA(cudaFree(d_alphs_cumprod));
    CHECK_CUDA(cudaFree(d_sqrt_one_minus_alphs_cumprod));
    CHECK_CUDA(cudaFree(d_sigmas));
    CHECK_CUDA(cudaFree(d_random_states));

    delete[] alphs;
    delete[] alphs_cumprod;
    delete[] sqrt_one_minus_alphs_cumprod;
    delete[] sigmas;
}

#include <iostream>
#include <cstdlib>
#include <ctime>

// Forward declaration of your function
extern "C" void sampleDiffusionModel(
    float* init_noise,
    float* final_sample,
    float* model_weights,
    DiffusionSamplingParams* params,
    int batch_size, int channels, int height, int width
);

int main() {

    int batch_size = 32;
    int channels = 3;
    int height = 64;
    int width = 64;

    DiffusionSamplingParams params;
    params.beta_start = 0.0001f;
    params.beta_end = 0.02f;
    params.num_diff_timesteps = 1000;
    params.sampling_timesteps = 50;
    params.use_ddim = true;
    params.eta = 0.0f;

    size_t sample_size = batch_size * channels * height * width;
    float* init_noise = new float[sample_size];
    float* final_sample = new float[sample_size];


    srand((unsigned)time(NULL));
    for (size_t i = 0; i < sample_size; i++) {
        init_noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    size_t model_size = 1 * 1024 * 1024;
    float* model_weights = new float[model_size];
    for (size_t i = 0; i < model_size; i++) {
        model_weights[i] = 0.01f * (i % 100);
    }

    sampleDiffusionModel(init_noise, final_sample, model_weights,
                         &params, batch_size, channels, height, width);


    for (int i = 0; i < 20; i++) {
        std::cout << "final_sample[" << i << "] = " << final_sample[i] << "\n";
    }

    delete[] init_noise;
    delete[] final_sample;
    delete[] model_weights;

    return 0;
}