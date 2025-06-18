#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void compute_histogram_probs(float* values, float* support, float* probs, 
                                        int n_values, int n_bins, const float sigma, const float sigma_times_sqrt_two){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    float value = values[idx];

    for (int i = 0; i < n_bins; i++){
        float cdf_right = erff((support[i + 1] - value) / sigma_times_sqrt_two);
        float cdf_left = erff((support[i] - value) / sigma_times_sqrt_two);
        probs[idx * n_bins + 1] = (cdf_right - cdf_left) / 2.0f;
    }
}

extern "C" void hlgauss_loss(float* d_values, float* d_support, float* d_probs, int n_values, int n_bins, float sigma){
    float sigma_times_sqrt_two = sigma * sqrtf(2.0f);

    dim3 block_size(256);
    dim3 grid_size((n_values + block_size.x - 1) / block_size.x);
    
    compute_histogram_probs<<<grid_size, block_size>>>(d_values, d_support, d_probs, n_values, n_bins, sigma, sigma_times_sqrt_two);
}

int main(){
    const int n_values = 1000;
    const int n_bins = 10;
    const float min_values = -5.0f;
    const float max_values = 5.0f;
    const float sigma = 1.0f;

    float* values = (float*)malloc(n_values * sizeof(float));
    float* support = (float*)malloc((n_bins + 1) * sizeof(float));
    float* probs = (float*)malloc(n_values * n_bins * sizeof(float));

    for (int i = 0; i < n_values; i++)
        values[i] = min_values + (max_values - min_values) * ((float)rand() / RAND_MAX);
    
    float bin_width = (max_values - min_values) / n_bins;
    for (int i = 0; i <= n_bins; i++)
        support[i] = min_values + i * bin_width;

    float *d_values, *d_supoort, *d_probs;
    cudaMalloc(&d_values, n_values * sizeof(float));
    cudaMalloc(&d_supoort, (n_bins + 1) * sizeof(float));
    cudaMalloc(&d_probs, n_values * n_bins * sizeof(float));

    cudaMemcpy(d_values, values, n_values * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supoort, support, (n_bins + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probs, probs, n_values * n_bins * sizeof(float), cudaMemcpyHostToDevice);

    hlgauss_loss(d_values, d_supoort, d_probs, n_values, n_bins, sigma);

    cudaMemcpy(probs, d_probs, n_values * n_bins * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Probs:");
    for (int i = 0; i < 5; i++){
        printf("Values %f", values[i]);
        for(int j = 0; j < n_bins; j++)
            printf("%f ", probs[i * n_bins + j]);

        printf("\n");
    }

    free(values);
    free(support);
    free(probs);

    cudaFree(d_values);
    cudaFree(d_supoort);
    cudaFree(d_probs);

    return 0;


}