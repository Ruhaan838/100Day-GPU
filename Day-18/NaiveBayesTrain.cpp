#include <cuda_runtime.h>
#include "NaiveBayesTrain.cuh"
#include "NaiveBayesKernel.cuh"

void fit_naiveBayes(
    int* dataset, int* priors, int *likelihoods,
    int num_sample, int num_features, int num_class, int num_features_values
) {
    int* d_dataset;
    int* d_priors;
    int* d_likelihoods;

    int dataset_size = num_sample * (num_features + 1) * sizeof(int);
    int priors_size  = num_class * sizeof(int);
    int likeli_hood_size = num_class * num_features * num_features_values * sizeof(int);

    cudaMalloc((void**)&d_dataset, dataset_size);
    cudaMalloc((void**)&d_priors, priors_size);
    cudaMalloc((void**)&d_likelihoods, likeli_hood_size);

    cudaMemcpy(d_dataset, dataset, dataset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_priors, priors, priors_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_likelihoods, likelihoods, likeli_hood_size, cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((num_sample + block_size.x - 1) / block_size.x);

    compute_likelihood<<<grid_size, block_size>>>(d_dataset, d_priors, d_likelihoods, num_sample, num_features, num_class, num_features_values);

    cudaMemcpy(priors, d_priors, priors_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(likelihoods, d_likelihoods, likeli_hood_size, cudaMemcpyDeviceToHost);

    cudaFree(d_dataset);
    cudaFree(d_priors);
    cudaFree(d_likelihoods);

}