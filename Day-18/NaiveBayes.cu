#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NaiveBayesKernel.cuh"
#include "NaiveBayesTrain.cuh"

#define SHARED_SIZE 20

__global__ void compute_likelihood(
    int *d_dataset, int *d_priors, int* d_likelihoods,
    int num_samples, int num_features, int num_class, int num_features_vals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int d_priors_local[SHARED_SIZE];
    __shared__ int d_likelihoods_local[SHARED_SIZE];

    if (idx < num_samples){

        int cls_label = d_dataset[idx * (num_features + 1) + num_features]; // we setup out dataset like the class labels is in last [-1]
        atomicAdd(&d_priors_local[cls_label], 1);

        for (int fidx = 0; fidx < num_features; ++fidx){
            int feature_val = d_dataset[idx * (num_features + 1) + fidx]; //getting the feature value
            int likelihoods_idx = cls_label * num_features * num_features_vals + (fidx * num_features_vals) + feature_val;

            atomicAdd(&d_likelihoods_local[likelihoods_idx], 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0){
        for (int c = 0; c < num_class; ++c)
            atomicAdd(&d_priors[c], d_priors_local[c]);
        
        for (int l = 0; l < num_class * num_features * num_features_vals; ++l)
            atomicAdd(&d_likelihoods[l], d_likelihoods_local[l]);

    }
}