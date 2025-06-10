#ifndef NAIVE_BAYES_KERNEL_CUH
#define NAIVE_BAYES_KERNEL_CUH

__global__ void compute_likelihood(
    int *d_dataset, int *d_priors, int* d_likelihoods,
    int num_samples, int num_features, int num_class, int num_features_vals
);

#endif