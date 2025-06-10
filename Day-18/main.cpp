#include <stdio.h>
#include "NaiveBayesTrain.cuh"

int main(){
    const int num_sample = 6;
    const int num_features = 2;
    const int num_class = 2;
    const int num_feature_values = 3;

    int dataset[num_sample][num_feature_values + 1] = {
        {0, 1, 1}, // feature0 = 0, feature1 = 1, class_label = 1
        {2, 1, 1},
        {0, 2, 0},
        {2, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
    };

    int priors[num_class] = {0};
    int likelihood[num_class * num_features * num_feature_values] = {0};

    fit_naiveBayes(
        (int*)dataset, priors, likelihood, num_sample, num_features, num_class, num_feature_values
    );

    printf("Priors:\n");
    for (int c = 0; c < num_class; ++c)
        printf("class %d: %f\n", c, (float)priors[c] / num_sample);
    
    printf("\n, likelihoods:\n");
    for (int c = 0; c < num_class; ++c){
        printf("Class %d:\n", c);
        for (int f = 0; f < num_features; ++f){
            for (int v = 0; v < num_feature_values; ++v){
                int idx = c * num_features * num_feature_values + f * num_feature_values + v;
                printf("features %d - values %d: %f\n", f, v, (float)likelihood[idx] / priors[c]);
            }
        }
        printf("\n");
    }

    return 0;

}