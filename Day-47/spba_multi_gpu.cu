#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

void CudaCheck(cudaError_t error){
    if (error != cudaSuccess){
        cerr << "Cuda Error: " << cudaGetErrorString(error) << '\n';
        exit(EXIT_FAILURE);
    }
}

__global__ void Candidate_kernel(const float* candidate, float* fitness, int num_candidate, int d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_candidate){
        float sum = 0.0f;
        for (int j = 0; j < d; j++){
            float x = candidate[idx * d + j];
            sum += x * x;
        }
        fitness[idx] = sum;
    }
}

struct Candidate{
    vector<float> position;
    float fitness;
};

void Candidate_device(int device, const float* candidate, float* fitness, int num_candidate, int d){

    CudaCheck(cudaSetDevice(device));
    size_t data_size = num_candidate * d * sizeof(float);
    float* d_candidate, *d_fitness;
    CudaCheck(cudaMalloc(&d_candidate, data_size));
    CudaCheck(cudaMalloc(&d_fitness, num_candidate * sizeof(float)));

    CudaCheck(cudaMemcpy(d_candidate, candidate, data_size, cudaMemcpyHostToDevice));

    dim3 block_size(256);
    dim3 grid_size((num_candidate + block_size.x - 1) / block_size.x);
    Candidate_kernel<<<grid_size, block_size>>>(d_candidate, d_fitness, num_candidate, d);
    CudaCheck(cudaGetLastError());
    CudaCheck(cudaDeviceSynchronize());

    CudaCheck(cudaMemcpy(fitness, d_fitness, num_candidate * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_candidate);
    cudaFree(d_fitness);
}

void Eval_Polulation_Multi_GPU(const vector<Candidate>& polulation, int d, int device_count, vector<float>& fitness_results){
    int total = polulation.size();
    vector<float> candidate_data(total * d);
    for(int i = 0; i < total; i++){
        for(int j = 0; j < d; j++){
            candidate_data[i * d + j] = polulation[i].position[j];
        }
    }

    fitness_results.resize(total);
    int partition_size = (total + device_count - 1) / device_count;
    vector<thread> threads;
    for (int dev = 0; dev < device_count; dev++){
        int start = dev * partition_size;
        int end = min(start + partition_size, total);
        if (start >= end) break;

        int num_candidates = end - start;
        float* candidate_subset = candidate_data.data() + start * d;
        float* fitness_subset = fitness_results.data() + start;
        threads.emplace_back(Candidate_device, dev, candidate_subset, fitness_subset, num_candidates, d);
    }
    for (auto & t:threads){
        t.join();
    }
}   

Candidate random_candidate(int d, float range, mt19937& rng){
    Candidate cand;
    cand.position.resize(d);
    uniform_real_distribution<float> dist(-range, range);
    for(int i = 0; i < d; i++){
        cand.position[i] = dist(rng);
    }
    cand.fitness = 0.0f;
    return cand;
}

int main(){
    const int population_size = 256;
    const int dim = 30;
    const int max_iter = 100;
    const float alpha = 0.1f;
    const int num_best_sites = population_size / 10;
    const int recuruits_per_site = 10;
    const float search_range = 10.0f;

    random_device rd;
    mt19937 rng(rd());
    vector<Candidate> population;
    for(int i = 0; i < population_size; i++){
        population.push_back(random_candidate(dim, search_range, rng));
    }

    int device_count = 0;
    CudaCheck(cudaGetDeviceCount(&device_count));
    if (device_count < 1){
        cerr << "No CUDA Device! Lol!!";
        return -1;
    }

    for (int iter = 0; iter < max_iter; iter++){
        vector<float> fitness_result;
        Eval_Polulation_Multi_GPU(population, dim, device_count, fitness_result);
        for (int i = 0; i < population_size; i++){
            population[i].fitness = fitness_result[i];
        }
        sort(population.begin(), population.end(), [](const Candidate& a, const Candidate& b){
            return a.fitness < b.fitness;
        });
        cout << "Iteration " << iter << ", Best fitness: " << population[0].fitness << '\n';

        for (int i = 0; i < num_best_sites; i++){
            Candidate bestCandidate = population[i];
            for(int r = 0; r < recuruits_per_site; r++){
                Candidate new_candiate = bestCandidate;
                for(int j = 0; j < dim; j++){
                    uniform_real_distribution<float> dist(-alpha, alpha);
                    new_candiate.position[j] += dist(rng);
                }

                float fitness = 0.0f;
                for(int j = 0; j < dim; j++){
                    fitness += new_candiate.position[j] * new_candiate.position[j];
                }
                new_candiate.fitness = fitness;
                if (new_candiate.fitness < bestCandidate.fitness){
                    bestCandidate = new_candiate;
                }
            }
            population[i] = bestCandidate;
        }

        for(int i = population_size / 2; i < population_size; i++){
            population[i] = random_candidate(dim, search_range, rng);
        }
    }

    cout << "Best fitness " << population[0].fitness << '\n';
    return 0;

}