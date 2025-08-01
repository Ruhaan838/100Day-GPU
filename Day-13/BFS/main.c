#include "bfs.h"

int main() {
    int num_edges;
    int *h_edges, *h_dest;

    printf("Generating random graph with %d vetices ... \n", NUM_VERTEX);
    generate_random_graph(NUM_VERTEX, &num_edges, &h_edges, &h_dest);

    int *gpu_labels = (int*)malloc(NUM_VERTEX * sizeof(int));
    int *cpu_labels = (int*)malloc(NUM_VERTEX * sizeof(int));

    if (!gpu_labels || !cpu_labels){
        printf("Memory allocation failed!\n");
        exit(1);
    }

    int source = 0;

    printf("\n Running GPU BFS...\n");
    clock_t gpu_start = clock();
    bfs_gpu(source, NUM_VERTEX, num_edges, h_edges, h_dest, gpu_labels);
    clock_t gpu_end = clock();
    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;

    printf("\n Running CPU BFS...\n");
    clock_t cpu_start = clock();
    bfs_cpu(source, NUM_VERTEX, num_edges, h_edges, h_dest, cpu_labels);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

    int mis_match = 0;
    int max_level_gpu = 0;
    int max_level_cpu = 0;
    int unreached_gpu = 0;
    int unreached_cpu = 0;

    for (int i = 0; i < NUM_VERTEX; i++){
        if (gpu_labels[i] > max_level_gpu) max_level_gpu = gpu_labels[i];
        if (cpu_labels[i] > max_level_cpu) max_level_cpu = cpu_labels[i];
        if (gpu_labels[i] == -1) unreached_gpu++;
        if (cpu_labels[i] == -1) unreached_cpu++;
        if (gpu_labels[i] != cpu_labels[i]) mis_match++;
    }

    printf("\nPerformance Results:\n");
    printf("GPU Time: %.6f seconds\n", gpu_time);
    printf("CPU Time: %.6f seconds\n", cpu_time);
    printf("Speedup: %.2fx\n", cpu_time/gpu_time);
    
    printf("\nQuality Results:\n");
    printf("Total mismatches: %d (%.2f%%)\n", mis_match, (100.0 * mis_match) / NUM_VERTEX);
    printf("Max level (GPU): %d\n", max_level_gpu);
    printf("Max level (CPU): %d\n", max_level_cpu);
    printf("Unreached vertices (GPU): %d (%.2f%%)\n", unreached_gpu, (100.0 * unreached_gpu) / NUM_VERTEX);
    printf("Unreached vertices (CPU): %d (%.2f%%)\n", unreached_cpu, (100.0 * unreached_cpu) / NUM_VERTEX);
    
    free(h_edges);
    free(h_dest);
    free(gpu_labels);
    free(cpu_labels);


}