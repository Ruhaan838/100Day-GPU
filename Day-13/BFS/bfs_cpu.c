#include "bfs.h"

void generate_random_graph(int num_vertex, int* num_edgs, int** edgs, int** dest){
    srand(time(NULL));

    int max_edges = num_vertex * AVG_EDES_PER_VERTEX;
    *dest = (int*)malloc(max_edges * sizeof(int));
    *edgs = (int*)malloc((num_vertex + 1) * sizeof(int));

    int crr_edge = 0;
    (*edgs)[0] = 0;

    for(int i = 0; i < num_vertex; i++){
        int edge_for_vertex = rand() % (AVG_EDES_PER_VERTEX * 2);

        for (int j = 0; j < edge_for_vertex; j++){
            int dest_vetex = rand() % num_vertex;
            if(dest_vetex != i)
                (*dest)[crr_edge++] = dest_vetex;
        }

        (*edgs)[i+1] = crr_edge;
    }
    *num_edgs = crr_edge;

    *dest = (int*)realloc(*dest, crr_edge * sizeof(int));
}

void bfs_cpu(int source, int num_vertex, int num_edgs, int* edgs, int* dest,  int* labels){

    for(int i = 0; i < num_vertex; i++){
        labels[i] = -1;
    }
    
    int* crr_frontier = (int*)malloc(num_vertex * sizeof(int));
    int* next_frontier = (int*)malloc(num_vertex * sizeof(int));

    int crr_size = 0;
    int next_size = 0;

    labels[source] = 0;
    crr_frontier[0] = source;
    crr_size = 1;

    int level = 0;

    while (crr_size > 0){
        next_size = 0;
        level++;

        for(int i = 0; i < crr_size = 0; i++){
            int vertx = crr_frontier[i];
            for (int edge = edgs[vertx]; edge < edgs[vertx + 1]; edge++){
                int neighbor = dest[edge];
                if (labels[neighbor] == -1){
                    labels[neighbor] = level;
                    next_frontier[next_size++] = neighbor;
                }
            }
        }

        int* temp = crr_frontier;
        crr_frontier = next_frontier;
        next_frontier = temp;
        crr_size = next_size;
        
    }

    free(crr_frontier);
    free(next_frontier);

}