#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define UNROLL_FACTOR 4

void vector_add(double *a, double *b, double *c, int n){
    int i, limit = n - (n % UNROLL_FACTOR);
    for (i = 0; i < limit; i += UNROLL_FACTOR) {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv){
    int rank, size, N = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? N : start + chunk_size;

    double *a = (double *)malloc(chunk_size * sizeof(double));
    double *b = (double *)malloc(chunk_size * sizeof(double));
    double *c = (double *)malloc(chunk_size * sizeof(double));

    for (int i = 0; i < chunk_size; i++){
        a[i] = i + rank;
        b[i] = i - rank;
    }

    double t1 = MPI_Wtime();
    vector_add(a, b, c, chunk_size);
    double t2 = MPI_Wtime();

    if (rank == 0) {
        printf("Time taken for vector addition: %f seconds\n", t2 - t1);
    }

    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}