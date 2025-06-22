#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>

void check_hip(hipError_t err){
    if (err != hipSuccess){
        fprintf(stderr, "HIP error in %s:%d: %s\n", __FILE__, __LINE__, hipGerErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void check_rocblas(rocblas_status err){
    if (err != rocblas_status_success){
        fprintf(stderr, "rocBLAS error in %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(err));
        exit(EXIT_FAILURE);
    }
}

int main(){
    const int N = 10;
    float a[N], b[N], c[N];

    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i;
    }

    rocblas_handle handle;
    check_rocblas(rocblas_create_handle(&handle));

    float *da, *db;
    size_t size = N * sizeof(float);
    check_hip(hipMalloc(&da, size));
    check_hip(hipMalloc(&db, size));

    check_hip(hipMemcpy(da, a, size, hipMemcpyHostToDevice));
    check_hip(hipMemcpy(db, b, size, hipMemcpyHostToDevice));

    const float alpha = 1.0f;

    check_rocblas(rocblas_saxpy(handle, N, &alpha, da, 1, db, 1));

    check_hip(hipMemcpy(c, dc, size, hipMemcpyHostToDevice));

    std::cout << "Result vec:";
    for (int i = 0; i < N; i++)
        std::cout << c[i] << " ";
    cout << "\n";

    hipFree(da);
    hipFree(db);
    rocblas_destory_handle(handle);

    return 0;
}