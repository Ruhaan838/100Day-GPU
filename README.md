## 1️⃣ Day - 01
Day 1 was very simple. I just learned how to add two 1D vectors using a basic CUDA kernel.

### Keywords and Variable.
- `__global__`: This keyword is used to define a function (kernel) that runs on the GPU. You can launch it using the `<<<...>>>` syntax.
- `blockIdx.x`: This variable provides the current block index within a grid.
- `blockDim.x`: This variable provides the number of threads per block.
- `threadIdx.x`: This variable provides the current thread index within the block.

### Allocation and Deallocation.

- `cudaMalloc`: Allocates memory on the CUDA device (GPU).
- `cudaMemcpy`: Copies memory between host (CPU) and device (GPU). For example, use `cudaMemcpyHostToDevice` to copy from host to device.
- `cudaFree`: Frees the allocated memory on the CUDA device.
  
## 2️⃣ Day - 02
Day 2 is kind a simple. Now I know how to add 2D matrix.

### Keywords and Variable.
- `dim3`: This is a CUDA data type that allows you to define dimensions in 1D, 2D, or 3D—for grids and blocks.
- `cudaDeviceSynchronize`: It foreces the CPU to wait for the GPU to finish its task.

### New Thing
- You can add two matrices using this formula
```
  c[i * N + j] = a[i * N + j] + b[i * N + j]
  here:
    i: row
    j: column
    N: len of matrix
```
- Rest of the things are same as per the Day01.