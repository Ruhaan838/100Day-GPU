## Day - 01
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