## 1️⃣ Day - 01
Day 1 was very simple. I just learned how to add two 1D vectors using a basic CUDA kernel.

### Keywords and Variables.
- `__global__`: This keyword is used to define a function (kernel) that runs on the GPU. You can launch it using the `<<<...>>>` syntax.
- `blockIdx.x`: This variable provides the current block index within a grid.
- `blockDim.x`: This variable provides the number of threads per block.
- `threadIdx.x`: This variable provides the current thread index within the block.

### Allocation and Deallocation.

- `cudaMalloc`: Allocates memory on the CUDA device (GPU).
- `cudaMemcpy`: Copies memory between host (CPU) and device (GPU). For example, use `cudaMemcpyHostToDevice` to copy from host to device.
- `cudaFree`: Frees the allocated memory on the CUDA device.
  
## 2️⃣ Day - 02
Day 2 is kind of simple. Now I know how to add 2D matrix.

### Keywords and Variables.
- `dim3`: This is a CUDA data type that allows you to define dimensions in 1D, 2D, or 3D—for grids and blocks.
- `cudaDeviceSynchronize`: It forces the CPU to wait for the GPU to finish its task.

### New Thing
- You can add two matrices using this formula
```
  c[i * N + j] = a[i * N + j] + b[i * N + j]
  here:
    i: row
    j: column
    N: len of matrix
```
- The rest of the things are the same as per Day01.

## 3️⃣ Day - 03
Day 3 is too simple. Now I know how to multiply 2D mat with 1D vec.

### Example:
```
2D mat
1 1 1 
1 1 1 
1 1 1 

1D vec
2 2 2 

ans vec
6 6 6

```

This is a simple matrix multiplication with a vector.
- Nothing new to learn on this day.

## 4️⃣ Day - 04
Day 4 is quite new and interesting.

### Keywords and Variables.
- `__shared__`: This kerword is allow to share the memory.
- `__syncthreads`[2]: Command is a block level synchronization barrier. Basically stabelze the hang or produce unintended side effects.

## How Code Works

- First thing is that we have

2 blocks, each with 8 threads <br>
Each block uses 32 bytes of shared memory <br>
`<<<gridsize, blocksize, shared_mem>>>` = `<<<2, 8, 32>>>`

---
>**NOTE[1]:** 1 static shared memory: <br>
&emsp;&emsp; `__shared__ int var1[10]` <br>
2 dynamic shared memory: should add "extern" keyword <br>
&emsp;&emsp; `extern __shared__ int Var1[]`
---

- Then we fill the sharedMemory like this

```bash
sharedMemory[0] = in[0] + in[8] = 1 + 9 = 10

sharedMemory[1] = in[1] + in[9] = 2 + 10 = 12

...

sharedMemory[7] = in[7] + in[15] = 8 + 16 = 24
```

- By this sharedMemory we perfrom the partialSum.
```
[10, 12, 14, 16, 18, 20, 22, 24]
[10, 22, 36, 52, 70, 90, 112, 136]
```

- Credits <br>
  - [stackOverflow-1](https://stackoverflow.com/questions/12066730/allocate-shared-variables-in-cuda)
  - [stackOverflow-2](https://stackoverflow.com/questions/15240432/does-syncthreads-synchronize-all-threads-in-the-grid)
  
## 5️⃣ Day - 05

Today, day I feel like I use all the concept of the other days.

- By using the formula for **Layer Normalization**, we can implement it in CUDA quite efficiently.

### 📐 LayerNorm Formula:

Given an input vector `x` of size `N`:

$$
\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Where:
- $$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i \quad \text{(Mean of } x \text{)} $$
- $$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \quad \text{(Variance of } x \text{)} $$
- $$ \epsilon \quad \text{(Small constant to avoid division by zero)} $$
- $$ \gamma, \beta \quad \text{(Learnable scale and shift parameters)} $$

