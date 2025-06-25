
# 🧑🏻‍💻 CUDA (NVIDIA)


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

Today, I use all the concepts from other days.

- By using the formula for **Layer Normalization**, we can implement it in CUDA quite efficiently.

### 📐 LayerNorm Formula:

Given an input vector `x` of size `N`:

$$
\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Where: <br>

$$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i \quad \text{(Mean of } x \text{)} $$ 
$$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \quad \text{(Variance of } x \text{)} $$
$$ \epsilon \quad \text{(Small constant to avoid division by zero)} $$
$$ \gamma, \beta \quad \text{(Learnable scale and shift parameters)} $$

## 6️⃣ Day - 06

Today, I learned how to **transpose any matrix** using CUDA.

### Keywords and Variables

- `cudaError_t` — This type is used to define error codes returned by CUDA API functions.
- `cudaGetLastError()` — This function retrieves the last error that occurred during CUDA kernel execution or API call.

---

Other than that, today felt similar to **Day - 02**, since it also involved working with **2D matrices**.

## 7️⃣ Day - 07

Day 07 is quite interesting and has so many things to learn. Now I know how tiled Conv works. Today I realize how tiled conv works and not other things need to be new on this day. 

- Refenrence:
  - https://www.youtube.com/watch?v=ysBrzOTMZlQ
  - https://www.cs.ucr.edu/~nael/217-f15/lectures/217-lec8.pdf

## 8️⃣ Day - 08

Day 08 I learn brent kung algorithm for fast prefix sum.

- Refenrance:
  - https://www.youtube.com/watch?v=1G4jfLcnI2w&t=88s
  
## 9️⃣ Day - 09 (flash Attention)
 
Day 09 I learn the Flash Attention forward pass by pure cuda with shared memory for each query, key and value.

- Learn proper memory management and other stuff ...

## 🔟 Day - 10

Day 10 I learn the falsh attention implementation in higher dimensional tensor.

## 1️⃣1️⃣ Day - 11

Learn what is the sparse Matrix, Learn what is the ELL and COO methods for storing the sparse Matrix.
Understand the importance of the sparse Matrix in real life.

## 1️⃣2️⃣ Day - 12 
 
Learn the how do parallel merge sort in cuda.

## 1️⃣3️⃣ Day - 13

Implement BFS in parallel and also implement the Gelu in multiple threads.

## 1️⃣4️⃣ Day - 14

Implement Simple Nural network. make the Linear Layer.

## 1️⃣5️⃣ Day - 15 

Today I Implement CNN from scratch using CUDA kernel.

## 1️⃣6️⃣ Day - 16

The Day 16 is quite simple one. Implement FHD (fully-hybrid Domain) algorim for non-cartesian magnetic resonance imaging (MRI) reconstruction in CUDA.

## 1️⃣7️⃣ Day - 17

The Day 17 is have so much to learn, I learn flash-attention-2 with forward and backward pass.

## 1️⃣8️⃣ Day - 18

Today I implement the NaiveBayes with shared memeory of prior and likelihood.

## 1️⃣9️⃣ Day - 19

Today I learn use of cublas API and implement vector addition and matmul.

## 2️⃣0️⃣ Day - 20

Today I learn use of cuDNN API which is very grate and complex libray. it's allow to write the fcNet by using just simple functions and help a lot for CUDA codes. Have inbuild functions for GEMM and etc... 

## 2️⃣1️⃣ Day - 21

Today I implement RoPE in cuda for query and key.

## 2️⃣2️⃣ Day - 22

Implement EM-algorithem for 1D gaussian vector. to make the cluster!!

## 2️⃣3️⃣ Day - 23

Implement swiGLU implement it on 2D data. eazy...

## 2️⃣4️⃣ Day - 24

Implement atomicAdd using cuda kernel in this code we basically count the number of threadID using just simple atomicAdd.

## 2️⃣5️⃣ Day - 25

Implement Monte Carlo Tree Search in CUDA. With 1024 parallel simulations.

## 2️⃣6️⃣ Day - 26

Implement histogram loss in parallel and shared memory on CUDA.

## 2️⃣7️⃣ Day - 27

Implement mirror descent in cuda with parallel threads.

## 2️⃣8️⃣ Day - 28

Implement micrograd like autograd in cuda on parallel threads.

## 2️⃣9️⃣ Day - 29

Learn how to use the CUDAGraph for fast arithmetic and computation without changing kernels.

Sure! Here's a sample write-up you can use to describe your work for **Day-30**, based on the project structure shown in the screenshot:



# 🖥️ HIP/RoCm (AMD)


## 3️⃣0️⃣ Day-30

Today’s focus was on implementing and experimenting with deep learning operations and parallel computing techniques using HIP (Heterogeneous-Compute Interface for Portability) for AMD GPUs. The project is organized into three key folders, each representing a layer of computational complexity and abstraction.

#### `DL/` — Deep Learning Operations


* **`conv_2d.cpp`**: HIP-based implementation of 2D convolution used in CNNs.
* **`flash_attention_forward.cpp`**: Explores efficient attention mechanisms inspired by FlashAttention.
* **`gelu.cpp`**: Implements the GELU activation function commonly used in Transformer models.
* **`layer_norm.cpp`**: HIP kernel for Layer Normalization.
* **`rope_hip.cpp`**: Rotary positional encoding, relevant for Transformer-based architectures.

#### `parallel/` — Matrix Operations with Parallelism


* **`matmul_rocblas.cpp`**: Matrix multiplication using AMD's rocBLAS.
* **`matrix_add.cpp`**: Parallel matrix addition.
* **`matrix_trans.cpp`**: Matrix transpose operation with shared memory usage.
* **`parallel_merge.cpp`**: Illustrates data merging using thread-level parallelism.

#### `simple/` — Introductory Parallel Programs


* **`partial_sum.cpp`**: A basic reduction operation to compute the sum of array elements.
* **`prefix_sum.cpp`**: Scan operation (inclusive prefix sum) across threads.
* **`vec_reocblas.cpp`**: Vector operations with rocBLAS integration.
* **`vector_add.cpp`**: Parallel vector addition using HIP.
* **`vector_matrix_mul.cpp`**: Hybrid computation of vector-matrix multiplication.

# 🧑🏻‍💻 CUDA (NVIDIA)

## 3️⃣1️⃣ Day-31

Implement game of life using shared memory in CUDA.


#  🖥️ HIP/RoCm (AMD)

## 3️⃣2️⃣ Day-32

Implement SGMM in AMDs HIP Kernel.

# 🧑🏻‍💻 CUDA (NVIDIA)

## 3️⃣3️⃣ Day-33

Implement mlp with relu with forward and backward.