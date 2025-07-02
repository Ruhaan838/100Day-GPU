import torch
import triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k + offs_k)[None, :] < K,
            other=0.0
        )

        B_tile = tl.load(
            B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k + offs_k)[:, None] < K & (offs_n[None, :] < N),
            other=0.0
        )

        acc += tl.dot(A_tile, B_tile)

    # acc = tl.maximum(acc, 0.0)

    C_tile = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        C_tile,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

def matmul_relu(A: torch.Tensor, B: torch.Tensor, BLOCK=64):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device='cuda', dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))

    matmul_relu_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK, BLOCK, BLOCK
    )
    return C

torch.manual_seed(0)
M, K, N = 128, 256, 64
A = torch.randn((M, K), device='cuda', dtype=torch.float16)
B = torch.randn((K, N), device='cuda', dtype=torch.float16)

C = matmul_relu(A, B)

C_ref = torch.matmul(A, B)
print(C)
print(C_ref)
print(torch.allclose(C, C_ref))
