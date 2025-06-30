import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, Block_size:tl.constexpr):
    program_id = tl.program_id(axis=0)
    
    num_blocks_n = tl.cdiv(N, Block_size)
    
    block_row = program_id // num_blocks_n
    block_col = program_id % num_blocks_n
    
    row_start = block_row * Block_size
    col_start = block_col * Block_size
    
    acc = tl.zeros((Block_size, Block_size), dtype=tl.float32)
    
    for k_start in range(0, K, Block_size):
        
        a_row = row_start + tl.arange(0, Block_size)[:, None]
        a_col = k_start + tl.arange(0, Block_size)[None, :]
        
        b_row = k_start + tl.arange(0, Block_size)[:, None]
        b_col = col_start + tl.arange(0, Block_size)[None, :]
        
        a_mask = (a_row < M) & (a_col < K)
        b_mask = (b_row < K) & (b_col < N)
        
        A = tl.load(
            a_ptr + a_row * stride_am + a_col * stride_ak,
            mask=a_mask, other=0.0
        )
        B = tl.load(
            b_ptr + b_row * stride_bk + b_col * stride_bn,
            mask=b_mask, other=0.0
        )
        
        acc += tl.dot(A, B)
        
    c_row = row_start + tl.arange(0, Block_size)[:, None]
    c_col = col_start + tl.arange(0, Block_size)[None, :]
    c_mask = (c_row < M) & (c_col < N)
    
    tl.store(
        c_ptr + c_row * stride_cm + c_col * stride_cn,
        acc, mask=c_mask
    )
    
def triton_matmul(A, B):
    assert A.shape[1] == B.shape[1], "mishmatch at shape"
    
    M, K = A.shape
    _, N = B.shape
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    Block_size = 16
    grid_m = (M + Block_size - 1) // Block_size
    grid_n = (N + Block_size - 1) // Block_size
    grid = [grid_m * grid_n]
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        Block_size=Block_size
    )
    
    return C

A = torch.randn(128, 128, device="cuda", dtype=torch.float32)
B = torch.randn(128, 128, device="cuda", dtype=torch.float32)

C = triton_matmul(A, B)
C1 = torch.matmul(A, B)
print(C1)
print(C)