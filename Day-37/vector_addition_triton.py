import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, Block_Size: tl.constexpr):
    program_id = tl.program_id(axis=0)
    offset = pid * Block_Size + tl.arange(0, Block_Size)
    mask = offset < N
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    output = x + y
    tl.store(out_ptr + offset, output, mask=mask)
    
def vector_add(x: torch.Tensor, y:torch.Tensor):
    assert x.shape == y.shape, "Input tensor have same shape"
    
    N = x.numel()
    out = torch.empty_like(x)
    Block_size = 1024
    grid = (triton.cdiv(N, Block_size),)
    
    vector_add_kernel[grid](
        x_ptr=x, y_ptr=y, out_ptr=out, N=N, Block_Size=Block_size
    )
    
    return out

x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
out = vector_add(x, y)
print(out)