import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride_n, F, Blocks_size:tl.constexpr):
    row_idx = tl.program_id(0) #for one probem we consider one row
    offsets = tl.arange(0, Blocks_size)
    mask = offsets < F
    
    row_start = in_ptr + row_idx * stride_n
    values = tl.load(row_start + offsets, mask=mask, other=-float("inf"))
    
    row_max = tl.max(values, axis=0)
    values = values - row_max
    
    exp_values = tl.exp(values)
    row_sum = tl.sum(exp_values, axis=0)
    
    softmax = exp_values / row_sum
    
    row_out_start = out_ptr + row_idx * stride_n
    tl.store(row_out_start + offsets, softmax, mask=mask)

def softmax(x):
    N, F = x.shape
    out = torch.empty_like(x)
    grid = (N,)
    Blocks_size = triton.next_power_of_2(F)
    
    softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        F,
        Blocks_size=Blocks_size
    )
    
    return out


x = torch.randn(4, 128, device="cuda")
y = softmax(x)
print(y)
y1 = torch.softmax(x, dim=-1)
print(y1)
print(torch.allclose(y, y1))