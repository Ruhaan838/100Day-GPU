import triton
import torch
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def conv1d_kernel(input_ptr, kernel_ptr, output_ptr, 
                  in_ch, out_ch, in_len, kernel_size, 
                  stride, out_len, Block_out:tl.constexpr):
    
    batch = tl.program_id(0)
    out_id = tl.program_id(1)
    out_idx = tl.program_id(2) * Block_out + tl.arange(0, Block_out)
    
    mask = out_idx < out_len
    acc = tl.zeros([Block_out], dtype=tl.float32)
    
    for i_idx in range(in_ch):
        for ker_idx in range(kernel_size):
            in_idx = out_idx * stride + ker_idx
            in_mask = in_idx < in_len
            
            input_offset = ((batch * in_ch + i_idx) * in_len) + in_idx
            kernel_offset = ((out_id * in_ch + i_idx) * kernel_size) + ker_idx
            
            input_val = tl.load(input_ptr + input_offset, mask=in_mask, other=0.0)
            kernel_val = tl.load(kernel_ptr + kernel_offset)
            
            acc += input_val * kernel_val
            
    output_offset = (batch * out_ch + out_id) * out_len + out_idx
    tl.store(output_ptr + output_offset, acc, mask=mask)
    
def conv1d(inputs, kernel, stride=1):
    b, in_ch, in_len = inputs.shape
    out_ch, _, kernel_size = kernel.shape
    out_len = (in_len - kernel_size) // stride + 1
    
    output = torch.empty((b, out_ch, out_len), device=inputs.device, dtype=torch.float32)
    grid = (b, out_ch, (out_len + 31) // 32)
    
    conv1d_kernel[grid](
        inputs, kernel, output,
        in_ch, out_ch, in_len, kernel_size, stride, out_len,
        Block_out=32
    )
    
    return output

b, in_ch, in_len = 1, 3, 64
out_ch, kernel_size, stride = 8, 5, 1

input_tensor = torch.randn(b, in_ch, in_len, device='cuda')
kernel_tensor = torch.randn(out_ch, in_ch, kernel_size, device='cuda')

output_tensor = conv1d(input_tensor, kernel_tensor, stride)
output_tensor_torch = F.conv1d(input_tensor, kernel_tensor)

print(output_tensor.shape)
print(output_tensor_torch.shape)
print(torch.allclose(output_tensor, output_tensor_torch))