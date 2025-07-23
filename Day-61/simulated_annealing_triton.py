import math
import torch
import triton
import triton.language as tl

@triton.jit
def objective_function_kernel(x_ptr, out_ptr):
    idx = tl.program_id(0)
    x = tl.load(x_ptr + idx)
    res = (x - 3.0) * (x - 3.0)
    
    tl.store(out_ptr + idx, res)
    
def objective_function(x:torch.Tensor):
    reuslt = torch.empty_like(x)
    
    grid = lambda meta : (x.numel(),)
    objective_function_kernel[grid](x, reuslt)
    return reuslt

def simulated_annealing(num_steps=1000, inital_x=0.0, init_temp=10.0, alpha=0.99):
    
    x = inital_x
    
    current_val = objective_function(torch.tensor([x], dtype=torch.float32, device='cuda')).item()
    best_x = x
    best_val = current_val
    temp = init_temp
    
    for _ in range(num_steps):
        candidate = x + (torch.rand(1).item() - 0.5) * 2.0
        cacdidate_val = objective_function(torch.tensor([candidate], dtype=torch.float32, device='cuda')).item()
        delta = cacdidate_val - current_val
        
        if delta < 0 or torch.rand(1).item() < math.exp(-delta / temp):
            x = candidate
            current_val = cacdidate_val
            
            if current_val < best_val:
                best_x = x
                best_val = current_val
        temp *= alpha
    
    return best_x, best_val

if __name__ == "__main__":
    best_sol, best_val = simulated_annealing(num_steps=1000, inital_x=0.0, init_temp=10.0, alpha=0.99)
    print(f"Best solution: x = {best_sol}, f(x) = {best_val}")
    
    