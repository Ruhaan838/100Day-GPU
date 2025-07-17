import os
import subprocess
import time 
import torch

cuda_file = "ddpm.cu"
exe = "./ddpm"

cmd = f"nvcc -arch=sm_75 -o {exe} {cuda_file}"
os.system(cmd)

cuda_out = subprocess.check_output([exe]).decoder("utf-8")
cuda_time = float(cuda_out.split(":")[-1].strip().split()[0])
print(f"CUDA Kernel Time {cuda_time} ms")

def ddpm(x, eps_pred, alpha, beta, alpha_bar):
    inv_sqrt_alpha = 1 / torch.sqrt(torch.tensor(alpha, device=x.device))
    scale = beta / torch.sqrt(torch.tensor(1 - alpha_bar, device=x.device))
    return inv_sqrt_alpha * (x - scale * eps_pred)

device = "cuda" if torch.cuda.is_available() else "cpu"
shape = (3, 1024, 1024)
x = torch.randn(shape, device=device)
eps = torch.randn(shape, device=device)
alpha = 0.9
beta = 0.1
alpha_bar = 0.5

def run_torch(iter=10000):
    st = time.time()
    for _ in range(iter):
        _ = ddpm(x, eps, alpha, beta, alpha_bar)
        e = time.time() - st
        return (e / iter) * 1000.0
torch_time = run_torch()
print(f"Pytorch Time: {torch_time} ms")

print(f"CUDA time: {cuda_time}ms vs Torch Time {torch_time}ms")
print(f"Speedup {torch_time / cuda_time}x")
