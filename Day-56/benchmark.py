import os
import subprocess
import torch
import matplotlib.pyplot as plt

cuda_file = "temp.cu"
exec_path = "./ddpm_kernel"

os.system(f"nvcc -O2 -o {exec_path} {cuda_file}")

output = subprocess.check_output([exec_path]).decode("utf-8")
cuda_time = float(output.strip().split(":")[-1].strip().split()[0])
print(f"CUDA Kernel Time: {cuda_time:.4f} ms")

def run_benchmark(count=1000, shape=(3,1024,1024), a=0.9, b=0.1, abar=0.5, dev="cuda"):
    inv_sqrt_a = 1 / torch.sqrt(torch.tensor(a, device=dev))
    scale = b / torch.sqrt(torch.tensor(1 - abar, device=dev))

    vec1 = torch.empty(shape, device=dev)
    vec2 = torch.empty(shape, device=dev)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        vec1.copy_(torch.randn(shape, device=dev))
        vec2.copy_(torch.randn(shape, device=dev))
        _ = inv_sqrt_a * (vec1 - scale * vec2)
        torch.cuda.synchronize()

    time_sum = 0.0
    for _ in range(count):
        vec1.copy_(torch.randn(shape, device=dev))
        vec2.copy_(torch.randn(shape, device=dev))
        torch.cuda.synchronize()
        start.record()
        _ = inv_sqrt_a * (vec1 - scale * vec2)
        stop.record()
        torch.cuda.synchronize()
        time_sum += start.elapsed_time(stop)

    return time_sum / count

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_time = run_benchmark(dev=device)

print(f"PyTorch Time: {torch_time:.4f} ms")

labels = ["CUDA", "PyTorch"]
times = [cuda_time, torch_time]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=["blue", "orange"])
plt.xlabel("Method")
plt.ylabel("Time (ms)")
plt.title("DDPM Update Performance")
plt.yscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, t * 1.05, f"{t:.4f} ms", ha="center", fontsize=10)
plt.show()

print(f"Speedup: {torch_time / cuda_time:.2f}x")
