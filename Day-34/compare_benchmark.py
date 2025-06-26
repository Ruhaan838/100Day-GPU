import subprocess
import numpy as np
from matplotlib import pyplot as plt

sizes = [100*i for i in range(1, 11)]
cpu_times = []
gpu_times = []

subprocess.run("mpicc -o vec_add_cpu vec_add_cpu.c -O3", shell=True, check=True)
subprocess.run("nvcc -o vec_add_gpu vec_add_gpu.cu -O3", shell=True, check=True)

for size in sizes:
    print(f"Running for size: {size}")
    
    result_cpu = subprocess.run(
        f"mpirun --oversubscribe --allow-run-as-root -np 4 ./vec_add_cpu {size}",
        shell=True, capture_output=True, text=True
    )
    
    print(f"CPU Output: {result_cpu.stdout.strip}")
    print("Error (if any):", result_cpu.stderr)
    
    output = result_cpu.stdout.strip().split()
    if len(output) >= 2:
        cpu_time = float(output[-2])
    else:
        print("Error: Failed to output:", result_cpu.stdout)
        cpu_time = float('inf')
    cpu_times.append(cpu_time)
    
    result_gpu = subprocess.run(
        f"./vec_add_gpu {size}",
        shell=True, capture_output=True, text=True
    )
    
    print(f"GPU Output: {result_gpu.stdout}")
    print("Error (if any):", result_gpu.stderr)
    
    output = result_gpu.stdout.strip().split()
    if len(output) >= 2:
        gpu_time = float(output[-2])
    else:
        print("Error: Failed to output:", result_gpu.stdout)
        gpu_time = float('inf')
    gpu_times.append(gpu_time)

plt.figure(figsize=(10, 5))
plt.plot(sizes, cpu_times, label='CPU Time', marker='o')
plt.plot(sizes, gpu_times, label='GPU Time', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Vector Addition Time')
plt.legend()
plt.grid()
plt.savefig('benchmark_comparison.png')
plt.show()
    