import subprocess
import numpy as np
import torch
import psutil


def get_memory_info():
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3), memory.total / (1024 ** 3)

def estimate_memory_usage(N, M):
    nnz = (N * M) // 3
    memory_gb = (nnz * (2 * 8 + 4)) / (1024 ** 3)
    return memory_gb

def verify_results(cuda_output_file, torch_output, N):
    cuda_results = []
    with open(cuda_output_file, 'r', encoding="utf-8") as f:
        cuda_results = [float(line.strip()) for line in f if line.strip()]
    
    torch_results = torch_output.cpu().numpy().flatten().tolist()
    
    max_diff = 0
    max_rel_diff = 0
    to_lerance = 1e-5
    
    for i, (cuda_val, torch_val) in enumerate(zip(cuda_results, torch_results)):
        abs_diff = abs(cuda_val - torch_val)
        max_diff = max(max_diff, abs_diff)
        
        if abs(cuda_val) > 1e-10:
            relative_diff = abs_diff / abs(cuda_val)
            max_rel_diff = max(max_rel_diff, relative_diff)
        
        if abs_diff > to_lerance:
            print(f"Mishmatch at index {i}: CUDA = {cuda_val}, Pytorch = {torch_val}")
            print(f"Abs Difference: {abs_diff}")
            return False
    
    print(f"""Results Match within to_laerance of {to_lerance}
              Max abs diff: {max_diff}
              Max relative diff: {max_rel_diff}""")
    
    return True

def create_spare_mat_and_vec(N, M):
    estimate_mem = estimate_memory_usage(N, M)
    _, total_memory = get_memory_info()
    if estimate_mem > total_memory * 0.7:
        raise MemoryError(f"Esitmate memory used {estimate_mem} GB you have {total_memory} GB")
    
    chunk_size = 1000000
    indx = []
    values = []
    
    for i in range(0, N, chunk_size // M):
        end_i = min(i + chunk_size // M, N)
        for j in range(M):
            for ii in range(i, end_i):
                if (ii + j) % 3 == 0:
                    indx.append([ii, j])
                    values.append(float(ii + j))

        if len(indx) > chunk_size:
            indices_tensor = torch.tensor(indx, dtype=torch.long).t()
            values_tensor = torch.tensor(values, dtype=torch.float32)
            if 'final_indices' not in locals():
                final_indices = indices_tensor
                final_values = values_tensor
            else:
                final_indices = torch.cat([final_indices, indices_tensor], dim=1) # type: ignore
                final_values = torch.cat([final_values, values_tensor]) # type: ignore
            indx = []
            values = []

    if indx:
        indices_tensor = torch.tensor(indx, dtype=torch.long).t()
        values_tensor = torch.tensor(values, dtype=torch.float32)
        if 'final_indices' not in locals():
            final_indices = indices_tensor
            final_values = values_tensor
        else:
            final_indices = torch.cat([final_indices, indices_tensor], dim=1) # type: ignore
            final_values = torch.cat([final_values, values_tensor]) # type: ignore

    A = torch.sparse_coo_tensor(final_indices, final_values, (N, M)) # type: ignore
    X = torch.ones(M, 1, dtype=torch.float32)

    return A, X

def run_torch_program(N, M):
    devide = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    times = 0
    
    A, X = create_spare_mat_and_vec(N, M)
    A = A.to(devide).coalesce()
    X = X.to(devide)
    
    output_torch = torch.sparse.mm(A, X)
    
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record() # type: ignore
    output_torch = torch.sparse.mm(A, X)
    end.record() # type: ignore
    
    torch.cuda.synchronize()
    times = start.elapsed_time(end)
        
    del A, X, output_torch
    torch.cuda.empty_cache()
    
    return times

N, M = 1000, 1000
print(f"Estimated memory use:{estimate_memory_usage(N, M)} GB")
used_mem, total_mem = get_memory_info()
print(f"current memory usage: {used_mem:.2f} GB / {total_mem:.2f} GB")
print('cuda' if torch.cuda.is_available() else 'cpu')
torch_time = run_torch_program(N , M)

print(f"Pytorch Sparse implementation time:{torch_time} seonds")



