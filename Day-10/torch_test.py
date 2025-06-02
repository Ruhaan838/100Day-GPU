import torch
from torch.nn import functional as F
import pandas as pd

def load_csv_to_tensor(file_path):
    df = pd.read_csv(file_path, header=None)
    return torch.tensor(df.values)

q = load_csv_to_tensor('query_out.csv')
k = load_csv_to_tensor('key_out.csv')
v = load_csv_to_tensor('value_out.csv')

exp_out = load_csv_to_tensor("output.csv")

dk = q.size(-1)
attention = torch.matmul(q, k.T) / (dk ** 0.5)

attention_weigths = F.softmax(attention, dim=-1)

actual_out = torch.matmul(attention_weigths, v)

check = torch.allclose(actual_out, exp_out, atol=1e-5)

print("Did the Output is correct? ", check)
