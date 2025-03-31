import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


d, k = 10, 10
W_rank = 2
W = torch.randn(d, W_rank) @ torch.randn(W_rank, k)
W_rank = np.linalg.matrix_rank(W) 
print(f"{W_rank=}")

top_indices = [[1,2,3], [4,5,6], [7,8,9]]
new_token_pos = torch.arange(0, 3, 3/32).type(torch.int64)
print(new_token_pos)
print(len(new_token_pos))

decoded = ['kill', 'x', 'ca', 'fro']

decoded_str = "".join(decoded).replace('</w>', ' ')[:-1]
print(decoded_str)