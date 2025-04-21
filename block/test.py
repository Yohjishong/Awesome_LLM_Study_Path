import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from rich import print


"""

batch_size = 16
seq_len = 32 # seq_len
l = 8 # 每个块大小
d = 4 # 每次移动的个数
dim = 8
heads = 2
head_dim = dim // heads

max_idx = (seq_len-l)//d
#print(max_idx) # 7个块
max_idx = max_idx + 1

# 初始化
X = torch.randn(batch_size, seq_len, dim)
Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)
Wv = torch.randn(dim, dim)
q = X @ Wq
k = X @ Wk
v = X @ Wv

# 压缩KV [l, 1] l个块混合在一起
# position_emb [l, dim] 位置编码
Wk_cmp = torch.randn(l, 1)
Wv_cmp = torch.randn(l, 1)
position_emb = torch.randn(l, dim)
o_proj = torch.randn(dim, dim)

k_cmp = []
v_cmp = []
for i in range(max_idx):
    # [batch_size, l, dim]
    k_cur = k[:, i*d:i*d+l, :] + position_emb.unsqueeze(0)
    v_cur = v[:, i*d:i*d+l, :]  + position_emb.unsqueeze(0)
    # [batch_size, dim, l] * [l, 1] = [batch_size, dim, 1]
    k_cur = k_cur.transpose(1, 2) @ Wk_cmp 
    v_cur = v_cur.transpose(1, 2) @ Wv_cmp
    k_cmp.append(k_cur)
    v_cmp.append(v_cur)

# [batch_size, max_idx, dim]
k_cmp = torch.cat(k_cmp, dim=2).transpose(-1, -2)
v_cmp = torch.cat(v_cmp, dim=2).transpose(-1, -2)

# 多头注意力
q = q.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
k_cmp = k_cmp.view(batch_size, max_idx, heads, head_dim).transpose(1, 2)
v_cmp = v_cmp.view(batch_size, max_idx, heads, head_dim).transpose(1, 2)
# [batch_size, heads, seq_len, head_dim] * [batch_size, heads, head_dim, max_idx] = [batch_size, heads, seq_len, max_idx]
attn = q @ k_cmp.transpose(-1, -2) * (1.0 /(math.sqrt(dim)))
mask = torch.tril(torch.ones(seq_len, max_idx)).view(1, 1, seq_len, max_idx)
attn = attn.masked_fill(mask[:, :, :seq_len, :max_idx] == 0, float('-inf'))
attn = F.softmax(attn, dim=-1)
# [batch_size, heads, seq_len, max_idx] * [batch_size, heads, max_idx, dim] = [batch_size, heads, seq_len, dim]
attn = attn @ v_cmp
attn = attn.transpose(1, 2).reshape(batch_size, seq_len, dim)
# [batch_size, seq_len, dim] * [dim, dim] = [batch_size, seq_len, dim]
attn = attn @ o_proj


p_cmp = torch.randn(batch_size, heads, seq_len, dim)
p_slc = p_cmp.sum(dim = 1) # 在head维度上进行合并
print(f"p_cmp size: {p_cmp.shape}") # torch.Size([16, 2, 32, 8])
print(f"p_slc size: {p_slc.shape}") # torch.Size([16, 32, 8])

select_top = 2

"""


seq_len = 128
print(f"假定输入长度为{seq_len}")
l_cmp = 8
print(f"定义为将{l_cmp}个kv块压缩为一个")
l_slc = 16
print(f"定义选择的块是{l_slc}个块的集合体")
d = 4
print(f"定义压缩时步长为{d}")

max_idx = (seq_len-l_cmp) // d
print('压缩块一共有', max_idx+1)
# max_idx+1 维度的 p_t_cmp
p_t_cmp = torch.randn(max_idx+1)
# m最大值
m_max = l_slc // d 
# n最小值
n_max = l_cmp // d 
# 第一个问题：8个块压缩成一个，那么我选择块是16吗，也就是说选择一个块就相当于选择了原来的16个块
max_j = seq_len // l_slc
p_t_slc = torch.zeros(max_j)
print('选择块一共有', max_j) # 8 一共有8个选择块
for j in range(max_j):
    for m in range(m_max):
        for n in range(n_max):
            # 这里index应该有个mod
            # index = (m_max*j-m-n) % (max_idx+1)
            index = (m_max*j+m+n)
            if 0 <= index < (max_idx+1):
                print(f"有效加和: 当m={m}, n={n}, index={index}, 加入p_t_slc[{j}]")
                p_t_slc[j] += p_t_cmp[index] 

print(p_t_slc)
