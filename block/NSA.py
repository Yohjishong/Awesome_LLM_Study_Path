import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from rich import print

@dataclass
class SparseAttentionConfig:
    batch_size: int
    seq_len: int
    l_cmp: int
    l_slc: int
    l_win: int
    dim: int
    heads: int
    stride: int
    top_k: int

# MHA 版本
class Naive_Sparse_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # batch 大小
        self.batch_size = config.batch_size
        # 输入的序列长度
        self.seq_len = config.seq_len
        # 多少个块被压缩为一个块
        self.l_cmp = config.l_cmp
        # 一个选择块里面含有多少个小块
        self.l_slc = config.l_slc
        # 滑动窗口注意力中滑动长度
        self.l_win = config.l_win
        # 每个具体序列的嵌入维度
        self.dim = config.dim
        # 多头 头的个数
        self.heads = config.heads
        # 压缩时候, 每隔多少个块压缩一个, 注意这个与self.l_cmp的区别
        self.stride = config.stride
        self.head_dim = self.dim // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # 压缩块一共有多少个, 要加 1 是因为这么算会少算第一个块，需要加上
        self.kv_cmp_len = (self.seq_len - self.l_cmp) // self.stride + 1
        # 一共有多少个选择块
        self.kv_slc_len = self.seq_len // self.l_slc
        # 选择 top 多少的块 
        self.top_k = config.top_k
        assert self.kv_slc_len > self.top_k, "块不够选啦"
        # 压缩块的 mask 矩阵, register_buffer:不进入反向传播
        self.register_buffer('cmp_mask', torch.tril(torch.ones(self.seq_len, self.kv_cmp_len), diagonal=-1).view(1, 1, self.seq_len, self.kv_cmp_len))
        # 从输入 x 到 qkv 的线性层
        self.c_attn = nn.Linear(self.dim, 3*self.dim)
        # 最后一层线性层
        self.c_proj = nn.Linear(self.dim, self.dim)
        # 压缩 k 的 mlp
        self.k_cmp_mlp = nn.Linear(self.l_cmp, 1)
        # 压缩 v 的 mlp
        self.v_cmp_mlp = nn.Linear(self.l_cmp, 1)
        # gate 门控线性层
        self.gate_choose = nn.Linear(self.dim, 3)

    def kv_cmp(self, k, v):
        k_cmp = []
        v_cmp = []
        # 对应原论文(t-l) // d
        for i in range(self.kv_cmp_len):
            # 每次压缩块，每隔self.stride个块将self.l_cmp个块压缩成一个块
            # [batch_size, l_cmp, dim]
            k_cur = k[:, i*self.stride:i*self.stride+self.l_cmp, :]
            # [batch_size, dim, l_cmp] * [l_cmp, 1] = [batch_size, dim, 1] 
            k_cur = self.k_cmp_mlp(k_cur.transpose(1, 2)) 
            # k, v是对称的，同理操作
            v_cur = v[:, i*self.stride:i*self.stride+self.l_cmp, :]
            v_cur = self.v_cmp_mlp(v_cur.transpose(1, 2))
            k_cmp.append(k_cur)
            v_cmp.append(v_cur)
        # [batch_size, kv_cmp_len, dim]
        # 结果合并变为压缩后的k,v返回
        k_cmp = torch.cat(k_cmp, dim=-1).transpose(-1, -2)
        v_cmp = torch.cat(v_cmp, dim=-1).transpose(-1, -2)
        return k_cmp, v_cmp
            

    def get_window_attn(self, q, k, v):
        # q, v: [batch_size, heads, seq_len, head_dim]
        # score [batch_size, heads, seq_len, seq_len]
        attn_score  = torch.full((self.batch_size, self.heads, self.seq_len, self.seq_len), float('-inf'), device=q.device)
        output  = torch.full((self.batch_size, self.heads, self.seq_len, self.head_dim), float(0), device=q.device)
        for i in range(self.seq_len):
            # 这里主要是根据原文的图，会发现这里的滑动窗口指的会是 q 之前的 l_win 个 KV 块
            start = max(0, i-self.l_win)
            end = i+1
            # 解释一下这里的乘法, 对于序列中 index 为 i 的 query, 找到对应的 k 的滑动窗口 
            # 然后有一个[batch_size, heads, 1, dim]*[batch_size, heads, dim, l_win] = batch_size, heads, 1, l_win]
            attn_score[:, :, i:i+1, start:end] = torch.matmul(q[:, :, i:i+1, :], (k[:, :, start:end, :].transpose(-1, -2)))
        # [batch_size, heads, seq_len, seq_len]
        attn_score = attn_score * self.scale
        attn_score = F.softmax(attn_score, dim=-1)
        for i in range(self.seq_len):
            start = max(0, i-self.l_win)
            end = i+1
            # 解释一下这里的乘法, 对于序列中 index 为 i 的 attn_score, 找到对应的 v 的滑动窗口 
            # 然后有一个[batch_size, heads, 1, l_win]*[batch_size, heads, l_win, dim] = batch_size, heads, 1, dim]
            output[:, :, i:i+1, :] = torch.matmul(attn_score[:, :, i:i+1, start:end], v[:, :, start:end, :])
        return output.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.dim)


    def forward(self, x):
        print(f"输出 x 维度: {x.shape}")
        # [batch_size, seq_len, dim]
        batch_size, seq_len, C = x.size()
        assert batch_size==self.batch_size, "batch_size 不一致"
        assert C == self.dim, "dim 不一致"
        # [batch_size, seq_len, dim]
        q, k, v = self.c_attn(x).split(self.dim, dim=-1)
        # [batch_size, heads, seq_len, head_dim]
        # 压缩注意力
        # [batch_size, kv_cmp_len, dim]
        k_cmp, v_cmp = self.kv_cmp(k, v)
        print(f"压缩kv块维度: {k_cmp.size()}")
        # 多头切割
        q = q.view(self.batch_size, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(self.batch_size, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(self.batch_size, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_cmp = k_cmp.view(self.batch_size, self.kv_cmp_len, self.heads, self.head_dim).transpose(1, 2)
        v_cmp = v_cmp.view(self.batch_size, self.kv_cmp_len, self.heads, self.head_dim).transpose(1, 2)

        # [batch_size, heads, seq_len, head_dim] * [batch_size, heads, head_dim, max_idx] = [batch_size, heads, seq_len, max_idx]
        cmp_attn = q @ k_cmp.transpose(-1, -2) * self.scale
        cmp_attn = cmp_attn.masked_fill(self.cmp_mask[:, :, :self.seq_len, :self.kv_cmp_len] == 0, float('-inf'))
        p_t_cmp = F.softmax(cmp_attn, dim=-1)
        print(f"压缩注意力分数维度(用作选择注意力): {p_t_cmp.shape}")
        # [batch_size, heads, seq_len, max_idx] * [batch_size, heads, max_idx, dim] = [batch_size, heads, seq_len, dim]
        cmp_attn = p_t_cmp @ v_cmp
        cmp_attn = cmp_attn.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.dim)
        print(f"压缩注意力最后输出维度: {cmp_attn.shape}")

        # 选择注意力
        # softmax(q * k_cmp^T) = score p_t_cmp
        # p_t_cmp [batch_size, heads, seq_len, kv_cmp_len]
        # 公式里面的外层循环
        m_max = self.l_slc // self.stride 
        # 公式里面的内层循环
        n_max = self.l_cmp // self.stride 
        p_t_slc = torch.full((self.batch_size, self.heads, self.seq_len, self.kv_slc_len), float(0), device=p_t_cmp.device)

        print(f'选择块一共有{self.kv_slc_len}个')
        # 笔者声明, 总感觉这个选择块这里不太对, 但我是基本按照 NSA 的官方报告写的, 但是感觉官方报告这里写的就不清楚
        for j in range(self.kv_slc_len):
            for m in range(m_max):
                for n in range(n_max):
                    index = (m_max*j+m+n)
                    if 0 <= index < self.kv_cmp_len:
                        #print(f"有效加和: 当m={m}, n={n}, index={index}, 加入p_t_slc[{j}]")
                        p_t_slc[:, :, :, j:j+1] += p_t_cmp[:, :, :, index:index+1] 
        # 这里的 idx 也就是 topk 的索引

        _, idx = torch.topk(p_t_slc, dim=-1, k=self.top_k)
        # torch.Size([32, 8, 128, 2])
        # 索引开始
        idx_slc_start = idx * self.l_slc
        # print('sta: ', idx_slc_start.size())
        # 索引结束
        idx_slc_end = idx * self.l_slc + self.l_slc
        # print('end: ', idx_slc_end.size())
        k_slc = torch.full((self.batch_size, self.heads, self.seq_len, self.l_slc * self.top_k, self.head_dim), float(0), device=k.device)
        v_slc = torch.full((self.batch_size, self.heads, self.seq_len, self.l_slc * self.top_k, self.head_dim), float(0), device=v.device)
        # 最复杂的一集, 慢慢解释, 这里在原文应该是直接算一个普通的注意力然后再mask, 用 kernel 进行优化
        # 但是我既不会 Cuda 也不会 Triton, 只能用最笨的方法做这个 
        for i in range(self.batch_size):
            for j in range(self.heads):
                for t in range(self.seq_len):
                    for p in range(self.top_k):
                        # 第 i 个 batch, 第 j 个 head, 第 t 个序列, k 的 self.l_slc 个是 topk 里面第 p 个开始和第 p 个结束的地方
                        # 就可以理解成我们选择了第 p 个 topk 的选择 KV 块，然后把它所对应的原 KV 块提取出来
                        k_slc[i:i+1, j:j+1, t:t+1, p * self.l_slc : (p+1) * self.l_slc, :] = k[i:i+1, j:j+1, idx_slc_start[i, j, t, p ]:idx_slc_end[i, j, t, p ], :].reshape(1, 1, 1, self.l_slc, self.head_dim)
                        v_slc[i:i+1, j:j+1, t:t+1, p * self.l_slc : (p+1) * self.l_slc, :] = v[i:i+1, j:j+1, idx_slc_start[i, j, t, p ]:idx_slc_end[i, j, t, p ], :].reshape(1, 1, 1, self.l_slc, self.head_dim)
        # 在 head 上聚合
        # [batch_size, seq_len, heads(1), select_kv_len, head_dim]
        V_slc = v_slc.sum(dim = 1, keepdim = True).transpose(1,2)
        K_slc = k_slc.sum(dim = 1, keepdim = True).transpose(1, 2)
        cls_attn = torch.zeros(self.batch_size, self.seq_len, self.dim).to(device=q.device)
        for i in range(seq_len):
            q_slc_j = q[:, :, i:i+1, :]
            # [batch_size, heads(1), select_kv_len, head_dim]
            K_slc_j = K_slc[:, i, :, :, :].repeat(1, self.heads, 1, 1)
            V_slc_j = V_slc[:, i, :, :, :].repeat(1, self.heads, 1, 1)
            # [batch_size, heads, 1, head_dim] * [batch_size, heads, head_dim, select_kv_len] = [batch_size, heads, 1, select_kv_len]
            attn_score_j = q_slc_j @ K_slc_j.transpose(2,3) * self.scale
            # 这里应该有个mask才对，但是以博主的猪脑子实在是想不通这里mask应该怎么设计了(sadddddd)
            p_slc_j = F.softmax(attn_score_j, dim = -1) 
            # [batch_size, heads, 1, select_kv_len] * [batch_size, heads, select_kv_len, head_dim] = [batch_size, heads, 1, head_dim]
            out_slc_j = (p_slc_j @ V_slc_j).contiguous().transpose(1, 2).view(self.batch_size, 1, self.dim)
            cls_attn[:, j:j+1, :] = out_slc_j
        print(f"选择注意力最后输出维度: {cls_attn.shape}")

        # 滑动窗口注意力

        win_attn = self.get_window_attn(q, k, v)
        print(f"滑动窗口注意力最后输出维度: {win_attn.shape}")

        # 最后一步了, 用一个 Gate Fusion
        # # [batch_size, seq_len, 3]
        gate = self.gate_choose(x)
        o_fused = gate[:, :, 0].unsqueeze(2) * cmp_attn + gate[:, :, 1].unsqueeze(2) * cls_attn + gate[:, :, 2].unsqueeze(2) * win_attn
        print(f"最后输出注意力维度: {o_fused.shape}")
        return o_fused




config = SparseAttentionConfig(
    batch_size=32,
    seq_len=128,
    l_cmp=8,
    l_slc=16,
    l_win=64,
    dim=512,
    heads=8,
    stride=4,
    top_k=2,
)

nsa_attention =  Naive_Sparse_Attention(config)
# [batch_size, seq_len, dim]
x = torch.randn(32, 128, 512)
nsa_attention(x)








