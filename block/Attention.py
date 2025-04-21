import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA_Attention(nn.Module):
    def __init__(self, seq_len, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
    
    def forward(self, x, kv_cache = False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        if kv_cache:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-1)
            v  =torch.cat([past_v, v], dim=-1)
            kv_cache = (k, v)
        attn = q @ k.transpose(-1, -2) *(1.0 / math.sqrt(self.n_embd//self.n_head))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(attn)
        if kv_cache:
            return out, kv_cache
        return out


class GroupQueryAttention(nn.Module):
    
    def __init__(self, n_embd, n_heads, n_groups):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_groups = n_groups
        assert self.n_embd % self.n_heads == 0, "n_embd must be divisible by n_head"
        assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd//n_heads*n_groups)
        self.v_attn = nn.Linear(n_embd, n_embd//n_heads*n_groups)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.kv_cache = None
        self.register_buffer("bias", torch.tril(torch.ones(n_embd, n_embd)).view(1, 1, n_embd, n_embd))
    
    def forward(self, x):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, self.n_heads, T, C // self.n_heads).transpose(1, 2)
        k = self.k_attn(x).view(B, self.n_groups, T, C // self.n_heads).transpose(1, 2)
        v = self.v_attn(x).view(B, self.n_groups, T, C // self.n_heads).transpose(1, 2)
        if self.kv_cache:
            past_k, past_v = self.kv_cache
            k = torch.cat([past_k, k], dim=-1)
            v = torch.cat([past_v, v], dim=-1)
            self.kv_cache = (k, v)
        k = k.repeat_interleave(self.n_heads//self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_heads//self.n_groups, dim=1)
        attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.n_embd//self.n_heads))
        attn = attn.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(attn)
        if self.kv_cache:
            return out, self.kv_cache
        return out
    


    






        