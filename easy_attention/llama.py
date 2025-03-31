import torch
import torch.nn as nn
import torch.nn.functional as F


class LLamaMHA_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.n_embd, config.n_embd)).view(1, 1, config.n_embd, config.n_embd))
    
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
        attn = q @ k.transpose(-1, -2) *(1.0 / (self.n_embd // self.n_head) ** 0.5)
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(attn)
        if kv_cache:
            return out, kv_cache
        return out

        