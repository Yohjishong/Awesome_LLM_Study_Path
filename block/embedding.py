import torch
import torch.nn as nn
import torch.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embedding, base=10000):
        self.dim = dim
        self.max_position_embedding = max_position_embedding
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float()/dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cso_sin_cache(max_position_embedding)
    
    def _set_cso_sin_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer("cos", cos)
        self.register_buffer('sin', sin)

    def forward(self, seq_len):
        if seq_len > self.max_seq_len_cache:
            self._set_cso_sin_cache(seq_len)
        return self.cos[:seq_len], self.sin[:seq_len]
    
def apply_Rope(q, k, sin, cos, position_ids):
    # q, k: [batch_size, seq_len, head_nums, head_dim]
    # sin, cos [seq_len, head_dim]
    # position_ids: [batch_size, seq_len]
    batch_size, seq_len, head_nums, head_dim = q.size()
    # cos.index_select(0, position_ids.reshape[-1]). 选出position对应位置的cos值
    cos = cos.index_select(0, position_ids.reshape[-1]).reshape(batch_size, seq_len, 1, head_dim)
    sin = sin.index_select(0, position_ids.reshape[-1]).reshape(batch_size, seq_len, 1, head_dim)
    q_rotate = q * cos + rotate_half(q) * sin
    k_rotate = k * cos + rotate_half(k) * sin
    return q_rotate, k_rotate

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis(dim, seq_len, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype = inv_freq.dtype, device=inv_freq.device)
    # freqs [seq_len, dim // 2] 
    freqs = torch.outer(t, inv_freq).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    xq_emb = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_emb = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_emb.type_as(xq), xk_emb.type_as(xk)



    
