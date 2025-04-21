import torch
import torch.nn as nn
import torch.nn.functional as F
from Norm import RMSNorm

class LLamaMLP(nn.Module):
    
    def __init__(self, dim, hidden_dim=None, multiple=256):
        super().__init__()
        self.dim = dim
        if not hidden_dim:
            hidden_dim = dim * 4
        self.hidden_dim = hidden_dim * 2 // 3
        self.hidden_dim = (self.hidden_dim +multiple -1) // multiple * multiple
        self.gate_proj = nn.Linear(self.dim, self.hidden_dim)
        self.up_proj = nn.Linear(self.dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, self.dim)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(up * gate)
        return out
    
            