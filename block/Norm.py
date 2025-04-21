import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weght = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return x * norm
    
    def forward(self, x):
        x = self._norm(x)
        return self.weght * x
    
    
class LayerNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        mean_value = x.mean(-1, keepdim=True)
        var_value = x.var(-1, keepdim=True)
        x = (x-mean_value)/torch.sqrt(var_value+self.eps)
        x = self.gamma * x + self.beta
        return x
    

        
        
        


        
