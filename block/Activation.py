import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLu(nn.Moduel):
    def __init__(self):
        super(ReLu, self).__init__()
        
    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0))
    
    
class Sigmoid(nn.Module):
    
    def __init__(self):
        super(Sigmoid, self).__init__()
        
    def forward(self, x):
        return 1/1+torch.exp(-x)
    
class Softmax(nn.Module):
    
    def __init__(self):
        super(Softmax, self).__init__()
        
    def forward(self, x):
        return torch.exp(x)/torch.sum(torch.exp(x), dim=-1, keepdim=True)
    
class Silu(nn.Module):
    
    def __init__(self):
        super(Silu, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)