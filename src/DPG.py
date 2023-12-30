import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

class DPG(nn.Module):
    def __init__(self,n_agent,din=11,hidden_dim=128,dout=11):
        super(DPG, self).__init__()
        self.fc1 = nn.Linear(din, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,dout) 
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(h))

 