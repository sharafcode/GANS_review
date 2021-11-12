import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.set_params(module)

    @staticmethod
    def set_params(module):
        """u, v
        """
        w = module.weight
        height = w.size(0)
        width = w.view(w.size(0), -1).shape[-1]

        ## Create the U & V vectors
        u = nn.Parameter(torch.randn(height, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(1, width), requires_grad=False)
        module.register_buffer('u', u)
        module.register_buffer('v', v)

    def forward(self, *args):
        u, v, w = self.module.u, self.module.v, self.module.weight
        height = w.size(0)
        
        eps = 1e-12 ## Smoothing
        
        for i in range(self.power_iterations):  # Power iteration
            v = w.view(height, -1).t() @ u
            v /= (v.norm(p=2) + eps) # L2-normalize
            u = w.view(height, -1) @ v
            u /= (u.norm(p=2) + eps) # L2-normalize

        w.data /= (u.t() @ w.view(height, -1) @ v).data  # Spectral normalization
        return self.module.forward(*args)
