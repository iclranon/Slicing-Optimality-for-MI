# Sample code for two correlated Gaussians

from optimal_SI import optimal_SI
import torch
import numpy as np
import math

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    rho=.9
    X,Y=sample_correlated_gaussian(rho=rho,dim=1,batch_size=500)
    print(optimal_SI(X,Y,num_slices=100))
