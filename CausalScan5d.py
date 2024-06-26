import torch
import torch.cuda as cuda
import torch.utils.cpp_extension as ext
import os

script_dir = os.path.dirname(__file__)
causal_scan_kernel = ext.load('extCausalScan4d', [
    os.path.join(script_dir, 'CausalScan5d.' + ('cu' if cuda.is_available() else 'cpp'))
]) 

class CausalScan(torch.autograd.Function):
    '''
    Formula:
    h(1) = a(1) * z         + b(1) * x(1)
    h(2) = a(2) * h(1)      + b(2) * x(2)
    ...
    h(n) = a(n) * h(n-1)    + b(n) * x(n)
    
    y(1) = c(1) * h(1)
    ...
    y(n) = c(n) * h(n)

    Args:
    x : (b, l, h, d)
    h : (b, 1, h, d, n)
    A : (b, l, h, d, n)
    B : (b, l, h, d, n)
    C : (b, l, h, d, n)

    Return:
    y : (b, l, d)
    h : (b, 1, d, n)
    '''
    @staticmethod
    def forward(ctx, x, h, A, B, C):
        (x, h, A, B, C) = [item.contiguous() for item in [x, h, A, B, C]]
        x = x.unsqueeze(-1)
        for item in [x, h, A, B, C]:
            assert len(item.shape) == 5
            assert item.size(0) == 1 or item.size(0) == h.size(0)
            assert item.size(1) == 1 or item.size(1) == x.size(1)
            assert item.size(2) == 1 or item.size(2) == h.size(2)
            assert item.size(3) == 1 or item.size(3) == h.size(3)
            assert item.size(4) == 1 or item.size(4) == h.size(4)
        assert h.size(1) == 1, 'hidden_state size should be one'
        
        length = x.size(1)
        group_size = 1023   # Must match the size in kernel.
        h = torch.repeat_interleave(h, (length+group_size-1)//group_size+1, dim=1)
        x = causal_scan_kernel.forward(x, h, A, B, C)
        ctx.save_for_backward(x, h, A, B, C)
        return x.squeeze(-1), h[:,-1:]

    @staticmethod
    def backward(ctx, gradO, gradZO):
        x, h, A, B, C = ctx.saved_variables
        gradx, gradh, gradA, gradB, gradC = causal_scan_kernel.backward(gradO, x, h, A, B, C)
        return gradx, gradh, gradA, gradB, gradC

if __name__ == "__main__":
    device = torch.device("cuda")
    Z = torch.tensor([
        [[[1,1,1,1]]]
    ], device=device, dtype=torch.float)
    A = torch.tensor([
        [[[2]]],
        [[[2]]]
    ], device=device, dtype=torch.float)
    B = torch.tensor([
        [[[3,3,3,3]]],
        [[[3,3,3,3]]]
    ], device=device, dtype=torch.float)
    X = torch.tensor([
        [[[4]]],
        [[[4]]]
    ], device=device, dtype=torch.float)
    C = torch.tensor([
        [[[5,5,5,5]]],
        [[[5,5,5,5]]],
    ], device=device, dtype=torch.float)
    (Z, A, B, X, C) = [
       item.unsqueeze(0)
        for item in [Z, A, B, X, C]
    ]
    (A, B, X, C) = [
        torch.repeat_interleave(item, 2, dim=1)
        for item in [A, B, X, C]
    ]
    (Z, A, B, X) = [
        torch.repeat_interleave(item, 2, dim=2)
        for item in [Z, A, B, X]
    ]
    print(CausalScan.apply(Z, A, B, X, C))
