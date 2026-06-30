import torch
import torch.nn.functional as F

def gumbel_softmax_sample(logits, tau_noise=1.0, eps=1e-20):
    # Step 1: sample Gumbel noise
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    
    # Step 2: scale noise by temperature, add it, softmax
    y = F.softmax(logits + g/tau_noise, dim=-1)
    return y

def gumbel_softmax(logits, tau_noise=1.0, hard=False):
    # Soft sample
    y_soft = gumbel_softmax_sample(logits, tau_noise)

    if hard:
        # Step 3: discrete sample via one-hot argmax
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # Step 4: straight-through trick
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y
