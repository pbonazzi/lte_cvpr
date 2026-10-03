from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.logic_layer import LogicLayer, GroupSum
from difflogic.conv_logic_layer import ConvLogicLayer


class CandidateWiring(nn.Module):
    """Learned wiring for a dense logic layer (bounded candidate pool, straight-through).

    Each of the layer's 2 * out_dim gate inputs picks one of k candidate inputs.
    Candidate 0 is the usual fixed random wire (every input used about equally,
    as in LogicLayer.get_connections), candidates 1..k-1 are further random
    inputs. Selection logits start equal, so argmax picks candidate 0 and the
    untrained model is exactly the fixed-wiring one. Forward always uses the
    argmax candidate, as at test time; the gradient flows through the softmax.
    Output: (batch, 2 * out_dim), first half the gates' A inputs, then the B inputs.
    """
    def __init__(self, in_dim: int, out_dim: int, k: int, device: str = "cuda"):
        super().__init__()
        n = 2 * out_dim
        wire = torch.randperm(in_dim)[torch.randperm(n) % in_dim]
        cand = torch.randint(in_dim, (n, k))
        cand[:, 0] = wire
        self.register_buffer("candidates", cand.to(device))
        self.logits = nn.Parameter(torch.zeros(n, k, device=device))

    def forward(self, x):
        pick = self.logits.argmax(-1, keepdim=True)
        if not self.training:
            return x[:, self.candidates.gather(1, pick).squeeze(1)]
        p = F.softmax(self.logits, dim=-1)
        w = torch.zeros_like(p).scatter_(1, pick, 1.0) - p.detach() + p
        return torch.einsum("bnk,nk->bn", x[:, self.candidates], w.to(x.dtype))


def dense_logic_layer(in_dim: int, out_dim: int, connection_candidates: int = 0, **kwargs):
    """A LogicLayer with fixed random wiring, or with learned wiring when connection_candidates > 0."""
    if connection_candidates <= 0:
        return LogicLayer(in_dim, out_dim, **kwargs)
    gates = LogicLayer(2 * out_dim, out_dim, **kwargs)
    with torch.no_grad():   # gate i reads inputs i and out_dim + i of the CandidateWiring output
        gates.indices_0.copy_(torch.arange(out_dim))
        gates.indices_1.copy_(torch.arange(out_dim, 2 * out_dim))
    if gates.implementation == "cuda":
        gates._build_reverse_adjacency()
    return nn.Sequential(CandidateWiring(in_dim, out_dim, connection_candidates, device=gates.device), gates)


def harden_gates(model: nn.Module):
    """Train from now on with hard gates, straight-through.

    The forward pass uses each gate's argmax, exactly as at test time, while the
    gradient flows through the softmax. This reuses the straight-through Gumbel
    path with its noise switched off (the noise is divided by tau_noise).
    """
    for m in model.modules():
        if hasattr(m, "use_gumbel"):
            m.use_gumbel, m.tau_noise = True, float("inf")

class ConvLogicBlock(nn.Module):
    """
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 tau_noise: float = 1.0,
                 event_mode: bool = False,
                 weights_init_mode: str = 'residual',
                 ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ConvLogicLayer(dim_in, dim_out, tau_noise=tau_noise, event_mode=event_mode,
                           weights_init_mode=weights_init_mode),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.net(x)
    
    
class LogicTreeNet(nn.Module):
    """
    LogicTreeNet architecture (Reference : https://arxiv.org/pdf/2411.04732)
    """
    def __init__(self,
                 model_scale: str = 's',
                 in_ch: int = 3,
                 out_classes: int = 10,
                 tau_gs: int = None,
                 tau_noise: float = 1.0,
                 learn_tau_gs: bool = True,
                 input_size: int = 32,
                 event_mode: bool = True,
                 grouping_mode: str = "balanced",
                 group_sum_device: Optional[str] = None,
                 weights_init_mode: str = 'residual',
                 connection_candidates: int = 0,
                 dense_k: int = 0,
                 ) -> None:

        super().__init__()

        assert model_scale in ['s', 'k64', 'k128', 'm', 'b', 'l', 'g']
        assert grouping_mode in {"balanced", "legacy_padded"}, grouping_mode
        assert weights_init_mode in {"residual", "gaussian"}, weights_init_mode
        if tau_noise < 0:
            raise ValueError(f"tau_noise must be 0 (off) or positive, got {tau_noise}")
        # k64 and k128 fill the 8x gap between s and m; gate count and memory grow linearly with k
        scale = {'s':32, 'k64':64, 'k128':128, 'm':256, 'b':512, 'l':1024, 'g':2560}
        k = scale[model_scale]
        # dense_k > 0 sizes the 3 dense layers independently of the conv blocks (0 = same k);
        # gates: conv 371 * k, dense 2240 * dense_k
        kd = dense_k if dense_k > 0 else k
        if in_ch > k:   # the first conv block's trees only read input channels 0 .. k-1
            raise ValueError(f"in_ch ({in_ch}) must not exceed the first block's width k ({k})")

        if tau_gs is None:
            tau = {'s':20, 'k64':25, 'k128':30, 'm':40, 'b':280, 'l':340, 'g':450}  # k64/k128 interpolated, untuned
            tau_gs = tau[model_scale]
        if group_sum_device is None:
            group_sum_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Calculate flattened dimension after conv blocks
        # 4 ConvLogicBlocks with MaxPool2d (stride=2) each, so spatial size is divided by 2^4 = 16
        spatial_size_after_conv = input_size // 16
        flattened_dim = 32 * k * spatial_size_after_conv * spatial_size_after_conv

        # Only first layer uses event_mode (for 2-channel event data)
        self.net = nn.Sequential(
            ConvLogicBlock(in_ch, k, tau_noise, event_mode=event_mode, weights_init_mode=weights_init_mode),
            ConvLogicBlock(k, 4*k, tau_noise, weights_init_mode=weights_init_mode),
            ConvLogicBlock(4*k, 16*k, tau_noise, weights_init_mode=weights_init_mode),
            ConvLogicBlock(16*k, 32*k, tau_noise, weights_init_mode=weights_init_mode),
            nn.Flatten(start_dim=1),
            dense_logic_layer(flattened_dim, 1280*kd, connection_candidates, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
            dense_logic_layer(1280*kd, 640*kd, connection_candidates, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
            dense_logic_layer(640*kd, 320*kd, connection_candidates, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
        )

        self.grouping_mode = grouping_mode
        final_dim = 320 * kd
        if self.grouping_mode == "legacy_padded":
            # Legacy mode pads to make equal-size groups before GroupSum.
            grouped_dim = ((final_dim + out_classes - 1) // out_classes) * out_classes
            self.pad_features = grouped_dim - final_dim
        else:
            # Balanced GroupSum keeps the original feature dimension and lets
            # GroupSum split it into near-equal class buckets.
            self.pad_features = 0

        self.group_sum =  GroupSum(out_classes, tau_gs, learn_tau_gs, device=group_sum_device)

    def forward_features(self, x):
        return self.net(x)

    def forward(self, x):
        x = self.forward_features(x)
        if self.pad_features > 0:
            x = F.pad(x, (0, self.pad_features))
        return self.group_sum(x)
