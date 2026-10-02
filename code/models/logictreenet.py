from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic.logic_layer import LogicLayer, GroupSum
from difflogic.conv_logic_layer import ConvLogicLayer

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
                 ) -> None:

        super().__init__()

        assert model_scale in ['s', 'm', 'b', 'l', 'g']
        assert grouping_mode in {"balanced", "legacy_padded"}, grouping_mode
        assert weights_init_mode in {"residual", "gaussian"}, weights_init_mode
        if tau_noise < 0:
            raise ValueError(f"tau_noise must be 0 (off) or positive, got {tau_noise}")
        scale = {'s':32, 'm':256, 'b':512, 'l':1024, 'g':2560}
        k = scale[model_scale]

        if tau_gs is None:
            tau = {'s':20, 'm':40, 'b':280, 'l':340, 'g':450}
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
            LogicLayer(flattened_dim, 1280*k, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
            LogicLayer(1280*k, 640*k, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
            LogicLayer(640*k, 320*k, tau_noise=tau_noise, weights_init_mode=weights_init_mode),
        )

        self.grouping_mode = grouping_mode
        final_dim = 320 * k
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
