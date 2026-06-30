import torch
import torch.nn.functional as F

import difflogic_cuda
import numpy as np

from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor
from .gumbel_noise import gumbel_softmax

class LogicLayer(torch.nn.Module):
    """
    Differentiable logic gate layer. Supports softmax training (DLGN) or Gumbel noise (GLGN).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cuda",
        grad_factor: float = 1.0,
        implementation: str = None,
        connections: str = "random",
        weights_init_mode: str = "residual", 
        tau_noise: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.tau_noise = tau_noise
        self.use_gumbel = False if tau_noise == 0 else True

        # Implementation selection
        self.implementation = implementation
        if self.implementation is None and device == "cuda":
            self.implementation = "cuda"
        elif self.implementation is None and device == "cpu":
            self.implementation = "python"
        assert self.implementation in ["cuda", "python"], self.implementation

        # Connections
        self.connections = connections
        assert self.connections in ["random", "unique"], self.connections
        self.indices = self.get_connections(self.connections, device)

        # Weights
        self.weights_init_mode = weights_init_mode
        assert self.weights_init_mode in ["residual", "gaussian"], self.weights_init_mode
        self.weights = self.initialize_weights(out_dim, weights_init_mode)

        if self.implementation == "cuda":
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(),
                device=device,
                dtype=torch.int64,
            )
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist],
                dtype=torch.int64,
                device=device,
            )

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, "PackBitsTensor not supported during differentiable training."
            assert self.device == "cuda", "PackBitsTensor only works with CUDA."
        else:
            if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        if self.indices[0].dtype != torch.int64 or self.indices[1].dtype != torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]

        if self.training:
            if self.use_gumbel:
                weights = gumbel_softmax(self.weights, tau_noise=self.tau_noise, hard=True)
            else:
                weights = F.softmax(self.weights, dim=-1)
            x = bin_op_s(a, b, weights)
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == "cuda", x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices
        if self.training:
            if self.use_gumbel:
                w = gumbel_softmax(self.weights, tau_noise=self.tau_noise, hard=True).to(x.dtype)
            else:
                w = F.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def get_connections(self, connections, device="cuda"):
        assert self.out_dim * 2 >= self.in_dim, "Too few neurons vs inputs."
        if connections == "random":
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == "unique":
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    def initialize_weights(self, out_dim, weights_init_mode):
        if weights_init_mode == "gaussian":
            weights = torch.randn(out_dim, 16, device=self.device)
        elif weights_init_mode == "residual":
            sigma = 5
            weights = torch.zeros(out_dim, 16, device=self.device)
            weights[:, 3] = sigma  # A gate
        else:
            raise ValueError(weights_init_mode)
        return torch.nn.parameter.Parameter(weights)

    def extra_repr(self):
        mode = "train" if self.training else "eval"
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, mode={mode}, use_gumbel={self.use_gumbel}, tau_noise={self.tau_noise}"

class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., learn_tau:bool=False, device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.device = device
        
        self.tau = tau
        self.learn_tau = learn_tau
        if learn_tau:
            self.tau = torch.nn.parameter.Parameter(torch.tensor(self.tau, device=self.device, dtype=torch.float32))

    def forward(self, x): 
        n = x.shape[-1]
        assert n >= self.k, (x.shape, self.k)

        group_size, remainder = divmod(n, self.k)
        if remainder == 0:
            return x.reshape(*x.shape[:-1], self.k, group_size).sum(-1) / self.tau

        # Keep all features by assigning the remainder to the first groups,
        # so group sizes differ by at most one feature.
        leading = remainder * (group_size + 1)
        head = x[..., :leading].reshape(*x.shape[:-1], remainder, group_size + 1).sum(-1)
        tail = x[..., leading:].reshape(*x.shape[:-1], self.k - remainder, group_size).sum(-1)
        return torch.cat((head, tail), dim=-1) / self.tau

    def extra_repr(self):
        return 'n_classes={}, temperature={}, learn_tau'.format(self.k, self.tau, self.learn_tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    pass
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################
