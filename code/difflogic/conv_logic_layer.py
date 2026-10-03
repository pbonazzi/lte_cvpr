import torch
import torch.nn as nn
import torch.nn.functional as F

import conv_difflogic_cuda
from .functional import bin_op_s
from .gumbel_noise import gumbel_softmax

class ConvLogicLayer(torch.nn.Module):
    """
    The core module for convolutional differentiable logic gate networks. Provides a convolutional differentiable logic gate layer.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            receptive_field: int = 3,
            stride: int = 1,
            padding: int = 1,
            device: str = 'cuda',
            implementation: str = None,
            connections: str = 'random_restricted',
            weights_init_mode: str = 'residual',
            tau_noise: float = 1.0,
            event_mode: bool = False

    ) -> None:
        """
        :param in_channels:         input channels dimensionality of the layer
        :param out_channels:        output channels dimensionality (== # tree kernels) of the layer
        :param kernel_size:         depth of the tree kernel
        :param receptive_field:     receptive field of shape [receptive_field, receptive_field] of the tree kernel
        :param stride:              stride
        :param padding:             zero padding size
        :param device:              device (options: 'cuda' / 'cpu')
        :param implementation:      implementation to use (options: 'cuda' / 'python')
        :param connections:         method for initializing the connectivity of the tree kernels
        :param weights_init_mode:   method for initializing the weights of the conv logic layer
        :param event_mode:          use event-specific connection strategy (for 2-channel event data)
        """

        super().__init__()
        self._check_validity_attributes(in_channels, out_channels, kernel_size, receptive_field, stride)
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.d = kernel_size
        self.rf = receptive_field
        self.str = stride
        self.pad = padding
        self.device = device
        self.tau_noise = tau_noise
        self.use_gumbel = True if tau_noise > 0 else False
        self.event_mode = event_mode
    
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random_restricted'], self.connections
        # connections index tensors: C_M, C_H, C_W (# Reference : https://arxiv.org/pdf/2411.04732)
        for name, tensor in self._get_connections_rstd().items():
            self.register_buffer(name, tensor)

        self.weights_init_mode = weights_init_mode
        assert self.weights_init_mode in ['residual', 'gaussian'], self.weights_init_mode
        self.weights = self._initialize_weights(weights_init_mode)

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, mode={}'.format(
            self.in_ch,self.out_ch,'train' if self.training else 'eval'
        )

    def forward(self,x):
        if self.implementation == 'cuda':
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device

        rf = self.rf
        stride = self.str
        batch_size, n_chs, h, w = x.shape
        assert h >= rf and w >= rf, "receptive field and input incompatible"
        if x.ndim != 4 or 0 in x.shape or n_chs != self.in_ch:
            raise ValueError("invalid input")

        c_m, c_h, c_w, ch_occ = self.c_m, self.c_h, self.c_w, self.ch_occ
        pad, d = self.pad, self.d

        if self.training:
            if self.use_gumbel:
                # straight-through gumbel: hard one-hot in forward, soft for grads
                weights = gumbel_softmax(self.weights.view(-1, 16), tau_noise=self.tau_noise, hard=True)
                weights = weights.view_as(self.weights).to(x.dtype)
            else:
                weights = F.softmax(self.weights, dim=-1).to(x.dtype)
            return ConvLogicLayerCudaFunction.apply(x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d)
        else:
            # you forgot this branch; returning None in eval poisoned MaxPool2d
            weights = F.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return ConvLogicLayerCudaFunction.apply(x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d)
              
    def forward_python(self,x):
        # add zero padding of specified size
        pad = self.pad
        x = F.pad(x, (pad,pad,pad,pad))

        # x of shape [batch_size, channels, height, width]
        rf = self.rf
        stride = self.str
        batch_size, n_chs, h, w = x.shape
        assert h >= rf and w >= rf, "receptive field and input incompatible"
        if len(x.shape) != 4 or 0 in x.shape or n_chs != self.in_ch: raise ValueError("invalid input")
        
        n_trees = self.out_ch
        d = self.d
        c_m, c_h, c_w = self.c_m, self.c_h, self.c_w
        weights = self.weights

        dev = x.device
        train = self.training

        # output tensor of shape [batch_size, out_channels, out_height, out_width]
        out_h = (x.shape[-2]-rf) // stride + 1
        out_w = (x.shape[-1]-rf) // stride + 1
        output = torch.empty((batch_size, n_trees, out_h, out_w), device=dev)

        # x_unf, before transpose and reshape, of shape [batch_size, channels*rf*rf (== #elements per patch), out_h * out_w (==#patches)]
        n_patches = out_h*out_w
        x_unf = F.unfold(x, (rf,rf)).transpose(1,2).reshape(batch_size*n_patches, rf*rf*n_chs)

        for t in range(n_trees):
            output[:,t] = self._evaluate_tree_unfolded(x_unf,
                                                       c_m[t], c_h[t], c_w[t], weights[t],
                                                       d, rf, batch_size, n_patches, 
                                                       dev, train, self.use_gumbel, self.tau_noise
                                                       ).reshape(batch_size, out_h, out_w)
        return output

    @staticmethod
    def _evaluate_tree_unfolded(x, c_m, c_h, c_w, weights, d, rf, batch_size, n_patches, dev, train, use_gumbel, tau_noise):
        n_inputs = 2**d
        output = torch.empty(batch_size*n_patches, n_inputs//2, device=dev)
        consumed = 0

        for i in range(n_inputs//2):
            a = x[:, ((c_m[2*i]*rf) + c_h[2*i])*rf + c_w[2*i]]
            b = x[:, ((c_m[2*i+1]*rf) + c_h[2*i+1])*rf + c_w[2*i+1]]
            if train: 
                if use_gumbel:
                    w = gumbel_softmax(weights[i].unsqueeze(0), tau_noise=tau_noise, hard=True).squeeze(0)
                else:
                    w = F.softmax(weights[i], dim=-1)
                output[:, i] = bin_op_s(a, b, w)          
            else: 
                hard_weights = torch.zeros_like(weights[i])
                hard_weights[torch.argmax(weights[i])] = 1
                output[:,i] = bin_op_s(a, b, hard_weights)
            consumed += 1
        n_inputs //= 2

        while n_inputs > 1:
            input = output
            output = torch.empty(batch_size*n_patches, n_inputs//2, device=dev)
            for i in range(n_inputs//2):
                a = input[:, 2*i]
                b = input[:, 2*i+1]
                if train: 
                    if use_gumbel:
                        w = gumbel_softmax(weights[consumed].unsqueeze(0), tau_noise=tau_noise, hard=True).squeeze(0)
                    else:
                        w = F.softmax(weights[consumed], dim=-1)
                    output[:, i] = bin_op_s(a, b, w)     
                else: 
                    hard_weights = torch.zeros_like(weights[consumed])
                    hard_weights[torch.argmax(weights[consumed])] = 1
                    output[:,i] = bin_op_s(a, b, hard_weights)
                consumed += 1
            n_inputs //= 2

        return output

    @staticmethod
    def _get_channels_occupation(n_in_chs: int, n_trees: int , rstd: int, event_mode: bool = False):
        if rstd != 2: raise AssertionError(NotImplemented)

        # Event-specific connection strategy for 2-channel event data
        if event_mode and n_in_chs == 2:
            # 50% trees see both polarities, 50% specialized on single polarity
            n_both = n_trees // 2
            n_single = n_trees - n_both

            both_polarities = torch.tensor([[0, 1]] * n_both)
            single_pol = torch.tensor([[i % 2, i % 2] for i in range(n_single)])

            return torch.cat([both_polarities, single_pol], dim=0)

        # Default: original random_restricted strategy
        # architecture choice: normally we have n_trees >> n_in_chs
        # => prio 1: have kernels specialized on one single channel
        #    prio 2: have kernels specialized on combination of channels (completly random)

        prio1 = torch.tensor([[i, i] for i in range(n_in_chs)])
        override_ids = torch.randperm(n_trees)
        if n_trees < n_in_chs:
            return prio1[override_ids,:]

        prio2 = torch.randperm(2*n_trees).reshape(n_trees,2) % n_in_chs
        prio2[override_ids[:n_in_chs], :] = prio1

        return prio2
    
    @staticmethod
    def _get_channel_connection(id: int, ch_pairs: torch.Tensor, n_in_x_tree: int):
        c_m = torch.ones(n_in_x_tree)
        n_ch1 = torch.randint(1, n_in_x_tree, (1,)).item()
        ids = torch.randperm(n_in_x_tree)

        c_m[ids[:n_ch1]] *= ch_pairs[id,0]
        c_m[ids[n_ch1:]] *= ch_pairs[id,1]

        return c_m
    
    @staticmethod
    def _get_rf(rf: int, n_in_x_tree: int):
        tot = rf*rf
        n_masked = tot - n_in_x_tree
        assert n_masked >= 0, "receptive field is too small w.r.t kernel size"

        mask_ids = torch.randperm(tot)[:n_masked]
        w = torch.ones(tot)
        w[mask_ids] = 0
        w = w.reshape(rf,rf)
        indices = torch.nonzero(w).t()

        return indices[0], indices[1] # output is tuple with entries in {0,..., rf} => specify the coordinates
    
    def _get_connections_rstd(self):
        dev = self.device
        n_in_x_tree = 2**self.d
        n_in_chs = self.in_ch
        n_trees = self.out_ch
        rf = self.rf
        rstd = 2 # limit the amount of channels each tree kernel can see

        # inputs - index tensors
        c_m = torch.empty((n_trees, n_in_x_tree), device=dev)
        c_h = torch.empty((n_trees, n_in_x_tree), device=dev)
        c_w = torch.empty((n_trees, n_in_x_tree), device=dev)

        ch_pairs = self._get_channels_occupation(n_in_chs, n_trees, rstd, self.event_mode)

        for i in range(n_trees):
            c_m[i,:] = self._get_channel_connection(i, ch_pairs, n_in_x_tree)
            c_h[i,:], c_w[i,:] = self._get_rf(rf, n_in_x_tree)
        
        return {
            "c_m" : c_m.long(),
            "c_h" : c_h.long(),
            "c_w" : c_w.long(),
            "ch_occ" : ch_pairs.long().to(dev)
        }

    def _initialize_weights(self, weights_init_mode):
        """
        residual initialization (Reference : https://arxiv.org/pdf/2411.04732)
        """
        n_nodes = 2**(self.d) -1 
        if weights_init_mode == 'gaussian':
            weights = torch.randn(self.out_ch, n_nodes, 16, device=self.device)
        elif weights_init_mode == 'residual':
            sigma = 5
            # Reference : https://arxiv.org/pdf/2411.04732
            weights = torch.zeros((self.out_ch, n_nodes, 16), device=self.device)
            weights[:,:,3] = sigma  # z3 corresponding to 'A' gate
        else:
            raise ValueError(weights_init_mode)
        
        return torch.nn.parameter.Parameter(weights)
    
    @staticmethod
    def _check_validity_attributes(in_channels, out_channels, kernel_size, receptive_field, stride):
        s = ["in_channels", "out_channels", "kernel_size", "receptive_field", "stride"]
        l = [in_channels, out_channels, kernel_size, receptive_field, stride]
        for e_s, e_l in zip(s,l):
            if e_l < 1: raise ValueError("{}={} invalid".format(e_s, e_l))




class ConvLogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d):
        ctx.save_for_backward(x, weights, c_m, c_h, c_w, ch_occ)
        ctx.pad, ctx.rf, ctx.stride, ctx.d = pad, rf, stride, d
        return conv_difflogic_cuda.forward(x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d)

    @staticmethod
    def backward(ctx, grad_y):
        x, weights, c_m, c_h, c_w, ch_occ = ctx.saved_tensors
        pad, rf, stride, d = ctx.pad, ctx.rf, ctx.stride, ctx.d
        grad_y = grad_y.contiguous()

        grad_w = None
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = conv_difflogic_cuda.backward_x(x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d, grad_y)
        if ctx.needs_input_grad[1]:
            grad_w = conv_difflogic_cuda.backward_w(x, weights, c_m, c_h, c_w, ch_occ, pad, rf, stride, d, grad_y)
        return grad_x, grad_w, None, None, None, None, None, None, None, None, None

    


class LogicOrPooling(torch.nn.Module):
    def __init__(self, 
                 kernel_size: int = 2,
                 stride: int = None, 
                 padding: int = 0,
                 device: str = 'cuda',
                 return_indices: bool = False,
                 ) -> None:
        """
        :param kernel_size:     pooling application region of shape (kernel_size x kernel_size)
        :param stride:          stride of the window, default value is kernel_size
        :param padding:         negative infinity padding to be added on both sides
        :param device:          device (options: 'cuda' / 'cpu')
        :param return_indices:  if True will return the max indices along with the outputs # TODO: implement me if needed
        """
        
        super().__init__()
        self._check_validity_attributes(kernel_size, stride, padding)
        self.kernel_size = kernel_size
        if stride is not None: self.stride = stride
        else: self.stride = kernel_size
        self.padding = padding
        self.return_ids = return_indices
        self.dev = device
    
    def forward(self,x):
        # input validity: x of shape [batch_size, channels, height, width]
        if len(x.shape) != 4 or 0 in x.shape: raise ValueError("invalid input")

        pad = self.padding
        st = self.stride
        kern_s = self.kernel_size
        dev = self.dev

        # add zero padding of specified size
        x_pad = F.pad(x, (pad,pad,pad,pad))
        assert (x_pad.shape[-2]-kern_s) % st + (x_pad.shape[-1]-kern_s) % st == 0, "part of input would not be considered"

        # output tensor of shape [batch_size, channels, out_height, out_width]
        out_h = (x_pad.shape[-2]-kern_s) // st + 1
        out_w = (x_pad.shape[-1]-kern_s) // st + 1
        batch_size = x_pad.shape[0]
        n_channels = x_pad.shape[1]
        output = torch.empty((batch_size, n_channels, out_h, out_w), device=dev)

        
        for b in range(batch_size):
            for c in range(n_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        output[b,c,i,j] = torch.max(x[b,c,st*i:st*i+kern_s,st*j:st*j+kern_s])
        return output

    def extra_repr(self):
        return 'kernel_size={}, stride={}, mode={}'.format(
            self.kernel_size,self.stride,'train' if self.training else 'eval'
        )

    @staticmethod
    def _check_validity_attributes(kernel_size, stride, padding):
        if kernel_size < 1 or padding < 0 or (stride is not None and stride < 1):
            raise ValueError("invalid attribute value") 

      
