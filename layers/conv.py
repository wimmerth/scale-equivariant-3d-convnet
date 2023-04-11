"""
Scale-Equivariant Layers, adapted to 3D from https://arxiv.org/abs/1910.11093 (with several improvements and additions)
"""
import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv_transpose3d
from .basis import normalize_basis_by_min_scale
from .basis import steerable_basis
from typing import Iterable, Union, Sequence
from utils import replicate_pad_scale_dim


class SESConv_Z3_H(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 effective_size: int = 1,
                 scales: Iterable[float] = None,
                 stride: Union[Sequence[int], int] = 1,
                 padding: Union[Sequence[int], int] = 0,
                 bias: bool = False,
                 basis_type: str = 'single',
                 transposed: bool = False,
                 kernel_padding: bool = True,
                 **kwargs):
        """
        Convolutional Layer that lifts from Z3 to Scale-Group Representation

        :param in_channels: Number of channels in input data
        :param out_channels: Number of channels in output data
        :param kernel_size: Size of 3D convolution kernel
        :param effective_size: Number of kernel basis functions per dimension
        :param scales: List of scales for the kernel basis
        :param stride: Stride of convolution as in "usual" convolution
        :param padding: Padding of convolution as in "usual" convolution
        :param bias: Whether to add a bias term or not
        :param basis_type: 'single' for scale separate basis computation, 'multi' for sharing of max_order over all scales
        :param transposed: Whether to use the layer as transposed convolution
        :param kernel_padding: Whether to use padding in the computation of the kernel basis
        :param kwargs: 'max_order' or 'mult' for multi scale basis type
        """
        super().__init__()

        # initialize class variables
        if scales is None:
            scales = [1.0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.transposed = transposed

        # precompute basis
        if basis_type == 'single':
            basis = steerable_basis(kernel_size, scales, effective_size, "onescale", kernel_padding)
        elif basis_type == 'multi':
            basis = steerable_basis(kernel_size, scales, effective_size, "multiscale", kernel_padding, **kwargs)
        else:
            raise ValueError("Only basis_type = 'single' and 'multi' are supported")
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)
        self.num_funcs = self.basis.size(0)  # number of functions in kernel

        # setup weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))  # learnable weights
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # initialize learnable parameter weights
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis  # @ = matrix multiplication
        kernel = kernel.view(self.out_channels, self.in_channels, self.num_scales, self.kernel_size, self.kernel_size,
                             self.kernel_size)

        # expand kernel
        kernel = kernel.permute(0, 2, 1, 3, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)

        # convolution
        if self.transposed:
            kernel = kernel.permute(1, 0, 2, 3, 4)
            y = conv_transpose3d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        else:
            y = conv3d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        b, c, d, h, w = y.shape
        y = y.view(b, self.out_channels, self.num_scales, d, h, w)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size} | transposed={transposed} |' \
            ' padding={padding} | stride={stride}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_size: int = 1,
                 kernel_size: int = 3,
                 effective_size: int = 1,
                 scales: Iterable[float] = None,
                 stride: Union[int, Sequence[int]] = 1,
                 padding: Union[int, Sequence[int]] = 0,
                 bias: bool = False,
                 basis_type: str = 'single',
                 transposed: bool = False,
                 kernel_padding: bool = True,
                 **kwargs):
        """
        Convolutional Layer that lifts from Z3 to Scale-Group Representation

        :param in_channels: Number of channels in input data
        :param out_channels: Number of channels in output data
        :param scale_size: Number of scales to interact with each other
        :param kernel_size: Size of 3D convolution kernel
        :param effective_size: Number of kernel basis functions per dimension
        :param scales: List of scales for the kernel basis
        :param stride: Stride of convolution as in "usual" convolution
        :param padding: Padding of convolution as in "usual" convolution
        :param bias: Whether to add a bias term or not
        :param basis_type: 'single' for scale separate basis computation, 'multi' for sharing of max_order over all scales
        :param transposed: Whether to use the layer as transposed convolution
        :param kernel_padding: Whether to use padding in the computation of the kernel basis
        :param kwargs: 'max_order' or 'mult' for multi scale basis type
        """
        super().__init__()

        # initialize class variables
        if scales is None:
            scales = [1.0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.transposed = transposed

        # precompute basis
        if basis_type == 'single':
            basis = steerable_basis(kernel_size, scales, effective_size, "onescale", kernel_padding)
        elif basis_type == 'multi':
            basis = steerable_basis(kernel_size, scales, effective_size, "multiscale", kernel_padding, **kwargs)
        else:
            raise ValueError("Only basis_type = 'single' or 'multi' is supported")
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)
        self.num_funcs = self.basis.size(0)  # number of functions in kernel

        # setup trainable weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # initialize learnable parameter weights
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5, 6).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            # Use replicated padding as suggested in https://arxiv.org/abs/1909.11193
            x = replicate_pad_scale_dim(x, [0, 0, 0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand x
            b, c, s, d, h, w = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4, 5).contiguous()
            x_ = x_.view(b, -1, d, h, w)
            if self.transposed:
                kernel_ = kernel.view(self.num_scales, self.out_channels, self.in_channels, self.scale_size,
                                      self.kernel_size, self.kernel_size, self.kernel_size)
                kernel_ = kernel_.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
                kernel_ = kernel_.view(-1, self.out_channels, self.scale_size,
                                       self.kernel_size, self.kernel_size, self.kernel_size)
                output += conv_transpose3d(x_, kernel_[:, :, i], padding=self.padding, groups=s, stride=self.stride)
            else:
                output += conv3d(x_, kernel[:, :, i], padding=self.padding, groups=s, stride=self.stride)

        # squeeze output
        b, c_, d_, h_, w_ = output.shape
        output = output.view(b, s, -1, d_, h_, w_)
        output = output.permute(0, 2, 1, 3, 4, 5).contiguous()
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size} | transposed={transposed} |' \
            ' padding={padding} | stride={stride}'
        return s.format(**self.__dict__)


class SESConv_H_H_1x1(nn.Conv3d):
    """
    1x1 Convolutional Layer for 6D Scale-Translation-Group feature maps.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 num_scales=1,
                 bias=True):
        super().__init__(in_channels, out_channels, (1, 1, 1), stride=stride, bias=bias)
        self.num_scales = num_scales

    def forward(self, x):
        kernel = self.weight.unsqueeze(0)
        kernel = kernel.expand(self.num_scales, -1, -1, -1, -1, -1).contiguous()
        kernel = kernel.view(-1, self.in_channels, 1, 1, 1)

        b, c, s, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(b, -1, d, h, w)
        x = conv3d(x, kernel, stride=self.stride, groups=self.num_scales)

        b, c_, d_, h_, w_ = x.shape
        x = x.view(b, s, -1, d_, h_, w_).permute(0, 2, 1, 3, 4, 5).contiguous()
        return x
