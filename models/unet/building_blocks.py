"""
Adapted and slightly modified version of MONAI U-Net implementation: https://github.com/Project-MONAI/MONAI that uses
the scale-equivariant convolutional layer
"""

from torch.nn import ReLU, ELU, SiLU, LeakyReLU
import numpy as np
from typing import Sequence, Union, Tuple, Optional
from torch import nn
from layers import *
import torch


def same_padding(
        kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.
    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.
    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class Convolution(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            conv_mode: str = "zh",
            scale_size: int = 1,
            kernel_size: int = 3,
            effective_size: int = 1,
            scales: list = None,
            strides: Union[Sequence[int], int] = 1,
            padding: Optional[Union[Sequence[int], int]] = None,
            bias: bool = True,
            basis_type: str = "single",
            is_transposed: bool = False,
            conv_only: bool = False,
            act: str = "relu",
            norm: str = "batch",
            dropout: float = 0.0,
            **kwargs
    ) -> None:
        """
        Constructs a convolution with normalization and optional activation

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            conv_mode: 'zh' or 'hh'
            scale_size: see scale-equivariant layer
            kernel_size: see scale-equivariant layer
            effective_size: see scale-equivariant layer
            scales: see scale-equivariant layer
            strides: convolution stride. Defaults to 1.
            padding: controls the amount of implicit zero-paddings on both sides for padding number of points
                for each dimension. Defaults to None.
            bias: whether to have a bias term. Defaults to True.
            basis_type: see scale-equivariant layer
            is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
            conv_only: whether to use the convolutional layer only. Defaults to False.
            act: activation type
            norm: feature normalization type and arguments. Defaults to instance norm.
        """
        super().__init__()
        if scales is None:
            scales = [1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, 1)

        if conv_mode == "zh":
            self.add_module("conv_z_h", SESConv_Z3_H(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                effective_size=effective_size,
                scales=scales,
                stride=strides,
                padding=padding,
                bias=bias,
                basis_type=basis_type,
                transposed=is_transposed,
                kernel_padding=False,
                **kwargs
            ))
        elif conv_mode == "hh":
            self.add_module("conv_h_h", SESConv_H_H(
                in_channels=in_channels,
                out_channels=out_channels,
                scale_size=scale_size,
                kernel_size=kernel_size,
                effective_size=effective_size,
                scales=scales,
                stride=strides,
                padding=padding,
                bias=bias,
                basis_type=basis_type,
                transposed=is_transposed,
                kernel_padding=False,
                **kwargs
            ))
        else:
            raise ValueError(f"Convolution mode '{conv_mode}' is not supported, available modes are 'zh' and 'hh'")

        if not conv_only:
            if norm is not None:
                if norm == "batch":
                    self.add_module("batch_norm", SEBatchNorm(out_channels, track_running_stats=False))
                elif norm == "instance":
                    self.add_module("instance_norm", SEInstanceNorm(out_channels, track_running_stats=False))
                else:
                    raise ValueError(
                        f"Normalization '{norm}' is not available, only 'batch' and 'instance' are implemented")
            if dropout > 0:
                self.add_module("dropout", SEDropout(dropout))
            if act is not None:
                acts = ["relu", "silu", "leaky", "elu"]
                activations = [ReLU, SiLU, LeakyReLU, ELU]
                if act not in acts:
                    raise ValueError(
                        f"Activation '{act}' is not available, only 'relu', 'silu', 'leaky' and 'elu' are supported")
                activation = activations[acts.index(act)]
                self.add_module(act, activation(inplace=True))


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::
        --+--submodule--o--
          |_____________|
    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str = "cat") -> None:
        """
        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class ResidualUnit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            conv_mode: str = "hh",
            scale_size: int = 1,
            kernel_size: int = 3,
            effective_size: int = 1,
            scales: Sequence[float] = None,
            strides: Union[Sequence[int], int] = 1,
            padding: Optional[Union[Sequence[int], int]] = None,
            bias: bool = True,
            basis_type: str = "single",
            subunits: int = 2,
            act: str = "relu",
            norm: str = "batch",
            dropout: float = 0.0,
            last_conv_only: bool = False,
            **kwargs
    ) -> None:
        """
        Residual module with multiple convolutions and a residual connection.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param conv_mode: 'zh' or 'hh'
        :param scale_size: see scale-equivariant layer
        :param kernel_size: see scale-equivariant layer
        :param effective_size: see scale-equivariant layer
        :param scales: see scale-equivariant layer
        :param strides: Strides used in down- or upsampling convolution
        :param padding: Padding to be used
        :param bias: see scale-equivariant layer
        :param basis_type: see scale-equivariant layer
        :param subunits: Number of residual units
        :param act: Activation function that should be used
        :param norm: Normalization method that should be used
        :param last_conv_only: Whether the residual block is the last of the U-Net
        :param kwargs:
        """
        super().__init__()
        if scales is None:
            scales = [1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if padding is None:
            padding = same_padding(kernel_size, 1)
        if strides != 1:
            skernel_size = kernel_size  # - 1
            spadding = 0
        else:
            skernel_size = kernel_size
            spadding = padding
        sconv_mode = conv_mode
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                in_channels=schannels,
                out_channels=out_channels,
                conv_mode=sconv_mode,
                scale_size=scale_size,
                kernel_size=skernel_size,
                effective_size=effective_size,
                scales=scales,
                strides=sstrides,
                padding=spadding,
                bias=bias,
                basis_type=basis_type,
                is_transposed=False,
                conv_only=conv_only,
                act=act,
                norm=norm,
                dropout=dropout,
                **kwargs
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            sconv_mode = "hh"
            schannels = out_channels
            sstrides = 1
            skernel_size = kernel_size
            spadding = padding

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:

            if conv_mode == "hh" and strides == 1:
                # if only adapting number of channels a 1x1 kernel is used with no padding
                # noinspection PyTypeChecker
                self.residual = SESConv_H_H_1x1(in_channels, out_channels, num_scales=len(scales), bias=bias,
                                                stride=strides)
            else:
                self.residual = Convolution(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_mode=conv_mode,
                    scale_size=scale_size,
                    kernel_size=kernel_size,  # if strides == 1 else kernel_size - 1,
                    effective_size=effective_size,
                    scales=scales,
                    strides=strides,
                    padding=padding if strides == 1 else 0,
                    bias=bias,
                    basis_type=basis_type,
                    conv_only=True,
                    **kwargs
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
