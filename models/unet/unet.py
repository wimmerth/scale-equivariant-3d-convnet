"""
Adapted and slightly modified version of MONAI U-Net implementation: https://github.com/Project-MONAI/MONAI that uses
the scale-equivariant convolutional layer
"""

import warnings
from typing import Sequence, Union
import torch
from torch import nn
import pytorch_lightning as pl

from models.unet.building_blocks import Convolution, ResidualUnit, SkipConnection
from utils.loss import DiceBCELoss, SoftDiceLoss
from layers import *


class UNet(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Union[Sequence[int], Sequence[Sequence[int]]],
            scale_size: int = 1,
            kernel_size: int = 3,
            effective_size: int = 1,
            scales=None,
            bias: bool = True,
            basis_type: str = "single",
            num_res_units: int = 0,
            act: str = "relu",
            norm: str = "batch",
            lr: float = 0.001,
            reduction: str = "max",
            dropout: float = 0.0,
            data_params: dict = None,
            **kwargs
    ) -> None:
        """
        Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
        The residual part uses a convolution to change the input dimensions to match the output dimensions
        if this is necessary but will use nn.Identity if not.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param channels: List of channels used in different depths of U-Net
        :param strides: List of strides used for down- and upsampling within the U-Net
        :param scale_size: see scale-equivariant layer
        :param kernel_size: see scale-equivariant layer
        :param effective_size: see scale-equivariant layer
        :param scales: see scale-equivariant layer
        :param bias: see scale-equivariant layer
        :param basis_type: see scale-equivariant layer
        :param num_res_units: Number of residual units in each block of the U-Net
        :param act: Activation function to be used within the U-Net
        :param norm: Normalization method to be used within the U-Net
        :param lr: Initial learning rate (Exponential decay of factor 0.9)
        :param reduction: Reduction method to used to collapse scale-dimension in the end of the network
        :param data_params: Parameters of the dataset, only needed for logging
        """
        super().__init__()
        self.save_hyperparameters()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.scale_size = scale_size
        self.effective_size = effective_size
        self.basis_type = basis_type
        self.scales = [1] if scales is None else scales
        self.conv_kwargs = kwargs
        self.strides = strides
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.learning_rate = lr

        def _create_block(
                inc: int,
                outc: int,
                channels: Sequence[int],
                strides: Sequence[int],
                is_top: bool
        ) -> nn.Sequential:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = _get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = _get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = _get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        def _get_down_layer(
                in_channels: int,
                out_channels: int,
                strides: int,
                is_top: bool) -> nn.Module:
            """
            Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            if self.num_res_units > 0:
                return ResidualUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_mode="zh" if is_top else "hh",
                    scale_size=self.scale_size,
                    kernel_size=self.kernel_size,
                    effective_size=self.effective_size,
                    scales=self.scales,
                    strides=strides,
                    bias=self.bias,
                    basis_type=self.basis_type,
                    subunits=self.num_res_units,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    **self.conv_kwargs
                )
            return Convolution(
                in_channels,
                out_channels,
                conv_mode="zh" if is_top else "hh",
                scale_size=self.scale_size,
                kernel_size=self.kernel_size,
                effective_size=self.effective_size,
                scales=self.scales,
                strides=strides,
                bias=self.bias,
                basis_type=self.basis_type,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                **self.conv_kwargs
            )

        def _get_bottom_layer(
                in_channels: int,
                out_channels: int) -> nn.Module:
            """
            Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
            """
            return _get_down_layer(in_channels, out_channels, 1, False)

        def _get_up_layer(
                in_channels: int,
                out_channels: int,
                strides: int,
                is_top: bool) -> nn.Module:
            """
            Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            conv: Union[Convolution, nn.Sequential]

            conv = Convolution(
                in_channels,
                out_channels,
                conv_mode="hh",
                scale_size=self.scale_size,
                kernel_size=self.kernel_size,  # if strides == 1 else self.kernel_size - 1,
                effective_size=self.effective_size,
                scales=self.scales,
                strides=strides,
                padding=(None if strides == 1 else 0),
                bias=self.bias,
                basis_type=self.basis_type,
                is_transposed=True,
                conv_only=is_top and self.num_res_units == 0,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                **self.conv_kwargs
            )

            if self.num_res_units > 0:
                ru = ResidualUnit(
                    out_channels,
                    out_channels,
                    conv_mode="hh",
                    scale_size=self.scale_size,
                    kernel_size=self.kernel_size,
                    effective_size=self.effective_size,
                    scales=self.scales,
                    strides=1,
                    bias=self.bias,
                    basis_type=self.basis_type,
                    subunits=1,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    last_conv_only=is_top,
                    **self.conv_kwargs
                )
                conv = nn.Sequential(conv, ru)

            if is_top:
                conv = nn.Sequential(conv, SESProjection_H_Z3(reduction))

            return conv

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

        self.loss_function = DiceBCELoss({}, {})
        # self.loss_function = SoftDiceLoss(apply_nonlin=torch.sigmoid)
        self.val_metric = SoftDiceLoss(apply_nonlin=torch.sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: dict, batch_idx, optimizer_idx=0):
        data = batch.get("data")
        target = batch.get("target")
        target = target.float()
        res = self.forward(data)
        loss = self.loss_function(res, target)
        self.log("train_loss", loss.detach().item(), on_step=True, on_epoch=False, logger=True)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Loss is NaN ({torch.isnan(loss)}) or Inf({torch.isinf(loss)}) for IDs: {batch.get('id')}")
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        data = batch.get("data")
        target = batch.get("target")
        target = target.float()
        res = self.forward(data)
        all_dice = -self.val_metric(res, target)
        loss = self.loss_function(res, target)
        if target.shape[1] == 3:
            dice_full = -self.val_metric(res[:, 0], target[:, 0])
            dice_core = -self.val_metric(res[:, 1], target[:, 1])
            dice_enhancing = -self.val_metric(res[:, 2], target[:, 2])
            self.log("val_details",
                     {"full": dice_full.item(), "core": dice_core.item(), "enhancing": dice_enhancing.item()},
                     on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('hp_metric', all_dice.item(), on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [lr_decay]
