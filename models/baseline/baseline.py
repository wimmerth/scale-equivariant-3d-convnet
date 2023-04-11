"""
Baseline method that uses the U-Net implementation of MONAI (https://github.com/Project-MONAI/MONAI).
Implementation was created to work with PyTorch Lightning.
"""
import warnings
from monai.networks.nets.unet import UNet
import pytorch_lightning as pl
from typing import Sequence, Union, Optional
from utils.loss import SoftDiceLoss, DiceBCELoss
import torch


class BaseLineModel(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Union[Sequence[int], Sequence[Sequence[int]]],
            kernel_size: int = 3,
            bias: bool = True,
            num_res_units: int = 0,
            act: str = "relu",
            norm: Optional[str] = "batch",
            lr: float = 0.001,
            dropout: float = 0.0,
            data_params: dict = None,
    ):
        """
        Baseline method that uses the U-Net implementation of MONAI

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param channels: List of channels that are used in the different hierarchical layers of the U-Net
        :param strides: List of strides that are used in convolutions to up- and downsample within U-Net
        :param kernel_size: Size of convolutional kernel
        :param bias: Whether to use a bias in the U-Net
        :param num_res_units: Number of residual units in each block of the U-Net
        :param act: Activation function that should be used within U-Net
        :param norm: Normalization method that should be used within U-Net
        :param lr: Initial learning rate (with exponential decay of 0.9)
        :param data_params: Parameters of data set, only used for logging
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout
        )
        self.out_channels = out_channels
        self.learning_rate = lr

        self.loss_function = DiceBCELoss({}, {})
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
            warnings.warn(
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
        self.log('hp_metric', all_dice.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [lr_decay]
