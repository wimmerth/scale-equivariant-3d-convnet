"""
Experiment script to run training of scale-equivariant U-Net
"""
import os
import sys

sys.path.append(os.getcwd())

import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.unet.unet import UNet
from utils.data.dataset import MICCAIDataset
from utils.data.precomputed_means import *

logger = TensorBoardLogger(
    save_dir="logs/",
    name="Exp-UNet"
)

checkpoint_callback = ModelCheckpoint(every_n_epochs=1, monitor="hp_metric", mode="max")

data_path = "{DATA_PATH/MICCAI_BraTS2020_TrainingData}"

unet_params = {
    "kernel_size": 5,
    "strides": [2, 2, 2],
    "padding": 0
}

scaling = {
    "n": 1.5,
    "random": True,
    "scale": 0.81,
    "range": (0.7, 1.0)
}

use_scaling = False

NUM_TRAINING_SAMPLES = 250

assert NUM_TRAINING_SAMPLES < 251

train_dataset = MICCAIDataset(data_path, filter_list=list(range(1, NUM_TRAINING_SAMPLES + 1, 1)),
                              unet_params=unet_params,
                              simplify=True, normalization="instance", scaling=scaling if use_scaling else None,
                              precomputed={"mean": mean_train_nonzero, "var": var_train_nonzero})
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataset = MICCAIDataset(data_path, filter_list=list(range(251, 370, 1)), unet_params=unet_params,
                             simplify=True, normalization="instance", scaling=scaling if use_scaling else None,
                             precomputed={"mean": mean_test_nonzero, "var": var_test_nonzero})
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

experiment = UNet(
    in_channels=4,
    out_channels=1,
    channels=[4, 8, 16, 32],
    strides=[2, 2, 2],
    scale_size=1,
    kernel_size=5,
    effective_size=3,
    scales=[0.9 ** i for i in range(4)],
    bias=True,
    basis_type="multi",
    num_res_units=2,
    act="silu",
    lr=0.008,
    norm="batch",
    reduction="avg",
    dropout=0.07,
    data_params=train_dataset.get_parameters()
)

runner = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    gpus=2,
    num_nodes=1,
    precision=16,
    log_gpu_memory='all',
    weights_summary="full",
    profiler="simple",
    log_every_n_steps=1,
    flush_logs_every_n_steps=10,
    accelerator='ddp',
    benchmark=True,
    max_epochs=50,
    gradient_clip_val=0.1,
)

print(f"======= Training =======")
runner.fit(
    model=experiment,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)
