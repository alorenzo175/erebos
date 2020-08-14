import logging
import math
import os
from pathlib import Path
import time


import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import xarray as xr


logger = logging.getLogger(__name__)


class MaskData(Dataset):
    def __init__(
        self, dataset, batch_size, dtype=torch.float32, adjusted=0, logger=logger,
    ):
        super().__init__()
        self.logger = logger
        self.batch_size = batch_size
        self.dtype = dtype
        self.adjusted = adjusted
        self.vars_ = [f"CMI_C{i:02d}" for i in range(1, 17)] + [
            "solar_zenith",
            "solar_azimuth",
        ]
        self.setup(dataset)

    def setup(self, dataset):
        dp = Path(dataset)
        paths = sorted(
            list(dp.parent.glob(str(dp.name) + "*")),
            key=lambda x: int(x.suffix.lstrip(".")),
        )
        datasets = [xr.open_zarr(str(p)) for p in paths]
        self.dataset = xr.concat(datasets, dim="rec")

    def __len__(self):
        return math.ceil(self.dataset.dims["rec"] / self.batch_size)

    def __getitem__(self, key):
        if key >= len(self):
            raise KeyError(f"{key} out of range")

        sl = slice(key * self.batch_size, (key + 1) * self.batch_size)
        dsl = self.dataset.isel(rec=sl)

        X = torch.tensor(
            dsl[self.vars_]
            .to_array()
            .transpose("rec", "variable", "gy", "gx", transpose_coords=False)
            .values,
            dtype=self.dtype,
        )
        mask = torch.tensor(
            dsl.label_mask.sel(adjusted=self.adjusted).values[:, np.newaxis],
            dtype=torch.bool,
        )
        y = torch.tensor((dsl.cloud_layers != 0).values, dtype=self.dtype,)
        return X, mask, y


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, padding):
        super().__init__()

        down0_out_chan = 32
        self.down0 = nn.Sequential(
            nn.Conv2d(n_channels, down0_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(down0_out_chan, down0_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        down1_out_chan = 64
        self.down1 = nn.Sequential(
            nn.Conv2d(down0_out_chan, down1_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(down1_out_chan, down1_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        down2_out_chan = 128
        self.down2 = nn.Sequential(
            nn.Conv2d(down1_out_chan, down2_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(down2_out_chan, down2_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(
            down2_out_chan, down1_out_chan, kernel_size=2, stride=2
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(down2_out_chan, down1_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(down1_out_chan, down1_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
        )

        self.up0 = nn.ConvTranspose2d(
            down1_out_chan, down0_out_chan, kernel_size=2, stride=2
        )

        self.upconv0 = nn.Sequential(
            nn.Conv2d(down1_out_chan, down0_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(down0_out_chan, down0_out_chan, kernel_size=3, padding=padding),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(down0_out_chan, n_classes, kernel_size=1, stride=1), nn.Sigmoid()
        )

    def _up_and_conv(self, x, x_skip, up, conv):
        xup = up(x)
        c = (x_skip.size()[2] - xup.size()[2]) // 2
        xc = torch.cat((xup, F.pad(x_skip, (-c, -c, -c, -c))), dim=1)
        return conv(xc)

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(self.pool1(x0))
        x = self.down2(self.pool2(x1))
        x = self._up_and_conv(x, x1, self.up1, self.upconv1)
        x = self._up_and_conv(x, x0, self.up0, self.upconv0)
        out = self.out(x)
        return out


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "38288"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger = logging.getLogger(f"train{rank}")
    formatter = logging.Formatter("%(asctime)s  %(message)s")
    handler = logging.FileHandler(f"train.log.{rank}")
    handler.setFormatter(formatter)
    handler.setLevel("DEBUG")
    logger.setLevel("INFO")
    logger.addHandler(handler)
    return logger


def cleanup():
    dist.destroy_process_group()


def dist_train(rank, world_size, train_path, batch_size):
    logger = setup(rank, world_size)
    logger.info("Training on rank %s", rank)

    torch.cuda.set_device(rank)
    model = UNet(18, 1, 0)
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    criterion = torch.nn.BCELoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)
    dataset = MaskData(train_path, batch_size)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=0, sampler=sampler,
    )

    for epoch in range(100):
        logger.info("Begin training of epoch %s", epoch)
        sampler.set_epoch(epoch)
        losses = []
        a = time.time()
        for i, (X, mask, y) in enumerate(loader):
            logger.debug("On step %s of epoch %s with %s recs", i, epoch, X.shape[0])
            X = X.to(rank, non_blocking=True)
            y = y.to(rank, non_blocking=True)
            mask = mask.to(rank, non_blocking=True)
            outputs = ddp_model(X)
            c = (mask.shape[3] - outputs.shape[3]) // 2
            m = F.pad(mask, (-c, -c, -c, -c))
            loss = criterion(outputs[m], y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(
                    "Step %s of epoch %s, mean loss %s",
                    i,
                    epoch,
                    np.array(losses).mean(),
                )
        dur = time.time() - a
        logger.info(
            "Epoch %s completed with average loss %s and time %s",
            epoch,
            np.array(losses).mean(),
            dur,
        )
    cleanup()
