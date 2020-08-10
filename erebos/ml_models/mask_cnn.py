import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import xarray as xr


def load_batch(dataset, device, records=500, dtype=torch.float32, adjusted=0):
    vars_ = [f"CMI_C{i:02d}" for i in range(1, 17)]
    ds = xr.open_zarr(dataset)
    s = 0
    while s < ds.dims["rec"]:
        rslice = slice(s, s + records)
        dsl = ds.isel(rec=rslice).load()
        X = torch.tensor(
            dsl[vars_]
            .to_array()
            .transpose("rec", "variable", "gy", "gx", transpose_coords=False)
            .values,
            dtype=dtype,
            device=device,
        )
        mask = torch.tensor(
            dsl.label_mask.sel(adjusted=adjusted).values[:, np.newaxis],
            dtype=torch.bool,
            device=device,
        )
        y = torch.tensor((dsl.cloud_layers != 0).values, dtype=dtype, device=device,)
        nanrecs = (
            torch.isnan(X).any(3).any(2).any(1)
            | torch.isnan(y)
            | ~mask.any(3).any(2).any(1)
        )
        s += records
        if nanrecs.sum().item() == records:
            continue
        yield X[~nanrecs], mask[~nanrecs], y[~nanrecs]
    ds.close()


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
