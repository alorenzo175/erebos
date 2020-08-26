import torch
from torch import nn
import torch.nn.functional as F


from .base_cnn import BatchedZarrData, make_train_func, dist_trainer


class MaskData(BatchedZarrData):
    def __get_y__(self, dsl):
        return torch.tensor((dsl.cloud_layers != 0).values, dtype=self.dtype,)


class MaskUNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, padding, maxpool=True, padding_mode="reflect"
    ):
        super().__init__()

        down0_out_chan = 32
        down1_out_chan = down0_out_chan * 2
        down2_out_chan = down1_out_chan * 2
        down3_out_chan = down2_out_chan * 2
        fin_chan = down0_out_chan

        self.down0 = nn.Sequential(
            nn.Conv2d(
                n_channels,
                down0_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down0_out_chan,
                down0_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
        )

        if maxpool:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.Conv2d(
                down0_out_chan, down0_out_chan, kernel_size=3, stride=2, padding=1
            )
        self.down1 = nn.Sequential(
            nn.Conv2d(
                down0_out_chan,
                down1_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down1_out_chan,
                down1_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
        )

        if maxpool:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool2 = nn.Conv2d(
                down1_out_chan, down1_out_chan, kernel_size=3, stride=2, padding=1
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                down1_out_chan,
                down2_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down2_out_chan,
                down2_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
        )

        if maxpool:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool3 = nn.Conv2d(
                down2_out_chan, down2_out_chan, kernel_size=3, stride=2, padding=1
            )
        self.down3 = nn.Sequential(
            nn.Conv2d(
                down2_out_chan,
                down3_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down3_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down3_out_chan,
                down3_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down3_out_chan),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(
            down3_out_chan, down2_out_chan, kernel_size=2, stride=2
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(
                down3_out_chan,
                down2_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down2_out_chan,
                down2_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down2_out_chan),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(
            down2_out_chan, down1_out_chan, kernel_size=2, stride=2
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(
                down2_out_chan,
                down1_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down1_out_chan,
                down1_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down1_out_chan),
            nn.ReLU(inplace=True),
        )

        self.up0 = nn.ConvTranspose2d(
            down1_out_chan, down0_out_chan, kernel_size=2, stride=2
        )

        self.upconv0 = nn.Sequential(
            nn.Conv2d(
                down1_out_chan,
                down0_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                down0_out_chan,
                down0_out_chan,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(down0_out_chan),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(fin_chan, n_classes, kernel_size=1, stride=1),
        )

    def _up_and_conv(self, x, x_skip, up, conv):
        xup = up(x)
        c = (x_skip.size()[2] - xup.size()[2]) // 2
        xc = torch.cat((xup, F.pad(x_skip, (-c, -c, -c, -c))), dim=1)
        return conv(xc)

    def forward(self, x):
        x = F.pad(x, (-2, -2, -2, -2))
        x0 = self.down0(x)
        x0d = self.pool1(x0)
        x1 = self.down1(x0d)
        x1d = self.pool2(x1)
        x2 = self.down2(x1d)
        x2d = self.pool3(x2)
        x = self.down3(x2d)
        x = self._up_and_conv(x, x2, self.up2, self.upconv2)
        x = self._up_and_conv(x, x1, self.up1, self.upconv1)
        x = self._up_and_conv(x, x0, self.up0, self.upconv0)
        out = self.out(x)
        return out

    def forward_prob(self, x):
        out = self.forward(x)
        out = nn.Sigmoid()(out)
        return out


class MaskLoss(nn.BCEWithLogitsLoss):
    def forward(self, output, mask, y):
        c = (mask.shape[3] - output.shape[3]) // 2
        m = F.pad(mask, (-c, -c, -c, -c))
        sy = y[m.any(3).any(2).any(1)]
        return super().forward(output[m], sy)


def dist_train(
    rank,
    world_size,
    backend,
    train_path,
    val_path,
    batch_size,
    load_from,
    epochs,
    adj_for_cloud,
    use_mixed_precision,
    loader_workers,
    cpu,
    log_level,
    use_max_pool,
    learning_rate,
    use_optimizer,
    padding,
    padding_mode,
):
    params = {
        "initial_learning_rate": learning_rate,
        "momentum": 0.9,
        "optimizer": use_optimizer,
        "cuda_backend": backend,
        "log_level": log_level,
        "loss": "bce",
        "loaded_from_run": load_from,
        "using_mixed_precision": use_mixed_precision,
        "adj_cloud_locations": adj_for_cloud,
        "batch_size": batch_size,
        "train_path": train_path,
        "validation_path": val_path,
        "num_data_loader_workers": loader_workers,
        "trained_on_cpu": cpu,
        "use_max_pooling": use_max_pool,
        "padding": padding,
        "padding_mode": padding_mode,
    }
    model = MaskUNet(18, 1, padding, use_max_pool, padding_mode=padding_mode)
    loss_func = MaskLoss()
    dataset = MaskData(train_path, batch_size, adjusted=adj_for_cloud)
    val_dataset = MaskData(val_path, batch_size, adjusted=adj_for_cloud)
    return dist_trainer(
        rank, world_size, model, params, loss_func, dataset, val_dataset, epochs
    )


train = make_train_func(dist_train, "cloud_mask")
