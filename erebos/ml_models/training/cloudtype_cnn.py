import logging
import math
import os
from pathlib import Path
import tempfile
import time


import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import xarray as xr


from erebos._version import get_versions
from .mask_cnn import BatchedZarrData, MaskUNet, setup, cleanup

git_commit = get_versions()["full-revisionid"]
if get_versions()["dirty"]:
    git_commit += ".dirty"


logger = logging.getLogger(__name__)


class CloudTypeData(BatchedZarrData):
    @property
    def num_cloud_types(self):
        return len(self.dataset.cloud_type.flag_values)

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
            dsl.label_mask.sel(adjusted=self.adjusted).values, dtype=torch.bool,
        )
        y = torch.tensor(dsl.cloud_type.values, dtype=torch.long)
        return X, mask, y


class CloudTypeUNet(MaskUNet):
    def __init__(
        self, n_channels, n_classes, padding, maxpool=True, padding_mode="reflect"
    ):
        super().__init__(
            n_channels, 1, padding, maxpool=maxpool, padding_mode=padding_mode
        )
        fin_chan = 32
        self.out = nn.Sequential(
            nn.Conv2d(fin_chan, n_classes, kernel_size=1, stride=1),
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.out.parameters():
            parameter.requires_grad = True

    def forward_prob(self, x):
        out = self.forward(x)
        out = nn.Softmax(dim=1)(out)
        return out


def validate(device, validation_loader, model, loss_function):
    out = None
    count = None
    model.eval()
    with torch.no_grad():
        for i, (X, mask, y) in enumerate(validation_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs = model(X)
            c = (mask.shape[-1] - outputs.shape[-1]) // 2
            m = F.pad(mask, (-c, -c, -c, -c))
            sy = y[m.any(2).any(1)]
            loss = loss_function(outputs.transpose(0, 1)[..., m].T, sy)
            if out is None:
                out = torch.tensor(loss.item() * sy.shape[0]).to(device)
                count = torch.tensor(sy.shape[0]).to(device)
            else:
                out += loss.item() * sy.shape[0]
                count += sy.shape[0]
    model.train()
    dist.barrier()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return out / count


def dist_train(
    rank,
    world_size,
    backend,
    train_path,
    val_path,
    batch_size,
    mask_model_load_from,
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
    logger = setup(rank, world_size, backend, log_level)
    logger.info("Training on rank %s", rank)

    params = {
        "initial_learning_rate": learning_rate,
        "momentum": 0.9,
        "optimizer": use_optimizer,
        "loss": "bce",
        "loaded_from_run": load_from,
        "mask_model_loaded": mask_model_load_from,
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
    dataset = CloudTypeData(train_path, batch_size, adjusted=adj_for_cloud)
    model = CloudTypeUNet(
        18, dataset.num_cloud_types, padding, use_max_pool, padding_mode=padding_mode
    )

    if cpu:
        device = torch.device("cpu")
        ddp_model = DDP(model)
        map_location = {"cuda:0": "cpu"}
    else:
        torch.cuda.set_device(rank)
        device = torch.device(rank)
        map_location = {"cuda:0": f"cuda:{rank:d}"}
        ddp_model = DDP(model.to(device), device_ids=[device])

    mask_model_dict = torch.load(mask_model_load_from, map_location=map_location)[
        "model_state_dict"
    ]
    rmk = [k for k in mask_model_dict.keys() if k.startswith("module.out")]
    for k in rmk:
        mask_model_dict[k] = model.state_dict()[k[len("module.") :]]

    ddp_model.load_state_dict(mask_model_dict)
    if rank == 0:
        mlflow.log_params(params)
    scaler = GradScaler(enabled=use_mixed_precision)
    if use_optimizer == "adam":
        optimizer = optim.Adam(
            ddp_model.parameters(),
            lr=params["initial_learning_rate"],
            eps=1e-4,
            amsgrad=True,
        )
    else:
        optimizer = optim.SGD(
            ddp_model.parameters(),
            lr=params["initial_learning_rate"],
            momentum=params["momentum"],
        )
    startat = 0
    if load_from is not None:
        if cpu:
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank:d}"}
        chkpoint = torch.load(load_from, map_location=map_location)
        ddp_model.load_state_dict(chkpoint["model_state_dict"])
        optimizer.load_state_dict(chkpoint["optimizer_state_dict"])
        scaler.load_state_dict(chkpoint["scaler_state_dict"])
        startat = chkpoint["epoch"] + 1

    criterion = torch.nn.CrossEntropyLoss().to(device)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        sampler=sampler,
        pin_memory=True,
    )

    val_dataset = CloudTypeData(val_path, batch_size, adjusted=adj_for_cloud)
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=loader_workers,
        sampler=val_sampler,
        pin_memory=True,
    )

    for epoch in range(startat, epochs):
        logger.info("Begin training of epoch %s", epoch)
        sampler.set_epoch(epoch)
        train_sum = None
        a = time.time()
        for i, (X, mask, y) in enumerate(loader):
            logger.debug("On step %s of epoch %s with %s recs", i, epoch, X.shape[0])
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(use_mixed_precision):
                outputs = ddp_model(X)
                c = (mask.shape[-1] - outputs.shape[-1]) // 2
                m = F.pad(mask, (-c, -c, -c, -c))
                sy = y[m.any(2).any(1)]
                loss = criterion(outputs.transpose(0, 1)[..., m].T, sy)
            if train_sum is None:
                train_sum = torch.tensor(loss.item() * sy.shape[0]).to(device)
                train_count = torch.tensor(sy.shape[0]).to(device)
            else:
                train_sum += loss.item() * sy.shape[0]
                train_count += sy.shape[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if i % 100 == 0:
                logger.info(
                    "Step %s of epoch %s, mean loss %s",
                    i,
                    epoch,
                    (train_sum / train_count).item(),
                )

        val_loss = validate(device, validation_loader, ddp_model, criterion)
        logger.info("val loss %s", val_loss.item())
        dur = time.time() - a
        learning_rate = optimizer.param_groups[0]["lr"]
        logger.info("Latest learning rate is %s", learning_rate)
        dist.barrier()
        dist.all_reduce(train_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        train_loss = train_sum / train_count
        logger.info(
            "Epoch %s in %s completed with average loss %s and validation loss %s",
            epoch,
            dur,
            train_loss.item(),
            val_loss.item(),
        )
        if rank == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": ddp_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "learning_rate": learning_rate,
                "train_loss": train_loss.item(),
                "duration": dur,
                "validation_loss": val_loss.item(),
            }
            yield checkpoint_dict, model
        else:
            yield
    cleanup()


def train(run_name, *args, **kwargs):
    seed = kwargs.pop("seed", 97238)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rank = kwargs["rank"]
    if rank != 0:
        list(dist_train(*args, **kwargs))
    else:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("mlflow.source.git.commit", git_commit)
            model_logged = False
            for chkpoint, model in dist_train(*args, **kwargs):
                mlflow.log_metric(
                    key="train_loss",
                    value=chkpoint["train_loss"],
                    step=chkpoint["epoch"],
                )
                mlflow.log_metric(
                    key="validation_loss",
                    value=chkpoint["validation_loss"],
                    step=chkpoint["epoch"],
                )
                mlflow.log_metric(
                    key="learning_rate",
                    value=chkpoint["learning_rate"],
                    step=chkpoint["epoch"],
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    tfile = Path(tmpdir) / f"cloud_type_unet.chk.{chkpoint['epoch']}"
                    torch.save(chkpoint, tfile)
                    mlflow.log_artifact(str(tfile))
                if not model_logged:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tfile = Path(tmpdir) / f"cloud_type_model.pth"
                        torch.save(model, tfile)
                        mlflow.log_artifact(str(tfile))
                    model_logged = True
