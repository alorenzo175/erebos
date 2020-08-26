import logging
import math
from pathlib import Path
import tempfile
import time


import mlflow
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import xarray as xr


from erebos._version import get_versions


git_commit = get_versions()["full-revisionid"]
if get_versions()["dirty"]:
    git_commit += ".dirty"


def make_train_func(dist_train_func, prefix):
    def train(run_name, *args, **kwargs):
        seed = kwargs.pop("seed", 97238)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rank = kwargs["rank"]
        if rank != 0:
            list(dist_train_func(*args, **kwargs))
        else:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("mlflow.source.git.commit", git_commit)
                model_logged = False
                for chkpoint, model in dist_train_func(*args, **kwargs):
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
                        tfile = Path(tmpdir) / f"{prefix}_unet.chk.{chkpoint['epoch']}"
                        torch.save(chkpoint, tfile)
                        mlflow.log_artifact(str(tfile))
                    if not model_logged:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tfile = Path(tmpdir) / f"{prefix}_model.pth"
                            torch.save(model, tfile)
                            mlflow.log_artifact(str(tfile))
                        model_logged = True

    return train


def setup(rank, world_size, backend, log_level):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel("DEBUG")
    logger.setLevel(log_level)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


def cleanup():
    dist.destroy_process_group()


class BatchedZarrData(Dataset):
    def __init__(self, dataset, batch_size, dtype=torch.float32, adjusted=0):
        super().__init__()
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

    def __get_y__(self, dsl):
        raise NotImplementedError

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
        y = self.__get_y__(dsl)
        return X, mask, y


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
            loss = loss_function(outputs, mask, y)
            if out is None:
                out = torch.tensor(loss.item() * X.shape[0]).to(device)
                count = torch.tensor(X.shape[0]).to(device)
            else:
                out += loss.item() * X.shape[0]
                count += X.shape[0]
    model.train()
    dist.barrier()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return out / count


def train_batches(
    model,
    loader,
    scaler,
    optimizer,
    device,
    loss_func,
    logger,
    epoch,
    use_mixed_precision,
):
    train_sum = None
    for i, (X, mask, y) in enumerate(loader):
        batch_size = X.shape[0]
        logger.debug("On step %s of epoch %s with %s recs", i, epoch, batch_size)
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(use_mixed_precision):
            outputs = model(X)
            loss = loss_func(outputs, mask, y)
        if train_sum is None:
            train_sum = torch.tensor(loss.item() * batch_size).to(device)
            train_count = torch.tensor(batch_size).to(device)
        else:
            train_sum += loss.item() * batch_size
            train_count += batch_size
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
    dist.barrier()
    dist.all_reduce(train_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
    train_loss = train_sum / train_count
    return train_loss


def dist_trainer(
    rank, world_size, model, params, loss_func, dataset, val_dataset, epochs
):
    load_from = params["loaded_from_run"]
    cpu = params["trained_on_cpu"]
    use_optimizer = params["optimizer"]
    use_mixed_precision = params["using_mixed_precision"]
    loader_workers = params["num_data_loader_workers"]
    backend = params["cuda_backend"]
    log_level = params["log_level"]
    logger = setup(rank, world_size, backend, log_level)
    logger.info("Training on rank %s", rank)

    if cpu:
        device = torch.device("cpu")
        ddp_model = DDP(model)
    else:
        torch.cuda.set_device(rank)
        device = torch.device(rank)
        ddp_model = DDP(model.to(device), device_ids=[device])

    loss_func = loss_func.to(device)
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
        if params["trained_on_cpu"]:
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank:d}"}
        chkpoint = torch.load(load_from, map_location=map_location)
        ddp_model.load_state_dict(chkpoint["model_state_dict"])
        optimizer.load_state_dict(chkpoint["optimizer_state_dict"])
        scaler.load_state_dict(chkpoint["scaler_state_dict"])
        startat = chkpoint["epoch"] + 1

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
        a = time.time()
        train_loss = train_batches(
            ddp_model,
            loader,
            scaler,
            optimizer,
            device,
            loss_func,
            logger,
            epoch,
            use_mixed_precision,
        )
        val_loss = validate(device, validation_loader, ddp_model, loss_func)
        logger.info("val loss %s", val_loss.item())
        dur = time.time() - a
        learning_rate = optimizer.param_groups[0]["lr"]
        logger.info("Latest learning rate is %s", learning_rate)
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
