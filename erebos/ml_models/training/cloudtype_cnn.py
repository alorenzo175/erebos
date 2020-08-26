import torch
from torch import nn
import torch.nn.functional as F


from .base_cnn import BatchedZarrData, dist_trainer, make_train_func
from .mask_cnn import MaskUNet


class CloudTypeData(BatchedZarrData):
    @property
    def num_cloud_types(self):
        return len(self.dataset.cloud_type.flag_values)

    def __get_y__(self, dsl):
        return torch.tensor(dsl.cloud_type.values, dtype=torch.long)


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


class TypeLoss(nn.CrossEntropyLoss):
    def forward(self, outputs, mask, y):
        c = (mask.shape[-1] - outputs.shape[-1]) // 2
        m = F.pad(mask, (-c, -c, -c, -c))[:, 0]
        sy = y[m.any(2).any(1)]
        return super().forward(outputs.transpose(0, 1)[..., m].T, sy)


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
    params = {
        "initial_learning_rate": learning_rate,
        "momentum": 0.9,
        "optimizer": use_optimizer,
        "loss": "crossentropy",
        "cuda_backend": backend,
        "log_level": log_level,
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

    map_location = {"cuda:0": "cpu"} if cpu else {"cuda:0": f"cuda:{rank:d}"}
    mask_model_dict = torch.load(mask_model_load_from, map_location=map_location)[
        "model_state_dict"
    ]

    state_dict = {}
    for k in mask_model_dict.keys():
        nk = k[len("module.") :]
        # replace weights, but keep out layer at random
        if nk.startswith("out"):
            state_dict[nk] = model.state_dict()[nk]
        else:
            state_dict[nk] = mask_model_dict[k]

    model.load_state_dict(state_dict)
    loss_func = TypeLoss()
    val_dataset = CloudTypeData(val_path, batch_size, adjusted=adj_for_cloud)
    return dist_trainer(
        rank, world_size, model, params, loss_func, dataset, val_dataset, epochs
    )


train = make_train_func(dist_train, "cloud_type")
