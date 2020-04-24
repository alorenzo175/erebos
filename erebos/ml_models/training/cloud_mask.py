import logging
import os
from pathlib import Path
import tempfile


import joblib
import numpy as np
import mlflow
from sklearn import (
    preprocessing,
    pipeline,
    decomposition,
    metrics,
    neural_network,
    ensemble,
)
import xarray as xr


from .utils import log_to_mlflow


logger = logging.getLogger(__name__)


def load_data(dataset):
    vars_ = [f"CMI_C{i:02d}" for i in range(1, 17)]
    logger.info("Loading data from %s", dataset)
    with xr.open_dataset(dataset, engine="h5netcdf") as ds:
        X = ds[vars_].isel(near=0).to_dataframe()
        y = (ds.cloud_layers != 0).to_dataframe()
    # xr to dataframe sometimes adds additional
    X = X.loc[X.notna().all(axis=1), vars_].astype("float32")
    y = y.loc[X.index, "cloud_layers"].astype("int8")
    return X, y


@log_to_mlflow
def objective(trial, extra_metrics=None, save_model=False):
    user_attrs = trial.study.user_attrs
    np.random.seed(user_attrs["seed"])
    X, y = load_data(user_attrs["train_file"])
    X_val, y_val = load_data(user_attrs["validation_file"])

    scaler_name = trial.suggest_categorical(
        "scaler", ["StandardScaler", "RobustScaler"]
    )
    pipe_steps = [("scale", getattr(preprocessing, scaler_name)())]

    clf_name = trial.suggest_categorical("classifier", ["MLP"])
    if clf_name == "MLP":
        activation = trial.suggest_categorical(
            "mlp_activation", ["relu", "logistic", "tanh"]
        )
        solver = trial.suggest_categorical("mlp_solver", ["adam", "sgd"])
        layer_size = trial.suggest_int("mlp_layer_size", 5, 25, 10)
        clf = neural_network.MLPClassifier(
            hidden_layer_sizes=(layer_size,),
            activation=activation,
            solver=solver,
            learning_rate="adaptive",
            shuffle=True,
            max_iter=500,
        )
    else:
        max_depth = int(trial.suggest_loguniform("rf_max_depth", 2, 32))
        clf = ensemble.RandomForestClassifier(max_depth=max_depth)
    pipe_steps.append(("classifier", clf))

    model = pipeline.Pipeline(pipe_steps)
    logger.info("Fitting pipeline %s", str(model))
    model.fit(X, y)
    scorer = metrics.check_scoring(model, scoring=user_attrs["optimization_metric"])
    score = scorer(model, X_val, y_val)
    logger.info("Score: %s", score)
    if extra_metrics is not None:
        allscores = {user_attrs["optimization_metric"]: score}
        for metric in extra_metrics:
            scorer = metrics.check_scoring(model, scoring=metric)
            allscores[metric] = scorer(model, X_val, y_val)
        mlflow.log_metrics(allscores)
    if save_model:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "cloud_mask.joblib"
            joblib.dump(model, tmpfile, compress=True)
            mlflow.log_artifact(str(tmpfile))

    return score
