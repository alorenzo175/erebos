import logging
from pathlib import Path
import tempfile


import joblib
import numpy as np
import mlflow
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import (
    preprocessing,
    pipeline,
    decomposition,
    metrics,
    neural_network,
    ensemble,
    svm,
    linear_model,
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


def mlp(trial, X, y):
    pipe_steps = [("scale", preprocessing.StandardScaler())]
    whiten = trial.suggest_categorical("whiten", [True, False])
    if whiten:
        pipe_steps.append(("pca", decomposition.PCA(whiten=True)))

    layer_size = trial.suggest_int("mlp_layer_size", 50, 250, 10)
    clf = neural_network.MLPClassifier(
        hidden_layer_sizes=(layer_size,),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        early_stopping=True,
        shuffle=True,
        max_iter=1000,
    )
    pipe_steps.append(("classifier", clf))

    model = pipeline.Pipeline(pipe_steps)
    logger.info("Fitting pipeline %s", str(model))
    model.fit(X, y)
    return model


def sgd(trial, X, y):
    pipe_steps = [("scale", preprocessing.StandardScaler())]

    alpha = trial.suggest_loguniform("sgd_alpha", 1e-5, 1)
    loss = trial.suggest_categorical(
        "sgd_loss", ["hinge", "squared_hinge", "perceptron", "modified_huber", "log"]
    )
    clf = linear_model.SGDClassifier(
        alpha=alpha, loss=loss, fit_intercept=False, n_jobs=-1, early_stopping=True
    )
    pipe_steps.append(("classifier", clf))
    model = pipeline.Pipeline(pipe_steps)
    logger.info("Fitting pipeline %s", str(model))
    model.fit(X, y)
    return model


def hist_gradient_boosting(trial, X, y):
    pipe_steps = [("scale", preprocessing.StandardScaler())]

    learning_rate = trial.suggest_loguniform("hgb_learning_rate", 1e-5, 1)
    l2 = trial.suggest_loguniform("hgb_l2", 1e-4, 1)
    clf = ensemble.HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        l2_regularization=l2,
        scoring=trial.study.user_attrs["optimization_metric"],
        n_iter_no_change=5,
    )
    pipe_steps.append(("classifier", clf))
    model = pipeline.Pipeline(pipe_steps)
    logger.info("Fitting pipeline %s", str(model))
    model.fit(X, y)
    return model


def objective(trial, train_file, validate_file, classifier_name):
    user_attrs = trial.study.user_attrs
    np.random.seed(user_attrs["seed"])
    X, y = load_data(train_file)
    X_val, y_val = load_data(validate_file)

    with log_to_mlflow(trial):
        if not isinstance(classifier_name, str):
            classifier_name = trial.suggest_categorical("classifier", classifier_name)
        else:
            mlflow.log_param("classifier", classifier_name)
        objective_func = globals()[classifier_name]
        model = objective_func(trial, X, y)
        scorer = metrics.check_scoring(model, scoring=user_attrs["optimization_metric"])
        score = scorer(model, X_val, y_val)
        logger.info("Score: %s", score)
        allscores = {user_attrs["optimization_metric"]: score}
        extra_metrics = trial.study.user_attrs.get("extra_metrics", [])
        for metric in extra_metrics:
            scorer = metrics.check_scoring(model, scoring=metric)
            try:
                allscores[metric] = scorer(model, X_val, y_val)
            except AttributeError:
                allscores[metric] = np.nan
        mlflow.log_metrics(allscores)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmplogfile = Path(tmpdir) / "model.txt"
            with open(tmplogfile, "w") as f:
                f.write(str(model))
            tmpfile = Path(tmpdir) / "cloud_mask.joblib"
            joblib.dump(model, tmpfile, compress=True)
            mlflow.log_artifacts(tmpdir)
    return score
