from functools import partial
import logging
import os
from pathlib import Path


import click
import mlflow
import optuna


from erebos.cli.base import (
    cli,
    verbose,
    schedule_options,
    set_log_level,
    PathParamType,
    run_loop,
)
from erebos.ml_models import training


logger = logging.getLogger(__name__)


def set_env_var(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    envvar = getattr(param, "set_env_to", param.envvar)
    if os.getenv(envvar, "") != value:
        os.environ[envvar] = value


class EnvOption(click.Option):
    def __init__(self, *args, **kwargs):
        if "set_env_to" in kwargs:
            self.set_env_to = kwargs.pop("set_env_to")
        super().__init__(*args, **kwargs)


def set_mysql_url(ctx, param, value):
    mysql_keys = [k for k in ctx.params if k.startswith("mysql")]
    mysql_params = {k: ctx.params.pop(k) for k in mysql_keys}

    if value:
        storage_url = "mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}".format(
            **mysql_params
        )
    else:
        storage_url = None
    ctx.params["mysql_storage_url"] = storage_url


def mysql_options(cmd):
    """Combine mysql options into a single decorator"""

    def options_wrapper(f):
        decs = [
            click.option(
                "--use-mysql/--no-use-mysql",
                default=False,
                envvar="USE_MYSQL_BACKEND",
                required=True,
                show_envvar=True,
                expose_value=False,
                callback=set_mysql_url,
            ),
            click.option(
                "--mysql-user",
                envvar="MYSQL_USERNAME",
                default="erebos",
                is_eager=True,
                show_envvar=True,
                required=True,
            ),
            click.option(
                "--mysql-password",
                envvar="MYSQL_PASSWORD",
                is_eager=True,
                show_envvar=True,
                required=True,
            ),
            click.option(
                "--mysql-port",
                envvar="MYSQL_PORT",
                default=3306,
                is_eager=True,
                show_envvar=True,
                required=True,
            ),
            click.option(
                "--mysql-database",
                envvar="MYSQL_DATABASE",
                default="erebos",
                is_eager=True,
                show_envvar=True,
                required=True,
            ),
            click.option(
                "--mysql-host",
                envvar="MYSQL_HOST",
                default="127.0.0.1",
                is_eager=True,
                show_envvar=True,
                required=True,
            ),
        ]
        for dec in reversed(decs):
            f = dec(f)
        return f

    return options_wrapper(cmd)


def mlflow_options(cmd):
    """Combine mlflow options into a single decorator"""

    def options_wrapper(f):
        decs = [
            click.option(
                "--mlflow-tracking-uri",
                envvar="MLFLOW_TRACKING_URI",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
            ),
            click.option(
                "--mlflow-tracking-username",
                envvar="MLFLOW_TRACKING_USERNAME",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
            ),
            click.option(
                "--mlflow-tracking-password",
                envvar="MLFLOW_TRACKING_PASSWORD",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
            ),
            click.option(
                "--mlflow-s3-endpoint-url",
                envvar="MLFLOW_S3_ENDPOINT_URL",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
            ),
            click.option(
                "--mlflow-s3-default-region",
                envvar="MLFLOW_S3_DEFAULT_REGION",
                set_env_to="AWS_DEFAULT_REGION",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
                cls=EnvOption,
            ),
            click.option(
                "--mlflow-s3-access-key-id",
                envvar="MLFLOW_S3_ACCESS_KEY_ID",
                set_env_to="AWS_ACCESS_KEY_ID",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
                cls=EnvOption,
            ),
            click.option(
                "--mlflow-s3-secret-access-key",
                envvar="MLFLOW_S3_SECRET_ACCESS_KEY",
                set_env_to="AWS_SECRET_ACCESS_KEY",
                show_envvar=True,
                required=True,
                expose_value=False,
                is_eager=True,
                callback=set_env_var,
                cls=EnvOption,
            ),
        ]
        for dec in reversed(decs):
            f = dec(f)
        return f

    return options_wrapper(cmd)


@cli.group()
def train():
    """Commands to train erebos models"""
    pass


@train.command()
@verbose
@set_log_level
@click.argument(
    "combined_dir", type=PathParamType(exists=True, resolve_path=True, file_okay=False),
)
@click.argument(
    "save_directory",
    type=PathParamType(exists=True, writable=True, resolve_path=True, file_okay=False),
)
@click.option("--train-pct", type=int, default=80)
@click.option("--test-pct", type=int, default=10)
@click.option("--daytime-only", is_flag=True, default=False)
@click.option("--seed", type=int, default=0)
def split_dataset(
    combined_dir, save_directory, train_pct, test_pct, seed, daytime_only
):
    """
    Split dataset into train, test, and validation sets combining each into
    a single netCDF4 file. The splitting will be performed by combined calipso
    run to keep calipso scans together. The split percentages are rough guides
    since each file has a variable number of points.
    """
    from erebos import prep

    df = prep.load_combined_files(combined_dir)
    train, test, val = prep.split_data(
        prep.filter_times(df, daytime_only), train_pct, test_pct, seed
    )
    prep.concat_datasets(train.filename, save_directory / "train.nc")
    prep.concat_datasets(test.filename, save_directory / "test.nc")
    prep.concat_datasets(val.filename, save_directory / "validate.nc")


@train.command()
@verbose
@set_log_level
@mlflow_options
@mysql_options
@click.argument("experiment_name")
@click.argument("objective_func")
@click.option("--n-trials", type=int, default=100, help="Number of trials")
@click.option("--n-jobs", type=int, default=1, help="Number of parallel jobs")
@click.option(
    "--train-file",
    type=PathParamType(exists=True, resolve_path=True),
    default=Path(__file__).parent / "../../data/cloud_mask/train.nc",
)
@click.option(
    "--validate-file",
    type=PathParamType(exists=True, resolve_path=True),
    default=Path(__file__).parent / "../../data/cloud_mask/validate.nc",
)
def cloud_mask(
    experiment_name,
    objective_func,
    n_trials,
    n_jobs,
    train_file,
    validate_file,
    mysql_storage_url,
):

    logger.info("Using tracking URI %s", mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment(experiment_name)
    study = optuna.create_study(
        study_name=experiment_name,
        direction="maximize",
        load_if_exists=True,
        storage=mysql_storage_url,
    )
    study.set_user_attr("model", "erebos_cloud_mask")
    study.set_user_attr("seed", 6626)
    study.set_user_attr("train_file", str(train_file.absolute()))
    study.set_user_attr("validation_file", str(validate_file.absolute()))
    study.set_user_attr("optimization_metric", "roc_auc")
    study.set_user_attr(
        "extra_metrics", ["f1", "accuracy", "precision", "neg_brier_score"]
    )
    study.optimize(
        partial(
            training.cloud_mask.objective,
            train_file=train_file,
            validate_file=validate_file,
            objective_func=objective_func,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
