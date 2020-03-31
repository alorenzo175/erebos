import logging
import os


import click
import mlflow


from erebos.cli.base import (
    cli,
    verbose,
    schedule_options,
    set_log_level,
    PathParamType,
    run_loop,
)


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
@mlflow_options
def cloud_mask(**kwargs):
    logger.info("Using tracking URI %s", mlflow.tracking.get_tracking_uri())
    mlflow.set_experiment("erebos-cloud-mask")
    with mlflow.start_run():
        mlflow.log_param("a", 12)
        mlflow.log_artifact("rep.json")
