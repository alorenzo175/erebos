from functools import partial


import click
import requests


from erebos.cli.base import (
    cli,
    verbose,
    schedule_options,
    set_log_level,
    PathParamType,
    run_loop,
)


def _url_callback(callback_url, final_path):
    if callback_url is not None:
        requests.post(callback_url, json={"path": str(final_path)})


@cli.group()
def data():
    """Commands to process data for erebos"""
    pass


@data.command()
@verbose
@schedule_options
@set_log_level
@click.option("--s3-prefix", default="ABI-L2-MCMIPC", help="Prefix of S3 files")
@click.option("--overwrite", default=False, type=bool, help="Overwrite existing files")
@click.option("--callback-url", help="URL to post finish file path to")
@click.argument("sqs_url")
@click.argument(
    "save_directory",
    type=PathParamType(exists=True, writable=True, resolve_path=True, file_okay=False),
)
def create_multichannel_files(
    sqs_url, save_directory, s3_prefix, overwrite, callback_url, cron, cron_tz
):
    """
    Process new files in SQS_URL and save the high-res combined NetCDF
    to SAVE_DIRECTORY
    """
    from erebos.custom_multichannel_generation import get_process_and_save

    run_loop(
        get_process_and_save,
        sqs_url,
        save_directory,
        overwrite,
        s3_prefix,
        partial(_url_callback, callback_url),
        cron=cron,
        cron_tz=cron_tz,
    )


@data.command()
@verbose
@set_log_level
@click.option("--s3-prefix", default="ABI-L2-MCMIPC", help="Prefix of S3 files")
@click.option("--s3-bucket", default="noaa-goes16")
@click.option("--overwrite", default=False, type=bool, help="Overwrite existing files")
@click.option("--callback-url", help="URL to post finish file path to")
@click.argument("date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.argument(
    "save_directory",
    type=PathParamType(exists=True, writable=True, resolve_path=True, file_okay=False),
)
def fetch_multichannel_files_by_date(
    date, save_directory, s3_prefix, overwrite, callback_url, s3_bucket
):
    """
    Process S3 files on DATE and save the high-res combined NetCDF
    to SAVE_DIRECTORY
    """
    from erebos.custom_multichannel_generation import process_files_on_day

    process_files_on_day(
        date,
        save_directory,
        overwrite,
        s3_prefix,
        s3_bucket,
        callback=partial(_url_callback, callback_url),
    )


@data.command()
@verbose
@set_log_level
@click.option("--calipso-glob", default="CAL*.nc")
@click.option("--goes-glob", default="**/erebos_MCMIPC*.nc")
@click.option(
    "--file-type", type=click.Choice(["pointwise", "cnn"]), default="pointwise"
)
@click.argument(
    "calipso_directory",
    type=PathParamType(exists=True, resolve_path=True, file_okay=False),
)
@click.argument(
    "goes_directory",
    type=PathParamType(exists=True, resolve_path=True, file_okay=False),
)
@click.argument(
    "save_directory",
    type=PathParamType(exists=True, writable=True, resolve_path=True, file_okay=False),
)
def generate_calipso_training_data(
    calipso_directory,
    goes_directory,
    save_directory,
    calipso_glob,
    goes_glob,
    file_type,
):
    """
    Generate files of combined CALIPOS and GOES data to train cloud models
    """
    from erebos.prep import combine_calipso_goes_files

    combine_calipso_goes_files(
        calipso_directory,
        goes_directory,
        save_directory,
        goes_glob,
        calipso_glob,
        fnc=file_type,
    )
