import logging
import os
from pathlib import Path
import psutil
import signal


import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage, Retries
from dramatiq_dashboard import DashboardApp
import pandas as pd
from periodiq import PeriodiqMiddleware, cron
import xarray as xr


from erebos import custom_multichannel_generation, config, utils
from erebos.ml_models import predict

HIGH = 0
MID = 50
LOW = 100


class RestartMiddleware(dramatiq.Middleware):
    def __init__(self, memory_limit):
        self.logger = dramatiq.logging.get_logger(__name__, type(self))
        self.memory_limit = memory_limit

    def after_process_message(self, broker, message, *, result=None, exception=None):
        proc = psutil.Process(os.getppid())
        rss = proc.memory_info().rss
        for child in proc.children():
            rss += child.memory_info().rss
        rss_mb = rss / 1024 ** 2
        self.logger.debug("Process group currently using %0.2f MiB RSS", rss_mb)
        if rss_mb > self.memory_limit:
            self.logger.info("Restarting workers with RSS of %0.2f MiB RSS...", rss_mb)
            os.kill(os.getppid(), signal.SIGHUP)


class RetryException(Exception):
    pass


class RetryWOException(Retries):
    def after_process_message(self, broker, message, *, result=None, exception=None):
        if hasattr(message, "retry") and message.retry and exception is None:
            exception = RetryException()
        super().after_process_message(
            broker, message, result=result, exception=exception
        )


redis_broker = RedisBroker(
    host=config.REDIS_HOST, port=config.REDIS_PORT, heartbeat_timeout=10 * 60_000
)
redis_broker.add_middleware(RestartMiddleware(config.MEM_LIMIT))
redis_broker.add_middleware(PeriodiqMiddleware(skip_delay=30))
redis_broker.add_middleware(CurrentMessage())
redis_broker.add_middleware(RetryWOException())
dramatiq.set_broker(redis_broker)
dashboard_app = DashboardApp(broker=redis_broker, prefix=config.DASHBOARD_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


@dramatiq.actor(priority=HIGH)
def process_combined_file(combined_file_path):
    predict.full_prediction_to_zarr(Path(combined_file_path), zarr_dir=config.ZARR_DIR)


@dramatiq.actor(priority=MID)
def generate_combined_file(key, bucket, async_process=True):
    final_path = custom_multichannel_generation.generate_combined_file(
        key, config.MULTI_DIR, bucket, overwrite=False
    )
    if final_path is not None:
        if async_process:
            process_combined_file.send(str(final_path))
        else:
            process_combined_file(final_path)
    else:
        logger.warning("Rescheduling processing of %s", key)
        msg = CurrentMessage.get_current_message()
        msg.retry = True


@dramatiq.actor(priority=LOW, periodic=cron("* * * * *"))
def periodically_generate_combined_files():
    custom_multichannel_generation.get_process_and_save(
        config.SQS_URL,
        config.MULTI_DIR,
        False,
        config.S3_PREFIX,
        callback=process_combined_file.send,
    )


@dramatiq.actor(priority=LOW, periodic=cron("15 0 * * *"))
def find_missing_combined_files():
    start = pd.Timestamp.utcnow().floor("1d") - pd.Timedelta("1d")
    for key in custom_multichannel_generation.loop_nonexistent_keys(
        start, config.S3_PREFIX, config.S3_BUCKET, config.MULTI_DIR
    ):
        logger.info(
            "Archive is missing file for %s, making job to retrieve", key
        )
        generate_combined_file.send(key, config.S3_BUCKET, False)


@dramatiq.actor(priority=LOW, periodic=cron("10 * * * *"))
def find_missing_zarr_files():
    now = pd.Timestamp.utcnow()
    for multi in (Path(config.MULTI_DIR) / now.strftime("%Y/%m/%d")).glob("erebos*.nc"):
        zarrpath = Path(config.ZARR_DIR) / multi.parent.relative_to(config.MULTI_DIR)
        logger.debug("Checking zarr dataset at %s for data from %s", zarrpath, multi)
        if not zarrpath.exists():
            logger.info(
                "No zarr dir (%s) found for %s, making zarr dataset", zarrpath, multi
            )
            process_combined_file.send(str(multi))
        else:
            try:
                zds = xr.open_zarr(str(zarrpath))
            except Exception:
                logger.error("Could not open zarr dataset at %s ", zarrpath)
                continue
            try:
                t = pd.Timestamp(str(multi.stem).split("_")[-1])
            except Exception:
                logger.error("Could not parse timestamp from %s", multi)
                continue
            if t not in zds.t.data:
                logger.info(
                    "Adding data from %s to zarr dataset at %s", multi, zarrpath
                )
                process_combined_file.send(str(multi))
