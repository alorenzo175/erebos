from contextlib import contextmanager
from pathlib import Path
import tempfile
import time


import mlflow
import pkg_resources
import optuna


@contextmanager
def log_to_mlflow(trial):
    with mlflow.start_run(run_name=str(trial.number), nested=True):
        start = time.time()
        yield
        duration = time.time() - start
        mlflow.log_params(trial.params)
        tags = {
            "trial_number": str(trial.number),
            "datetime_start": str(trial.datetime_start),
            "duration": duration,
        }
        tags.update(trial.study.user_attrs)
        tags.update(trial.user_attrs)
        distributions = {
            f"{k}_distribution": str(v) for k, v in trial.distributions.items()
        }
        tags.update(distributions)
        mlflow.set_tags(tags)
        installed_packages = [
            f"{d.project_name}=={d.version}" for d in pkg_resources.working_set
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tfile = Path(tmpdir) / "versions.txt"
            with open(tfile, "w") as tf:
                tf.write("\n".join(installed_packages))
            mlflow.log_artifact(str(tfile))
