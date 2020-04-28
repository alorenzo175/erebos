import time
from functools import wraps
import mlflow
import optuna


def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        user_attrs = study.user_attrs.copy()
        mlflow.log_params(trial.params)
        mlflow.log_metrics({user_attrs.pop("metric"): trial_value})
        mlflow.log_params(user_attrs)


def log_to_mlflow(f):
    @wraps(f)
    def wrapper(trial):
        with mlflow.start_run(run_name=str(trial.number)):
            start = time.time()
            score = f(
                trial,
                extra_metrics=trial.study.user_attrs.get("extra_metrics", []),
                save_model=True,
            )
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
        return score

    return wrapper
