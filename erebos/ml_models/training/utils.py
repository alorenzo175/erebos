from functools import wraps
import mlflow


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
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.study.user_attrs)
            score = f(
                trial,
                extra_metrics=trial.study.user_attrs.get("extra_metrics", []),
                save_model=True,
            )
            mlflow.log_params(trial.params)
        return score

    return wrapper
