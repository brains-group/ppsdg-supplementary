import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import ray
from lightgbm import LGBMClassifier
from ray import tune
from sklearn.metrics import get_scorer
from tabkit.config import DATA_DIR
from tabkit.data import DatasetConfig, TableProcessor, TableProcessorConfig
from tabkit.utils import Configuration, get_random_id
from xgboost import XGBClassifier

from ..utils import get_data_dir

DATA_DIR = get_data_dir()


class EvalMLEConfig(Configuration):
    dataset: str
    model: Literal["xgboost"]
    target_metric: str
    # this is where we inject the synthetic data.
    train_override_df: str | None = None
    preprocess: str | None = None
    model_params: dict | None = None
    train_params: dict | None = None
    eval_params: dict | None = None
    data_params: dict | None = None
    metrics: list[str] | None = None

    @property
    def save_dir(self) -> Path:
        return DATA_DIR / "mle" / self.get_unique_name()

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = dict()
        if self.train_params is None:
            self.train_params = dict()
        if self.eval_params is None:
            self.eval_params = dict()
        if self.data_params is None:
            self.data_params = dict()
        if self.metrics is None:
            self.metrics = list()

        # make sure target metric is in metrics
        if self.target_metric not in self.metrics:
            self.metrics.append(self.target_metric)

        # ray changes the working directory of each distributed job. so
        # anything that uses path needs to be absolute
        if self.dataset is not None:
            self.dataset = str(Path(self.dataset).resolve())
        if self.preprocess is not None:
            self.preprocess = str(Path(self.preprocess).resolve())
        if self.train_override_df is not None:
            self.train_override_df = str(Path(self.train_override_df).resolve())


def build_model(config: EvalMLEConfig, hp_override: dict | None = None):
    model_params = config.model_params.copy()
    if hp_override is not None:
        model_params.update(hp_override)
    if config.model == "xgboost":
        return XGBClassifier(**model_params)
    elif config.model == "lightgbm":
        return LGBMClassifier(**model_params)
    raise ValueError(f"Unknown classifier name: {config.model}")


def get_hp_space(config: EvalMLEConfig):
    # taken from autogluon
    if config.model == "xgboost":
        return dict(
            n_estimators=tune.randint(100, 1000),
            max_depth=tune.randint(3, 10),
            learning_rate=tune.loguniform(5e-3, 2e-1),
            min_child_weight=tune.randint(1, 5),
            colsample_bytree=tune.uniform(0.5, 1.0),
        )
    elif config.model == "lightgbm":
        return dict(
            learning_rate=tune.loguniform(lower=5e-3, upper=0.2),
            feature_fraction=tune.uniform(lower=0.75, upper=1.0),
            min_data_in_leaf=tune.randint(lower=2, upper=60),
            num_leaves=tune.randint(lower=16, upper=96),
        )
    raise ValueError(f"Unknown classifier name: {config.classifier}")


def load_data(config: EvalMLEConfig):
    dataset_config = DatasetConfig.from_yaml(config.dataset)
    preprocess_config = (
        TableProcessorConfig.from_yaml(config.preprocess)
        if config.preprocess is not None and config.preprocess != "default"
        else None
    )
    processor = TableProcessor(
        dataset_config=dataset_config,
        config=preprocess_config,
    ).prepare()
    X_tr, y_tr = processor.get_split("train")
    X_va, y_va = processor.get_split("val")
    X_te, y_te = processor.get_split("test")

    if config.train_override_df is not None:
        override_df = pd.read_csv(config.train_override_df)
        _X_tr = override_df.drop(columns=[processor.label_info.name]).copy()
        _y_tr = override_df[processor.label_info.name].copy()
        if (_X_tr.shape[1] != X_tr.shape[1]) or (_y_tr.shape[0] != y_tr.shape[0]):
            raise ValueError(
                "Override train df has different shape than original train split. X_tr: {}, y_tr: {}, override X_tr: {}, override y_tr: {}".format(
                    X_tr.shape, y_tr.shape, _X_tr.shape, _y_tr.shape
                )
            )
        X_tr = _X_tr
        y_tr = _y_tr

    return X_tr, y_tr, X_va, y_va, X_te, y_te


def train_model(
    config: EvalMLEConfig,
    hp_override: dict | None = None,
    save_results: bool = False,
):
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data(config)

    clf = build_model(config, hp_override=hp_override)
    clf.fit(X_tr, y_tr)

    y_pred_tr = clf.predict(X_tr)
    y_pred_va = clf.predict(X_va)
    y_pred_te = clf.predict(X_te)

    splits = {
        "train": (y_tr, y_pred_tr),
        "val": (y_va, y_pred_va),
        "test": (y_te, y_pred_te),
    }

    report = {}
    for split_name, (y_true, y_pred) in splits.items():
        for m in config.metrics:
            func = get_scorer(m)._score_func
            report[f"{split_name}/{m}"] = float(func(y_true, y_pred))

    if save_results:
        config.save_dir.mkdir(exist_ok=True, parents=True)
        joblib.dump(clf, config.save_dir / "model.joblib")
        with open(config.save_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    return report


def tune_model(
    config: EvalMLEConfig,
    n_trials: int,
    n_workers: int,
    debug: bool = False,
):
    tmp_dir = Path(os.environ["RAY_TEMP_DIR_BASE"])
    ray_session_name = get_random_id(6)
    # before we go ahead, make sure it doesn't already exist
    # needs to be short bc long paths will make errors
    while (tmp_dir / ray_session_name).is_dir():
        ray_session_name = get_random_id(6)
    ray.init(
        _temp_dir=str((tmp_dir / ray_session_name).resolve()),
    )

    original_config = config.copy()

    def trainable_func(hp):
        _config = original_config.copy()
        _config.model_params.update(dict(hp))
        metrics = train_model(_config, save_results=False)
        tune.report(metrics)

    ray_storage_path = (DATA_DIR / "ray_results").resolve()
    run_name = config.get_unique_name()[:20]
    analysis = tune.run(
        trainable_func,
        config=get_hp_space(config),
        metric="val/balanced_accuracy",
        mode="max",
        num_samples=n_trials,
        storage_path=ray_storage_path,
        max_concurrent_trials=n_workers,
        name=run_name,  # ray has a limit of 20 chars for this
    )
    best_trial = analysis.get_best_trial(
        metric="val/balanced_accuracy",
        mode="max",
    )

    ray.shutdown()
    shutil.rmtree(ray_storage_path / run_name)
    shutil.rmtree(tmp_dir / ray_session_name)
    return best_trial.config


def display_report(config: EvalMLEConfig, report: dict):
    print("=" * 20)
    print("{} finished".format(config.get_unique_name()))
    print()
    print(json.dumps(config.to_dict(), indent=2))
    print(pd.DataFrame([report], index=["score"]).T.to_markdown())
    print()
    print("=" * 20)


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("config",           type=str, help="name of the config file to use")
    parser.add_argument("--trials",         type=int, default=100,  help="number of hyperparameter tuning trials")
    parser.add_argument("--workers",        type=int, default=10,   help="number of parallel workers for hyperparameter tuning")
    parser.add_argument("--overwrite",      action="store_true",    help="whether to overwrite existing results")
    parser.add_argument("--tune",           action="store_true",    help="whether to do hyperparameter tuning. if false, will use default params or those specified in config.")
    parser.add_argument("--metrics_only",   action="store_true",    help="when overwriting, only compute metrics, don't retrain. if overwrite is false, this is ignored.")
    args = parser.parse_args()
    # fmt: on

    config = EvalMLEConfig.from_yaml(args.config)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(config.save_dir / "config.yaml")
    report_path = config.save_dir / "report.json"
    if args.tune:
        report_path = config.save_dir / "tuned_report.json"

    if report_path.is_file() and not args.overwrite:
        print(
            "{} is already done at {}. skipping..".format(
                config.get_unique_name(), report_path
            )
        )
        with open(report_path) as f:
            report = json.load(f)
        display_report(config, report)
        return

    best_hp = None
    if args.tune and not args.metrics_only:
        best_hp = tune_model(
            config,
            n_trials=args.trials,
            n_workers=args.workers,
        )

    if best_hp is not None:
        with open(config.save_dir / "best_hp.json", "w") as f:
            json.dump(best_hp, f, indent=2)

    if args.overwrite and args.metrics_only:
        if not (config.save_dir / "best_hp.json").is_file():
            raise ValueError(
                f"Cannot only compute metrics if best_hp.json doesn't exist in {config.save_dir}."
            )
        with open(config.save_dir / "best_hp.json", "r") as f:
            best_hp = json.load(f)

    # config.model_params.update(best_hp)
    report = train_model(
        config,
        hp_override=best_hp,
        save_results=True,
    )

    display_report(config, report)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
