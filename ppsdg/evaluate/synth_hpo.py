"""
This script defines hyperparameter optimization (HPO) for the synthesizers themselves.
To be downstream-agnostic, we can optimize for some quality metric of the data
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
import ray
from ray import tune
from sdmetrics.reports.single_table import QualityReport
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from tabkit import DatasetConfig, TableProcessor, TableProcessorConfig
from tabkit.config import DATA_DIR
from tabkit.utils import Configuration, get_random_id

from ppsdg.models.tabdiff.integration import TabDiffWrapper

from ..utils import get_sdv_metadata
from .privacy import slugify


class TuneSynthConfig(Configuration):
    dataset: str
    model: Literal["gaussian", "ctgan", "tvae", "tabdiff"]
    target_metric: Literal["quality", "downstream_xgb"]
    preprocess: str | None = None
    target_metric_direction: Literal["max", "min"] = "max"
    model_params: dict | None = None
    train_params: dict | None = None

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = dict()
        if self.train_params is None:
            self.train_params = dict()

        # ray changes the working directory of each distributed job. so
        # anything that uses path needs to be absolute
        if self.dataset is not None:
            self.dataset = str(Path(self.dataset).resolve())
        if self.preprocess is not None:
            self.preprocess = str(Path(self.preprocess).resolve())

    @property
    def save_dir(self) -> Path:
        return DATA_DIR / "synth_hpo" / self.get_unique_name()


def _compute_metric(
    metric: str,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    metadata: dict,
) -> dict[str, float]:
    report = {}
    if metric == "quality":
        # easy (and probably cheapest) case.
        q_rep = QualityReport()
        q_rep.generate(real_df, syn_df, metadata, verbose=False)
        df = q_rep.get_properties()
        for k, v in df[["Property", "Score"]].values:
            report[f"qual/{slugify(k)}"] = float(v)
        report["score"] = df["Score"].mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # numpy floats are nasty. force to python float
    return {k: float(v) for k, v in report.items()}


def get_hp_space(config: TuneSynthConfig):
    # taken from autogluon
    if config.model == "ctgan":
        return dict(
            generator_decay=tune.loguniform(lower=1e-4, upper=1e-2),
            generator_lr=tune.loguniform(lower=1e-4, upper=1e-2),
            discriminator_steps=tune.randint(lower=1, upper=10),
            epochs=tune.randint(lower=10, upper=500),
            # pac=tune.randint(lower=1, upper=20),
        )
    elif config.model == "tvae":
        return dict(
            embedding_dim=tune.choice([16, 32, 64, 128, 256]),
            compress_dims=tune.choice(
                [
                    (128,) * 2,
                    (128,) * 3,
                    (256,) * 2,
                    (256,) * 3,
                ]
            ),
            decompress_dims=tune.choice(
                [
                    (128,) * 2,
                    (128,) * 3,
                    (256,) * 2,
                    (256,) * 3,
                ]
            ),
            l2scale=tune.loguniform(lower=1e-6, upper=1e-2),
            batch_size=tune.choice([500, 1000, 2000]),
            epochs=tune.randint(lower=10, upper=500),
            loss_factor=tune.randint(lower=1, upper=10),
        )
    elif config.model == "tabdiff":
        # NOTE: tabdiff is weird b/c the model and train params are separated.
        # we should use a flat format here (no nesting) so that Ray can pick the values,
        # but when we pass it to the model, we need to figure out which one goes where.
        # only mess with training params for now.
        return dict(
            lr=tune.loguniform(1e-5, 1e-3),
            weight_decay=tune.loguniform(1e-6, 1e-2),
            batch_size=tune.choice([500, 1000, 2000]),  # can we even do this?? idk.
            steps=tune.randint(1000, 20000),
        )
    else:
        raise ValueError(f"Unknown syntehsizer name: {config.model}")


def build_model(
    config: TuneSynthConfig,
    proc: TableProcessor,
    hp_override: dict | None = None,
):
    metadata = get_sdv_metadata(proc)
    if config.model_params is None:
        config.model_params = dict()
    model_params = config.model_params.copy()
    if hp_override is not None:
        model_params.update(hp_override)
    if config.model == "gaussian":
        return GaussianCopulaSynthesizer(
            metadata,
            **model_params,
        )
    elif config.model == "ctgan":
        return CTGANSynthesizer(metadata, **model_params)
    elif config.model == "tvae":
        return TVAESynthesizer(metadata, **model_params)
    elif config.model == "tabdiff":
        return TabDiffWrapper(
            proc=proc,
            train_params=model_params,
        )
    else:
        raise ValueError(f"Unknown synthesizer name: {config.model}")


def train_model(
    config: TuneSynthConfig,
    hp_override: dict | None = None,
    save_results: bool = False,
):
    proc = TableProcessor(
        config=TableProcessorConfig.from_yaml(config.preprocess)
        if config.preprocess
        else None,
        dataset_config=DatasetConfig.from_yaml(config.dataset),
    ).prepare()

    X_tr, y_tr = proc.get_split("train")
    n_sample_rows = len(X_tr)
    # join them, since most synthesizers don't care about label cols.
    data = X_tr.copy()
    data[y_tr.name] = y_tr.copy()
    synth = build_model(config, proc, hp_override=hp_override)
    # train on the train split
    generated = None
    synth.fit(data)
    generated = synth.sample(num_rows=n_sample_rows)

    if generated is None:
        return {
            "score": -1.0 if config.target_metric_direction == "max" else 1.0,
        }

    metadata = get_sdv_metadata(proc).to_dict()["tables"]["table"]

    # here we do the eval
    metrics = _compute_metric(
        metric=config.target_metric,
        real_df=data,
        syn_df=generated,
        # NOTE: metrics want the metadata for one table only..
        metadata=metadata,
    )
    return metrics


def tune_model(
    config: TuneSynthConfig,
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
        tune.with_resources(trainable_func, {"gpu": 1 / n_workers, "cpu": 2}),
        config=get_hp_space(config),
        metric="score",
        mode=config.target_metric_direction,
        num_samples=n_trials,
        storage_path=ray_storage_path,
        max_concurrent_trials=n_workers,
        name=run_name,  # ray has a limit of 20 chars for this(!!!)
    )
    best_trial = analysis.get_best_trial(
        metric="score",
        mode=config.target_metric_direction,
    )

    ray.shutdown()
    shutil.rmtree(ray_storage_path / run_name)
    shutil.rmtree(tmp_dir / ray_session_name)
    return best_trial.config


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("config",           type=str, help="name of the config file to use")
    parser.add_argument("--trials",         type=int, default=100,  help="number of hyperparameter tuning trials")
    parser.add_argument("--workers",        type=int, default=3,   help="number of parallel workers for hyperparameter tuning")
    parser.add_argument("--overwrite",      action="store_true",    help="whether to overwrite existing results")
    parser.add_argument("--metrics_only",   action="store_true",    help="when overwriting, only compute metrics, don't retrain. if overwrite is false, this is ignored.")
    args = parser.parse_args()
    # fmt: on

    config = TuneSynthConfig.from_yaml(args.config)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    if (config.save_dir / "report.json").exists() and not args.overwrite:
        print(
            "{} already exists at {}. skipping..".format(
                config.get_unique_name(), str(config.save_dir / "report.json")
            )
        )
        return
    print(
        f"tuning {config.get_unique_name()} for {args.trials} trials with {args.workers} workers"
    )
    best_hp = tune_model(
        config,
        n_trials=args.trials,
        n_workers=args.workers,
    )
    print("best hyperparameters found:", best_hp)
    config.save_yaml(config.save_dir / "config.yaml")
    with open(config.save_dir / "best_hp.json", "w") as f:
        json.dump(best_hp, f, indent=2)
    report_path = config.save_dir / "report.json"

    if report_path.is_file() and not args.overwrite:
        print(
            "{} is already done at {}. skipping..".format(
                config.get_unique_name(), report_path
            )
        )
        return
    metrics = train_model(
        config,
        save_results=not args.metrics_only,
    )
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"report saved to {str(report_path)}")
    print(pd.DataFrame([metrics], index=["score"]).T.to_markdown())
    print()
    print("=" * 20)
    print()
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
