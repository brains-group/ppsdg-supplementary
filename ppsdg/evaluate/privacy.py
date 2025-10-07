"""Script for general-purpose evaluation of synthetic data privacy metrics.

This is for dataset-agnostic benchmarking of arbitrary synthetic data.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from tabkit import DatasetConfig, TableProcessor
from tabkit.config import DATA_DIR
from tabkit.utils import Configuration

from ..utils import get_data_dir, get_sdv_metadata
from .privacy_metrics import compute_metric

DATA_DIR = get_data_dir()


def slugify(name: str) -> str:
    """Convert a string to a slug suitable for use in filenames."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
    )


class EvalPrivacyConfig(Configuration):
    dataset: str
    synthetic_data: str
    metrics: list[str]
    preprocess: str | None = None

    @property
    def save_dir(self) -> Path:
        return DATA_DIR / "privacy" / self.get_unique_name()


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("config",           type=str, help="name of the config file to use")
    parser.add_argument("--overwrite",      action="store_true",    help="whether to overwrite existing results")
    args = parser.parse_args()
    # fmt: on

    config = EvalPrivacyConfig.from_yaml(args.config)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    if (config.save_dir / "report.json").exists() and not args.overwrite:
        print(
            "{} already exists at {}. skipping..".format(
                config.get_unique_name(), str(config.save_dir / "report.json")
            )
        )
        return

    # prepare inputs for metric functions
    dataset_config = DatasetConfig.from_yaml(config.dataset)
    preprocess_config = (
        TableProcessorConfig.from_yaml(config.preprocess)
        if config.preprocess is not None and config.preprocess != "default"
        else None
    )
    proc = TableProcessor(
        dataset_config=dataset_config,
        config=preprocess_config,
    ).prepare()
    X_tr, y_tr = proc.get_split("train")
    real_df = X_tr.copy()
    real_df[y_tr.name] = y_tr.copy()
    X_va, y_va = proc.get_split("val")
    val_df = X_va.copy()
    val_df[y_va.name] = y_va.copy()
    X_te, y_te = proc.get_split("test")
    test_df = X_te.copy()
    test_df[y_te.name] = y_te.copy()
    syn_df = pd.read_csv(config.synthetic_data)
    metadata = get_sdv_metadata(proc).to_dict()["tables"]["table"]

    # first do the quality report
    if not (config.save_dir / "quality_report.pkl").exists() or args.overwrite:
        q_rep = QualityReport()
        q_rep.generate(real_df, syn_df, metadata)
        q_rep.save(filepath=str(config.save_dir / "quality_report.pkl"))
    else:
        q_rep = QualityReport.load(filepath=str(config.save_dir / "quality_report.pkl"))

    # now do the metrics
    report = {
        f"qual/{slugify(k)}": float(v)
        for k, v in q_rep.get_properties()[["Property", "Score"]].values
    }
    for m in config.metrics:
        out = compute_metric(
            metric=m,
            real_df=real_df,
            syn_df=syn_df,
            val_df=val_df,
            test_df=test_df,
            metadata=metadata,
        )
        for k, v in out.items():
            report[f"{m}/{k}"] = v

    # now to save
    with open(config.save_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("### report table")
    print()
    print(pd.DataFrame([report], index=["value"]).T.to_markdown())


if __name__ == "__main__":
    main()
