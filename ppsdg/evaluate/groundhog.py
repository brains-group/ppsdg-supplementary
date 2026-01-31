"""Script for general-purpose evaluation of synthetic data privacy metrics.

This is for dataset-agnostic benchmarking of arbitrary synthetic data.
"""

import argparse
import json
import functools
from pathlib import Path
from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd

from sdmetrics.reports.single_table import QualityReport
from tabkit import DatasetConfig, TableProcessor
from tabkit.config import DATA_DIR
from tabkit.utils import Configuration

from sdv.single_table.utils import detect_discrete_columns
from ctgan.data_transformer import DataTransformer
from ..utils import get_data_dir, get_sdv_metadata
from .privacy_metrics import compute_metric
from sklearn.ensemble import RandomForestClassifier

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

class GroundhogAuditConfig (Configuration):
    dataset: str
    train_models: list[list[str]]
    test_models: list[list[str]]
    sample_size: int = 1000
    iterations: int = 10
    train_size: int = 100
    test_size: int = 100

    def __post_init__ (self):
        available_paths = [[], []]
        for models in self.train_models, self.test_models:
            for i in range(len(models)):
                paths = models[i]
                for j in range(len(paths)):
                    paths[j] = str(Path(paths[j]).resolve())

                    import glob
                    pattern = paths[j][:-29] + "*" + paths[j][-21:]
                    available_paths[i] += glob.glob(pattern)

                    # Early exit
                    try:
                        with open(paths[j], "rb") as _:
                            available_paths[i].append(paths[j])
                    except FileNotFoundError:
                        import sys
                        print(paths[j], "not found")

        # XXX
        #self.train_models = [available_paths[0][:-1] + available_paths[1][:-1], available_paths[1][:-1] + available_paths[0][:-1]]
        #self.test_models = [available_paths[1][-1:] + available_paths[0][-1:], available_paths[0][-1:] + available_paths[1][-1:]]
        self.train_models = [available_paths[0][:-1], available_paths[1][:-1]]
        self.test_models = [available_paths[0][-1:], available_paths[1][-1:]]

    @property
    def save_dir(self) -> Path:
        return DATA_DIR / "privacy" / self.get_unique_name()

def groundhog_extract (data : pd.DataFrame):
    features = []
    for head in list(data):
        col = data[head]
        if pd.api.types.is_numeric_dtype(col):
            features += [col.mean(), col.var(), col.min(), col.max()]
        else:
            features += [len(col.category_codes.keys())]
    return pd.DataFrame(features)

class SynthSampleSynth:
    def __init__ (self, df : pd.DataFrame, transformer : DataTransformer):
        self.df = pd.DataFrame(transformer.transform(df))

    def sample(self, num_rows):
        return self.df.sample(num_rows, replace=True)

class WrappedModel:
    def __init__ (self, model, transformer : DataTransformer):
        self.model = model
        self.transformer = transformer

    def sample(self, num_rows):
        return pd.DataFrame(self.transformer.transform(self.model.sample(num_rows)))

@functools.cache
def load_model (path : str, transformer : DataTransformer):
    if path.endswith(".csv"):
        return SynthSampleSynth(pd.read_csv(path), transformer)
    else:
        return WrappedModel(joblib.load(path), transformer)

def make_Xy (transformer : DataTransformer, size : int, sample_size : int, model_paths : list[int]):
    num_classes = len(model_paths)

    class_inds = np.zeros(num_classes)
    class_seq = np.random.choice(len(model_paths[0]), size)

    y_tr = np.random.choice(len(model_paths), size)
    X_tr = []
    for label in tqdm(y_tr):
        model = None
        tries = 1
        # For testing before completing shadow training -- not supposed to happen
        #tries = len(model_paths[0])
        while model is None and tries > 0:
            path = model_paths[label][class_seq[int(class_inds[label])]]
            try:
                model = load_model(model_paths[label][class_seq[int(class_inds[label])]], transformer)
            except:
                class_seq[int(class_inds[label])] += 1
                class_seq[int(class_inds[label])] %= len(model_paths[0])
                print(f"XXX synth {path}, trying next index {class_seq[int(class_inds[label])]}")
                tries -= 1
        class_inds[label] += 1
        df = model.sample(sample_size)
        X_tr.append(groundhog_extract(df))
    return pd.concat(X_tr, axis=1).T, y_tr

def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("config",           type=str, help="name of the config file to use")
    parser.add_argument("--overwrite",      action="store_true",    help="whether to overwrite existing results")
    args = parser.parse_args()
    # fmt: on

    config = GroundhogAuditConfig.from_yaml(args.config)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    if (config.save_dir / "report.json").exists() and not args.overwrite:
        print(
            "{} already exists at {}. skipping..".format(
                config.get_unique_name(), str(config.save_dir / "report.json")
            )
        )
        with open(config.save_dir / "report.json") as f:
            report = json.load(f)
    else:
        proc = TableProcessor(
            dataset_config=DatasetConfig.from_yaml(config.dataset),
        ).prepare()
        # Used as out-of-training data from the same distribution
        X_oo, y_oo = proc.get_split("test")
        X_oo[y_oo.name] = y_oo
        metadata = get_sdv_metadata(proc)
        discrete_columns = detect_discrete_columns(metadata, X_oo, {})
        transformer = DataTransformer()
        transformer.fit(X_oo, discrete_columns)

        correct = 0
        total = 0

        #for j in range(len(config.train_models[0])):
        #    train_models = [config.train_models[k][:j] + config.train_models[k][j+1:] for k in range(len(config.train_models))]
        #    test_models = [[config.train_models[k][j]] for k in range(len(config.train_models))]
        for j in range(config.iterations):
            clf = RandomForestClassifier()
            clf.fit(*make_Xy(transformer, config.train_size, config.sample_size, config.train_models))

            X_te, y_te = make_Xy(transformer, config.test_size, config.sample_size, config.test_models)
            y_pred = clf.predict(X_te)
            print(y_te)
            print(y_pred)
            correct += (y_te == y_pred).sum().item()
            total += len(y_te)
            print(correct, total)

        from .epsilon_estimate import get_eps_audit
        report = {
            "guesses": total,
            "correct": correct / total,
            "correct_count": correct,
            "eps_est": get_eps_audit(
                total, total, correct,
                # XXX We should be communicating the claimed delta
                1e-5, #0.5/config.sample_size,
                # XXX This shouldn't be hard-coded but we are presenting 6x5=30 trials
                0.03),
            "eps_p": 0.03,
        }

        # now to save
        with open(config.save_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    print("### report table")
    print()
    print(pd.DataFrame([report], index=["value"]).T.to_markdown())


if __name__ == "__main__":
    main()
