"""Script to train a synthesizer and generate the data for later evaluation."""

import argparse
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from sdv.single_table.utils import detect_discrete_columns

from tabkit import DatasetConfig, TableProcessor, TableProcessorConfig
from tabkit.utils import Configuration

from ..models.dp_ctgan import DPCTGAN
from ..models.dp_tvae import DPTVAE
from ..models.iter_ctgan import IterCTGAN
from ..models.opacus_ctgan import OpacusCTGAN
from ..models.tabdiff import eval_tabdiff, train_tabdiff
from ..utils import get_data_dir, get_sdv_metadata

SynthesizerType = Literal["original", "ctgan", "tvae", "gaussian", "tabdiff", "dp_ctgan", "dp_tvae", "iter_ctgan", "opacus_ctgan"]
DATA_DIR = get_data_dir()


# this config shouldn't be coupled with the synthesizer training itself.
# only encapulsate flags related to the *generation*, agnostic of model.
class GenerationConfig(Configuration):
    n_rows: int | None = None


class SynthesizerConfig(Configuration):
    synthesizer: SynthesizerType
    train_params: dict | None = None
    synthesizer_params: dict | None = None
    dataset: str
    preprocess: str | None = None
    main_seed: int | None = None

    canary_index: int | None = None
    include_canary: bool | None = None
    reference_size: int | None = None
    reference_seed: int | None = None
    sample_size: int | None = None
    sample_seed: int | None = None

    downsample_balanced: bool | None = None

    def __post_init__(self):
        if self.preprocess is not None:
            self.preprocess = str(Path(self.preprocess).resolve())

    def get_processor(self) -> TableProcessor:
        proc = TableProcessor(
            config=TableProcessorConfig.from_yaml(self.preprocess)
            if self.preprocess
            else None,
            dataset_config=DatasetConfig.from_yaml(self.dataset),
        )
        return proc

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / "train_and_gen" / self.get_unique_name()

    @property
    def completion_marker(self) -> Path:
        return self.cache_path / "TRAINED"

    @property
    def model_save_path(self) -> Path:
        return self.cache_path / "TRAINED"

    def get_data_save_path(self, gen_config: GenerationConfig) -> Path:
        return self.cache_path / f"{gen_config.get_unique_name()}.csv"


class IdentitySynthesizer:
    def __init__(self, data):
        self._data = data

    def sample(self, num_rows):
        return self._data

def train(config: SynthesizerConfig, overwrite: bool = False):
    if config.main_seed is not None:
        np.random.seed(config.main_seed)
        torch.manual_seed(config.main_seed)

    config.cache_path.mkdir(exist_ok=True, parents=True)
    if config.completion_marker.exists() and not overwrite:
        print("{} is already trained. skipping...".format(config.get_unique_name()))
        return
    print("starting training for:", config.get_unique_name())
    proc = config.get_processor().prepare()

    X_tr, y_tr = proc.get_split("all" if config.synthesizer == "original" else "train")
    # join them, since most synthesizers don't care about label cols.
    data = X_tr.copy()
    data[y_tr.name] = y_tr.copy()
    orig_data = data.copy()

    print(X_tr.shape, y_tr.sum().item(), y_tr.sum().item()/y_tr.shape[0])

    if config.canary_index is not None:
        assert config.include_canary is not None
        assert config.sample_size is not None
        assert config.sample_seed is not None

        ref_sampler = np.random.default_rng(config.reference_seed or 1)
        max_class = y_tr.max()
        data.iloc[config.canary_index, -1] += ref_sampler.choice(max_class) + 1
        data.iloc[config.canary_index, -1] %= max_class + 1
        canary_row = data.iloc[config.canary_index, :]
        data.iloc[config.canary_index, :] = data.iloc[-1, :]

        ref_size = config.reference_size or len(y_tr) - 1
        ref_inds = ref_sampler.choice(len(y_tr) - 1, ref_size, replace=False)
        data = data.iloc[ref_inds, :]

        sampler = np.random.default_rng(config.sample_seed)
        samp_inds = sampler.choice(ref_size, config.sample_size, replace=False)
        data = data.iloc[samp_inds, :]
        if config.include_canary:
            data.iloc[sampler.choice(config.sample_size), :] = canary_row

    if config.downsample_balanced:
        majority_inds = np.nonzero(data.iloc[:, -1])[0]
        if majority_inds.shape[0]*2 < data.shape[0]:
            majority_inds = np.nonzero(1-data.iloc[:, -1])[0]
        drop_inds = np.random.choice(majority_inds, majority_inds.shape[0]*2-data.shape[0], replace=False)
        data.drop(data.iloc[drop_inds, -1].index.tolist(), inplace=True)

    metadata = get_sdv_metadata(proc)
    discrete_columns = detect_discrete_columns(metadata, orig_data, {})
    synthesizer = None
    train_logs = None
    if config.synthesizer == "original":
        synthesizer = IdentitySynthesizer(data)
    elif config.synthesizer == "ctgan":
        synthesizer = CTGANSynthesizer(
            metadata,  # required
            enforce_rounding=False,
            epochs=300,
            verbose=True,
            **(config.synthesizer_params or {}),
        )
        synthesizer.fit(data)
        train_logs = synthesizer._model.loss_values
    elif config.synthesizer == "tvae":
        synthesizer = TVAESynthesizer(
            metadata,  # required
            enforce_rounding=False,
            epochs=300,
            verbose=True,
            **(config.synthesizer_params or {}),
        )
        synthesizer.fit(data)
        train_logs = synthesizer._model.loss_values
    elif config.synthesizer == "gaussian":
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data)
    elif config.synthesizer == "dp_ctgan":
        synthesizer = DPCTGAN(
            **(config.synthesizer_params or {}),
            verbose=True,
        )
        synthesizer.fit_transformer(orig_data, discrete_columns)
        synthesizer.fit(data, discrete_columns=discrete_columns)
        train_logs = synthesizer.loss_values
    elif config.synthesizer == "dp_tvae":
        synthesizer = DPTVAE(
            **(config.synthesizer_params or {}),
            verbose=True,
        )
        synthesizer.fit_transformer(orig_data, discrete_columns)
        synthesizer.fit(data, discrete_columns=discrete_columns)
        train_logs = synthesizer.loss_values
    elif config.synthesizer == "iter_ctgan":
        synthesizer = IterCTGAN(**(config.synthesizer_params or {}), verbose=True)
        synthesizer.fit_transformer(data, discrete_columns)
        synthesizer.fit(data, discrete_columns=discrete_columns)
        train_logs = synthesizer.loss_values
    elif config.synthesizer == "opacus_ctgan":
        synthesizer = OpacusCTGAN(**(config.synthesizer_params or {}), verbose=True)
        synthesizer.fit_transformer(data, discrete_columns)
        synthesizer.fit(data, discrete_columns=discrete_columns)
        train_logs = synthesizer.loss_values
    elif config.synthesizer == "tabdiff":
        # weare just using tabdiff's internal logic for checkpointing. I'm not
        # going to spend time making it work like sdv, so we will just set the
        # synehtsizer object to None
        train_tabdiff(
            proc,
            train_params=config.train_params,
            cache_path=config.cache_path,
            split_override={'train': [data.iloc[:, :-1], data.iloc[:, -1]]},
        )

    if synthesizer is not None:
        joblib.dump(synthesizer, str(config.cache_path / "model.joblib"))
        print("saved checkpoint to:", str(config.cache_path / "model.joblib"))

    if train_logs is not None:
        train_logs.to_csv(config.cache_path / "train_logs.csv", index=False)
        print("saved training logs to:", str(config.cache_path / "train_logs.json"))

    config.completion_marker.touch()


def generate(
    synth_config: SynthesizerConfig,
    gen_config: GenerationConfig,
):
    if not synth_config.completion_marker.exists():
        raise FileNotFoundError(
            f"The training did not complete at {str(synth_config.cache_path)}. skipping"
        )
        return

    proc = synth_config.get_processor().prepare()

    n_sample_rows = gen_config.n_rows
    if n_sample_rows is None:
        X_tr, _ = proc.get_split("train")
        n_sample_rows = len(X_tr)

    if synth_config.synthesizer == "tabdiff":
        _gen = eval_tabdiff(
            proc,
            train_params=synth_config.train_params,
            n_rows=n_sample_rows,
        )
    elif synth_config.synthesizer in ["ctgan", "tvae", "gaussian", "dp_ctgan", "dp_tvae", "iter_ctgan", "opacus_ctgan"]:
        if not (synth_config.cache_path / "model.joblib").exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {str(synth_config.cache_path / 'model.joblib')}"
            )
        synthesizer = joblib.load(synth_config.cache_path / "model.joblib")
        _gen = synthesizer.sample(num_rows=n_sample_rows)
    elif (synth_config.cache_path / "model.joblib").exists():
        synthesizer = joblib.load(synth_config.cache_path / "model.joblib")
        _gen = synthesizer.sample(num_rows=n_sample_rows)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {str(synth_config.cache_path / 'model.joblib')}"
        )

    # just in case!
    if not isinstance(_gen, pd.DataFrame):
        print(f"generated data is not a DataFrame but {type(_gen)}, take a look:")
        breakpoint()
        _gen = pd.DataFrame(_gen)
    data_save_path = synth_config.get_data_save_path(gen_config)
    _gen.to_csv(data_save_path, index=False)
    print("data is saved to:", str(data_save_path))
    print(np.bincount(_gen.iloc[:, -1]))


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Train a synthesizer and generate data.")
    parser.add_argument("synth_configs", nargs='*', type=str,                   help="Configuration for training the synthesizer.")
    parser.add_argument("-g", "--gen",  type=str,   default=None,   help="Configuration for generating using the trained synthesizer.",)
    parser.add_argument("-s", "--random-seed",  type=int,   default=None,   help="Seed PRNGs before generating.",)
    parser.add_argument("--train-only",   action="store_true",    help="Only train the synthesizer, do not generate data.")
    parser.add_argument("--overwrite-train",      action="store_true",    help="Whether to overwrite existing results.")
    parser.add_argument("-n", "--name-only", action="store_true", help="Don't train the synthesizer, just print out filenames")
    args = parser.parse_args()
    # fmt: on

    for synth_config_file in args.synth_configs:
        synth_config = SynthesizerConfig.from_yaml(synth_config_file)
        if args.random_seed is not None:
            synth_config.main_seed = args.random_seed

        if args.name_only:
            print(synth_config.cache_path)
            continue

        if args.gen is not None:
            gen_config = GenerationConfig.from_yaml(args.gen)
        else:
            print("no generation config provided, using default.")
            gen_config = GenerationConfig()
        print("parsed configurations.")

        cp_path = train(synth_config, overwrite=args.overwrite_train)
        if not args.train_only:
            generate(synth_config, gen_config)


if __name__ == "__main__":
    main()
