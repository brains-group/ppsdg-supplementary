"""This is where we apply the changes to integrate tabdiff."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabkit import TableProcessor

from .config_tabdiff import TabDiffConfig
from .main import TABDIFF_CACHE_DIR, build_trainer


def get_dataset_info(proc: TableProcessor):
    columns_info = proc.columns_info + [proc.label_info]

    # IMPORTANT: we need to remove the last one when doing indices because that's how they did it
    num_col_idx = [i for i, col in enumerate(columns_info[:-1]) if col.is_cont]
    cat_col_idx = [i for i, col in enumerate(columns_info[:-1]) if not col.is_cont]

    # we can't have a column that is neither, nor have overlaps here.
    # mostly a just-in-case check.
    if set(num_col_idx) & set(cat_col_idx):
        raise ValueError(
            "There are columns that are both numerical and categorical, or neither: {}".format(
                columns_info[i].name for i in set(num_col_idx) & set(cat_col_idx)
            )
        )
    if len(num_col_idx) + len(cat_col_idx) != len(columns_info) - 1:
        raise ValueError(
            "There are columns that are neither numerical nor categorical: {}".format(
                columns_info[i].name
                for i in range(len(columns_info))
                if i not in num_col_idx and i not in cat_col_idx
            )
        )

    target_col_idx = len(columns_info) - 1
    sd_cols_data = {}

    for i in num_col_idx:
        sd_cols_data[i] = {}
        sd_cols_data[i]["sdtype"] = "numerical"
        sd_cols_data[i]["computer_representation"] = "Float"
    for i in cat_col_idx:
        sd_cols_data[i] = {}
        sd_cols_data[i]["sdtype"] = "categorical"
    if proc.label_info.is_cont:
        sd_cols_data[target_col_idx] = {}
        sd_cols_data[target_col_idx]["sdtype"] = "numerical"
        sd_cols_data[target_col_idx]["computer_representation"] = "Float"
    else:
        sd_cols_data[target_col_idx] = {}
        sd_cols_data[target_col_idx]["sdtype"] = "categorical"

    # they sort the columns such that the continuous ones are all grouped in
    # the front and the categorical ones follow, ending with the target.
    idx_mapping = dict(enumerate(num_col_idx + cat_col_idx + [target_col_idx]))

    info = {
        "name": proc.dataset_config.config_name,
        "task_type": "binclass" if proc.label_info.is_bin else "regression",
        "column_names": [col.name for col in columns_info],
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": [target_col_idx],  # last column is the target
        "file_type": "csv",
        "train_num": len(proc.get("train_idxs")),
        "test_num": len(proc.get("test_idxs")),
        "int_col_idx": num_col_idx,
        "int_columns": [columns_info[i].name for i in num_col_idx],
        "int_col_idx_wrt_num": list(range(len(num_col_idx))),
        "val_num": len(proc.get("val_idxs")),
        "idx_mapping": {str(v): k for k, v in idx_mapping.items()},
        "inverse_idx_mapping": {str(k): v for k, v in idx_mapping.items()},
        "idx_name_mapping": {
            str(i): columns_info[i].name for i in range(len(columns_info))
        },
        "metadata": {
            "columns": {str(k): v for k, v in sd_cols_data.items()},
        },
    }

    return info


def prepare_data(proc: TableProcessor, data_dir: str):
    cat_cols = [i + 1 for i, col in enumerate(proc.columns_info) if not col.is_cont]
    num_cols = [i + 1 for i, col in enumerate(proc.columns_info) if col.is_cont]
    label_col = proc.label_info.name
    for sp in ["train", "test", "val"]:
        X, y = proc.get_split(f"{sp}")
        df = pd.concat([y.to_frame(), X], axis=1)
        df.columns = list(range(len(proc.columns_info) + 1))
        for i in cat_cols:
            df.loc[:, i] = df.loc[:, i].astype("object")
            df.loc[:, i] = (
                df.loc[:, i]
                .map(dict(enumerate(proc.columns_info[i - 1].mapping)))
                .fillna("Missing" if not proc.columns_info[i - 1].is_cont else 0.0)
            )
        if not proc.label_info.is_cont and proc.label_info.mapping is not None:
            df.loc[:, 0] = (
                df.loc[:, 0]
                .astype(int)
                .map(dict(enumerate(proc.label_info.mapping)))
                .fillna("Missing" if not proc.label_info.is_cont else 0.0)
            )
        np.save(
            data_dir / f"X_num_{sp}.npy",
            df.loc[:, num_cols].to_numpy().astype(np.float32),
        )
        np.save(data_dir / f"X_cat_{sp}.npy", df.loc[:, cat_cols].to_numpy())
        np.save(data_dir / f"y_{sp}.npy", df.loc[:, 0].to_numpy())


def get_tabdiff_config(
    proc: TableProcessor,
    train_params: dict | None = None,
) -> TabDiffConfig:
    # TODO: make the info dict from the proc
    df = proc.get("raw_df")
    # we need to re-order this to make sure the label is first.
    df = df[[proc.label_info.name] + [c.name for c in proc.columns_info]]
    dataset_name = proc.dataset_config.config_name
    dataset_info = get_dataset_info(proc)

    # prepare to save for metrics
    (TABDIFF_CACHE_DIR / f"synthetic/{dataset_name}").mkdir(exist_ok=True, parents=True)
    log_dir = TABDIFF_CACHE_DIR / "log" / dataset_name
    log_dir.mkdir(exist_ok=True, parents=True)
    cp_dir = TABDIFF_CACHE_DIR / "checkpoint" / dataset_name
    cp_dir.mkdir(exist_ok=True, parents=True)

    col_order = [proc.label_info.name] + [c.name for c in proc.columns_info]

    # save the splits into the synthetic data save dir
    tr_idxs = proc.get("train_idxs")
    tr_df = df.loc[tr_idxs, col_order].copy()
    tr_df.to_csv(TABDIFF_CACHE_DIR / f"synthetic/{dataset_name}/real.csv", index=False)
    te_idxs = proc.get("test_idxs")
    te_df = df.loc[te_idxs, col_order].copy()
    te_df.to_csv(TABDIFF_CACHE_DIR / f"synthetic/{dataset_name}/test.csv", index=False)
    va_idxs = proc.get("val_idxs")
    va_df = df.loc[va_idxs, col_order].copy()
    va_df.to_csv(TABDIFF_CACHE_DIR / f"synthetic/{dataset_name}/val.csv", index=False)

    # also need tosave to use with TabDiffDataset...
    data_dir = TABDIFF_CACHE_DIR / "data" / dataset_name
    data_dir.mkdir(exist_ok=True, parents=True)

    prepare_data(proc, data_dir)
    with open(data_dir / "info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    cat_cardinalities = [
        len(col.mapping)
        for col in [proc.label_info] + proc.columns_info
        if not col.is_cont
    ]
    print("cat_cardinalities:", cat_cardinalities)
    print("n_cont:", len(dataset_info["num_col_idx"]))

    config = TabDiffConfig(
        exp_name="testing",
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        data_dir=str(data_dir),
        train_params=train_params,
        model_save_path_base=str(cp_dir),
        result_save_path_base=str(log_dir),
        n_cont=len(dataset_info["num_col_idx"]),
        cat_cardinalities=np.array(cat_cardinalities),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return config


def train_tabdiff(
    proc: TableProcessor,
    train_params: dict | None = None,
    cache_path: Path | None = None,  # SynthesizerConfig-compatible cache path
) -> TabDiffConfig:
    config = get_tabdiff_config(proc, train_params)
    # need to prep the cache directory ahead of time
    trainer = build_trainer(config)
    # NOTE: this is how we will interact with the tabdiff
    # since the DP version will have some more tunable knobs (eps, delta), the
    # `train_params` dictionary should contain that type of information.
    # We also need to have a way to track the hyperparams we set for these.
    trainer.run_loop()
    print("best model should now be at", config.model_save_path)

    # Also copy training logs to SynthesizerConfig cache path if provided
    if cache_path is not None and trainer.train_logs:
        train_logs_df = pd.DataFrame(trainer.train_logs)
        cache_path.mkdir(parents=True, exist_ok=True)
        synth_train_logs_path = cache_path / "train_logs.csv"
        train_logs_df.to_csv(synth_train_logs_path, index=False)
        print(f"Training logs also saved to: {synth_train_logs_path}")

    return config
    # tabdiff_main(config)


def eval_tabdiff(
    proc: TableProcessor,
    n_rows: int,
    train_params: dict | None = None,
) -> pd.DataFrame:
    config = get_tabdiff_config(proc, train_params).copy()

    # HACK: for eval, we first need to set the checkpoint. this is kind of
    # insane -- hash of the config will change after we do this but we need to
    # know how to load the old model.
    cps = sorted(Path(config.model_save_path).glob("*.pt"))
    best_cp = None
    # we now need to pick the best one. first check if we have "best_**" models.
    best_models = [c for c in cps if "best_" in c.name]
    if best_models:
        best_cp = str(best_models[-1])
    else:
        # otherwise, just pick the last one
        best_cp = str(cps[-1])
    config.args.ckpt_path = best_cp
    trainer = build_trainer(config)

    syn_df = trainer.sample_synthetic(
        num_samples=n_rows,
        ema=False,  # not sure what this does!
    )

    # need to bring it back to the encoded format for consistency
    for col in proc.columns_info + [proc.label_info]:
        if col.is_cont:
            continue
        if col.mapping is not None:
            syn_df[col.name] = syn_df[col.name].map(
                {v: k for k, v in enumerate(col.mapping)}
            )

    return syn_df


# just in case we want OOP style
class TabDiffWrapper:
    def __init__(self, proc: TableProcessor, train_params: dict | None = None):
        self.proc = proc
        self.train_params = train_params
        self.config = get_tabdiff_config(proc, train_params)
        self.trainer = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        self.trainer = build_trainer(self.config)
        self.trainer.run_loop()
        self.is_fitted = True
        print("best model should now be at", self.config.model_save_path)

    def sample(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.trainer is None:
            raise ValueError("Model is not fitted yet. Please call fit() first.")
        syn_df = self.trainer.sample_synthetic(
            num_samples=num_rows,
            ema=False,  # not sure what this does!
        )

        # need to bring it back to the encoded format for consistency
        for col in self.proc.columns_info + [self.proc.label_info]:
            if col.is_cont:
                continue
            if col.mapping is not None:
                syn_df[col.name] = syn_df[col.name].map(
                    {v: k for k, v in enumerate(col.mapping)}
                )

        return syn_df
