# %%

"""
Run this script like `python -m ppsdg.experiments.e01_dp_synth` And run the
commands that it says are incomplete.

For MLE with HPO, use `--tune` flag with `eval-mle`.
"""

import json
from pathlib import Path

import pandas as pd
from tabkit.data import DatasetConfig, TableProcessor

from ppsdg.evaluate.mle import EvalMLEConfig
from ppsdg.evaluate.privacy import EvalPrivacyConfig
from ppsdg.evaluate.train_and_generate import GenerationConfig, SynthesizerConfig

from . import global_config as config

# DP parameters
EPSILON = 1.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

exp_dir = Path("exp/e01_dp_synth")
exp_dir.mkdir(parents=True, exist_ok=True)

datasets = config.datasets
pretty_dset_names = config.pretty_dset_names
mle_target = config.mle_target
mle_metric = config.mle_metric
privacy_metrics = config.privacy_metrics

synthesizers = ["dp_ctgan", "dp_tvae"]
pretty_synth_names = ["DP-CTGAN", "DP-TVAE"]
snames = dict(zip(synthesizers, pretty_synth_names))

incomplete_runs = []
mle_report, data_report = [], []
# a higly nested dictioary of dataset -> synthesizer  -> (X, y)
synth_data = {}
privacy_configs = {}

for dname, dset in datasets.items():
    # original mle for comparison
    config_name = f"{dname}_original_mle"
    original_mle_config = EvalMLEConfig(
        config_name=config_name,
        dataset=dset,
        model=mle_target,
        target_metric=mle_metric,
    )
    original_mle_config_path = exp_dir / f"config/{config_name}.yaml"
    original_mle_config.save_yaml(original_mle_config_path)
    if not (original_mle_config.save_dir / "report.json").exists():
        incomplete_runs.append(
            "eval-mle {} --tune".format(str(original_mle_config_path))
        )
    else:
        with open(original_mle_config.save_dir / "report.json") as f:
            mle_report.append(
                {
                    "dataset": dname,
                    "synthesizer": None,
                    "config": config_name,
                    **json.load(f),
                }
            )

    proc = TableProcessor(
        dataset_config=DatasetConfig.from_yaml(dset),
    ).prepare()

    synth_data[dname] = {"original": proc.get_split("train")}

    for synth in synthesizers:
        if synth == "dp_ctgan":
            conds = [
                ("_bce", {"use_bce_loss": True}),
                ("_wgan", {"use_bce_loss": False})
            ]
        else:
            conds = [("", {})]
        for suffix, extra_params in conds:
            config_name = f"{dname}_{synth}{suffix}"
            # first generate the synth config
            synth_config = SynthesizerConfig(
                config_name=config_name,
                synthesizer=synth,
                dataset=dset,
                synthesizer_params={
                    "epsilon": EPSILON,
                    "delta": DELTA,
                    "max_grad_norm": MAX_GRAD_NORM,
                    **extra_params
                },
            )
            gen_config = GenerationConfig()

            # save the configs
            synth_config_path = exp_dir / f"config/{config_name}.yaml"
            synth_config.save_yaml(synth_config_path)
            # don't need to save gen config since its just empty

            data_save_path = synth_config.get_data_save_path(gen_config)

            if not data_save_path.exists():
                incomplete_runs.append("train-gen {}".format(str(synth_config_path)))
                continue

            df = pd.read_csv(data_save_path)
            X = df.drop(columns=[proc.label_info.name])
            y = df[proc.label_info.name]
            synth_data[dname][synth] = (X, y)

            # mle config
            config_name = f"{dname}_{synth}{suffix}_mle"
            mle_config = EvalMLEConfig(
                config_name=config_name,
                dataset=dset,
                model=mle_target,
                target_metric=mle_metric,
                train_override_df=str(data_save_path),
            )
            mle_config_path = exp_dir / f"config/{config_name}.yaml"
            mle_config.save_yaml(mle_config_path)
            if not (mle_config.save_dir / "report.json").exists():
                incomplete_runs.append("eval-mle {} --tune".format(str(mle_config_path)))
            else:
                with open(mle_config.save_dir / "report.json") as f:
                    mle_report.append(
                        {
                            "dataset": dname,
                            "synthesizer": synth,
                            "config": config_name,
                            **json.load(f),
                        }
                    )

            config_name = f"{dname}_{synth}{suffix}_privacy"
            # privacy measure config
            privacy_config = EvalPrivacyConfig(
                config_name=config_name,
                dataset=dset,
                synthetic_data=str(data_save_path),
                metrics=privacy_metrics,
            )
            privacy_configs[config_name] = privacy_config

            privacy_config_path = exp_dir / f"config/{config_name}.yaml"
            privacy_config.save_yaml(privacy_config_path)
            if not (privacy_config.save_dir / "report.json").exists():
                incomplete_runs.append("eval-privacy {}".format(str(privacy_config_path)))
            else:
                with open(privacy_config.save_dir / "report.json") as f:
                    data_report.append(
                        {
                            "dataset": dname,
                            "synthesizer": synth,
                            "config": config_name,
                            **json.load(f),
                        }
                    )

if incomplete_runs:
    print("\n### incomplete runs:\n")
    for run in incomplete_runs:
        print(run)
    print()

mle_rep = pd.DataFrame(mle_report)
data_rep = pd.DataFrame(data_report)
