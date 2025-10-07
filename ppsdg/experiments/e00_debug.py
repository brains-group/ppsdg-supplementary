# %%
"""Run this script like `python -m ppsdg.experiments.e00_debug` And run the
commands that it says are incomplete.

For MLE with HPO, use `--tune` flag with `eval-mle`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sdmetrics.reports.single_table import QualityReport
from tabkit.data import DatasetConfig, TableProcessor, TableProcessorConfig

from ppsdg.evaluate.mle import EvalMLEConfig
from ppsdg.evaluate.privacy import EvalPrivacyConfig
from ppsdg.evaluate.train_and_generate import GenerationConfig, SynthesizerConfig

from . import global_config

# data_dir = Path(os.environ["DATA_DIR"])


exp_dir = Path("exp/e00_debug")
exp_dir.mkdir(parents=True, exist_ok=True)

datasets = global_config.datasets
pretty_dset_names = global_config.pretty_dset_names
mle_target = global_config.mle_target
mle_metric = global_config.mle_metric
privacy_metrics = global_config.privacy_metrics

synthesizers = ["gaussian", "ctgan", "tvae", "tabdiff"]
pretty_synth_names = ["Gaussian", "CTGAN", "TVAE", "TabDiff"]
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
        config_name = f"{dname}_{synth}"
        # first generate the synth config
        synth_config = SynthesizerConfig(
            config_name=config_name,
            synthesizer=synth,
            dataset=dset,
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
        config_name = f"{dname}_{synth}_mle"
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

        config_name = f"{dname}_{synth}_privacy"
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

if False and incomplete_runs:
    print("\n### incomplete runs:\n")
    for run in incomplete_runs:
        print(run)
    print()

mle_rep = pd.DataFrame(mle_report)
data_rep = pd.DataFrame(data_report)

mle_metrics = list(set(mle_rep.columns) - set(["dataset", "synthesizer", "config"]))
data_metrics = list(set(data_rep.columns) - set(["dataset", "synthesizer", "config"]))


baselines = mle_rep[mle_rep["synthesizer"].isna()]

new_mle_rep = []
for dset, grp in mle_rep[mle_rep["dataset"].isin(baselines["dataset"])].groupby(
    "dataset"
):
    baseline = grp[grp["synthesizer"].isna()]
    synth = grp[grp["synthesizer"].notna()].copy()
    if synth.empty:
        # only gather results if we have synth results
        continue
    for m in mle_metrics:
        synth[f"{m}-delta"] = synth[m] - baseline[m].mean()
        synth[f"{m}-ratio"] = synth[m] / baseline[m].mean()
        synth[f"{m}-baseline"] = baseline[m].mean()
    new_mle_rep.append(synth)
mle_rep = pd.concat(new_mle_rep, ignore_index=True).reset_index(drop=True)

# now only keep datasets we have all results for
common_datasets = set(baselines["dataset"])
for synth in synthesizers:
    synth_dsets = set(mle_rep[mle_rep["synthesizer"] == synth]["dataset"])
    common_datasets = common_datasets.intersection(synth_dsets)
mle_rep = mle_rep[mle_rep["dataset"].isin(common_datasets)].reset_index(drop=True)
data_rep = data_rep[data_rep["dataset"].isin(common_datasets)].reset_index(drop=True)
print(f"Using {len(common_datasets)} datasets: {sorted(common_datasets)}")

# table = mle_rep[["dataset", "synthesizer"] + [f"{m}-ratio" for m in mle_metrics]].copy()
metric_names = [m.split("/")[0] + "_ratio" for m in mle_metrics]
# table = table.rename(
#     columns={f"{m}-ratio": n for m, n in zip(mle_metrics, metric_names)},
# )


# %%

# dataset stats
print("\n### Dataset Stats\n")

ds_stats = []

for d in sorted(common_datasets):
    config = DatasetConfig.from_yaml(datasets[d])
    proc = TableProcessor(
        dataset_config=config,
        config=TableProcessorConfig(),
    ).prepare()
    X, y = proc.get_split("all")
    ds_stats.append(
        {
            "dataset": d,
            "n_rows": proc.n_samples,
            "n_cols": proc.n_cols,
            "n_cat": sum(c.is_cat or c.is_bin for c in proc.columns_info),
            "n_cont": sum(c.is_cont for c in proc.columns_info),
            "minority_class_pct": (y.value_counts() / len(y)).min(),
        }
    )
ds_stats = pd.DataFrame(ds_stats)
ds_stats["dataset"] = ds_stats["dataset"].map(pretty_dset_names)
ds_stats.rename(
    columns={
        "dataset": "Dataset",
        "n_rows": "\# Rows",
        "n_cols": "\# Columns",
        "n_cat": "\# Cat.",
        "n_cont": "\# Cont.",
        "minority_class_pct": "Minority Pct.",
    },
    inplace=True,
)
print(ds_stats.to_markdown())

ds_stats.style.hide(axis="index").format(
    {
        "\# Rows": "{:,}",
        "\# Cols": "{:,}",
        "\# Cat": "{:,}",
        "\# Cont": "{:,}",
        "Minority Pct.": lambda s: "{:.2f}\%".format(s * 100),
    }
).to_latex(
    exp_dir / "dataset_stats.tex",
    caption="Statistics of datasets used in the experiments.",
    label="tab:dataset_stats",
    hrules=True,
    position_float="centering",
)

# %%

print("\n### MLE Report\n")
# print(table.groupby("synthesizer")[metric_names].mean().to_markdown())

# table[["dataset", "synthesizer", "test_ratio"]].groupby("synthesizer")["test_ratio"].median().T

avg = (
    mle_rep[["dataset", "synthesizer", "test/balanced_accuracy-ratio"]]
    .groupby("synthesizer")[["test/balanced_accuracy-ratio"]]
    .mean()
    .T
)
std = (
    mle_rep[["dataset", "synthesizer", "test/balanced_accuracy-ratio"]]
    .groupby("synthesizer")[["test/balanced_accuracy-ratio"]]
    .std()
    .T
)

# join these two tables together. Need for format by using $\pm$.
table = pd.DataFrame()
for synth in avg.columns:
    formatted = ["{:.4f} ± {:.4f}".format(a, s) for a, s in zip(avg[synth], std[synth])]
    table[synth] = formatted
table.index = avg.index
table.columns = [snames.get(c, c) for c in table.columns]


def fmt_cell(v, style: str):
    if isinstance(v, (float, int, np.number)):
        v = f"{v:.4f}"
    else:
        if style == "tex":
            v = v.replace("±", r"$\pm$")
    return v


def bold_best_vals(
    df: pd.DataFrame,
    axis: int,
    style: str = "md",
    direction: str = "max",
):
    def _bold(v):
        if style == "md":
            return f"**{v}**"
        elif style == "tex":
            return r"\textbf{" + str(v) + "}"

    # prepare the function to apply
    def _bold_best(x):
        if isinstance(x.values[0], str):
            # we will use the first part of the string to determine max
            nums = [float(v.split("±")[0].strip()) for v in x.values]
        else:
            nums = x.values
        if direction == "max":
            best_val = max(nums)
        else:
            best_val = min(nums)
        out = [
            _bold(fmt_cell(xx, style)) if n == best_val else fmt_cell(xx, style)
            for xx, n in zip(x.values, nums)
        ]
        return pd.Series(out, index=x.index, name=x.name)

    return df.apply(_bold_best, axis=axis)


print(bold_best_vals(table, axis=1, style="md"))

with open(exp_dir / "mle_report.tex", "w") as f:
    f.write(
        bold_best_vals(table, axis=1, style="tex")
        .style.hide(axis="index")
        .to_latex(
            caption="Mean and median of test balanced accuracy ratio (synthesized / original) across datasets.",
            label="tab:mle_report",
            hrules=True,
            position_float="centering",
        )
    )

# %%

print("\n### MLE per dataset \n")

mle_w_baselines = pd.concat([baselines, mle_rep], ignore_index=True).reset_index(
    drop=True
)

# use example datases
examples = list(sorted(common_datasets))

ex_tabs = []
for e in examples:
    tab = (
        mle_w_baselines[mle_w_baselines["dataset"] == e][
            ["synthesizer", "test/balanced_accuracy"]
        ]
        .copy()
        .set_index("synthesizer")
        .T
    )
    tab.columns = [
        snames.get(c, c) if c is not None else "Original" for c in tab.columns
    ]
    tab["Dataset"] = e
    ex_tabs.append(tab)
ex_tab = pd.concat(ex_tabs, ignore_index=True).reset_index(drop=True)
ex_tab = ex_tab[["Dataset"] + [c for c in ex_tab.columns if c != "Dataset"]]
print("\n**Balanced accuracy**\n")
print(ex_tab.to_markdown(index=False))

ex_tab["Dataset"] = ex_tab["Dataset"].map(pretty_dset_names)
ex_tab.style.hide(axis="index").format(
    {c: "{:.4f}" for c in ex_tab.columns if c != "Dataset"}
).highlight_max(
    axis=1,
    props="textbf:--rwrap;",
    subset=[c for c in ex_tab.columns if c not in ["Dataset", "Original"]],
).to_latex(
    exp_dir / "mle_report_examples.tex",
    caption="Test balanced accuracy for individual datasets.",
    hrules=True,
    position_float="centering",
)

# %%

print("\n### Privacy Report\n")
# now do privacy report. same deal
avg = data_rep.groupby("synthesizer")[data_metrics].mean().T.sort_index()
std = data_rep.groupby("synthesizer")[data_metrics].std().T.sort_index()
# med_tab = data_rep.groupby("synthesizer")[data_metrics].median().T.sort_index()
table = pd.DataFrame()
for synth in avg.columns:
    formatted = ["{:.4f} ± {:.4f}".format(a, s) for a, s in zip(avg[synth], std[synth])]
    table[synth] = formatted
table.index = avg.index
table.columns = [snames.get(c, c) for c in table.columns]
print(bold_best_vals(table, axis=1, style="md").to_markdown())
with open(exp_dir / "data_report.tex", "w") as f:
    f.write(
        bold_best_vals(table, axis=1, style="tex").style.to_latex(
            caption="Mean and standard deviation of data quality metrics across datasets.",
            label="tab:mle_report",
            hrules=True,
            position_float="centering",
        )
    )

# %%

# check the label balance of the original vs synthetic data
lab_bals = []
for ds, by_synth in synth_data.items():
    if ds not in common_datasets:
        continue
    print(f"\n### {ds} label balance\n")
    for synth, (_, y) in by_synth.items():
        print(f"{synth}: {y.value_counts(normalize=True).to_dict()} (n={len(y)})")
        lab_bals.append(
            {
                "dataset": ds,
                "synthesizer": synth,
                "n_rows": len(y),
                "label_balance": y.value_counts(normalize=True).to_dict(),
                "minority_class_pct": (y.value_counts() / len(y)).min(),
            }
        )
    print()
lab_bals = pd.DataFrame(lab_bals)
lab_bals = lab_bals[lab_bals["dataset"].isin(common_datasets)]
original = lab_bals[lab_bals["synthesizer"] == "original"].copy()
synth_delta = []
for (ds, synth), _data in lab_bals.groupby(["dataset", "synthesizer"]):
    if synth == "original":
        continue
    _data = _data.copy()
    _data["minority_class_pct-delta"] = (
        _data["minority_class_pct"]
        - original[original["dataset"] == ds]["minority_class_pct"].values[0]
    )
    _data["label_balance-delta"] = _data["label_balance"].apply(
        lambda x: {
            k: f"{v - original[original['dataset'] == ds]['label_balance'].values[0][k]:.2%}"
            for k, v in x.items()
        }
    )
    _data["label_balance"] = _data["label_balance"].apply(
        lambda x: {k: f"{v:.2%}" for k, v in x.items()}
    )
    synth_delta.append(_data)
sdf = pd.concat(synth_delta, ignore_index=True).reset_index(drop=True)
sdf.pivot_table(
    index="dataset",
    columns="synthesizer",
    values="minority_class_pct-delta",
).reset_index().style.hide(axis="index").format(
    lambda x: f"{100 * x:.2f}\\%", subset=sdf["synthesizer"].unique().tolist()
).to_latex(
    exp_dir / "label_balance_delta.tex",
    caption="Minority class percentage delta (synthesized - original) across datasets.",
    label="tab:label_balance_delta",
    hrules=True,
    position_float="centering",
)

# %%
"""
Table: label balance shift vs. MLE shift per dataset

| Dataset |            Syn1          | ... |
|         | Label Shift | MLE Shift  | ... | 
| Bank    |  \down 30%  | \down 0.1% | ... | 
"""

table = []
for ds, _grp in lab_bals.groupby("dataset"):
    ori = _grp[_grp["synthesizer"] == "original"]["minority_class_pct"].values[0]
    synth = _grp[_grp["synthesizer"] != "original"].copy()
    # synth["minority_class_pct"] -= ori
    one_row = {
        ("Dataset", ""): ds,
    }
    for syn, _sgrp in synth.groupby("synthesizer"):
        mle_ori = mle_rep[(mle_rep["dataset"] == ds) & (mle_rep["synthesizer"] == syn)][
            "test/balanced_accuracy-baseline"
        ].values[0]
        mle_synth = mle_rep[
            (mle_rep["dataset"] == ds) & (mle_rep["synthesizer"] == syn)
        ]["test/balanced_accuracy"].values[0]
        one_row[(syn, "Label Shift")] = (
            "{:.2f}\% $\\rightarrow$ {:.2f}\% ({:.2f}\%)".format(
                ori * 100,
                _sgrp["minority_class_pct"].values[0] * 100,
                ((_sgrp["minority_class_pct"].values[0] - ori) / ori) * 100,
            )
        )
        one_row[(syn, "MLE Shift")] = (
            "{:.2f}\% $\\rightarrow$ {:.2f}\% ({:.2f}\%)".format(
                mle_ori * 100,
                mle_synth * 100,
                ((mle_synth - mle_ori) / mle_ori) * 100,
            )
        )
    table.append(one_row)
table = pd.DataFrame(table)
table.columns = pd.MultiIndex.from_tuples(table.columns)

table.style.hide(axis="index").to_latex(
    exp_dir / "label_balance_vs_mle_shift.tex",
    caption="Label balance shift vs. MLE shift per dataset.",
    label="tab:label_balance_vs_mle_shift",
    hrules=True,
    position_float="centering",
    multicol_align="c",
    environment="tinytable",
)


# %%
"""
Figure: scatter plot between the MLE shift and the label balance shift for each dataset/synth combination.
"""

plot_data = []
for ds, _grp in lab_bals.groupby("dataset"):
    ori = _grp[_grp["synthesizer"] == "original"]["minority_class_pct"].values[0]
    synth = _grp[_grp["synthesizer"] != "original"].copy()
    for syn, _sgrp in synth.groupby("synthesizer"):
        mle_ori = mle_rep[(mle_rep["dataset"] == ds) & (mle_rep["synthesizer"] == syn)][
            "test/balanced_accuracy-baseline"
        ].values[0]
        mle_synth = mle_rep[
            (mle_rep["dataset"] == ds) & (mle_rep["synthesizer"] == syn)
        ]["test/balanced_accuracy"].values[0]
        plot_data.append(
            {
                "dataset": ds,
                "synthesizer": syn,
                "label_shift": ((_sgrp["minority_class_pct"].values[0] - ori) / ori),
                "mle_shift": ((mle_synth - mle_ori) / mle_ori),
            }
        )
plot_data = pd.DataFrame(plot_data)

fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(
    data=plot_data,
    x="label_shift",
    y="mle_shift",
    hue="dataset",
    style="synthesizer",
    ax=ax,
)
ax.set_xlabel("Label Balance Shift (synthesized - original)")
ax.set_ylabel("MLE Shift (synthesized - original)")
ax.set_title("Label Balance Shift vs. MLE Shift")
ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1, 1),
    title="Dataset / Synthesizer",
    title_fontsize="13",
    fontsize="11",
)
fig.tight_layout()
fig.savefig(exp_dir / "label_balance_vs_mle_shift.pdf", bbox_inches="tight")

# %%
"""

Table: divide the quality score into cont. vs. cat. for each synthesizer

| Dataset |     Syn1   | ... |
|         | Cat | Cont | ... | 
| Bank    | 0.1 | 0.2  | ... |

(use the below snippet to load the full report)

"""

table = []
for ds, _grp in mle_rep.groupby("dataset"):
    one_row = {("Dataset", ""): ds}
    for syn, _sgrp in _grp.groupby("synthesizer"):
        pconf = privacy_configs["{}_{}_privacy".format(ds, syn)]
        q_rep = QualityReport.load(filepath=str(pconf.save_dir / "quality_report.pkl"))
        dt = q_rep.get_details(property_name="Column Shapes")
        cat = dt[dt["Metric"] == "TVComplement"]["Score"].values
        cont = dt[dt["Metric"] == "KSComplement"]["Score"].values
        one_row[(syn, "Cat")] = "{:.4f} $\pm$ {:.4f}, {:.4f}".format(
            np.mean(cat), np.std(cat), np.median(cat)
        )
        one_row[(syn, "Cont")] = "{:.4f} $\pm$ {:.4f}, {:.4f}".format(
            np.mean(cont), np.std(cont), np.median(cont)
        )
    table.append(one_row)
table = pd.DataFrame(table)
table.columns = pd.MultiIndex.from_tuples(table.columns)
table.style.hide(axis="index").to_latex(
    exp_dir / "quality_report_per_coltype.tex",
    caption="Breakdown of quality report into categorical vs. continuous columns per dataset and synthesizer.",
    label="tab:quality_report",
    hrules=True,
    position_float="centering",
    multicol_align="c",
    environment="tinytable",
)
