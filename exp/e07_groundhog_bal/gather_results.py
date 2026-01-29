# %%

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabkit import DatasetConfig, TableProcessor
from tabkit.config import DATA_DIR
from tabkit.utils import Configuration

from ppsdg.evaluate.groundhog import GroundhogAuditConfig
from ppsdg.evaluate.train_and_generate import SynthesizerConfig

exp_dir = Path("exp/e07_groundhog_bal")

# %%


def get_synthname(run_name):
    if "dp_tvae" in run_name:
        return "DP-TVAE"
    elif "tvae" in run_name:
        return "TVAE"
    elif "dp_ctgan" in run_name:
        return "DP-CTGAN"
    elif "ctgan" in run_name:
        return "CTGAN"
    elif "tabdiff" in run_name:
        return "TabDiff"
    elif "gaussian" in run_name:
        return "GaussianCopula"
    elif "original" in run_name:
        return "Original"
    else:
        print(f"Unknown synthesizer in {run_name}")


def get_dataname(run_name):
    return run_name.split("_")[0]


def get_eps(run_name):
    if not "dp" in run_name:
        return 0.0
    dname = get_dataname(run_name)
    return float(run_name.split("dp")[0].replace(dname, "").replace("_", ""))


# groundhog stuff
all_reports = []
for conf in exp_dir.glob("config/*groundhog*.yaml"):
    # need to split the filename
    cc = GroundhogAuditConfig.from_yaml(conf)
    if not cc.save_dir.exists():
        # print(f"Skipping {conf} as {cc.save_dir} does not exist")
        continue
    # else:
    #     print(f"Processing {conf}")
    with open(cc.save_dir / "report.json") as f:
        report = json.load(f)

    proc = TableProcessor(
        dataset_config=DatasetConfig.from_yaml(cc.dataset),
    ).prepare()
    lname = proc.label_info.name
    original_ratio = proc.get_split("train")[1].value_counts(normalize=True, dropna=False)
    min_cls = int(original_ratio.argmin())

    tr_lab_ratios, te_lab_ratios = [], []
    tr_lab_ratio_change, te_lab_ratio_change = [], []
    for dpath in [pp for p in cc.train_models for pp in p]:
        dd = pd.read_csv(dpath)
        ratio = dd[lname].value_counts(normalize=True, dropna=False)
        for i,_ in enumerate(proc.label_info.mapping):
            if i not in ratio:
                ratio[i] = 0.0
        tr_lab_ratios.append(ratio.min())
        tr_lab_ratio_change.append(ratio[min_cls] - original_ratio[min_cls])
    for dpath in [pp for p in cc.test_models for pp in p]:
        dd = pd.read_csv(dpath)
        ratio = dd[lname].value_counts(normalize=True, dropna=False)
        for i,_ in enumerate(proc.label_info.mapping):
            if i not in ratio:
                ratio[i] = 0.0
        te_lab_ratios.append(ratio.min())
        te_lab_ratio_change.append(ratio[min_cls] - original_ratio[min_cls])
    tr_lab_ratios = np.array(tr_lab_ratios)
    te_lab_ratios = np.array(te_lab_ratios)
    tr_lab_ratio_change = np.array(tr_lab_ratio_change)
    te_lab_ratio_change = np.array(te_lab_ratio_change)

    report["synthesizer"] = get_synthname(conf.stem)
    report["dataset"] = get_dataname(conf.stem)
    report["epsilon"] = get_eps(conf.stem)
    report["mean_train_label_ratio"] = tr_lab_ratios.mean()
    report["min_train_label_ratio"] = tr_lab_ratios.min()
    report["mean_test_label_ratio"] = te_lab_ratios.mean()
    report["min_test_label_ratio"] = te_lab_ratios.min()
    report["mean_train_label_ratio_change"] = tr_lab_ratio_change.mean()
    report["min_train_label_ratio_change"] = tr_lab_ratio_change.min()
    report["mean_test_label_ratio_change"] = te_lab_ratio_change.mean()
    report["min_test_label_ratio_change"] = te_lab_ratio_change.min()
    report["train_size"] = cc.train_size
    report["test_size"] = cc.test_size
    all_reports.append(report)
gdf = pd.DataFrame(all_reports)

# %%

import matplotlib.pyplot as plt

# first attempt at plotting this. Let's try
import seaborn as sns

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
sns.scatterplot(
    data=gdf[gdf["synthesizer"].str.contains("DP")],
    x="mean_train_label_ratio",
    y="correct",
    hue="epsilon",
    style="dataset",
    ax=ax,
    s=100,
)
ax.set_title("Per dataset")
# move legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
fig.savefig("per_dset.pdf", bbox_inches="tight")
fig.clear()
# move legend below

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
sns.scatterplot(
    data=gdf[gdf["synthesizer"].str.contains("DP")],
    x="mean_train_label_ratio",
    y="correct",
    hue="epsilon",
    style="synthesizer",
    ax=ax,
    s=100,
)
ax.set_title("Per synthesizer")
# move legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
fig.savefig("per_synth.pdf", bbox_inches="tight")
fig.clear()

# same thing but now for "mean_train_label_ratio_change"
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
sns.scatterplot(
    data=gdf[gdf["synthesizer"].str.contains("DP")],
    x="mean_train_label_ratio_change",
    y="correct",
    hue="epsilon",
    style="dataset",
    ax=ax,
    s=100,
)
ax.set_title("Per dataset")
# move legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
fig.savefig("per_dset_change.pdf", bbox_inches="tight")
fig.clear()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
sns.scatterplot(
    data=gdf[gdf["synthesizer"].str.contains("DP")],
    x="mean_train_label_ratio_change",
    y="correct",
    hue="epsilon",
    style="synthesizer",
    ax=ax,
    s=100,
)
ax.set_title("Per synthesizer")
# move legend outside
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
fig.savefig("per_synth_change.pdf", bbox_inches="tight")
fig.clear()
