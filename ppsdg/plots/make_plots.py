# %%

import itertools
from pathlib import Path

import pandas as pd

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

model_pretty_names = {
    "original": "Original",
    "gaussian": "Gaussian",
    "tabdiff": "TabDiff",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "dp_ctgan": "DP-CTGAN",
    "dp_tvae": "DP-TVAE",
}

df_acc = pd.read_csv("ba_summary.csv")
df_bal = pd.read_csv("class_summary.csv")

# update model names
df_acc["Model"] = df_acc["Model"].map(model_pretty_names)
df_bal["Model"] = df_bal["Model"].map(model_pretty_names)

# fill NA epsilon (non private) with -1 for easier filtering
df_acc["Eps"] = df_acc["Eps"].fillna(-1)
df_bal["Eps"] = df_bal["Eps"].fillna(-1)

# First, plot average downstream accuarcy vs. class balance
# let's make a new df with this info.
methods = ["Gaussian", "TabDiff", "CTGAN", "TVAE", "DP-CTGAN", "DP-TVAE"]
datasets = ["AD", "BC", "BM", "CC", "CR", "GM", "PW"]
epsilons = [0, 1, 5, 10, -1]

df_plt = []
for m, d, e in itertools.product(methods, datasets, epsilons):
    accs = df_acc[
        (df_acc["Model"] == m) & (df_acc["Dataset"] == d) & (df_acc["Eps"] == e)
    ]["test/balanced_accuracy"].values
    bal_base = df_bal[(df_bal["Model"] == "Original") & (df_bal["Dataset"] == d)][
        "p"
    ].values[0]
    bal = df_bal[
        (df_bal["Model"] == m) & (df_bal["Dataset"] == d) & (df_bal["Eps"] == e)
    ]["p"].values

    if len(accs) == 0 or len(bal) == 0:
        continue

    df_plt.append(
        {
            "Model": m,
            "Dataset": d,
            "BACC-Mean": accs.mean(),
            "BACC-STD": accs.std(),
            "P-Mean": bal.mean(),
            "P-STD": bal.std(),
            "P-Delta": bal.mean() - bal_base,
            "Eps": e,
        }
    )

df_plt = pd.DataFrame(df_plt)

# %%


"""
       Model Dataset  BACC-Mean  BACC-STD    P-Mean         P-STD   P-Delta  Eps
0   gaussian      AD   0.516910  0.003369  0.241503  2.775558e-17  0.002221   -1
1   gaussian      BC   0.539075  0.001765  0.210127  0.000000e+00  0.006427   -1
2   gaussian      BM   0.506775  0.001465  0.117577  0.000000e+00  0.000592   -1
3   gaussian      CC   0.546787  0.007220  0.219835  0.000000e+00 -0.001365   -1
4   gaussian      CR   0.533966  0.021083  0.697635  0.000000e+00 -0.002365   -1
5   gaussian      GM   0.502276  0.000693  0.066893  0.000000e+00  0.000053   -1
6   gaussian      PW   0.849830  0.008157  0.558182  0.000000e+00  0.001239   -1
7    tabdiff      AD   0.684064  0.003311  0.232741  1.261184e-03 -0.006541   -1
8    tabdiff      BM   0.581695  0.001695  0.111773  2.613684e-03 -0.005212   -1
9    tabdiff      CC   0.655536  0.002875  0.202635  5.747308e-03 -0.018565   -1
10   tabdiff      CR   0.619145  0.006068  0.728463  1.769118e-02  0.028463   -1
11   tabdiff      GM   0.584797  0.001340  0.062337  0.000000e+00 -0.004503   -1
12     ctgan      AD   0.718382  0.046949  0.253378  3.547754e-02  0.014097   -1
13     ctgan      BC   0.648815  0.035045  0.305035  5.993539e-02  0.101335   -1
14     ctgan      BM   0.604405  0.036258  0.152094  6.865534e-02  0.035109   -1
15     ctgan      CC   0.586976  0.000074  0.320555  3.849092e-02  0.099355   -1
16     ctgan      CR   0.506613  0.025829  0.669341  3.568100e-02 -0.030659   -1
17     ctgan      GM   0.660456  0.005446  0.345885  0.000000e+00  0.279045   -1
18     ctgan      PW   0.923292  0.002759  0.521693  2.235293e-02 -0.035249   -1
19      tvae      AD   0.769173  0.008551  0.227273  2.070965e-02 -0.012009   -1
20      tvae      BC   0.627242  0.023688  0.134402  1.803741e-02 -0.069298   -1
21      tvae      BM   0.603130  0.009650  0.170430  4.695717e-02  0.053446   -1
22      tvae      CC   0.635018  0.000896  0.146819  1.500630e-02 -0.074381   -1
23      tvae      CR   0.552123  0.005855  0.872748  4.433563e-03  0.172748   -1
24      tvae      PW   0.903703  0.004708  0.550680  1.652555e-03 -0.006262   -1
25  dp_ctgan      AD   0.784189  0.004280  0.263117  6.207950e-03  0.023835    0
26  dp_ctgan      AD   0.681957  0.034002  0.196313  4.620356e-02 -0.042969    1
27  dp_ctgan      AD   0.709075  0.034661  0.196526  4.432642e-02 -0.042756    5
28  dp_ctgan      AD   0.734034  0.025794  0.187125  3.222766e-02 -0.052157   10
29  dp_ctgan      BC   0.648338  0.009238  0.201181  1.213431e-02 -0.002519    0
...
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# Plot 1: Non-private methods (Eps == NA)
df_nonprivate = df_plt[df_plt["Eps"] == -1]
df_nonprivate_agg = (
    df_nonprivate.groupby("Model")
    .agg(
        {
            "BACC-Mean": "mean",
            "BACC-STD": "mean",
            "P-Mean": "mean",
            "P-STD": "mean",
            "P-Delta": "mean",
        }
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(4, 3))

from matplotlib.lines import Line2D

legend_handles = []

for model in df_nonprivate_agg["Model"]:
    data = df_nonprivate_agg[df_nonprivate_agg["Model"] == model]
    line = ax.errorbar(
        # data["P-Mean"],
        data["P-Delta"],
        data["BACC-Mean"],
        xerr=data["P-STD"],
        yerr=data["BACC-STD"],
        fmt="o",
        markersize=6,
        capsize=3,
        capthick=1,
        elinewidth=1,
    )
    # Create custom legend handle with just a solid circle
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=line.lines[0].get_color(),
            markersize=6,
            label=model,
        )
    )

ax.set_xlabel("Min. Class Balance Shift", fontsize=10)
ax.set_ylabel("Balanced Accuracy (BACC)", fontsize=10)
ax.legend(handles=legend_handles, fontsize=8, handlelength=1, handletextpad=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / "bacc_vs_p_nonprivate.pdf", bbox_inches="tight")
plt.close(fig)

# %%
# Plot 2: Private methods with subplots for each epsilon

df_private = df_plt[df_plt["Eps"] != -1]
eps_values = [0, 10, 5, 1]

fig, axes = plt.subplots(
    1, 1, figsize=(3, 3), sharey=True
)

if len(eps_values) == 1:
    axes = [axes]

# Collect all legend handles and colors
all_legend_handles = {}

for idx, eps in enumerate(eps_values):
    ax = axes
    df_eps = df_private[df_private["Eps"] == eps]
    df_eps_agg = (
        df_eps.groupby("Model")
        .agg(
            {
                "BACC-Mean": "mean",
                "BACC-STD": "mean",
                "P-Mean": "mean",
                "P-STD": "mean",
                "P-Delta": "mean",
            }
        )
        .reset_index()
    )

    for model in df_eps_agg["Model"]:
        if model != "DP-CTGAN":
            continue
        data = df_eps_agg[df_eps_agg["Model"] == model]
        line = ax.errorbar(
            data["P-Mean"],
            data["BACC-Mean"],
            xerr=data["P-STD"],
            yerr=data["BACC-STD"],
            fmt="o",
            markersize=5,
            capsize=2,
            capthick=1,
            elinewidth=1,
            label=f"ε={'∞' if eps == 0 else eps}",
        )
        # Store legend handle for this model (only once)
        if model not in all_legend_handles:
            all_legend_handles[model] = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=line.lines[0].get_color(),
                markersize=5,
                label=model,
            )

    ax.set_xlabel("Minority Class Balance Shift", fontsize=9)
    ax.legend(handlelength=1)
    if idx == 0:
        ax.set_ylabel("Balanced Accuracy", fontsize=9)
    ax.set_title(f"DP-CTGAN over all datasets", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

"""
# Add single shared legend
fig.legend(
    handles=list(all_legend_handles.values()),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.00),
    ncol=len(all_legend_handles),
    fontsize=7,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=1,
)
"""

fig.tight_layout()
fig.savefig(plots_dir / "bacc_vs_p_private.pdf", bbox_inches="tight")
plt.close(fig)

# %%

# new plot here


# %%

"""
Some notes:

y: downstream models' balanced accuracy when trained on generated data
x: change in minority class balance between original and generated data

*No privacy*
- Gaussian and TabDiff generally keep the Min. balance the same.
    - But they also perform worse in downstream.

*Privacy*
- CTGAN consistently performs better, somehow flipped it on TVAE!
- Somehow, DP-CTGAn is also better at preserving class balance than DP-TVAE.


TODO:
plot change in performance with DP for CTGAN and TVAE
"""

# %%
# Plot 3: Change in BACC and P-Delta when applying DP
# Compare dp_ctgan vs ctgan and dp_tvae vs tvae at each epsilon level

# Get non-private baselines for ctgan and tvae
df_ctgan_base = df_plt[(df_plt["Model"] == "CTGAN") & (df_plt["Eps"] == -1)]
df_tvae_base = df_plt[(df_plt["Model"] == "TVAE") & (df_plt["Eps"] == -1)]

# Build comparison dataframe
eps_values = [0, 1, 5, 10]
df_dp_change = []

for eps in eps_values:
    for base_model, dp_model in [("CTGAN", "DP-CTGAN"), ("TVAE", "DP-TVAE")]:
        base_df = df_plt[(df_plt["Model"] == base_model) & (df_plt["Eps"] == -1)]
        dp_df = df_plt[(df_plt["Model"] == dp_model) & (df_plt["Eps"] == eps)]

        # Get common datasets
        common_datasets = set(base_df["Dataset"]) & set(dp_df["Dataset"])

        for dataset in common_datasets:
            base_row = base_df[base_df["Dataset"] == dataset].iloc[0]
            dp_row = dp_df[dp_df["Dataset"] == dataset].iloc[0]

            df_dp_change.append(
                {
                    "Model": base_model,
                    "Dataset": dataset,
                    "Eps": eps,
                    "BACC-Change": base_row["BACC-Mean"] - dp_row["BACC-Mean"],
                    "P-Delta-Change": base_row["P-Delta"] - dp_row["P-Delta"],
                }
            )

df_dp_change = pd.DataFrame(df_dp_change)

# %%
# Plot 3a: Change in BACC for each epsilon
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)

model_colors = {"CTGAN": "tab:blue", "TVAE": "tab:orange"}

for idx, eps in enumerate(eps_values):
    ax = axes[idx]
    df_eps = df_dp_change[df_dp_change["Eps"] == eps]

    for model in ["CTGAN", "TVAE"]:
        data = df_eps[df_eps["Model"] == model]
        # Plot individual dataset points
        ax.scatter(
            [model] * len(data),
            data["BACC-Change"],
            color=model_colors[model],
            alpha=0.5,
            s=30,
        )
        # Plot mean
        mean_val = data["BACC-Change"].mean()
        ax.scatter(
            [model],
            [mean_val],
            color=model_colors[model],
            s=100,
            marker="D",
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"ε = {eps}", fontsize=10)
    if idx == 0:
        ax.set_ylabel("ΔBACC (Non-Private - DP)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=8)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="gray", alpha=0.5, label="Individual datasets"),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=8,
        label="Mean",
    ),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=8,
)

fig.tight_layout()
fig.savefig(plots_dir / "bacc_change_dp.pdf", bbox_inches="tight")
plt.close(fig)

# %%
# Plot 3b: Change in P-Delta for each epsilon
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)

for idx, eps in enumerate(eps_values):
    ax = axes[idx]
    df_eps = df_dp_change[df_dp_change["Eps"] == eps]

    for model in ["CTGAN", "TVAE"]:
        data = df_eps[df_eps["Model"] == model]
        # Plot individual dataset points
        ax.scatter(
            [model] * len(data),
            data["P-Delta-Change"],
            color=model_colors[model],
            alpha=0.5,
            s=30,
        )
        # Plot mean
        mean_val = data["P-Delta-Change"].mean()
        ax.scatter(
            [model],
            [mean_val],
            color=model_colors[model],
            s=100,
            marker="D",
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"ε = {eps}", fontsize=10)
    if idx == 0:
        ax.set_ylabel("ΔP-Delta (Non-Private - DP)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=8)

fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=8,
)

fig.tight_layout()
fig.savefig(plots_dir / "p_delta_change_dp.pdf", bbox_inches="tight")
plt.close(fig)

# %%
