# %%
"""
This is pretty much the same as our e00_debug experiemnt.
But here we want an understanding of what values are good for training the
synthesizers -- especially n_iters, because of privacy budget concerns.
"""

import json
from pathlib import Path

import pandas as pd
from tabkit.data import DatasetConfig, TableProcessor

from ppsdg.evaluate.train_and_generate import GenerationConfig, SynthesizerConfig
from ppsdg.experiments import global_config

exp_dir = Path("exp/e02_synth_training")
exp_dir.mkdir(parents=True, exist_ok=True)

datasets = global_config.datasets
pretty_dset_names = global_config.pretty_dset_names
mle_target = global_config.mle_target
mle_metric = global_config.mle_metric
privacy_metrics = global_config.privacy_metrics

synthesizers = ["ctgan", "tvae", "tabdiff"]
pretty_synth_names = ["CTGAN", "TVAE", "TabDiff"]
snames = dict(zip(synthesizers, pretty_synth_names))
target_metric = "quality"
target_metric_direction = "max"  # this one isn't actually a parameter, it just needs to be set accordingly for the metric type.

# %%
"""
Prepare runs and collect completed stuff
"""

incomplete = []
all_train_logs = {}

for dname, dset in datasets.items():
    proc = TableProcessor(
        dataset_config=DatasetConfig.from_yaml(dset),
    ).prepare()
    for synth in synthesizers:
        # IMPORTANT: the name needs to be different from the main. we don't want to overwrite those results.
        config_name = "{}_{}_testing".format(dname, synth)

        # All synthesizers use SynthesizerConfig - including TabDiff
        synth_config = SynthesizerConfig(
            config_name=config_name,
            synthesizer=synth,
            dataset=dset,
        )
        synth_config_path = exp_dir / f"config/{config_name}.yaml"
        synth_config.save_yaml(synth_config_path)

        if not synth_config.completion_marker.exists():
            incomplete.append(
                "python -m ppsdg.evaluate.train_and_generate {} --train-only".format(
                    synth_config_path
                )
            )
            continue
        else:
            train_logs = pd.read_csv(synth_config.cache_path / "train_logs.csv")
            all_train_logs[(dname, synth)] = train_logs


if incomplete:
    with open(exp_dir / "incomplete.txt", "w") as f:
        f.write("\n".join(incomplete))
    print("incomplete experiments found. see exp/e02_synth_training/incomplete.txt")
tvae_dfs = []
for (dname, synth), logs in all_train_logs.items():
    if synth == "tvae":
        logs_with_dataset = logs.copy()
        logs_with_dataset["dataset"] = dname
        tvae_dfs.append(logs_with_dataset)
tvae_df = pd.concat(tvae_dfs, ignore_index=True) if tvae_dfs else pd.DataFrame()

ctgan_dfs = []
for (dname, synth), logs in all_train_logs.items():
    if synth == "ctgan":
        logs_with_dataset = logs.copy()
        logs_with_dataset["dataset"] = dname
        ctgan_dfs.append(logs_with_dataset)
ctgan_df = pd.concat(ctgan_dfs, ignore_index=True) if ctgan_dfs else pd.DataFrame()

tabdiff_dfs = []
for (dname, synth), logs in all_train_logs.items():
    if synth == "tabdiff":
        logs_with_dataset = logs.copy()
        logs_with_dataset["dataset"] = dname
        tabdiff_dfs.append(logs_with_dataset)
tabdiff_df = (
    pd.concat(tabdiff_dfs, ignore_index=True) if tabdiff_dfs else pd.DataFrame()
)

# %%

"""
TVAE report tells us loss per batch, and CTGAN report tells us generator and
discriminator loss per epoch.
"""
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

plt.style.use("seaborn-v0_8")

# Define a color palette with good separation
plotly_colors = px.colors.qualitative.Plotly  # More muted colors
matplotlib_colors = plt.cm.tab10.colors  # Use tab10 colormap

# TVAE Loss Plot - aggregate per epoch (Plotly interactive)
if not tvae_df.empty:
    # Group by dataset and epoch, then compute mean loss per epoch
    tvae_epoch_means = (
        tvae_df.groupby(["dataset", "Epoch"])["Loss"].mean().reset_index()
    )

    fig = go.Figure()

    for i, dname in enumerate(tvae_epoch_means["dataset"].unique()):
        dataset_data = tvae_epoch_means[tvae_epoch_means["dataset"] == dname]
        fig.add_trace(
            go.Scatter(
                x=dataset_data["Epoch"],
                y=dataset_data["Loss"],
                mode="lines",
                name=pretty_dset_names.get(dname, dname),
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,  # Add smoothing
            )
        )

    fig.update_layout(
        title="TVAE Training Loss by Dataset",
        xaxis_title="Epoch",
        yaxis_title="Mean Loss per Epoch",
        width=800,
        height=500,
        template="plotly_white",
    )

    fig.write_html(exp_dir / "tvae_losses.html")

    # Also save as static matplotlib for PDF
    plt_fig, ax = plt.subplots(figsize=(10, 6))
    for i, dname in enumerate(tvae_epoch_means["dataset"].unique()):
        dataset_data = tvae_epoch_means[tvae_epoch_means["dataset"] == dname]
        ax.plot(
            dataset_data["Epoch"],
            dataset_data["Loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss per Epoch")
    ax.set_title("TVAE Training Loss by Dataset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "tvae_losses.pdf", bbox_inches="tight")
    plt.close()

# CTGAN Loss Plot (Plotly interactive)
if not ctgan_df.empty:
    # Create Plotly subplot
    plotly_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Generator Loss", "Discriminator Loss"),
        horizontal_spacing=0.1,
    )

    dataset_names = ctgan_df["dataset"].unique()

    for i, dname in enumerate(dataset_names):
        dataset_data = ctgan_df[ctgan_df["dataset"] == dname]

        dataset_label = pretty_dset_names.get(dname, dname)

        # Generator Loss
        plotly_fig.add_trace(
            go.Scatter(
                x=dataset_data["Epoch"],
                y=dataset_data["Generator Loss"],
                mode="lines",
                name=dataset_label,
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,
                legendgroup=dataset_label,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Discriminator Loss
        plotly_fig.add_trace(
            go.Scatter(
                x=dataset_data["Epoch"],
                y=dataset_data["Discriminator Loss"],
                mode="lines",
                name=dataset_label,
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,
                legendgroup=dataset_label,
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    plotly_fig.update_layout(
        title="CTGAN Training Losses by Dataset",
        width=1200,
        height=500,
        template="plotly_white",
    )

    plotly_fig.update_xaxes(title_text="Epoch", row=1, col=1)
    plotly_fig.update_xaxes(title_text="Epoch", row=1, col=2)
    plotly_fig.update_yaxes(title_text="Generator Loss", row=1, col=1)
    plotly_fig.update_yaxes(title_text="Discriminator Loss", row=1, col=2)

    plotly_fig.write_html(exp_dir / "ctgan_losses.html")

    # Also save as static matplotlib for PDF
    plt_fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Generator Loss
    ax1 = axes[0]
    for i, dname in enumerate(dataset_names):
        dataset_data = ctgan_df[ctgan_df["dataset"] == dname]
        ax1.plot(
            dataset_data["Epoch"],
            dataset_data["Generator Loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Generator Loss")
    ax1.set_title("CTGAN Generator Loss by Dataset")
    ax1.grid(True, alpha=0.3)

    # Discriminator Loss
    ax2 = axes[1]
    for i, dname in enumerate(dataset_names):
        dataset_data = ctgan_df[ctgan_df["dataset"] == dname]
        ax2.plot(
            dataset_data["Epoch"],
            dataset_data["Discriminator Loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Discriminator Loss")
    ax2.set_title("CTGAN Discriminator Loss by Dataset")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / "ctgan_losses.pdf", bbox_inches="tight")
    plt.close()

# TabDiff Loss Plots (Plotly interactive)
if not tabdiff_df.empty:
    # Create Plotly subplot
    plotly_fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Discrete Loss", "Continuous Loss", "Total Loss"),
        horizontal_spacing=0.08,
    )

    dataset_names = tabdiff_df["dataset"].unique()

    for i, dname in enumerate(dataset_names):
        dataset_data = tabdiff_df[tabdiff_df["dataset"] == dname]
        dataset_label = pretty_dset_names.get(dname, dname)

        # Discrete Loss
        plotly_fig.add_trace(
            go.Scatter(
                x=dataset_data["epoch"],
                y=dataset_data["discrete_loss"],
                mode="lines",
                name=dataset_label,
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,
                legendgroup=dataset_label,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Continuous Loss
        plotly_fig.add_trace(
            go.Scatter(
                x=dataset_data["epoch"],
                y=dataset_data["continuous_loss"],
                mode="lines",
                name=dataset_label,
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,
                legendgroup=dataset_label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Total Loss (with legend and best model markers)
        # First plot the line
        plotly_fig.add_trace(
            go.Scatter(
                x=dataset_data["epoch"],
                y=dataset_data["total_loss"],
                mode="lines",
                name=dataset_label,
                line=dict(color=plotly_colors[i % len(plotly_colors)], width=2),
                line_smoothing=1.3,
                legendgroup=dataset_label,
                showlegend=True,
            ),
            row=1,
            col=3,
        )

        # Add markers for best model selections
        best_model_data = dataset_data[
            dataset_data["is_best_model"] | dataset_data["is_best_ema_model"]
        ]
        if not best_model_data.empty:
            plotly_fig.add_trace(
                go.Scatter(
                    x=best_model_data["epoch"],
                    y=best_model_data["total_loss"],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=10,
                        color=plotly_colors[i % len(plotly_colors)],
                        line=dict(color="black", width=1),
                    ),
                    name=f"{dataset_label} Best",
                    legendgroup=dataset_label,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

    plotly_fig.update_layout(
        title="TabDiff Training Losses by Dataset",
        width=1600,
        height=500,
        template="plotly_white",
    )

    plotly_fig.update_xaxes(title_text="Epoch", row=1, col=1)
    plotly_fig.update_xaxes(title_text="Epoch", row=1, col=2)
    plotly_fig.update_xaxes(title_text="Epoch", row=1, col=3)
    plotly_fig.update_yaxes(title_text="Discrete Loss", row=1, col=1)
    plotly_fig.update_yaxes(title_text="Continuous Loss", row=1, col=2)
    plotly_fig.update_yaxes(title_text="Total Loss", row=1, col=3)

    plotly_fig.write_html(exp_dir / "tabdiff_losses.html")

    # Also save as static matplotlib for PDF
    plt_fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Discrete Loss
    ax1 = axes[0]
    for i, dname in enumerate(dataset_names):
        dataset_data = tabdiff_df[tabdiff_df["dataset"] == dname]
        ax1.plot(
            dataset_data["epoch"],
            dataset_data["discrete_loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Discrete Loss")
    ax1.set_title("TabDiff Discrete Loss by Dataset")
    ax1.grid(True, alpha=0.3)

    # Continuous Loss
    ax2 = axes[1]
    for i, dname in enumerate(dataset_names):
        dataset_data = tabdiff_df[tabdiff_df["dataset"] == dname]
        ax2.plot(
            dataset_data["epoch"],
            dataset_data["continuous_loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Continuous Loss")
    ax2.set_title("TabDiff Continuous Loss by Dataset")
    ax2.grid(True, alpha=0.3)

    # Total Loss with best model markers
    ax3 = axes[2]
    for i, dname in enumerate(dataset_names):
        dataset_data = tabdiff_df[tabdiff_df["dataset"] == dname]
        ax3.plot(
            dataset_data["epoch"],
            dataset_data["total_loss"],
            label=pretty_dset_names.get(dname, dname),
            color=matplotlib_colors[i % len(matplotlib_colors)],
            linewidth=2,
            alpha=0.8,
        )

        # Add markers for best models
        best_model_data = dataset_data[
            dataset_data["is_best_model"] | dataset_data["is_best_ema_model"]
        ]
        if not best_model_data.empty:
            ax3.scatter(
                best_model_data["epoch"],
                best_model_data["total_loss"],
                marker="*",
                s=100,
                color=matplotlib_colors[i % len(matplotlib_colors)],
                edgecolors="black",
                linewidth=1,
                alpha=0.9,
                zorder=5,
            )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Total Loss")
    ax3.set_title("TabDiff Total Loss by Dataset")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / "tabdiff_losses.pdf", bbox_inches="tight")
    plt.close()

# %%
