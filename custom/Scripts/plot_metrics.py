import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

METHOD_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]
METRIC_PALETTE = [
    "#2A6F97",
    "#E9724C",
    "#82A33D",
    "#B5446E",
    "#6C5B7B",
    "#C9A227",
    "#3E8E7E",
    "#7D7D7D",
]


def color_map(items, palette):
    return {item: palette[i % len(palette)] for i, item in enumerate(items)}


def fmt_value(v):
    """Adaptive formatting"""
    if pd.isna(v):
        return "-"
    if v == 0:
        return "0"
    av = abs(v)
    if av < 1e-3 or av >= 1e4:
        return f"{v:.2e}"
    if av < 1:
        return f"{v:.3f}"
    if av < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def plot_metrics(
    df,
    result_dir="Results",
    show=False,
    verbose=True,
):

    output = os.path.join(result_dir, "metric_summary.png")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )

    if verbose:
        print(df.columns)

    # Create Methods from csv
    if "Spectral" in df.columns:
        df["Method"] = (
            df["Registration"].astype(str) + " + " + df["Spectral"].astype(str)
        )
    else:
        df["Method"] = df["Registration"].astype(str)

    # Detect metrics
    ignore = ["Patient", "Registration", "Spectral", "Method"]
    metrics = [c for c in df.columns if c not in ignore]

    if verbose:
        print("Detected metrics:")
        print(metrics)

    # Check if methods < 2
    methods = list(df["Method"].unique())
    if len(methods) < 2:
        raise RuntimeError("Select at least 2 different methods.")

    method_colors = color_map(methods, METHOD_PALETTE)
    metric_colors = color_map(metrics, METRIC_PALETTE)

    # Line 1 -> 1 Boxplot per metric
    # Line 2 -> Mean±Std | Benefit | Heatmap | Tabelle
    n_metrics = len(metrics)
    cols = max(n_metrics, 4)
    rows = 2

    fig = plt.figure(figsize=(4.6 * cols, 10.5))
    gs = GridSpec(
        rows, cols, figure=fig, hspace=0.65, wspace=0.85, top=0.88, bottom=0.08
    )

    # Line 1
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])

        data = [df.loc[df["Method"] == m, metric].dropna().values for m in methods]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "#222222", "linewidth": 1.6},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "#999999",
                "markeredgecolor": "none",
                "alpha": 0.6,
            },
        )

        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(method_colors[m])
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)

        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(metric)
        _style_axes(ax)

    # Mean plus standard deviation
    ax = fig.add_subplot(gs[1, 0])

    means = df.groupby("Method")[metrics].mean().loc[methods]
    stds = df.groupby("Method")[metrics].std().loc[methods]

    # Metrics are normalized to their respective maximum values for visual comparability.
    # original values are shown as bar labels.
    scale = means.abs().max().replace(0, 1)
    norm_means = means / scale
    norm_stds = stds / scale

    x = np.arange(len(methods))
    width = 0.8 / len(metrics)

    for j, metric in enumerate(metrics):
        bars = ax.bar(
            x + j * width,
            norm_means[metric],
            width,
            yerr=norm_stds[metric],
            label=metric,
            color=metric_colors[metric],
            edgecolor="#333333",
            linewidth=0.6,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#333333"},
        )
        for bar, method in zip(bars, methods):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + norm_stds[metric][method] + 0.02,
                f"{means.loc[method, metric]:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="#333333",
            )

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Value (normalized per metric)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.set_title("Mean ± Std")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=1)
    _style_axes(ax)

    # Benefit (relative change)
    ax = fig.add_subplot(gs[1, 1])
    reference = methods[0]
    benefit = []

    for method in methods[1:]:
        left = df[df["Method"] == reference].set_index("Patient")
        right = df[df["Method"] == method].set_index("Patient")

        common = left.index.intersection(right.index)
        left = left.loc[common]
        right = right.loc[common]

        vals = []
        for metric in metrics:
            b = ((right[metric] - left[metric]) / left[metric]) * 100
            vals.append(b.mean())
        benefit.append(vals)

    benefit = np.array(benefit)
    n_compare = len(methods) - 1
    bwidth = 0.8 / max(n_compare, 1)

    for i, method in enumerate(methods[1:]):
        ax.bar(
            np.arange(len(metrics)) + i * bwidth,
            benefit[i],
            width=bwidth,
            label=method,
            color=method_colors[method],
            edgecolor="#333333",
            linewidth=0.6,
        )

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(len(metrics)) + bwidth * (n_compare - 1) / 2)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Benefit [%]")
    ax.set_title(f"Benefit vs {reference}")
    ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=1)
    _style_axes(ax)

    # Heatmap
    ax = fig.add_subplot(gs[1, 2])

    heat_metric = metrics[0]
    heat = df.pivot_table(values=heat_metric, index="Patient", columns="Method")[
        methods
    ]

    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title(heat_metric)

    # Text color based on brightness
    vmin, vmax = np.nanmin(heat.values), np.nanmax(heat.values)
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            val = heat.values[r, c]
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if norm < 0.6 else "black"
            ax.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03, shrink=0.9)
    cbar.ax.tick_params(labelsize=8)

    # Table with mean values
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")

    table = df.groupby("Method")[metrics].mean().loc[methods]
    cell_text = table.apply(lambda col: col.map(fmt_value)).values

    n_rows, n_cols = table.shape
    fontsize = max(6.5, 9.5 - 0.6 * n_cols)
    rotate_header = n_cols >= 4 or any(len(str(m)) > 8 for m in metrics)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=table.index,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 2.2 if rotate_header else 1.6)

    # Autosize columns
    tbl.auto_set_column_width(col=list(range(-1, n_cols)))

    # Highlight top
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
            if rotate_header:
                cell.set_text_props(
                    rotation=30, ha="left", color="white", fontweight="bold"
                )
                cell.set_height(cell.get_height() * 1.6)
        elif c == -1:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")

    ax.set_title("Mean values")

    fig.suptitle(
        "Comparison of Registration Methods", fontsize=18, fontweight="bold", y=0.97
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if verbose:
        print("Saved:", output)

    return output
