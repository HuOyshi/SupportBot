"""Plots saved as PNG and logged to MLflow."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def training_curves(history, path, title):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [h["train_loss"] for h in history], color=SERIES[0], marker="o", ms=4, label="train")
    a1.plot(ep, [h["val_loss"] for h in history], color=SERIES[1], marker="o", ms=4, label="validation")
    a1.set(title="Loss", xlabel="epoch"); a1.legend()
    a2.plot(ep, [h["val_accuracy"] for h in history], color=SERIES[0], marker="o", ms=4, label="accuracy")
    a2.plot(ep, [h["val_macro_f1"] for h in history], color=SERIES[2], marker="o", ms=4, label="macro-F1")
    a2.set(title="Validation accuracy / macro-F1", xlabel="epoch"); a2.legend()
    fig.suptitle(title, color=INK, fontweight="bold")
    return _save(fig, path)


def confusion(cm, labels, path, title):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] and i != j:   # show only the mistakes as numbers
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color=INK)
    ax.set(xlabel="predicted intent", ylabel="true intent", title=title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of true intent")
    return _save(fig, path)


def ood_histogram(in_scores, oos_scores, threshold, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.histogram_bin_edges(np.r_[in_scores, oos_scores], bins=50)
    ax.hist(in_scores, bins=bins, color=SERIES[0], alpha=0.75, label="in-scope (Bitext test)", density=True)
    ax.hist(oos_scores, bins=bins, color=SERIES[1], alpha=0.75, label="out-of-scope (CLINC150)", density=True)
    ax.axvline(threshold, color=INK, lw=1.5, ls="--")
    ax.annotate("threshold", (threshold, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set(xlabel="in-scope score (higher = more confident it is a support request)", ylabel="density", title=title)
    ax.legend(loc="best")
    return _save(fig, path)


def comparison(df, path):
    """Small multiples: one panel per metric, one horizontal bar per model."""
    metrics = [("test_macro_f1", "Test macro-F1"), ("test_noisy_macro_f1", "Noisy test macro-F1"),
               ("oos_test_auroc", "Out-of-scope AUROC"), ("oos_test_fpr_at_95tpr", "OOS FPR@95%TPR (lower better)")]
    metrics = [m for m in metrics if m[0] in df.columns]
    models = list(df["run"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.55 * len(models) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, name) in zip(axes, metrics):
        vals = df[col].values
        ax.barh(range(len(models)), vals, color=[SERIES[i % len(SERIES)] for i in range(len(models))], height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.15)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), models)
    axes[0].invert_yaxis()
    return _save(fig, path)
