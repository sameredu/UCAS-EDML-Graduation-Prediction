"""
Regenerate the six figures in the paper.

    python -m src.figures --data data/sample_data.csv --out results/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from . import config as C
from . import models
from .experiments import run_configuration
from .preprocessing import prepare

plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans",
    "figure.dpi": 600, "savefig.bbox": "tight",
})

COLOURS = ["#1b6ca8", "#c0392b", "#27ae60"]
HATCH = ["", "///", "..."]          # colour is never the only channel
LINESTYLE = ["-", "--", "-."]


def fig1_pipeline(out: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    stages = [
        ("UCAS student\nrecords\n65,532", "#dbe9f4"),
        ("IT cohort\nselection\nn = 4,704", "#dbe9f4"),
        ("Leakage\nscreening\n2 attributes\ndropped", "#fadbd8"),
        ("70/30 split\n3,292 / 1,412\nscaler fit\non train", "#dbe9f4"),
        ("DT / RF / GBT\ntraining\naugmentation\nx4 (control)", "#dbe9f4"),
        ("Hold-out\n5-fold CV\nWilson CI\nMcNemar\nfairness", "#e4f4e0"),
    ]
    gap, n = 2.6, len(stages)
    W = (100 - gap * (n - 1)) / n
    for k, (text, colour) in enumerate(stages):
        x = k * (W + gap)
        ax.add_patch(FancyBboxPatch((x, 26), W, 52, boxstyle="round,pad=0.5",
                                    fc=colour, ec="#34495e", lw=1.0))
        ax.text(x + W / 2, 52, text, ha="center", va="center",
                fontsize=8, linespacing=1.5)
        if k < n - 1:
            ax.annotate("", xy=(x + W + gap - 0.35, 52), xytext=(x + W + 0.35, 52),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#34495e"))
    left, right = 3 * (W + gap), 4 * (W + gap) + W
    ax.plot([left, right], [19, 19], color="#7f8c8d", lw=0.8)
    ax.plot([left, left], [19, 25], color="#7f8c8d", lw=0.8)
    ax.plot([right, right], [19, 25], color="#7f8c8d", lw=0.8)
    ax.text((left + right) / 2, 9, "split precedes scaling and augmentation",
            ha="center", fontsize=7.5, style="italic", color="#7f8c8d")
    fig.savefig(os.path.join(out, "Fig1.png"))
    plt.close(fig)


def fig2_leakage(runs: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    x = np.arange(3)
    width = 0.26
    for k, (name, colour) in enumerate(zip(C.MODEL_ORDER, COLOURS)):
        values = [runs[c]["results"][name]["CA"] for c in ["A", "B", "C"]]
        bars = ax.bar(x + (k - 1) * width, values, width=width, color=colour,
                      label=name, hatch=HATCH[k], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9,
                    f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [C.CONFIG_LABELS[c].replace(": ", ":\n") for c in ["A", "B", "C"]],
                  fontsize=8.5)
    ax.set_ylim(50, 118)
    ax.axhline(100, color="#c0392b", ls="--", lw=0.9)
    ax.text(2.44, 100.7, "perfect separation", color="#c0392b", fontsize=7.5, ha="right")
    ax.set_ylabel("Classification accuracy (%)")
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "Fig2.png"))
    plt.close(fig)


def fig3_roc(run_b: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.9))
    y_test = run_b["y_test"]
    for k, (name, colour) in enumerate(zip(C.MODEL_ORDER, COLOURS)):
        model = run_b["fitted"][name]
        prob = model.predict_proba(run_b["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, color=colour, lw=1.7, ls=LINESTYLE[k],
                label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "Fig3.png"))
    plt.close(fig)


def fig4_confusion(run_b: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    y_test = run_b["y_test"]
    for ax, name in zip(axes, C.MODEL_ORDER):
        cm = confusion_matrix(y_test, run_b["predictions"][name])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum() * 100:.1f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["Pred:\nNot graduated", "Pred:\nGraduated"], fontsize=7.5)
        ax.set_yticks([0, 1], ["Not\ngraduated", "Graduated"], fontsize=7.5)
        acc = accuracy_score(y_test, run_b["predictions"][name]) * 100
        ax.set_title(f"{name} - CA {acc:.2f}%", fontsize=8.5)
    fig.subplots_adjust(top=0.80, wspace=0.45)
    fig.savefig(os.path.join(out, "Fig4.png"))
    plt.close(fig)


def fig5_tree(df, out: str) -> None:
    """
    The display tree is refitted on raw (unscaled) units so the thresholds are
    readable. Tree models are scale-invariant, so it is structurally identical
    to the fitted model.
    """
    features = C.CONFIGURATIONS["B"]
    X = df[features].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy()
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=y)
    tree = DecisionTreeClassifier(random_state=C.SEED,
                                  **C.HYPERPARAMETERS["Decision Tree"]).fit(X_tr, y_tr)

    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    plot_tree(tree, max_depth=2, filled=True, rounded=True, impurity=False,
              proportion=True, fontsize=7, precision=1, ax=ax,
              feature_names=[C.DISPLAY_NAMES[f] for f in features],
              class_names=["Not graduated", "Graduated"])
    fig.savefig(os.path.join(out, "Fig5.png"))
    plt.close(fig)
    print(f"  tree: {tree.tree_.node_count} nodes, {tree.get_n_leaves()} leaves")


def fig6_importance(run_b: dict, out: str) -> None:
    features = run_b["features"]
    rf = run_b["fitted"]["Random Forest"].feature_importances_
    order = np.argsort(rf)
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    width = 0.27
    ypos = np.arange(len(features))
    for k, (name, colour) in enumerate(zip(C.MODEL_ORDER, COLOURS)):
        values = run_b["fitted"][name].feature_importances_[order]
        ax.barh(ypos + (k - 1) * width, values, height=width, color=colour,
                label=name, hatch=HATCH[k], edgecolor="white", linewidth=0.5)
    ax.set_yticks(ypos, [C.DISPLAY_NAMES[features[i]] for i in order], fontsize=8.5)
    ax.set_xlabel("Gini importance")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "Fig6.png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/sample_data.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = prepare(args.data)
    runs = {}
    for cfg in ["A", "B", "C"]:
        run = run_configuration(df, cfg)
        # keep the scaled test matrix for the ROC curves
        features = C.CONFIGURATIONS[cfg]
        from .preprocessing import split_scale_augment
        _, X_te, _, _, _ = split_scale_augment(df, features)
        run["X_test"] = X_te
        runs[cfg] = run

    print("Writing figures")
    fig1_pipeline(args.out)
    fig2_leakage(runs, args.out)
    fig3_roc(runs["B"], args.out)
    fig4_confusion(runs["B"], args.out)
    fig5_tree(df, args.out)
    fig6_importance(runs["B"], args.out)
    print(f"  six figures written to {args.out}/")


if __name__ == "__main__":
    main()
