"""
evaluation.py
-------------
Confusion matrix, classification metrics, and visualisation helpers.

Author : Samer A. Yaghi (The Islamic University of Gaza, 2022)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)


def evaluate(y_true, y_pred, y_prob=None, model_name: str = "Model") -> dict:
    """
    Compute AUC, CA, F1, Precision, Recall from predictions.
    Matches the evaluation framework used in Chapter 5 of the thesis.
    """
    ca  = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    results = {
        "model":     model_name,
        "CA":        round(ca,  4),
        "AUC":       round(auc, 4) if auc else "N/A",
        "F1":        round(f1,  4),
        "Precision": round(pre, 4),
        "Recall":    round(rec, 4),
    }

    print(f"\n{'='*48}")
    print(f"  {model_name}")
    print(f"{'='*48}")
    print(f"  Accuracy  (CA) : {ca*100:.2f}%")
    if auc:
        print(f"  AUC            : {auc:.4f}")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  Precision      : {pre:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"{'='*48}\n")
    return results


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model",
                          save_path: str = None):
    """
    Plot and optionally save the confusion matrix.
    Colours match the visual style used in the thesis figures.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Graduated", "Graduated"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=12)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    # Annotate TP / TN / FP / FN percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            label = {(1,1): "TP", (0,0): "TN", (1,0): "FN", (0,1): "FP"}[(i,j)]
            ax.text(j, i + 0.35, f"{label}\n{pct:.1f}%",
                    ha="center", va="center", fontsize=8, color="grey")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eval] Confusion matrix saved → {save_path}")
    plt.show()
    return cm


def plot_roc_curves(models_data: list, save_path: str = None):
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    models_data : list of dict
        Each dict: {"name": str, "y_true": array, "y_prob": array}
    """
    COLORS = {"Random Forest": "#1f77b4",
               "Decision Tree": "#ff7f0e",
               "GBT Classifier": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models_data:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
        auc = roc_auc_score(m["y_true"], m["y_prob"])
        color = COLORS.get(m["name"], "#999999")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{m['name']} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — UCAS ML Models", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eval] ROC curves saved → {save_path}")
    plt.show()


def plot_model_comparison(results: list, save_path: str = None):
    """
    Bar chart comparing CA / AUC / F1 across the three algorithms.
    """
    names   = [r["model"] for r in results]
    metrics = ["CA", "F1", "Precision", "Recall"]
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ML Model Comparison — UCAS Educational Data", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[eval] Comparison chart saved → {save_path}")
    plt.show()


def pearson_correlation_report(df, target_col: str = "STATUS"):
    """Print Pearson correlation table matching Table 5.1 of the thesis."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        print("[corr] Target column not found in numeric columns.")
        return
    corr = numeric_df.corr()[target_col].drop(target_col).sort_values(
        ascending=False, key=abs
    )
    print("\nPearson Correlation with Graduation (STATUS):")
    print("-" * 40)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"  {feat:<25} {sign}{abs(val):.3f}  {bar}")
    print("-" * 40)
    return corr
