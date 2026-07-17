#!/usr/bin/env python3
"""
confusion_matrix_2x2.py
=======================

A small, self-contained utility that generates a 2x2 confusion matrix
visualization (heatmap) for binary classification, mirroring the style of
``plot_confusion_matrix`` used in ``deep_train.py``.

The user supplies the four cell counts (TP, TN, FP, FN) and a title, and the
script renders a labeled 2x2 matrix with True/False Positives,
True/False Negatives, plus derived metrics (accuracy, precision, recall,
specificity, F1) printed to the console and saved as a PNG.

Usage
-----
    python confusion_matrix_2x2.py --title "My Model" --tp 50 --tn 40 --fp 10 --fn 5
    python confusion_matrix_2x2.py   # uses interactive prompts if no args given

Requires only: matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Confusion matrix construction
# --------------------------------------------------------------------------
def build_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> np.ndarray:
    """Build a 2x2 confusion matrix with layout matching sklearn conventions.

    Rows  -> True label   (Positive=0, Negative=1)
    Cols  -> Predicted    (Positive=0, Negative=1)

        [[TP, FN],
         [FP, TN]]
    """
    return np.array([[tp, fn], [fp, tn]], dtype=int)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Compute standard binary classification metrics from the four cells."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


# --------------------------------------------------------------------------
# Plotting (mirrors plot_confusion_matrix in deep_train.py)
# --------------------------------------------------------------------------
def plot_confusion_matrix_2x2(
    cm: np.ndarray,
    class_names: list,
    title: str,
    out_dir: str = ".",
) -> str:
    """Plot and save a 2x2 confusion matrix heatmap.

    Labels and text match ``plot_confusion_matrix`` in ``deep_train.py``
    exactly: xlabel "Predicted", ylabel "True", raw cell counts only, and a
    colorbar. The TP/TN/FP/FN values are supplied as the matrix inputs.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    path = os.path.join(out_dir, f"confusion_2x2_{safe_title}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 2x2 confusion matrix from TP/TN/FP/FN inputs."
    )
    parser.add_argument("--title", type=str, default="2x2 Confusion Matrix",
                        help="Title for the confusion matrix plot.")
    parser.add_argument("--tp", type=int, help="True Positives.")
    parser.add_argument("--tn", type=int, help="True Negatives.")
    parser.add_argument("--fp", type=int, help="False Positives.")
    parser.add_argument("--fn", type=int, help="False Negatives.")
    parser.add_argument("--out", type=str, default=".",
                        help="Output directory for the saved PNG.")
    args = parser.parse_args()

    # Interactive fallback if any cell count is missing
    def prompt_int(name: str, value) -> int:
        if value is not None:
            return value
        while True:
            try:
                return int(input(f"Enter {name}: "))
            except ValueError:
                print("Please enter an integer.")

    tp = prompt_int("TP (True Positives)", args.tp)
    tn = prompt_int("TN (True Negatives)", args.tn)
    fp = prompt_int("FP (False Positives)", args.fp)
    fn = prompt_int("FN (False Negatives)", args.fn)

    cm = build_confusion_matrix(tp, tn, fp, fn)
    metrics = compute_binary_metrics(tp, tn, fp, fn)

    print("\nConfusion Matrix ([[TP, FN], [FP, TN]]):")
    print(cm)
    print("\nMetrics:")
    for key, val in metrics.items():
        print(f"  {key:>11}: {val:.4f}")

    class_names = ["truth", "deceptive"]
    out_path = plot_confusion_matrix_2x2(cm, class_names, args.title, args.out)
    print(f"\nSaved confusion matrix plot: {out_path}")


if __name__ == "__main__":
    main()