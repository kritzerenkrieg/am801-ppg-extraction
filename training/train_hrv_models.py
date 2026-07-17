#!/usr/bin/env python3
"""
train_hrv_models.py

Train and evaluate HRV-based deception-detection models (SVM, Random Forest,
MLP) on ultra-short HRV feature windows produced by ``extract_hrv.py``.

Two evaluation protocols are run for every model:

1. Subject-Specific evaluation (80/20 stratified split, per subject).
2. Leave-One-Subject-Out cross-validation (LOSOCV).

The classification target is configurable via ``CLASS_MODE`` in the
Configuration section: ``"three_class"`` trains on base/deceptive/truth,
``"binary"`` drops the baseline class and trains on lie vs. truth only.

All reports and plots are written to a ``./{class_count}_hrv_result``
subdirectory of the input directory (e.g. ``3_hrv_result`` or
``2_hrv_result``), so runs with different ``CLASS_MODE`` settings never
overwrite each other.

The script also computes/saves a Pearson correlation matrix, a covariance
matrix, per-model confusion matrices, and per-model ROC curves, plus CSV
reports summarizing every run.

Usage
-----
    python train_hrv_models.py [directory]

If ``directory`` is omitted, the current working directory is used and must
contain one or more ``*_hrv.csv`` files (as produced by extract_hrv.py).

Dependencies
------------
    pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Classification target -------------------------------------------------
# "three_class" -> train on all labels present (base / deceptive / truth).
# "binary"      -> drop the baseline class and train only on lie vs. truth.
CLASS_MODE = "binary"  # "three_class" or "binary"

# Label values (case-insensitive) treated as the "baseline/base" class.
# These rows are excluded entirely when CLASS_MODE == "binary".
BASELINE_LABEL_NAMES = {"base", "baseline"}

# Label values (case-insensitive) treated as the "lie/deceptive" class.
# Only used to validate that a binary run still has both classes present.
DECEPTIVE_LABEL_NAMES = {"deceptive", "lie", "deception"}
TRUTH_LABEL_NAMES = {"truth", "truthful"}

EXCLUDE_COLS = {"window_start_ns", "window_end_ns", "label", "subject", "task", "source_file"}

# Model registry: display name -> factory producing a fresh, unfitted estimator.
MODEL_FACTORIES: Dict[str, Callable[[], object]] = {
    "SVM": lambda: SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    ),
    "MLP": lambda: MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=3000,
        early_stopping=True,
        alpha=1e-4,
        random_state=RANDOM_STATE,
    ),
}

# Short tokens used in saved filenames (e.g. RF_LOSOCV_confusion.png).
MODEL_FILE_TOKENS: Dict[str, str] = {"SVM": "SVM", "RandomForest": "RF", "MLP": "MLP"}

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_hrv_models")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Aggregate:
    """Accumulates predictions across subjects/folds for one model+protocol."""

    y_true: List[int] = field(default_factory=list)
    y_proba: List[np.ndarray] = field(default_factory=list)
    cm_sum: Optional[np.ndarray] = None

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, all_classes: List[int]) -> None:
        self.y_true.extend(list(y_true))
        self.y_proba.append(y_proba)
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        self.cm_sum = cm if self.cm_sum is None else self.cm_sum + cm


# --------------------------------------------------------------------------
# Loading & subject extraction
# --------------------------------------------------------------------------


def extract_subject(filename: str) -> Tuple[str, str]:
    """Extract ``(subject, task)`` from an ``*_hrv.csv`` filename.

    Examples
    --------
    ``baseline_andi_hrv.csv``          -> subject="andi",  task="baseline"
    ``kartu_daffa_labeled_hrv.csv``    -> subject="daffa", task="kartu"
    ``interview_doni_labeled_hrv.csv`` -> subject="doni",  task="interview"

    The rule: split the stem on underscores, drop a trailing ``hrv`` token,
    drop any ``labeled`` token, then the first remaining token is the task
    and the last remaining token is the subject.
    """
    stem = Path(filename).stem
    tokens = [t for t in stem.split("_") if t != ""]

    if tokens and tokens[-1].lower() == "hrv":
        tokens = tokens[:-1]

    tokens = [t for t in tokens if t.lower() != "labeled"]

    if not tokens:
        return "unknown", "unknown"
    if len(tokens) == 1:
        return tokens[0], "unknown"

    task = tokens[0]
    subject = tokens[-1]
    return subject, task


def load_dataset(directory: Path) -> pd.DataFrame:
    """Load and concatenate every ``*_hrv.csv`` file in ``directory``.

    Adds ``subject``, ``task``, and ``source_file`` columns derived from each
    filename. Raises ``FileNotFoundError`` if no matching files are found.
    """
    files = sorted(directory.glob("*_hrv.csv"))
    if not files:
        raise FileNotFoundError(f"No *_hrv.csv files found in {directory}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", f.name, exc)
            continue

        if "label" not in df.columns:
            logger.warning("Skipping %s: missing 'label' column", f.name)
            continue

        subject, task = extract_subject(f.name)
        df["subject"] = subject
        df["task"] = task
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No valid *_hrv.csv files could be loaded from {directory}")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    logger.info("Loaded %d file(s)", len(frames))
    logger.info("Total windows: %d", len(combined))
    logger.info("Subjects (%d): %s", combined["subject"].nunique(), sorted(combined["subject"].unique()))
    logger.info("Label distribution:\n%s", combined["label"].value_counts().to_string())

    return combined


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return every numeric column not in ``EXCLUDE_COLS``."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return feature_cols


def apply_class_mode(df: pd.DataFrame, class_mode: str) -> pd.DataFrame:
    """Filter the dataset down to the configured classification target.

    ``"three_class"`` keeps every row as-is (base / deceptive / truth).
    ``"binary"`` drops baseline rows so only lie-vs-truth remains, and warns
    if the resulting data doesn't actually contain both classes.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset with a ``label`` column.
    class_mode : str
        Either ``"three_class"`` or ``"binary"`` (see ``CLASS_MODE``).

    Returns
    -------
    pd.DataFrame
        The (possibly filtered) dataset.
    """
    mode = class_mode.strip().lower()

    if mode == "three_class":
        logger.info("CLASS_MODE='three_class': training on all labels present (base/deceptive/truth).")
        return df

    if mode == "binary":
        label_lower = df["label"].astype(str).str.lower()
        is_baseline = label_lower.isin(BASELINE_LABEL_NAMES)
        filtered = df.loc[~is_baseline].copy()

        dropped = int(is_baseline.sum())
        logger.info(
            "CLASS_MODE='binary': dropping %d baseline row(s); training on lie vs. truth only.", dropped
        )

        remaining_labels = set(filtered["label"].astype(str).str.lower().unique())
        has_deceptive = bool(remaining_labels & DECEPTIVE_LABEL_NAMES)
        has_truth = bool(remaining_labels & TRUTH_LABEL_NAMES)
        if not (has_deceptive and has_truth):
            logger.warning(
                "Binary mode expected lie/deceptive and truth labels but found: %s. "
                "Check BASELINE_LABEL_NAMES/DECEPTIVE_LABEL_NAMES/TRUTH_LABEL_NAMES in the Configuration section.",
                sorted(remaining_labels),
            )
        return filtered

    raise ValueError(f"Unknown CLASS_MODE '{class_mode}'. Expected 'three_class' or 'binary'.")


# --------------------------------------------------------------------------
# Modeling helpers
# --------------------------------------------------------------------------


def build_pipeline(estimator: object) -> Pipeline:
    """Build an impute -> filter -> scale -> classify pipeline for one estimator."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def align_proba(y_proba: np.ndarray, fitted_classes: np.ndarray, all_classes: List[int]) -> np.ndarray:
    """Re-index a predict_proba matrix onto the full global class ordering.

    A pipeline trained on a subset of classes (e.g. a subject missing one
    label category) produces a narrower probability matrix. This expands it
    back to ``len(all_classes)`` columns, filling absent classes with 0.
    """
    aligned = np.zeros((y_proba.shape[0], len(all_classes)), dtype=np.float64)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    for col, cls in enumerate(fitted_classes):
        if cls in class_to_idx:
            aligned[:, class_to_idx[cls]] = y_proba[:, col]
    return aligned


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, all_classes: List[int]) -> float:
    """Manually compute macro-averaged specificity from the confusion matrix.

    Specificity_i = TN_i / (TN_i + FP_i), macro-averaged over classes that
    have at least one true-negative-or-false-positive opportunity.
    """
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    total = cm.sum()
    specs = []
    for i in range(len(all_classes)):
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        tn = total - tp - fp - fn
        denom = tn + fp
        if denom > 0:
            specs.append(tn / denom)
    return float(np.mean(specs)) if specs else float("nan")


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, all_classes: List[int]) -> float:
    """Compute macro one-vs-rest ROC-AUC, returning NaN if it cannot be computed.

    ``roc_auc_score(..., multi_class="ovr")`` expects a 1-D score array (the
    positive-class probability) for the binary case rather than a 2-column
    matrix, so that case is handled separately from the true multiclass case.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(all_classes) == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1], labels=all_classes))
            return float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=all_classes)
            )
    except Exception:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, all_classes: List[int]
) -> Dict[str, float]:
    """Compute the full required metric set for one train/test split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=all_classes)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=all_classes)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_classes)

    specificity = compute_specificity(y_true, y_pred, all_classes)
    roc_auc = compute_roc_auc(y_true, y_proba, all_classes)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC-AUC": roc_auc,
    }


def fit_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    all_classes: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fit one model's pipeline and return ``(y_pred, y_proba_aligned)``.

    Returns ``None`` if training fails (e.g. degenerate single-class data).
    """
    try:
        pipeline = build_pipeline(MODEL_FACTORIES[model_name]())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba_raw = pipeline.predict_proba(X_test)
        fitted_classes = pipeline.named_steps["clf"].classes_
        y_proba = align_proba(y_proba_raw, fitted_classes, all_classes)
        return y_pred, y_proba
    except Exception as exc:
        logger.warning("  [%s] training/prediction failed: %s", model_name, exc)
        return None


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _heatmap(matrix: pd.DataFrame, title: str, save_path: Path, cmap: str = "coolwarm") -> None:
    """Generic annotated heatmap for correlation/covariance matrices."""
    n = matrix.shape[0]
    fig_size = max(10, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation(feature_df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute and save the Pearson correlation matrix (CSV + heatmap PNG)."""
    corr = feature_df.corr(method="pearson")
    corr.to_csv(out_csv)
    _heatmap(corr, "Feature Correlation Matrix (Pearson)", out_png)
    return corr


def plot_covariance(feature_df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Compute and save the covariance matrix (CSV + heatmap PNG)."""
    cov = feature_df.cov()
    cov.to_csv(out_csv)
    _heatmap(cov, "Feature Covariance Matrix", out_png)
    return cov


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Path) -> None:
    """Save an annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], all_classes: List[int], title: str, save_path: Path
) -> None:
    """Save a one-vs-rest ROC curve plot (one line per class + macro-average)."""
    y_true = np.asarray(y_true)
    y_bin = label_binarize(y_true, classes=all_classes)

    # label_binarize collapses to a single column for the binary (2-class)
    # case; expand it back to one column per class for uniform handling.
    if len(all_classes) == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6, 5))
    valid_aucs = []

    for i, cls_name in enumerate(class_names):
        col_true = y_bin[:, i]
        if col_true.sum() == 0 or col_true.sum() == len(col_true):
            continue  # class absent or trivially present; ROC undefined
        try:
            fpr, tpr, _ = roc_curve(col_true, y_proba[:, i])
            roc_auc_val = sk_auc(fpr, tpr)
            valid_aucs.append(roc_auc_val)
            ax.plot(fpr, tpr, lw=1.5, label=f"{cls_name} (AUC={roc_auc_val:.3f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    macro_str = f" | Macro AUC={np.mean(valid_aucs):.3f}" if valid_aucs else ""
    ax.set_title(f"{title}{macro_str}")
    if valid_aucs:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Evaluation protocols
# --------------------------------------------------------------------------


def subject_specific_evaluation(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    subjects: List[str],
    all_classes: List[int],
    class_names: List[str],
) -> Tuple[List[Dict[str, object]], Dict[str, Aggregate]]:
    """Run the 80/20 stratified subject-specific evaluation for every subject."""
    results: List[Dict[str, object]] = []
    aggregates: Dict[str, Aggregate] = {m: Aggregate() for m in MODEL_FACTORIES}

    for subject in subjects:
        sub_df = df[df["subject"] == subject]
        X = sub_df[feature_cols]
        y = sub_df[label_col].to_numpy()

        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            logger.warning("Subject '%s': only one class present, skipping subject-specific split.", subject)
            continue
        if counts.min() < 2:
            logger.warning(
                "Subject '%s': a class has < 2 samples, stratified split may be unreliable; attempting anyway.",
                subject,
            )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )
        except Exception as exc:
            logger.warning("Subject '%s': stratified split failed (%s), skipping.", subject, exc)
            continue

        logger.info("Subject-Specific | subject=%s | train=%d test=%d", subject, len(X_train), len(X_test))

        for model_name in MODEL_FACTORIES:
            outcome = fit_predict(model_name, X_train, y_train, X_test, all_classes)
            if outcome is None:
                continue
            y_pred, y_proba = outcome

            metrics = compute_metrics(y_test, y_pred, y_proba, all_classes)
            row = {"Subject": subject, "Model": model_name, **metrics}
            results.append(row)

            aggregates[model_name].update(y_test, y_pred, y_proba, all_classes)

    return results, aggregates


def losocv_evaluation(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    subjects: List[str],
    all_classes: List[int],
    class_names: List[str],
) -> Tuple[List[Dict[str, object]], Dict[str, Aggregate]]:
    """Run Leave-One-Subject-Out cross-validation."""
    results: List[Dict[str, object]] = []
    aggregates: Dict[str, Aggregate] = {m: Aggregate() for m in MODEL_FACTORIES}

    for fold_idx, held_out in enumerate(subjects, start=1):
        train_df = df[df["subject"] != held_out]
        test_df = df[df["subject"] == held_out]

        X_train, y_train = train_df[feature_cols], train_df[label_col].to_numpy()
        X_test, y_test = test_df[feature_cols], test_df[label_col].to_numpy()

        if len(np.unique(y_train)) < 2:
            logger.warning("LOSOCV fold %d (held out=%s): training data has <2 classes, skipping.", fold_idx, held_out)
            continue
        if len(X_test) == 0:
            continue

        logger.info(
            "LOSOCV | fold=%d/%d | held-out subject=%s | train=%d test=%d",
            fold_idx,
            len(subjects),
            held_out,
            len(X_train),
            len(X_test),
        )

        for model_name in MODEL_FACTORIES:
            outcome = fit_predict(model_name, X_train, y_train, X_test, all_classes)
            if outcome is None:
                continue
            y_pred, y_proba = outcome

            metrics = compute_metrics(y_test, y_pred, y_proba, all_classes)
            row = {"Fold": fold_idx, "Held-out Subject": held_out, "Model": model_name, **metrics}
            results.append(row)

            aggregates[model_name].update(y_test, y_pred, y_proba, all_classes)

    return results, aggregates


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def build_summary(
    subject_results: List[Dict[str, object]], losocv_results: List[Dict[str, object]]
) -> pd.DataFrame:
    """Build the summary_results.csv content (mean/std per model+protocol)."""
    rows = []

    def summarize(records: List[Dict[str, object]], protocol_name: str) -> None:
        if not records:
            return
        rec_df = pd.DataFrame(records)
        for model_name in MODEL_FACTORIES:
            model_rows = rec_df[rec_df["Model"] == model_name]
            if model_rows.empty:
                continue
            entry: Dict[str, object] = {"Model": model_name, "Evaluation": protocol_name}
            for metric in METRIC_COLUMNS:
                values = model_rows[metric].astype(float)
                entry[f"Mean {metric}"] = values.mean(skipna=True)
                entry[f"Std {metric}"] = values.std(skipna=True, ddof=1) if len(values) > 1 else 0.0
            rows.append(entry)

    summarize(subject_results, "SubjectSpecific")
    summarize(losocv_results, "LOSOCV")

    return pd.DataFrame(rows)


def save_reports(
    subject_results: List[Dict[str, object]],
    losocv_results: List[Dict[str, object]],
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Persist subject_specific_results.csv, losocv_results.csv, summary_results.csv."""
    subj_cols = ["Subject", "Model"] + METRIC_COLUMNS
    losocv_cols = ["Fold", "Held-out Subject", "Model"] + METRIC_COLUMNS

    subj_df = pd.DataFrame(subject_results)
    if not subj_df.empty:
        subj_df = subj_df[subj_cols]
    subj_df.to_csv(out_dir / "subject_specific_results.csv", index=False)

    losocv_df = pd.DataFrame(losocv_results)
    if not losocv_df.empty:
        losocv_df = losocv_df[losocv_cols]
    losocv_df.to_csv(out_dir / "losocv_results.csv", index=False)

    summary_df.to_csv(out_dir / "summary_results.csv", index=False)

    logger.info("Saved subject_specific_results.csv (%d rows)", len(subj_df))
    logger.info("Saved losocv_results.csv (%d rows)", len(losocv_df))
    logger.info("Saved summary_results.csv (%d rows)", len(summary_df))


def render_aggregate_plots(
    aggregates: Dict[str, Aggregate], protocol_label: str, class_names: List[str], all_classes: List[int], out_dir: Path
) -> None:
    """Save the confusion-matrix and ROC-curve PNGs for every model in a protocol."""
    for model_name, agg in aggregates.items():
        token = MODEL_FILE_TOKENS[model_name]
        if agg.cm_sum is None or not agg.y_true:
            logger.warning("No aggregated predictions for %s | %s, skipping plots.", model_name, protocol_label)
            continue

        cm_path = out_dir / f"{token}_{protocol_label}_confusion.png"
        plot_confusion_matrix(
            agg.cm_sum, class_names, f"{model_name} - {protocol_label} Confusion Matrix", cm_path
        )

        y_true_arr = np.array(agg.y_true)
        y_proba_arr = np.vstack(agg.y_proba)
        roc_path = out_dir / f"{token}_{protocol_label}_roc.png"
        plot_roc_curve(
            y_true_arr,
            y_proba_arr,
            class_names,
            all_classes,
            f"{model_name} - {protocol_label} ROC Curve",
            roc_path,
        )

        logger.info("Saved %s and %s", cm_path.name, roc_path.name)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def print_final_summary(summary_df: pd.DataFrame) -> None:
    """Pretty-print the final summary table to the console."""
    if summary_df.empty:
        logger.info("No results to summarize.")
        return

    display_cols = ["Model", "Evaluation", "Mean Accuracy", "Std Accuracy", "Mean F1", "Std F1", "Mean ROC-AUC", "Std ROC-AUC"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    logger.info("\n===== FINAL SUMMARY =====\n%s", summary_df[display_cols].to_string(index=False))


def main() -> None:
    """Load data, run both evaluation protocols, and save all reports/plots."""
    target_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()

    if not target_dir.is_dir():
        logger.error("'%s' is not a directory.", target_dir)
        sys.exit(1)

    try:
        df = load_dataset(target_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # --- Classification target (binary lie/truth vs. three-class) --------
    df = apply_class_mode(df, CLASS_MODE)
    if df["label"].nunique() < 2:
        logger.error(
            "Fewer than 2 classes remain after applying CLASS_MODE='%s'. Aborting.", CLASS_MODE
        )
        sys.exit(1)

    feature_cols = get_feature_columns(df)
    logger.info("Number of HRV features: %d", len(feature_cols))

    if len(feature_cols) == 0:
        logger.error("No numeric HRV feature columns found. Aborting.")
        sys.exit(1)

    # --- Label encoding -----------------------------------------------
    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"].astype(str))
    label_mapping = {cls: int(code) for code, cls in enumerate(encoder.classes_)}
    logger.info("Label mapping: %s", label_mapping)

    all_classes = list(range(len(encoder.classes_)))
    class_names = list(encoder.classes_)
    class_count = len(class_names)

    # --- Output directory: ./{class_count}_hrv_result --------------------
    out_dir = target_dir / f"{class_count}_hrv_result"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("All outputs will be saved to: %s", out_dir)

    subjects = sorted(df["subject"].unique())
    logger.info("Number of subjects: %d", len(subjects))
    logger.info("Number of files: %d", df["source_file"].nunique())
    logger.info("Number of windows: %d", len(df))

    # --- Correlation / covariance (computed on raw feature values) ----
    feature_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN (would break corr/cov); keep rest.
    feature_df_clean = feature_df.dropna(axis=1, how="all")
    dropped = set(feature_df.columns) - set(feature_df_clean.columns)
    if dropped:
        logger.warning("Excluding all-NaN feature columns from correlation/covariance: %s", sorted(dropped))

    logger.info("Computing Pearson correlation matrix...")
    plot_correlation(feature_df_clean, out_dir / "correlation_matrix.csv", out_dir / "correlation_heatmap.png")

    logger.info("Computing covariance matrix...")
    plot_covariance(feature_df_clean, out_dir / "covariance_matrix.csv", out_dir / "covariance_heatmap.png")

    # --- Evaluation 1: Subject-Specific ---------------------------------
    logger.info("===== Running Subject-Specific Evaluation =====")
    subject_results, subj_aggregates = subject_specific_evaluation(
        df, feature_cols, "label_encoded", subjects, all_classes, class_names
    )
    render_aggregate_plots(subj_aggregates, "SubjectSpecific", class_names, all_classes, out_dir)

    # --- Evaluation 2: LOSOCV -------------------------------------------
    logger.info("===== Running LOSOCV Evaluation =====")
    losocv_results, losocv_aggregates = losocv_evaluation(
        df, feature_cols, "label_encoded", subjects, all_classes, class_names
    )
    render_aggregate_plots(losocv_aggregates, "LOSOCV", class_names, all_classes, out_dir)

    # --- Reports ----------------------------------------------------------
    summary_df = build_summary(subject_results, losocv_results)
    save_reports(subject_results, losocv_results, summary_df, out_dir)

    print_final_summary(summary_df)


if __name__ == "__main__":
    main()