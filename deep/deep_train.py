#!/usr/bin/env python3
"""
train_raw_ppg_dl.py
====================

Trains deep learning models directly on raw, multi-channel PPG sequences
for deception detection (truth / lie classification) using labeled
recordings such as ``baseline_<subject>.csv``, ``kartu_<subject>_labeled.csv``
and ``interview_<subject>_labeled.csv``.

Pipeline
--------
1. Discover every labeled raw PPG CSV in the current directory (``*_hrv.csv``
   files are explicitly excluded).
2. Extract the subject ID from each filename for Leave-One-Subject-Out
   Cross Validation (LOSOCV).
3. Preprocess each of the five channels (``red``, ``infrared``, ``ppg0``,
   ``ppg1``, ``ppg2``) with a 4th-order Butterworth band-pass filter
   (0.5 - 4.0 Hz) applied with zero phase distortion (``scipy.signal.filtfilt``),
   followed by per-channel z-score normalization.
4. Build fixed-length, timestamp-based sliding windows (10 s length,
   1 s stride) treating the five channels as a single multivariate
   time-series (shape: ``(window_samples, 5)``).
5. Train three architectures (1D-CNN, CNN-LSTM, GRU) with TensorFlow/Keras.
6. Evaluate with Subject-Specific (80/20 split per subject) and LOSOCV
   protocols, reporting accuracy, precision, recall, specificity, F1 and
   multiclass ROC-AUC, plus confusion matrices and ROC curves.
7. Persist trained models, training curves, confusion matrices, ROC
   curves and CSV result tables.

The script is fully self-contained and requires only: TensorFlow, NumPy,
Pandas, SciPy, Matplotlib and Scikit-learn.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:  # pragma: no cover - not available on every TF build
    pass

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CHANNELS: List[str] = ["red", "infrared", "ppg0", "ppg1", "ppg2"]
WINDOW_SECONDS: float = 10.0
STRIDE_SECONDS: float = 1.0
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 4.0
BANDPASS_ORDER: int = 4
EPOCHS: int = 100
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
OUTPUT_DIR: str = "."

# --------------------------------------------------------------------------
# Class-mode configuration
# --------------------------------------------------------------------------
# "3class": keep all labels as-is (e.g. baseline / truth / deception).
# "2class": drop windows whose label matches BASELINE_LABEL_KEYWORD (case
#           insensitive substring match) and train truth vs. deception only.
CLASS_MODE: str = "2class"  # "3class" or "2class"
BASELINE_LABEL_KEYWORD: str = "base"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_raw_ppg_dl")


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class Recording:
    """A single preprocessed PPG recording."""

    subject: str
    filename: str
    fs: float
    data: np.ndarray  # shape (n_samples, len(CHANNELS)) filtered + normalized
    timestamps_ns: np.ndarray
    labels: np.ndarray  # raw string label per sample


# --------------------------------------------------------------------------
# File discovery / subject extraction
# --------------------------------------------------------------------------
def find_input_files(directory: str = ".") -> List[str]:
    """Locate every labeled raw PPG CSV in ``directory``, excluding ``*_hrv.csv``."""
    all_csv = glob.glob(os.path.join(directory, "*.csv"))
    files = [f for f in all_csv if not os.path.basename(f).lower().endswith("_hrv.csv")]
    return sorted(files)


def extract_subject(filename: str) -> str:
    """Extract the subject ID from a filename such as ``kartu_daffa_labeled.csv``."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"_labeled$", "", base, flags=re.IGNORECASE)
    parts = [p for p in base.split("_") if p]
    subject = parts[-1].lower() if parts else base.lower()
    return subject


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
def estimate_sampling_rate(timestamps_ns: np.ndarray) -> float:
    """Estimate sampling frequency (Hz) from nanosecond host timestamps."""
    diffs = np.diff(timestamps_ns.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Cannot estimate sampling rate: insufficient monotonic timestamps")
    median_dt_ns = float(np.median(diffs))
    return 1e9 / median_dt_ns


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to a 1D signal."""
    nyquist = fs / 2.0
    low_n = low / nyquist
    high_n = min(high / nyquist, 0.999)
    b, a = sp_signal.butter(order, [low_n, high_n], btype="band")
    return sp_signal.filtfilt(b, a, x)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1D signal: (x - mean) / std."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        std = 1e-12
    return (x - mean) / std


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def load_dataset(directory: str = ".") -> List[Recording]:
    """Load, sort, filter and normalize every labeled raw PPG CSV in ``directory``."""
    files = find_input_files(directory)
    logger.info(f"Number of files found: {len(files)}")
    recordings: List[Recording] = []
    required_cols = ["host_timestamp_ns"] + CHANNELS + ["label"]

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Skipping {f}: failed to read ({exc})")
            continue

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Skipping {f}: missing columns {missing}")
            continue

        df = df.sort_values("host_timestamp_ns").reset_index(drop=True)
        timestamps_ns = df["host_timestamp_ns"].to_numpy()

        try:
            fs = estimate_sampling_rate(timestamps_ns)
        except ValueError as exc:
            logger.warning(f"Skipping {f}: {exc}")
            continue

        subject = extract_subject(f)
        channel_data = np.zeros((len(df), len(CHANNELS)), dtype=np.float64)
        for i, ch in enumerate(CHANNELS):
            raw = df[ch].to_numpy(dtype=np.float64)
            try:
                filtered = bandpass_filter(raw, fs)
            except Exception as exc:
                logger.warning(f"{f}: band-pass filter failed on channel {ch} ({exc}); using raw signal")
                filtered = raw
            channel_data[:, i] = normalize_signal(filtered)

        labels = df["label"].astype(str).to_numpy()

        recordings.append(
            Recording(
                subject=subject,
                filename=f,
                fs=fs,
                data=channel_data,
                timestamps_ns=timestamps_ns,
                labels=labels,
            )
        )
        logger.info(f"Loaded {f}: subject={subject}, fs={fs:.2f} Hz, samples={len(df)}")

    return recordings


# --------------------------------------------------------------------------
# Sliding window construction
# --------------------------------------------------------------------------
def create_windows(
    recordings: List[Recording],
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build fixed-length, timestamp-based sliding windows across all recordings.

    Windows are defined by elapsed time, not by raw sample counts. Each
    window's raw samples are resampled (via linear interpolation over the
    true elapsed time) onto a fixed number of points so that every window
    has an identical shape, as required by the neural network input.
    """
    reference_fs = float(np.median([r.fs for r in recordings]))
    window_samples = int(round(window_seconds * reference_fs))

    all_X: List[np.ndarray] = []
    all_y: List[str] = []
    all_subjects: List[str] = []

    for rec in recordings:
        t_ns = rec.timestamps_ns.astype(np.float64)
        t_sec = (t_ns - t_ns[0]) / 1e9
        total_duration = float(t_sec[-1])

        if total_duration < window_seconds:
            logger.warning(f"{rec.filename}: duration {total_duration:.2f}s shorter than window; skipped")
            continue

        start = 0.0
        min_samples_required = max(4, window_samples // 4)
        while start + window_seconds <= total_duration:
            end = start + window_seconds
            mask = (t_sec >= start) & (t_sec < end)
            idx = np.where(mask)[0]

            if idx.size < min_samples_required:
                start += stride_seconds
                continue

            seg_t = t_sec[idx]
            seg_data = rec.data[idx, :]
            target_t = np.linspace(start, end, window_samples, endpoint=False)

            seg_resampled = np.empty((window_samples, len(CHANNELS)), dtype=np.float64)
            for c in range(len(CHANNELS)):
                seg_resampled[:, c] = np.interp(target_t, seg_t, seg_data[:, c])

            seg_labels = rec.labels[idx]
            values, counts = np.unique(seg_labels, return_counts=True)
            window_label = values[int(np.argmax(counts))]

            all_X.append(seg_resampled)
            all_y.append(window_label)
            all_subjects.append(rec.subject)

            start += stride_seconds

    if not all_X:
        raise RuntimeError("No windows could be constructed from the available recordings")

    X = np.stack(all_X, axis=0)
    y = np.array(all_y)
    subjects = np.array(all_subjects)
    return X, y, subjects, window_samples


# --------------------------------------------------------------------------
# Class-mode filtering
# --------------------------------------------------------------------------
def filter_classes_for_mode(
    X: np.ndarray,
    y_raw: np.ndarray,
    subjects: np.ndarray,
    class_mode: str = CLASS_MODE,
    baseline_keyword: str = BASELINE_LABEL_KEYWORD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter windows according to ``class_mode``.

    - ``"3class"``: no filtering; all labels (e.g. baseline / truth /
      deception) are kept as-is.
    - ``"2class"``: windows whose label matches ``baseline_keyword``
      (case-insensitive substring match) are excluded, leaving only the
      truth vs. deception classes for training.
    """
    if class_mode == "3class":
        logger.info("Class mode: 3class (baseline / truth / deception kept as-is)")
        return X, y_raw, subjects

    if class_mode == "2class":
        mask = np.array([baseline_keyword.lower() not in str(lbl).lower() for lbl in y_raw])
        removed = int((~mask).sum())
        logger.info(
            f"Class mode: 2class (excluding {removed} baseline windows; "
            f"training truth vs. deception only)"
        )
        if mask.sum() == 0:
            raise RuntimeError(
                "2-class mode removed all windows: no non-baseline labels found. "
                "Check BASELINE_LABEL_KEYWORD against your CSV label values."
            )
        return X[mask], y_raw[mask], subjects[mask]

    raise ValueError(f"Unknown CLASS_MODE '{class_mode}'; expected '2class' or '3class'")


# --------------------------------------------------------------------------
# Model architectures
# --------------------------------------------------------------------------
def build_cnn(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """1D-CNN: Conv-BN-ReLU-Pool x2 + GAP + Dense + Dropout + Softmax."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=5, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="CNN")


def build_cnn_lstm(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """CNN-LSTM: Conv-BN-ReLU-Pool + LSTM + Dropout + Dense + Softmax."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=5, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="CNN_LSTM")


def build_gru(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """GRU: GRU(128) + Dropout + Dense + Softmax."""
    inputs = layers.Input(shape=input_shape)
    x = layers.GRU(128)(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="GRU")


MODEL_BUILDERS = {
    "CNN": build_cnn,
    "CNNLSTM": build_cnn_lstm,
    "GRU": build_gru,
}


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile a model with Adam(lr=0.001), sparse categorical cross-entropy."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Standard EarlyStopping + ReduceLROnPlateau callback set."""
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_specificity(cm: np.ndarray) -> float:
    """Manually compute macro-averaged specificity (TN / (TN + FP)) from a confusion matrix."""
    n_classes = cm.shape[0]
    total = cm.sum()
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        specificities.append(float(tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specificities))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute accuracy, precision, recall, specificity, F1 and ROC-AUC."""
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    specificity = compute_specificity(cm)

    try:
        if num_classes == 2:
            roc_auc_val = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc_val = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except ValueError:
        roc_auc_val = float("nan")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc_val,
    }
    return metrics, cm


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_training_curve(history: tf.keras.callbacks.History, model_name: str, out_dir: str = OUTPUT_DIR) -> None:
    """Plot and save loss/accuracy training curves for a model's final training run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history.get("loss", []), label="train_loss")
    axes[0].plot(history.history.get("val_loss", []), label="val_loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history.get("accuracy", []), label="train_acc")
    axes[1].plot(history.history.get("val_accuracy", []), label="val_acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_training_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curve: {path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    eval_type: str,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} {eval_type} Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_{eval_type}_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix: {path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    class_names: List[str],
    model_name: str,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """Plot and save a (multiclass, one-vs-rest) ROC curve for a model."""
    fig, ax = plt.subplots(figsize=(6, 5))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
            roc_auc_val = auc(fpr, tpr)
            label = class_names[c] if c < len(class_names) else str(c)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_ROC.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve: {path}")


# --------------------------------------------------------------------------
# Evaluation protocols
# --------------------------------------------------------------------------
def subject_specific(
    X: np.ndarray,
    y_enc: np.ndarray,
    subjects: np.ndarray,
    num_classes: int,
    class_names: List[str],
    input_shape: Tuple[int, int],
    out_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Per-subject 80/20 stratified train/test evaluation for every model."""
    rows: List[Dict] = []
    agg_true: Dict[str, List[int]] = {m: [] for m in MODEL_BUILDERS}
    agg_pred: Dict[str, List[int]] = {m: [] for m in MODEL_BUILDERS}

    unique_subjects = sorted(np.unique(subjects))
    for subj in unique_subjects:
        mask = subjects == subj
        X_subj, y_subj = X[mask], y_enc[mask]

        if len(np.unique(y_subj)) < 2:
            logger.warning(f"[Subject-Specific] subject={subj}: single class present, skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, stratify=y_subj, random_state=SEED
        )

        for model_name, builder in MODEL_BUILDERS.items():
            logger.info(f"[Subject-Specific] subject={subj} model={model_name}")
            tf.keras.backend.clear_session()
            model = compile_model(builder(input_shape, num_classes))
            model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=get_callbacks(),
                verbose=0,
            )
            y_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            metrics, _ = compute_metrics(y_test, y_pred, y_proba, num_classes)
            metrics.update({"subject": subj, "model": model_name, "n_test": len(y_test)})
            rows.append(metrics)

            agg_true[model_name].extend(y_test.tolist())
            agg_pred[model_name].extend(y_pred.tolist())

    for model_name in MODEL_BUILDERS:
        if not agg_true[model_name]:
            continue
        cm = confusion_matrix(agg_true[model_name], agg_pred[model_name], labels=list(range(num_classes)))
        plot_confusion_matrix(cm, class_names, model_name, "SubjectSpecific", out_dir)

    return pd.DataFrame(rows)


def losocv(
    X: np.ndarray,
    y_enc: np.ndarray,
    subjects: np.ndarray,
    num_classes: int,
    class_names: List[str],
    input_shape: Tuple[int, int],
    out_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Leave-One-Subject-Out Cross Validation for every model."""
    rows: List[Dict] = []
    agg_true: Dict[str, List[int]] = {m: [] for m in MODEL_BUILDERS}
    agg_pred: Dict[str, List[int]] = {m: [] for m in MODEL_BUILDERS}
    agg_proba: Dict[str, List[List[float]]] = {m: [] for m in MODEL_BUILDERS}

    unique_subjects = sorted(np.unique(subjects))
    for fold_idx, test_subj in enumerate(unique_subjects, start=1):
        logger.info(f"[LOSOCV] Fold {fold_idx}/{len(unique_subjects)}: held-out subject = {test_subj}")
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        X_train, y_train = X[train_mask], y_enc[train_mask]
        X_test, y_test = X[test_mask], y_enc[test_mask]

        if X_test.size == 0 or X_train.size == 0:
            logger.warning(f"[LOSOCV] subject={test_subj}: insufficient data, skipping fold")
            continue

        for model_name, builder in MODEL_BUILDERS.items():
            logger.info(f"[LOSOCV] fold(test_subject={test_subj}) model={model_name}")
            tf.keras.backend.clear_session()
            model = compile_model(builder(input_shape, num_classes))
            model.fit(
                X_train,
                y_train,
                validation_split=0.1,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=get_callbacks(),
                verbose=0,
            )

            fold_model_path = os.path.join(out_dir, f"{model_name}_LOSOCV_{test_subj}_best.keras")
            model.save(fold_model_path)
            logger.info(f"Saved fold model: {fold_model_path}")

            y_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            metrics, _ = compute_metrics(y_test, y_pred, y_proba, num_classes)
            metrics.update({"subject": test_subj, "model": model_name, "n_test": len(y_test)})
            rows.append(metrics)

            agg_true[model_name].extend(y_test.tolist())
            agg_pred[model_name].extend(y_pred.tolist())
            agg_proba[model_name].extend(y_proba.tolist())

    for model_name in MODEL_BUILDERS:
        if not agg_true[model_name]:
            continue
        cm = confusion_matrix(agg_true[model_name], agg_pred[model_name], labels=list(range(num_classes)))
        plot_confusion_matrix(cm, class_names, model_name, "LOSOCV", out_dir)
        plot_roc_curve(
            np.array(agg_true[model_name]),
            np.array(agg_proba[model_name]),
            num_classes,
            class_names,
            model_name,
            out_dir,
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Results persistence
# --------------------------------------------------------------------------
def save_results(subject_df: pd.DataFrame, losocv_df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """Write subject-specific, LOSOCV and aggregated summary result CSVs."""
    subject_path = os.path.join(out_dir, "subject_specific_results.csv")
    losocv_path = os.path.join(out_dir, "losocv_results.csv")
    subject_df.to_csv(subject_path, index=False)
    losocv_df.to_csv(losocv_path, index=False)
    logger.info(f"Saved: {subject_path}")
    logger.info(f"Saved: {losocv_path}")

    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    summary_rows = []
    for model_name in MODEL_BUILDERS:
        for eval_type, df in (("SubjectSpecific", subject_df), ("LOSOCV", losocv_df)):
            if df.empty or "model" not in df.columns:
                continue
            sub = df[df["model"] == model_name]
            if sub.empty:
                continue
            row = {"model": model_name, "evaluation": eval_type, "n_folds": len(sub)}
            for col in metric_cols:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"] = sub[col].std()
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")
    return summary_df


# --------------------------------------------------------------------------
# Final full-dataset model training
# --------------------------------------------------------------------------
def train_final_models(
    X: np.ndarray,
    y_enc: np.ndarray,
    num_classes: int,
    input_shape: Tuple[int, int],
    out_dir: str = OUTPUT_DIR,
) -> None:
    """Train each architecture on the full dataset and persist model + training curve."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )
    filenames = {"CNN": "cnn.keras", "CNNLSTM": "cnn_lstm.keras", "GRU": "gru.keras"}

    for model_name, builder in MODEL_BUILDERS.items():
        logger.info(f"Training final full-dataset model: {model_name}")
        tf.keras.backend.clear_session()
        model = compile_model(builder(input_shape, num_classes))
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(),
            verbose=1,
        )
        plot_training_curve(history, model_name, out_dir)
        model_path = os.path.join(out_dir, filenames[model_name])
        model.save(model_path)
        logger.info(f"Saved final model: {model_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    """Run the full raw-PPG deep learning training and evaluation pipeline."""
    out_dir = OUTPUT_DIR

    recordings = load_dataset(out_dir)
    if not recordings:
        logger.error("No valid labeled raw PPG recordings found. Exiting.")
        return

    subjects_list = sorted({r.subject for r in recordings})
    logger.info(f"Number of files: {len(recordings)}")
    logger.info(f"Number of subjects: {len(subjects_list)} -> {subjects_list}")

    X, y_raw, subjects, window_samples = create_windows(recordings)
    logger.info(f"Number of windows (before class-mode filtering): {X.shape[0]}")
    logger.info(f"Window size (samples): {window_samples} (~{WINDOW_SECONDS}s)")
    median_fs = float(np.median([r.fs for r in recordings]))
    logger.info(f"Sampling frequency (median across recordings): {median_fs:.2f} Hz")

    X, y_raw, subjects = filter_classes_for_mode(X, y_raw, subjects, CLASS_MODE, BASELINE_LABEL_KEYWORD)
    logger.info(f"Number of windows (after class-mode filtering): {X.shape[0]}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    class_names = [str(c) for c in encoder.classes_]
    logger.info(f"Label mapping: {dict(zip(encoder.classes_, range(num_classes)))}")

    values, counts = np.unique(y_enc, return_counts=True)
    dist = {class_names[v]: int(c) for v, c in zip(values, counts)}
    logger.info(f"Class distribution: {dist}")

    input_shape = (window_samples, len(CHANNELS))

    logger.info("=== Starting Subject-Specific evaluation ===")
    subject_df = subject_specific(X, y_enc, subjects, num_classes, class_names, input_shape, out_dir)

    logger.info("=== Starting LOSOCV evaluation ===")
    losocv_df = losocv(X, y_enc, subjects, num_classes, class_names, input_shape, out_dir)

    summary_df = save_results(subject_df, losocv_df, out_dir)

    logger.info("=== Training final full-dataset models ===")
    train_final_models(X, y_enc, num_classes, input_shape, out_dir)

    logger.info("=== Final Summary ===")
    if not subject_df.empty:
        logger.info(
            "Subject-Specific mean accuracy per model:\n"
            f"{subject_df.groupby('model')['accuracy'].mean().to_string()}"
        )
    if not losocv_df.empty:
        logger.info(
            "LOSOCV mean accuracy per model:\n"
            f"{losocv_df.groupby('model')['accuracy'].mean().to_string()}"
        )
    logger.info(f"Summary table:\n{summary_df.to_string(index=False)}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()