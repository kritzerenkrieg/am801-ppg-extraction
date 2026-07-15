#!/usr/bin/env python3
"""
extract_hrv.py

Extract ultra-short HRV (Heart Rate Variability) features from raw PPG
recordings using a 10-second sliding window (1-second stride).

For every ``*.csv`` file found in the script's working directory, this tool
independently processes five PPG-related channels (``red``, ``infrared``,
``ppg0``, ``ppg1``, ``ppg2``), extracts systolic peaks with NeuroKit2,
derives NN-interval-based HRV features per window, and writes one combined
row per window (58 columns total) to ``<original_name>_hrv.csv``.

Usage
-----
    python extract_hrv.py [directory]

If ``directory`` is omitted, the current working directory is used.

Dependencies
------------
    pandas, numpy, scipy, neurokit2
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

import neurokit2 as nk

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CHANNELS: List[str] = ["red", "infrared", "ppg0", "ppg1", "ppg2"]

REQUIRED_COLUMNS: List[str] = [
    "host_timestamp_ns",
    "red",
    "infrared",
    "ppg0",
    "ppg1",
    "ppg2",
    "label",
]

BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 4.0
BANDPASS_ORDER: int = 4

WINDOW_SECONDS: float = 10.0
STRIDE_SECONDS: float = 1.0

FEATURE_NAMES: List[str] = [
    "Mean_NN",
    "Median_NN",
    "Heart_Rate",
    "Mean_HR",
    "SDNN",
    "SDSD",
    "RMSSD",
    "NN_Count",
    "SD1",
    "SD2",
    "SD1_SD2",
]

# Minimum number of NN intervals required to compute meaningful HRV stats.
MIN_NN_FOR_STATS = 2
MIN_NN_FOR_POINCARE = 3


# --------------------------------------------------------------------------
# Core signal-processing helpers
# --------------------------------------------------------------------------


def estimate_sampling_rate(timestamps_ns: np.ndarray) -> float:
    """Estimate the sampling frequency (Hz) from nanosecond timestamps.

    Parameters
    ----------
    timestamps_ns : np.ndarray
        Monotonically sorted array of timestamps, in nanoseconds.

    Returns
    -------
    float
        Estimated sampling rate in Hz.

    Raises
    ------
    ValueError
        If fewer than two timestamps are provided or the estimated rate is
        non-positive / non-finite.
    """
    if timestamps_ns.size < 2:
        raise ValueError("Need at least 2 timestamps to estimate sampling rate.")

    diffs_s = np.diff(timestamps_ns.astype(np.float64)) / 1e9
    diffs_s = diffs_s[diffs_s > 0]
    if diffs_s.size == 0:
        raise ValueError("No positive timestamp differences found.")

    median_dt = np.median(diffs_s)
    if median_dt <= 0 or not np.isfinite(median_dt):
        raise ValueError("Invalid median timestamp difference.")

    fs = 1.0 / median_dt
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid estimated sampling rate: {fs}")

    return fs


def bandpass_filter(
    x: np.ndarray, fs: float, low: float = BANDPASS_LOW_HZ, high: float = BANDPASS_HIGH_HZ, order: int = BANDPASS_ORDER
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    x : np.ndarray
        Raw 1-D signal.
    fs : float
        Sampling rate in Hz.
    low, high : float
        Band-pass cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal, same length as input.

    Raises
    ------
    ValueError
        If the Nyquist-normalized cutoffs are invalid (e.g. fs too low) or
        the signal is too short for filtfilt's default padding.
    """
    nyquist = fs / 2.0
    low_n = low / nyquist
    high_n = high / nyquist

    if not (0 < low_n < high_n < 1):
        raise ValueError(
            f"Invalid band-pass cutoffs for fs={fs:.3f} Hz "
            f"(normalized band=[{low_n:.4f}, {high_n:.4f}])."
        )

    # filtfilt requires the signal length to exceed a padding-dependent
    # minimum (default padlen = 3 * max(len(a), len(b))).
    b, a = sp_signal.butter(order, [low_n, high_n], btype="band")
    min_len = 3 * max(len(a), len(b))
    if x.size <= min_len:
        raise ValueError(
            f"Signal too short ({x.size} samples) for filtfilt (needs > {min_len})."
        )

    return sp_signal.filtfilt(b, a, x)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal: ``(x - mean) / std``.

    If the standard deviation is zero (flat signal), returns a zero array
    instead of dividing by zero.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(x, dtype=np.float64)
    return (x - mean) / std


def detect_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Detect systolic peaks in a (filtered, normalized) PPG signal.

    Uses NeuroKit2's ``ppg_findpeaks`` for detection and, when possible,
    ``signal_fixpeaks`` (Kubios method) for artifact correction.

    Parameters
    ----------
    x : np.ndarray
        Preprocessed PPG signal.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Sample indices of detected (and corrected, if available) peaks.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peak_info = nk.ppg_findpeaks(x, sampling_rate=fs)

    peaks = np.asarray(peak_info["PPG_Peaks"], dtype=np.int64)

    if peaks.size < 3:
        # Not enough peaks to meaningfully attempt artifact correction.
        return peaks

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected, _artifacts = nk.signal_fixpeaks(
                peaks, sampling_rate=fs, method="Kubios"
            )
        corrected = np.asarray(corrected, dtype=np.int64)
        corrected = corrected[(corrected >= 0)]
        if corrected.size >= 2:
            return np.sort(corrected)
        return peaks
    except Exception:
        # Artifact correction is best-effort; fall back to raw peaks.
        return peaks


def compute_nn_intervals(peak_samples: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak sample indices into NN intervals, in milliseconds.

    Parameters
    ----------
    peak_samples : np.ndarray
        Sample indices of detected peaks (monotonically increasing).
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        NN intervals in milliseconds. Empty array if fewer than 2 peaks.
    """
    peak_samples = np.asarray(peak_samples, dtype=np.float64)
    if peak_samples.size < 2:
        return np.array([], dtype=np.float64)

    peak_samples = np.sort(peak_samples)
    diffs_samples = np.diff(peak_samples)
    nn_ms = (diffs_samples / fs) * 1000.0
    # Drop non-physiological / invalid intervals (keep it permissive but sane:
    # roughly 300-2000 ms covers ~30-200 bpm).
    valid = np.isfinite(nn_ms) & (nn_ms > 0)
    return nn_ms[valid]


def compute_hrv(nn_intervals_ms: np.ndarray) -> Dict[str, float]:
    """Compute the required HRV feature set from NN intervals.

    Returns NaN for any feature that cannot be computed given the available
    number of intervals, rather than raising.

    Parameters
    ----------
    nn_intervals_ms : np.ndarray
        NN intervals in milliseconds.

    Returns
    -------
    dict
        Mapping of feature name -> value (float, possibly NaN).
    """
    features: Dict[str, float] = {name: np.nan for name in FEATURE_NAMES}

    nn = np.asarray(nn_intervals_ms, dtype=np.float64)
    nn = nn[np.isfinite(nn)]
    n = nn.size

    features["NN_Count"] = float(n)

    if n == 0:
        return features

    # Mean / median NN and heart rate are computable with a single interval.
    mean_nn = float(np.mean(nn))
    features["Mean_NN"] = mean_nn
    features["Median_NN"] = float(np.median(nn))

    # Instantaneous heart rate derived from each NN interval (bpm).
    with np.errstate(divide="ignore", invalid="ignore"):
        hr_series = 60000.0 / nn
    hr_series = hr_series[np.isfinite(hr_series)]
    if hr_series.size > 0:
        features["Heart_Rate"] = float(hr_series[-1])
        features["Mean_HR"] = float(np.mean(hr_series))

    if n >= MIN_NN_FOR_STATS:
        features["SDNN"] = float(np.std(nn, ddof=1))

        successive_diffs = np.diff(nn)
        if successive_diffs.size >= 1:
            features["SDSD"] = float(np.std(successive_diffs, ddof=1)) if successive_diffs.size >= 2 else 0.0
            features["RMSSD"] = float(np.sqrt(np.mean(successive_diffs ** 2)))

    if n >= MIN_NN_FOR_POINCARE:
        diffs = np.diff(nn)
        sd1 = float(np.sqrt(np.var(diffs, ddof=1) / 2.0)) if diffs.size >= 2 else np.nan
        # SD2 from Poincare plot: 2*SDNN^2 - 0.5*SD1^2, sqrt thereof.
        sdnn = features["SDNN"]
        if np.isfinite(sdnn) and np.isfinite(sd1):
            sd2_sq = 2.0 * (sdnn ** 2) - 0.5 * (sd1 ** 2)
            sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else np.nan
        else:
            sd2 = np.nan

        features["SD1"] = sd1
        features["SD2"] = sd2
        if np.isfinite(sd1) and np.isfinite(sd2) and sd2 != 0:
            features["SD1_SD2"] = sd1 / sd2

    return features


# --------------------------------------------------------------------------
# Per-channel / per-window / per-file orchestration
# --------------------------------------------------------------------------


def extract_channel_features(
    raw_signal: np.ndarray,
    timestamps_ns: np.ndarray,
    fs: float,
    window_start_ns: int,
    window_end_ns: int,
) -> Dict[str, float]:
    """Run the full pipeline for one channel within one time window.

    Filtering and peak detection are performed on the samples within the
    window slice only (peaks are detected fresh per window, which keeps the
    implementation simple and robust to local signal quality changes).

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw samples for the *entire* channel (unfiltered).
    timestamps_ns : np.ndarray
        Timestamps (ns) aligned to ``raw_signal``.
    fs : float
        Sampling rate (Hz), estimated once for the whole file.
    window_start_ns, window_end_ns : int
        Inclusive/exclusive window bounds in nanoseconds.

    Returns
    -------
    dict
        HRV feature dict (see ``compute_hrv``), all-NaN with NN_Count=0 on
        any failure.
    """
    empty = compute_hrv(np.array([]))

    mask = (timestamps_ns >= window_start_ns) & (timestamps_ns < window_end_ns)
    segment = raw_signal[mask]

    if segment.size < 4:
        return empty

    try:
        filtered = bandpass_filter(segment, fs)
        normalized = normalize_signal(filtered)
        peaks = detect_peaks(normalized, fs)
        nn_intervals = compute_nn_intervals(peaks, fs)
        return compute_hrv(nn_intervals)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, must not crash
        print(f"    [warn] channel processing failed: {exc}")
        return empty


def derive_window_label(
    df: pd.DataFrame,
    window_start_ns: int,
    window_end_ns: int,
    fallback_label: str = "",
) -> str:
    """Infer a label for a window from the labels of rows inside it.

    If the window contains multiple labels, the most frequent one is used.
    If there are no usable labels in the window, the file-level fallback is
    returned.
    """
    timestamps_ns = df["host_timestamp_ns"].to_numpy()
    mask = (timestamps_ns >= window_start_ns) & (timestamps_ns < window_end_ns)
    labels = df.loc[mask, "label"].dropna()

    if labels.empty:
        return fallback_label

    normalized = labels.astype(str).str.strip()
    normalized = normalized[normalized != ""]
    if normalized.empty:
        return fallback_label

    counts = normalized.value_counts()
    return str(counts.idxmax())


def process_window(
    df: pd.DataFrame,
    fs: float,
    window_start_ns: int,
    window_end_ns: int,
    label: str,
) -> Dict[str, object]:
    """Compute the full multi-channel feature row for a single window.

    Parameters
    ----------
    df : pd.DataFrame
        Full sorted file data (columns include timestamps and channels).
    fs : float
        Estimated sampling rate for the file.
    window_start_ns, window_end_ns : int
        Window bounds in nanoseconds.
    label : str
        Fallback label inherited from the source file when the window has no
        usable labels.

    Returns
    -------
    dict
        One flattened row: timestamps + per-channel HRV features + label.
    """
    timestamps_ns = df["host_timestamp_ns"].to_numpy()

    row: Dict[str, object] = {
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(window_end_ns),
    }

    for channel in CHANNELS:
        raw_signal = df[channel].to_numpy(dtype=np.float64)
        feats = extract_channel_features(
            raw_signal, timestamps_ns, fs, window_start_ns, window_end_ns
        )
        for feat_name, value in feats.items():
            row[f"{channel}_{feat_name}"] = value

    row["label"] = derive_window_label(
        df, window_start_ns, window_end_ns, fallback_label=label
    )
    return row


def process_file(csv_path: Path) -> Optional[Path]:
    """Process a single input CSV and write its ``*_hrv.csv`` output.

    Parameters
    ----------
    csv_path : Path
        Path to the raw PPG CSV file.

    Returns
    -------
    Optional[Path]
        Path to the written output file, or ``None`` if the file was
        skipped due to an unrecoverable problem.
    """
    print(f"Processing: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [skip] could not read CSV: {exc}")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [skip] missing required columns: {missing}")
        return None

    if df.empty:
        print("  [skip] empty file")
        return None

    try:
        df = df.sort_values("host_timestamp_ns").reset_index(drop=True)
    except Exception as exc:
        print(f"  [skip] could not sort by timestamp: {exc}")
        return None

    # Drop rows with non-finite timestamps or channel values up front.
    numeric_cols = ["host_timestamp_ns"] + CHANNELS
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    if len(df) < before:
        print(f"  [warn] dropped {before - len(df)} rows with invalid numeric values")

    if len(df) < 4:
        print("  [skip] not enough valid rows after cleaning")
        return None

    timestamps_ns = df["host_timestamp_ns"].to_numpy()

    try:
        fs = estimate_sampling_rate(timestamps_ns)
    except Exception as exc:
        print(f"  [skip] could not estimate sampling rate: {exc}")
        return None

    print(f"  Estimated sampling rate: {fs:.2f} Hz")

    label_series = df["label"].dropna()
    label = str(label_series.iloc[0]) if not label_series.empty else ""

    window_ns = int(WINDOW_SECONDS * 1e9)
    stride_ns = int(STRIDE_SECONDS * 1e9)

    t_start = int(timestamps_ns[0])
    t_end = int(timestamps_ns[-1])

    if t_end - t_start < window_ns:
        print(
            f"  [skip] recording too short ({(t_end - t_start) / 1e9:.2f}s) "
            f"for a {WINDOW_SECONDS:.0f}s window"
        )
        return None

    rows: List[Dict[str, object]] = []
    window_start = t_start
    n_windows = 0
    n_failed = 0

    while window_start + window_ns <= t_end:
        window_end = window_start + window_ns
        try:
            row = process_window(df, fs, window_start, window_end, label)
            rows.append(row)
        except Exception as exc:
            n_failed += 1
            print(f"  [warn] window [{window_start}, {window_end}) failed: {exc}")
        n_windows += 1
        window_start += stride_ns

    if not rows:
        print("  [skip] no windows produced")
        return None

    out_df = pd.DataFrame(rows)

    # Enforce exact column order / completeness (58 columns).
    ordered_cols = ["window_start_ns", "window_end_ns"]
    for channel in CHANNELS:
        for feat in FEATURE_NAMES:
            ordered_cols.append(f"{channel}_{feat}")
    ordered_cols.append("label")

    for col in ordered_cols:
        if col not in out_df.columns:
            out_df[col] = np.nan
    out_df = out_df[ordered_cols]

    out_path = csv_path.with_name(f"{csv_path.stem}_hrv.csv")
    out_df.to_csv(out_path, index=False)

    print(
        f"  Done: {n_windows} windows ({n_failed} failed) -> "
        f"{out_path.name} ({len(out_df)} rows, {len(out_df.columns)} columns)"
    )
    return out_path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    """Discover CSVs in the target directory and process each one."""
    target_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()

    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a directory.")
        sys.exit(1)

    csv_files = sorted(
        p for p in target_dir.glob("*.csv") if not p.name.endswith("_hrv.csv")
    )

    if not csv_files:
        print(f"No .csv files found in {target_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {target_dir}\n")

    succeeded = 0
    failed = 0

    for csv_path in csv_files:
        try:
            result = process_file(csv_path)
            if result is not None:
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001 - keep batch processing alive
            failed += 1
            print(f"  [error] unexpected failure on {csv_path.name}: {exc}")
            traceback.print_exc()
        print()

    print(f"Summary: {succeeded} file(s) processed, {failed} file(s) skipped/failed.")


if __name__ == "__main__":
    main()