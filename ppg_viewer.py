#!/usr/bin/env python3
"""
Interactive matplotlib viewer for combined AM801 + Verity Sense PPG data.

Usage:
    python ppg_viewer.py data/raw/bowo/baseline_combined_bowo.csv
    python ppg_viewer.py data/raw/bowo/baseline_combined_bowo.csv --start 10 --duration 30
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for combined AM801 + Verity Sense PPG data"
    )
    parser.add_argument("csv", help="Path to the combined CSV file")
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Start time in seconds (elapsed_s) to view (default: 0)"
    )
    parser.add_argument(
        "--duration", type=float, default=0,
        help="Duration in seconds to display (default: 30, use 0 for all)"
    )
    parser.add_argument(
        "--downsample", type=int, default=1,
        help="Downsample factor: plot every Nth sample (default: 1 = all)"
    )
    parser.add_argument(
        "--absolute", action="store_true",
        help="Use the original absolute elapsed_s from the file instead of "
             "re-baselining time to the first loaded row (default: off)"
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Compute the time axis. By default we re-baseline to the first loaded row so
    # that manually cropping rows (deleting lines) does NOT leave empty space
    # before the first remaining sample. Use --absolute to keep the original
    # elapsed_s values from the file.
    if args.absolute and "elapsed_s" in df.columns:
        df["t_rel"] = df["elapsed_s"].astype(float)
    elif "host_timestamp_ns" in df.columns:
        t0 = df["host_timestamp_ns"].iloc[0]
        df["t_rel"] = (df["host_timestamp_ns"] - t0) / 1e9
    elif "elapsed_s" in df.columns:
        t0 = df["elapsed_s"].iloc[0]
        df["t_rel"] = df["elapsed_s"] - t0
    else:
        df["t_rel"] = np.arange(len(df))

    # Filter by time range (relative to the loaded data's first row)
    if args.duration > 0:
        end = args.start + args.duration
        mask = (df["t_rel"] >= args.start) & (df["t_rel"] <= end)
        df = df.loc[mask].reset_index(drop=True)
        print(f"Time window [{args.start:.1f}, {end:.1f}] s → {len(df)} rows")
    elif args.start > 0:
        df = df.loc[df["t_rel"] >= args.start].reset_index(drop=True)
        print(f"From {args.start:.1f} s → {len(df)} rows")

    # Downsample
    if args.downsample > 1:
        df = df.iloc[::args.downsample].reset_index(drop=True)
        print(f"Downsampled by {args.downsample} → {len(df)} rows")

    if len(df) == 0:
        print("No data in the selected range.")
        return 1

    # Determine which Verity columns exist
    has_verity = "ppg0" in df.columns and df["ppg0"].notna().any()

    # Create figure with subplots
    if has_verity:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
        ax2 = None

    fig.suptitle("PPG Signal Viewer", fontsize=14)

    # --- Top subplot: AM801 signals ---
    x = df["t_rel"].values

    ax1.plot(x, df["red"].values, label="AM801 Red", color="#E74C3C", linewidth=0.5, alpha=0.8)
    ax1.plot(x, df["infrared"].values, label="AM801 Infrared", color="#2ECC71", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("AM801 Raw Value")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Bottom subplot: Verity Sense signals ---
    if has_verity:
        ax2.plot(x, df["ppg0"].values, label="Verity PPG0", color="#3498DB", linewidth=0.5, alpha=0.8)
        ax2.plot(x, df["ppg1"].values, label="Verity PPG1", color="#E67E22", linewidth=0.5, alpha=0.8)
        ax2.plot(x, df["ppg2"].values, label="Verity PPG2", color="#9B59B6", linewidth=0.5, alpha=0.8)
        ax2.set_ylabel("Verity PPG Value")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Time (s)")

    # Add 30-second interval vertical lines
    time_min = x.min()
    time_max = x.max()
    
    # Generate tick positions at 30-second intervals
    interval = 30  # seconds
    first_tick = np.ceil(time_min / interval) * interval
    tick_positions = np.arange(first_tick, time_max + interval, interval)
    
    # Add vertical lines to both subplots
    for tick_pos in tick_positions:
        if tick_pos <= time_max:
            ax1.axvline(x=tick_pos, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
            if has_verity and ax2 is not None:
                ax2.axvline(x=tick_pos, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Format x-axis
    ax1.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())