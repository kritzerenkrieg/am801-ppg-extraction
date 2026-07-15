#!/usr/bin/env python3
"""
modify_ppg_csv.py

Processes all CSV files in the current directory by:
- Removing unnecessary columns
- Keeping only:
    host_timestamp_ns
    red
    infrared
    ppg0
    ppg1
    ppg2
    label

If the 'label' column does not exist, it is created with the default value "base".

Modified files overwrite the originals.

Usage:
    python modify_ppg_csv.py
"""

from pathlib import Path
import pandas as pd

# Required signal columns
SIGNAL_COLUMNS = [
    "host_timestamp_ns",
    "red",
    "infrared",
    "ppg0",
    "ppg1",
    "ppg2",
]

# Final output column order
OUTPUT_COLUMNS = SIGNAL_COLUMNS + ["label"]


def process_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)

        # Check for missing required signal columns
        missing = [col for col in SIGNAL_COLUMNS if col not in df.columns]
        if missing:
            print(f"[SKIP] {csv_path.name}")
            print(f"       Missing required columns: {missing}")
            return

        # Create label column if it doesn't exist
        if "label" not in df.columns:
            df["label"] = "base"

        # Keep only desired columns
        df = df[OUTPUT_COLUMNS]

        # Overwrite original file
        df.to_csv(csv_path, index=False)

        print(f"[OK] {csv_path.name}")

    except Exception as e:
        print(f"[ERROR] {csv_path.name}")
        print(f"        {e}")


def main():
    csv_files = sorted(Path(".").glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV file(s).\n")

    for csv_file in csv_files:
        process_csv(csv_file)

    print("\nDone.")


if __name__ == "__main__":
    main()