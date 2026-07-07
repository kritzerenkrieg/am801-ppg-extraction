#!/usr/bin/env python3

"""Interactive CSV segment labeling.

Given a CSV containing an `elapsed_s` column, this script splits the signal
into fixed-length segments (default: 30 seconds) and asks the user to label
each segment as:

  - T => truth
  - F => deceptive

It writes a new CSV with an extra `label` column.
The output file name is the input name with `*_labeled.csv` suffix.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from math import floor
from typing import Dict, Iterable, List, Tuple
from datetime import datetime


TRUTH_MAP = {
    "t": "truth",
    "truth": "truth",
    "1": "truth",
}

DECEPTIVE_MAP = {
    "f": "deceptive",
    "deceptive": "deceptive",
    "0": "deceptive",
}


@dataclass(frozen=True)
class Segment:
    index: int
    start_s: float
    end_s: float


def parse_label(token: str) -> str:
    t = token.strip().lower()
    if t in TRUTH_MAP:
        return TRUTH_MAP[t]
    if t in DECEPTIVE_MAP:
        return DECEPTIVE_MAP[t]
    raise ValueError("Expected T/truth/1 or F/deceptive/0")


def build_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    # base already includes folders; keep only file name semantics.
    return f"{base}_labeled{ext or '.csv'}"


def get_float(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except KeyError as e:
        raise KeyError(f"Missing required column '{field}'") from e
    except Exception as e:
        raise ValueError(f"Could not parse column '{field}' as float") from e


def compute_segment_index(seconds_since_epoch: float, segment_seconds: float) -> int:
    """Return the segment index for a given time value.

    This uses the absolute time in seconds (e.g. epoch seconds from host_timestamp_iso_am)
    and divides by the segment length. Negative values are handled by floor division
    just like before, ensuring consistent behavior.
    """
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    return int(floor(seconds_since_epoch / segment_seconds))


def parse_host_timestamp_seconds(row: dict, time_field: str) -> float:
    ts = row.get(time_field)
    if ts is None:
        raise KeyError(f"Missing required time column '{time_field}'")
    try:
        # Python 3.7+: fromisoformat supports the ISO 8601 format with offset
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"Could not parse timestamp '{ts}' in field '{time_field}' as ISO8601: {e}") from e


def iter_rows_with_segments(
    rows: Iterable[dict],
    segment_seconds: float,
    time_field: str,
) -> Tuple[List[int], List[float]]:
    # If the user passed the default elapsed_s as the time_field, we actually want to
    # base segmentation on host_timestamp_iso_am (the absolute time of the sample).
    actual_time_field = time_field
    if time_field == "elapsed_s":
        actual_time_field = "host_timestamp_iso_am"

    # Collect all timestamps first to compute a relative time base (start of dataset)
    time_values: List[float] = []
    for r in rows:
        seconds = parse_host_timestamp_seconds(r, actual_time_field)
        time_values.append(seconds)
    if not time_values:
        return [], []
    base_time = min(time_values)
    seg_indices: List[int] = []
    for t in time_values:
        si = int(floor((t - base_time) / segment_seconds))
        seg_indices.append(si)
    return seg_indices, time_values


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively label 30s segments in a merged CSV.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV (e.g. data/preprocessed/interview_doni_merged.csv)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="Segment length in seconds (default: 30)",
    )
    parser.add_argument(
        "--elapsed-field",
        default="elapsed_s",
        help="Column name used to compute segments (default: elapsed_s)",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the output label column (default: label)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional comma-separated labels for each segment index in ascending order "
            "(e.g. 'T,F,T'). If provided, no interactive prompting is done."
        ),
    )
    args = parser.parse_args()

    input_path: str = args.input
    segment_seconds: float = float(args.segment_seconds)
    elapsed_field: str = args.elapsed_field
    label_col: str = args.label_col

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    # Load all rows (CSV is typically not huge for interactive labeling).
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Input CSV appears to have no header.", file=sys.stderr)
            return 2
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    seg_indices, elapsed_values = iter_rows_with_segments(
        rows=rows,
        segment_seconds=segment_seconds,
        time_field=elapsed_field,
    )

    unique_segments = sorted(set(seg_indices))
    if not unique_segments:
        print("No segments found (check elapsed_s column).", file=sys.stderr)
        return 2

    segment_labels: Dict[int, str] = {}
    print(f"Found {len(unique_segments)} segment(s) with segment length {segment_seconds}s.")

    if args.labels is not None:
        tokens = [t.strip() for t in args.labels.split(",") if t.strip()]
        if len(tokens) != len(unique_segments):
            print(
                f"--labels provided {len(tokens)} token(s) but expected {len(unique_segments)} (one per segment).",
                file=sys.stderr,
            )
            return 2
        for si, tok in zip(unique_segments, tokens):
            segment_labels[si] = parse_label(tok)
    else:
        # Ask the user for each segment.
        print("Provide label per segment: T=truth, F=deceptive")
        print("-------------------------------------------------------")
        for si in unique_segments:
            start_s = si * segment_seconds
            end_s = (si + 1) * segment_seconds
            while True:
                prompt = f"Segment {si} ({start_s:.3f}s - {end_s:.3f}s). Enter T or F: "
                try:
                    token = input(prompt)
                except EOFError:
                    print("\nInput ended unexpectedly. Aborting.", file=sys.stderr)
                    return 2
                if not token.strip():
                    continue
                try:
                    segment_labels[si] = parse_label(token)
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}")

    output_path = build_output_path(input_path)
    out_fieldnames = fieldnames
    if label_col in out_fieldnames:
        # avoid duplicate column; replace by dropping existing label column
        out_fieldnames = [fn for fn in out_fieldnames if fn != label_col]
    out_fieldnames = out_fieldnames + [label_col]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row, si in zip(rows, seg_indices):
            # Copy row without possible existing label_col.
            out_row = {k: v for k, v in row.items() if k != label_col}
            out_row[label_col] = segment_labels[si]
            writer.writerow(out_row)

    print(f"Wrote labeled CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
