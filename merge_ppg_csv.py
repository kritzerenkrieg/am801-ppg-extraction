#!/usr/bin/env python3
"""
Merge two CSV files (AM801 and Verity Sense) based on host_timestamp_ns.

Since the two devices are started at different times and run at different
sampling rates, an exact join on host_timestamp_ns will rarely match.
Instead, this script uses a nearest-neighbour (asof) merge that pairs each
row from one file with the closest timestamp from the other.

Default behaviour: for each AM801 row, find the nearest Verity row.

Use --direction forward / --direction nearest to control asof alignment.
"""

import argparse
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge AM801 and Verity Sense CSV files on host_timestamp_ns"
    )
    parser.add_argument("am_csv", help="Path to the AM801 CSV file")
    parser.add_argument("verity_csv", help="Path to the Verity Sense CSV file")
    parser.add_argument(
        "-o", "--output", default="combined.csv", help="Output CSV path (default: combined.csv)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=None,
        help="Maximum absolute difference (ns) allowed for a match (default: no limit)",
    )
    parser.add_argument(
        "--direction",
        default="nearest",
        choices=["nearest", "forward", "backward"],
        help="Asof merge direction (default: nearest)",
    )
    args = parser.parse_args()

    # Read both CSV files
    am = pd.read_csv(args.am_csv)
    verity = pd.read_csv(args.verity_csv)

    # Ensure both have the join column
    if "host_timestamp_ns" not in am.columns or "host_timestamp_ns" not in verity.columns:
        print("Error: both CSV files must contain a 'host_timestamp_ns' column.")
        return 1

    # Sort by the merge key (required by merge_asof)
    am_sorted = am.sort_values("host_timestamp_ns").reset_index(drop=True)
    verity_sorted = verity.sort_values("host_timestamp_ns").reset_index(drop=True)

    # Merge using asof (nearest, forward, or backward)
    # Note: merge_asof requires that the key is the index or a column.
    # We'll attach a temporary index to keep track of original AM row order later.
    combined = pd.merge_asof(
        am_sorted,
        verity_sorted,
        on="host_timestamp_ns",
        direction=args.direction,
        tolerance=args.tolerance,
        suffixes=("_am", "_verity"),
    )

    # Sort back by timestamp for a clean output
    combined = combined.sort_values("host_timestamp_ns").reset_index(drop=True)

    # Write result
    combined.to_csv(args.output, index=False)

    matched = combined.dropna(subset=[c for c in combined.columns if c.endswith("_verity") or c == "pmd_timestamp"]).shape[0]
    print(f"AM rows: {len(am)},  Verity rows: {len(verity)}")
    print(f"Merged rows: {len(combined)},  Rows with a Verity match: {matched}")
    print(f"Output written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())