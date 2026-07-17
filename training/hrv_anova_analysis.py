#!/usr/bin/env python3
"""
hrv_anova_analysis.py

Perform Analysis of Variance (ANOVA) on HRV features to identify statistically
significant differences between deception detection classes (baseline, truth, deception).

This analysis helps determine which HRV features are most discriminative for
detecting deception.

Usage
-----
    python hrv_anova_analysis.py [directory]

If ``directory`` is omitted, the current working directory is used and must
contain one or more ``*_hrv.csv`` files.

Dependencies
------------
    pandas, numpy, scipy, matplotlib, seaborn
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

WINDOW_SECONDS: float = 10.0
STRIDE_SECONDS: float = 1.0

CHANNELS: List[str] = ["red", "infrared", "ppg0", "ppg1", "ppg2"]

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

EXCLUDE_COLS = {"window_start_ns", "window_end_ns", "label", "subject", "task", "source_file"}

# Statistical significance threshold
ALPHA: float = 0.05

# Effect size interpretation (eta-squared)
SMALL_EFFECT: float = 0.01
MEDIUM_EFFECT: float = 0.06
LARGE_EFFECT: float = 0.14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hrv_anova_analysis")

# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------


def extract_subject(filename: str) -> Tuple[str, str]:
    """Extract ``(subject, task)`` from an ``*_hrv.csv`` filename."""
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
    """Load and concatenate every ``*_hrv.csv`` file in ``directory``."""
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
    logger.info("Loaded %d file(s), %d total windows", len(frames), len(combined))
    logger.info("Classes: %s", combined["label"].value_counts().to_dict())

    return combined


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return every numeric column not in ``EXCLUDE_COLS``."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return feature_cols


def get_channel_agnostic_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Combine the same feature across different channels by averaging.
    
    For example, instead of analyzing red_Mean_NN, infrared_Mean_NN, ppg0_Mean_NN, etc.
    separately, this creates a single 'Mean_NN' feature by averaging across all channels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with channel-specific features
    feature_cols : List[str]
        List of all feature column names
    
    Returns
    -------
    pd.DataFrame
        DataFrame with channel-agnostic features (one column per feature type)
    """
    # Extract base feature names (remove channel prefix)
    base_features = set()
    for col in feature_cols:
        parts = col.split('_', 1)
        if len(parts) == 2:
            base_features.add(parts[1])
    
    base_features = sorted(base_features)
    logger.info(f"Identified {len(base_features)} base feature types: {base_features}")
    
    # Create new dataframe with aggregated features
    agg_df = pd.DataFrame()
    
    # Copy metadata columns
    for col in ['label', 'subject', 'task', 'source_file']:
        if col in df.columns:
            agg_df[col] = df[col]
    
    # Aggregate each feature across channels
    for base_feat in base_features:
        # Find all columns for this feature (e.g., red_Mean_NN, infrared_Mean_NN, etc.)
        matching_cols = [c for c in feature_cols if c.endswith(f'_{base_feat}')]
        
        if not matching_cols:
            continue
        
        # Calculate mean across channels for each row
        values = df[matching_cols].mean(axis=1)
        agg_df[base_feat] = values
        
        logger.debug(f"Aggregated {base_feat}: {len(matching_cols)} channels -> 1 feature")
    
    logger.info(f"Created {len(base_features)} channel-agnostic features from {len(feature_cols)} channel-specific features")
    return agg_df


# --------------------------------------------------------------------------
# ANOVA Analysis
# --------------------------------------------------------------------------


def calculate_eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """Calculate eta-squared effect size from F-statistic.
    
    eta^2 = (F * df_between) / (F * df_between + df_within)
    """
    return (f_stat * df_between) / (f_stat * df_between + df_within)


def interpret_effect_size(eta_sq: float) -> str:
    """Interpret eta-squared effect size."""
    if eta_sq < SMALL_EFFECT:
        return "negligible"
    elif eta_sq < MEDIUM_EFFECT:
        return "small"
    elif eta_sq < LARGE_EFFECT:
        return "medium"
    else:
        return "large"


def perform_one_way_anova(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Perform one-way ANOVA on multiple groups.
    
    Returns dictionary with F-statistic, p-value, and effect size.
    """
    # Remove NaN values from each group
    groups_clean = [g[~np.isnan(g)] for g in groups]
    groups_clean = [g for g in groups_clean if len(g) > 0]
    
    if len(groups_clean) < 2:
        return {"f_stat": np.nan, "p_value": np.nan, "eta_squared": np.nan, "interpretation": "insufficient_data"}
    
    try:
        f_stat, p_value = f_oneway(*groups_clean)
        
        # Calculate effect size (eta-squared)
        # df_between = k - 1, df_within = N - k
        k = len(groups_clean)
        n_total = sum(len(g) for g in groups_clean)
        df_between = k - 1
        df_within = n_total - k
        
        eta_squared = calculate_eta_squared(f_stat, df_between, df_within) if f_stat > 0 else 0.0
        interpretation = interpret_effect_size(eta_squared)
        
        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "interpretation": interpretation
        }
    except Exception as exc:
        logger.warning(f"ANOVA failed: {exc}")
        return {"f_stat": np.nan, "p_value": np.nan, "eta_squared": np.nan, "interpretation": "error"}


def perform_kruskal_wallis(groups: List[np.ndarray]) -> Dict[str, Any]:
    """Perform Kruskal-Wallis non-parametric test (robust alternative to ANOVA)."""
    groups_clean = [g[~np.isnan(g)] for g in groups]
    groups_clean = [g for g in groups_clean if len(g) > 0]
    
    if len(groups_clean) < 2:
        return {"h_stat": np.nan, "p_value": np.nan}
    
    try:
        h_stat, p_value = kruskal(*groups_clean)
        return {"h_stat": h_stat, "p_value": p_value}
    except Exception as exc:
        logger.warning(f"Kruskal-Wallis failed: {exc}")
        return {"h_stat": np.nan, "p_value": np.nan}


def analyze_feature_by_class(
    df: pd.DataFrame, 
    feature_col: str, 
    class_col: str = "label"
) -> Dict[str, Any]:
    """Perform ANOVA and post-hoc analysis for a single feature across classes."""
    
    classes = df[class_col].unique()
    groups = [df[df[class_col] == cls][feature_col].values for cls in classes]
    
    # One-way ANOVA
    anova_results = perform_one_way_anova(groups)
    
    # Kruskal-Wallis (non-parametric alternative)
    kruskal_results = perform_kruskal_wallis(groups)
    
    # Descriptive statistics per class
    desc_stats = []
    for cls in classes:
        values = df[df[class_col] == cls][feature_col].dropna()
        desc_stats.append({
            "class": cls,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max()
        })
    
    result = {
        "feature": feature_col,
        "n_classes": len(classes),
        "classes": ", ".join(sorted(classes)),
        **anova_results,
        "kw_h_stat": kruskal_results["h_stat"],
        "kw_p_value": kruskal_results["p_value"],
        "descriptive_stats": desc_stats
    }
    
    return result


def analyze_all_features(df: pd.DataFrame, feature_cols: List[str], channel_agnostic: bool = False) -> pd.DataFrame:
    """Perform ANOVA for all features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features
    feature_cols : List[str]
        List of feature column names
    channel_agnostic : bool
        If True, aggregate same features across channels before analysis
    """
    if channel_agnostic:
        logger.info("Using CHANNEL-AGNOSTIC mode: aggregating features across channels")
        analysis_df = get_channel_agnostic_features(df, feature_cols)
        analysis_cols = [c for c in analysis_df.columns if c not in EXCLUDE_COLS]
    else:
        logger.info("Using CHANNEL-SPECIFIC mode: analyzing each channel separately")
        analysis_df = df
        analysis_cols = feature_cols
    
    results = []
    
    for feature in analysis_cols:
        logger.info(f"Analyzing feature: {feature}")
        result = analyze_feature_by_class(analysis_df, feature)
        results.append(result)
    
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------


def plot_feature_distributions(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    out_dir: Path,
    top_n: int = 20,
    channel_agnostic: bool = False
) -> None:
    """Plot distribution of top significant features by class."""
    
    # Get ANOVA results to identify top features
    anova_results = analyze_all_features(df, feature_cols, channel_agnostic=channel_agnostic)
    anova_results["neg_log_p"] = -np.log10(anova_results["p_value"].replace(0, 1e-300))
    top_features = anova_results.nsmallest(top_n, "p_value")["feature"].tolist()
    
    # Use aggregated dataframe for channel-agnostic mode
    if channel_agnostic:
        plot_df = get_channel_agnostic_features(df, feature_cols)
    else:
        plot_df = df
    
    # Create subplot grid
    n_features = len(top_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Box plot - use plot_df instead of df
        data_to_plot = [plot_df[plot_df["label"] == cls][feature].dropna().values 
                        for cls in sorted(plot_df["label"].unique())]
        bp = ax.boxplot(data_to_plot, labels=sorted(plot_df["label"].unique()), patch_artist=True)
        
        # Color boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Get p-value for title
        p_val = anova_results[anova_results["feature"] == feature]["p_value"].values[0]
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        ax.set_title(f"{feature}\np={p_val:.2e} {significance}", fontsize=10)
        ax.set_ylabel("Value", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Top {top_n} Most Significant HRV Features by Class", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = out_dir / "anova_feature_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature distributions: {out_path}")


def plot_effect_sizes(anova_results: pd.DataFrame, out_dir: Path, top_n: int = 30, channel_agnostic: bool = False) -> None:
    """Plot effect sizes (eta-squared) for top features."""
    
    # Filter valid results and sort by effect size
    valid_results = anova_results[anova_results["eta_squared"].notna()].copy()
    valid_results = valid_results.nlargest(top_n, "eta_squared")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#2ecc71' if interp == 'large' else 
              '#f39c12' if interp == 'medium' else 
              '#3498db' if interp == 'small' else '#95a5a6'
              for interp in valid_results["interpretation"]]
    
    bars = ax.barh(range(len(valid_results)), valid_results["eta_squared"], color=colors)
    
    ax.set_yticks(range(len(valid_results)))
    ax.set_yticklabels(valid_results["feature"], fontsize=9)
    ax.set_xlabel("Eta-squared (η²) - Effect Size", fontsize=11)
    
    mode_label = "Channel-Agnostic" if channel_agnostic else "Channel-Specific"
    ax.set_title(f"Top {top_n} HRV Features by Effect Size (ANOVA) - {mode_label}", fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Large effect (η² ≥ 0.14)'),
        Patch(facecolor='#f39c12', label='Medium effect (0.06 ≤ η² < 0.14)'),
        Patch(facecolor='#3498db', label='Small effect (0.01 ≤ η² < 0.06)'),
        Patch(facecolor='#95a5a6', label='Negligible (η² < 0.01)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    out_path = out_dir / "anova_effect_sizes.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved effect sizes: {out_path}")


def plot_significance_heatmap(anova_results: pd.DataFrame, out_dir: Path, channel_agnostic: bool = False) -> None:
    """Create heatmap of -log10(p-value) for all features."""
    
    valid_results = anova_results[anova_results["p_value"].notna()].copy()
    valid_results["neg_log_p"] = -np.log10(valid_results["p_value"].replace(0, 1e-300))
    valid_results = valid_results.sort_values("neg_log_p", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(valid_results) * 0.3)))
    
    # Color based on significance
    colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.01 else '#3498db' if p < 0.05 else '#95a5a6'
              for p in valid_results["p_value"]]
    
    ax.barh(range(len(valid_results)), valid_results["neg_log_p"], color=colors)
    ax.set_yticks(range(len(valid_results)))
    ax.set_yticklabels(valid_results["feature"], fontsize=8)
    ax.set_xlabel("-log10(p-value)", fontsize=11)
    
    mode_label = "Channel-Agnostic" if channel_agnostic else "Channel-Specific"
    ax.set_title(f"ANOVA Significance Heatmap (All HRV Features) - {mode_label}", fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add significance threshold line
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax.axvline(x=-np.log10(0.01), color='orange', linestyle='--', linewidth=1, label='p=0.01')
    ax.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', linewidth=1, label='p=0.001')
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    out_path = out_dir / "anova_significance_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved significance heatmap: {out_path}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def generate_report(anova_results: pd.DataFrame, out_dir: Path, channel_agnostic: bool = False) -> None:
    """Generate comprehensive ANOVA analysis report."""
    
    mode_label = "Channel-Agnostic" if channel_agnostic else "Channel-Specific"
    report_path = out_dir / f"anova_analysis_report_{mode_label.lower()}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ANOVA ANALYSIS REPORT - HRV Features for Deception Detection\n")
        f.write(f"Mode: {mode_label}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("Test: One-way ANOVA (F-test)\n")
        f.write(f"Significance Level (α): {ALPHA}\n")
        f.write(f"Effect Size Measure: Eta-squared (η²)\n")
        f.write(f"Effect Size Thresholds:\n")
        f.write(f"  - Small: η² ≥ {SMALL_EFFECT}\n")
        f.write(f"  - Medium: η² ≥ {MEDIUM_EFFECT}\n")
        f.write(f"  - Large: η² ≥ {LARGE_EFFECT}\n\n")
        
        if channel_agnostic:
            f.write("CHANNEL AGGREGATION METHOD\n")
            f.write("-" * 80 + "\n")
            f.write("Features from different sensor channels (red, infrared, ppg0, ppg1, ppg2)\n")
            f.write("were combined by calculating the mean across all channels for each feature.\n")
            f.write("This creates channel-independent feature representations.\n\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write("ANOVA tests whether there are statistically significant differences between\n")
        f.write("class means for each HRV feature. A significant result (p < α) indicates that\n")
        f.write("at least one class differs from the others.\n\n")
        f.write("Effect size (η²) indicates the proportion of variance explained by class\n")
        f.write("membership, providing practical significance beyond statistical significance.\n\n")
        
        # Summary statistics
        total_features = len(anova_results)
        significant = anova_results[anova_results["p_value"] < ALPHA]
        highly_significant = anova_results[anova_results["p_value"] < 0.001]
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total features analyzed: {total_features}\n")
        f.write(f"Significant features (p < {ALPHA}): {len(significant)} ({len(significant)/total_features*100:.1f}%)\n")
        f.write(f"Highly significant (p < 0.001): {len(highly_significant)} ({len(highly_significant)/total_features*100:.1f}%)\n\n")
        
        # Effect size distribution
        f.write("EFFECT SIZE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for effect in ["large", "medium", "small", "negligible"]:
            count = len(anova_results[anova_results["interpretation"] == effect])
            f.write(f"  {effect.capitalize()}: {count} features ({count/total_features*100:.1f}%)\n")
        f.write("\n")
        
        # Detailed results
        f.write("DETAILED RESULTS (sorted by p-value)\n")
        f.write("-" * 80 + "\n")
        
        sorted_results = anova_results.sort_values("p_value")
        
        for _, row in sorted_results.iterrows():
            significance = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
            
            f.write(f"\nFeature: {row['feature']}\n")
            f.write(f"  Classes: {row['classes']}\n")
            f.write(f"  F-statistic: {row['f_stat']:.4f}\n")
            f.write(f"  p-value: {row['p_value']:.2e} {significance}\n")
            f.write(f"  η² (eta-squared): {row['eta_squared']:.4f} ({row['interpretation']})\n")
            f.write(f"  Kruskal-Wallis H: {row['kw_h_stat']:.4f}, p={row['kw_p_value']:.2e}\n")
            
            if isinstance(row["descriptive_stats"], list):
                f.write("  Descriptive statistics:\n")
                for stat in row["descriptive_stats"]:
                    f.write(f"    {stat['class']}: n={stat['n']}, mean={stat['mean']:.4f}, std={stat['std']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Top discriminative features
        top_features = sorted_results.head(10)
        f.write("TOP 10 MOST SIGNIFICANT FEATURES:\n")
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            f.write(f"{idx}. {row['feature']} (p={row['p_value']:.2e}, η²={row['eta_squared']:.4f})\n")
        
        f.write("\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        
        large_effect = sorted_results[sorted_results["interpretation"] == "large"]
        if len(large_effect) > 0:
            f.write(f"\n{len(large_effect)} features show LARGE effect sizes and should be prioritized:\n")
            for _, row in large_effect.iterrows():
                f.write(f"  - {row['feature']} (η²={row['eta_squared']:.4f})\n")
        
        f.write("\n")
        f.write("Features with both statistical significance (p < 0.05) and medium-to-large\n")
        f.write("effect sizes (η² ≥ 0.06) are recommended for classification tasks.\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Saved report: {report_path}")


def save_results_csv(anova_results: pd.DataFrame, out_dir: Path, channel_agnostic: bool = False) -> None:
    """Save ANOVA results to CSV."""
    
    mode_label = "channel_agnostic" if channel_agnostic else "channel_specific"
    
    # Flatten descriptive statistics
    rows = []
    for _, row in anova_results.iterrows():
        base_row = {
            "feature": row["feature"],
            "n_classes": row["n_classes"],
            "classes": row["classes"],
            "f_stat": row["f_stat"],
            "p_value": row["p_value"],
            "eta_squared": row["eta_squared"],
            "interpretation": row["interpretation"],
            "kw_h_stat": row["kw_h_stat"],
            "kw_p_value": row["kw_p_value"]
        }
        
        # Add descriptive stats
        if isinstance(row["descriptive_stats"], list):
            for stat in row["descriptive_stats"]:
                row_dict = base_row.copy()
                row_dict[f"{stat['class']}_n"] = stat["n"]
                row_dict[f"{stat['class']}_mean"] = stat["mean"]
                row_dict[f"{stat['class']}_std"] = stat["std"]
                row_dict[f"{stat['class']}_median"] = stat["median"]
                rows.append(row_dict)
    
    results_df = pd.DataFrame(rows)
    csv_path = out_dir / f"anova_results_{mode_label}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results: {csv_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    """Run ANOVA analysis on HRV features."""
    
    import argparse
    parser = argparse.ArgumentParser(description="ANOVA analysis of HRV features")
    parser.add_argument("directory", help="Directory containing *_hrv.csv files")
    parser.add_argument("--channel-agnostic", action="store_true", 
                       help="Combine same features across channels by averaging")
    args = parser.parse_args()
    
    target_dir = Path(args.directory).expanduser().resolve()
    
    if not target_dir.is_dir():
        logger.error("'%s' is not a directory.", target_dir)
        sys.exit(1)
    
    try:
        df = load_dataset(target_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    
    if df["label"].nunique() < 2:
        logger.error("Need at least 2 classes for ANOVA. Found %d.", df["label"].nunique())
        sys.exit(1)
    
    feature_cols = get_feature_columns(df)
    logger.info("Analyzing %d features across %d classes", len(feature_cols), df["label"].nunique())
    
    # Create output directory
    mode_suffix = "channel_agnostic" if args.channel_agnostic else "channel_specific"
    out_dir = target_dir / f"anova_result_{mode_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    
    # Perform ANOVA
    logger.info("Performing ANOVA analysis...")
    anova_results = analyze_all_features(df, feature_cols, channel_agnostic=args.channel_agnostic)
    
    # Save results
    save_results_csv(anova_results, out_dir, channel_agnostic=args.channel_agnostic)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_feature_distributions(df, feature_cols, out_dir, top_n=20, channel_agnostic=args.channel_agnostic)
    plot_effect_sizes(anova_results, out_dir, top_n=30, channel_agnostic=args.channel_agnostic)
    plot_significance_heatmap(anova_results, out_dir, channel_agnostic=args.channel_agnostic)
    
    # Generate report
    logger.info("Generating report...")
    generate_report(anova_results, out_dir, channel_agnostic=args.channel_agnostic)
    
    # Print summary
    significant = anova_results[anova_results["p_value"] < ALPHA]
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total features analyzed: {len(anova_results)}")
    logger.info(f"Significant (p < {ALPHA}): {len(significant)} ({len(significant)/len(anova_results)*100:.1f}%)")
    logger.info(f"Highly significant (p < 0.001): {len(anova_results[anova_results['p_value'] < 0.001])}")
    logger.info(f"\nTop 5 most significant features:")
    for idx, (_, row) in enumerate(anova_results.nsmallest(5, "p_value").iterrows(), 1):
        logger.info(f"  {idx}. {row['feature']} (p={row['p_value']:.2e}, η²={row['eta_squared']:.4f})")
    logger.info(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()