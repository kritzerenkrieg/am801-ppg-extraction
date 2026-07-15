#!/usr/bin/env python3
"""
hrv_correlation_analysis.py

Compute and visualize Spearman correlation matrix for HRV features extracted
from raw PPG signals. This script analyzes the relationships between different
HRV features across all channels to identify redundant features and understand
feature importance for deception detection.

Usage
-----
    python hrv_correlation_analysis.py [directory]

If ``directory`` is omitted, the current working directory is used and must
contain one or more ``*_hrv.csv`` files (as produced by extract_hrv.py).

Outputs
-------
- ``spearman_correlation_matrix.csv``: Full correlation matrix
- ``spearman_correlation_heatmap.png``: Visual heatmap of correlations
- ``high_correlation_pairs.csv``: Pairs with |correlation| > 0.8 (potentially redundant)
- ``correlation_analysis_report.txt``: Summary statistics and insights

Dependencies
------------
    pandas, numpy, scipy, matplotlib, seaborn
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hrv_correlation_analysis")

# Correlation threshold for identifying potentially redundant features
HIGH_CORRELATION_THRESHOLD = 0.8

# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------


def find_hrv_files(directory: Path) -> List[Path]:
    """Find all *_hrv.csv files in the specified directory."""
    files = sorted(directory.glob("*_hrv.csv"))
    logger.info(f"Found {len(files)} HRV feature files")
    return files


def load_and_combine_hrv_files(files: List[Path]) -> pd.DataFrame:
    """Load and combine multiple HRV CSV files into a single DataFrame."""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Add subject and task information from filename
            stem = f.stem.replace("_hrv", "")
            parts = stem.split("_")
            
            # Extract subject (last part) and task (first part)
            subject = parts[-1] if parts else "unknown"
            task = parts[0] if parts else "unknown"
            
            df["subject"] = subject
            df["task"] = task
            df["source_file"] = f.name
            
            frames.append(df)
            logger.info(f"Loaded {f.name}: {len(df)} windows")
        except Exception as exc:
            logger.warning(f"Failed to load {f.name}: {exc}")
    
    if not frames:
        raise ValueError("No valid HRV files could be loaded")
    
    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(f"Combined dataset: {len(combined)} total windows from {len(frames)} files")
    return combined


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract HRV feature columns (exclude metadata columns)."""
    exclude_cols = {"window_start_ns", "window_end_ns", "label", "subject", "task", "source_file"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    logger.info(f"Found {len(feature_cols)} numeric feature columns")
    return feature_cols


def aggregate_features_across_channels(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Aggregate HRV features across all channels by computing the mean.
    
    This reduces the feature set from (channels × features) to just (features),
    allowing correlation analysis of the HRV features themselves rather than
    sensor-specific variations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with channel-specific feature columns
    feature_cols : List[str]
        List of all feature column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated features (one column per HRV feature type)
    """
    logger.info("Aggregating features across channels...")
    
    # Extract base feature names (remove channel prefix)
    base_features = set()
    for col in feature_cols:
        # Split by underscore and get everything after the first underscore
        parts = col.split('_', 1)
        if len(parts) == 2:
            base_features.add(parts[1])
    
    base_features = sorted(base_features)
    logger.info(f"Found {len(base_features)} unique HRV feature types: {base_features}")
    
    # Compute mean across channels for each feature
    aggregated_data = {}
    for base_feat in base_features:
        # Find all columns for this feature across channels
        matching_cols = [c for c in feature_cols if c.endswith(f'_{base_feat}') or c == base_feat]
        if matching_cols:
            # Compute row-wise mean across all matching columns
            aggregated_data[base_feat] = df[matching_cols].mean(axis=1)
    
    aggregated_df = pd.DataFrame(aggregated_data)
    logger.info(f"Aggregated to {len(aggregated_df.columns)} features: {list(aggregated_df.columns)}")
    
    return aggregated_df


# --------------------------------------------------------------------------
# Correlation Analysis
# --------------------------------------------------------------------------


def compute_spearman_correlation(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute Spearman correlation matrix for all features."""
    logger.info("Computing Spearman correlation matrix...")
    
    # Extract feature data and drop rows with any NaN
    feature_data = df[feature_cols].copy()
    feature_data_clean = feature_data.dropna(axis=0, how='any')
    
    logger.info(f"Using {len(feature_data_clean)} complete windows for correlation analysis")
    
    if len(feature_data_clean) < 3:
        raise ValueError("Insufficient data for correlation analysis (need at least 3 complete windows)")
    
    # Compute Spearman correlation
    corr_matrix = feature_data_clean.corr(method='spearman')
    
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
    return corr_matrix


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = HIGH_CORRELATION_THRESHOLD) -> pd.DataFrame:
    """Find feature pairs with high absolute correlation."""
    logger.info(f"Finding feature pairs with |correlation| > {threshold}...")
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)
    
    # Find high correlations
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if pd.notna(upper_tri.loc[idx, col]):
                corr_val = upper_tri.loc[idx, col]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'Feature_1': idx,
                        'Feature_2': col,
                        'Correlation': corr_val,
                        'Abs_Correlation': abs(corr_val)
                    })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        logger.info(f"Found {len(high_corr_df)} high correlation pairs")
    else:
        logger.info("No high correlation pairs found")
    
    return high_corr_df


def compute_correlation_statistics(corr_matrix: pd.DataFrame) -> Dict[str, float]:
    """Compute summary statistics for the correlation matrix."""
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)
    
    corr_values = upper_tri.values.flatten()
    corr_values = corr_values[~np.isnan(corr_values)]
    
    stats = {
        'mean_correlation': np.mean(corr_values),
        'std_correlation': np.std(corr_values),
        'min_correlation': np.min(corr_values),
        'max_correlation': np.max(corr_values),
        'median_correlation': np.median(corr_values),
        'n_high_positive': np.sum(corr_values > HIGH_CORRELATION_THRESHOLD),
        'n_high_negative': np.sum(corr_values < -HIGH_CORRELATION_THRESHOLD),
        'n_total_pairs': len(corr_values)
    }
    
    return stats


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path, figsize: Tuple[int, int] = None) -> None:
    """Create and save a heatmap visualization of the correlation matrix."""
    logger.info(f"Creating correlation heatmap: {output_path.name}")
    
    # Adjust figure size based on number of features
    n_features = len(corr_matrix)
    if figsize is None:
        figsize = (max(12, n_features * 0.3), max(10, n_features * 0.3))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with diverging colormap
    sns.heatmap(
        corr_matrix,
        annot=False,  # Too many features for annotations
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
        ax=ax
    )
    
    ax.set_title('Spearman Correlation Matrix of HRV Features', fontsize=16, pad=20)
    ax.set_xlabel('HRV Features', fontsize=12)
    ax.set_ylabel('HRV Features', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved heatmap: {output_path}")


def plot_correlation_clustermap(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    """Create a clustered heatmap to group similar features."""
    logger.info(f"Creating clustered correlation heatmap: {output_path.name}")
    
    # Create clustermap
    g = sns.clustermap(
        corr_matrix,
        method='complete',
        metric='euclidean',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(16, 16),
        linewidths=0.5,
        cbar_kws={"label": "Spearman Correlation", "shrink": 0.8}
    )
    
    g.fig.suptitle('Clustered Spearman Correlation Matrix of HRV Features', fontsize=16, y=0.98)
    g.fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved clustermap: {output_path}")


def plot_feature_correlation_distribution(corr_matrix: pd.DataFrame, output_path: Path) -> None:
    """Plot distribution of correlation values."""
    logger.info(f"Creating correlation distribution plot: {output_path.name}")
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)
    corr_values = upper_tri.values.flatten()
    corr_values = corr_values[~np.isnan(corr_values)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(corr_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    axes[0].axvline(x=np.mean(corr_values), color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(corr_values):.3f}')
    axes[0].set_xlabel('Spearman Correlation', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Feature Correlations', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(corr_values, vert=True, showmeans=True, meanline=True)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    axes[1].set_ylabel('Spearman Correlation', fontsize=12)
    axes[1].set_title('Correlation Distribution (Box Plot)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved distribution plot: {output_path}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def generate_report(
    corr_matrix: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    stats: Dict[str, float],
    output_path: Path
) -> None:
    """Generate a text report summarizing correlation analysis findings."""
    logger.info(f"Generating analysis report: {output_path.name}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HRV FEATURE CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total features analyzed: {len(corr_matrix)}\n")
        f.write(f"Total feature pairs: {stats['n_total_pairs']}\n")
        f.write(f"Correlation method: Spearman\n\n")
        
        f.write("CORRELATION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean correlation:   {stats['mean_correlation']:+.4f}\n")
        f.write(f"Std correlation:    {stats['std_correlation']:.4f}\n")
        f.write(f"Median correlation: {stats['median_correlation']:+.4f}\n")
        f.write(f"Min correlation:    {stats['min_correlation']:+.4f}\n")
        f.write(f"Max correlation:    {stats['max_correlation']:+.4f}\n\n")
        
        f.write("HIGH CORRELATION ANALYSIS (|r| > 0.8)\n")
        f.write("-" * 80 + "\n")
        f.write(f"High positive correlations (r > 0.8): {stats['n_high_positive']}\n")
        f.write(f"High negative correlations (r < -0.8): {stats['n_high_negative']}\n")
        f.write(f"Total high correlations: {len(high_corr_pairs)}\n\n")
        
        if not high_corr_pairs.empty:
            f.write("TOP 20 HIGHEST CORRELATIONS (Potentially Redundant Features)\n")
            f.write("-" * 80 + "\n")
            for idx, row in high_corr_pairs.head(20).iterrows():
                f.write(f"{row['Feature_1']} <-> {row['Feature_2']}: r = {row['Correlation']:+.4f}\n")
            f.write("\n")
        
        f.write("INTERPRETATION AND RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Analyze correlation patterns
        if stats['mean_correlation'] > 0.5:
            f.write("• HIGH AVERAGE CORRELATION: Features are highly correlated on average.\n")
            f.write("  → Consider dimensionality reduction (PCA, feature selection)\n")
        elif stats['mean_correlation'] > 0.3:
            f.write("• MODERATE AVERAGE CORRELATION: Some redundancy exists.\n")
            f.write("  → Review high correlation pairs and consider removing redundant features\n")
        else:
            f.write("• LOW AVERAGE CORRELATION: Features are relatively independent.\n")
            f.write("  → Good feature diversity, minimal redundancy\n")
        
        if len(high_corr_pairs) > 0:
            f.write(f"\n• {len(high_corr_pairs)} HIGH CORRELATION PAIRS DETECTED:\n")
            f.write("  → These feature pairs may be redundant\n")
            f.write("  → Consider keeping only one feature from each highly correlated pair\n")
            f.write("  → Recommended action: Remove one feature from each pair with |r| > 0.9\n")
        
        # Channel-specific insights
        channels = ['red', 'infrared', 'ppg0', 'ppg1', 'ppg2']
        f.write("\n• CHANNEL-SPECIFIC OBSERVATIONS:\n")
        for channel in channels:
            channel_features = [c for c in corr_matrix.columns if c.startswith(channel + '_')]
            if channel_features:
                f.write(f"  - {channel}: {len(channel_features)} features\n")
        
        f.write("\n• FEATURE IMPORTANCE INDICATORS:\n")
        f.write("  - Features with low mean correlation to others may be more informative\n")
        f.write("  - Highly correlated features provide redundant information\n")
        f.write("  - Consider domain knowledge when selecting features for final model\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Report saved: {output_path}")


# --------------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------------


def main() -> None:
    """Main analysis pipeline."""
    # Determine input directory
    target_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
    
    if not target_dir.is_dir():
        logger.error(f"'{target_dir}' is not a directory")
        sys.exit(1)
    
    logger.info(f"Analyzing HRV features in: {target_dir}")
    
    # Find and load HRV files
    hrv_files = find_hrv_files(target_dir)
    if not hrv_files:
        logger.error("No *_hrv.csv files found")
        sys.exit(1)
    
    df = load_and_combine_hrv_files(hrv_files)
    feature_cols = get_feature_columns(df)
    
    if len(feature_cols) < 2:
        logger.error("Need at least 2 features for correlation analysis")
        sys.exit(1)
    
    # Aggregate features across channels to get HRV feature correlations
    aggregated_df = aggregate_features_across_channels(df, feature_cols)
    agg_feature_cols = list(aggregated_df.columns)
    
    # Compute correlations on aggregated features
    corr_matrix = compute_spearman_correlation(aggregated_df, agg_feature_cols)
    high_corr_pairs = find_high_correlations(corr_matrix, HIGH_CORRELATION_THRESHOLD)
    stats = compute_correlation_statistics(corr_matrix)
    
    # Save results
    output_dir = target_dir
    
    # Save correlation matrix
    corr_matrix_path = output_dir / "spearman_correlation_matrix.csv"
    corr_matrix.to_csv(corr_matrix_path)
    logger.info(f"Saved correlation matrix: {corr_matrix_path}")
    
    # Save high correlation pairs
    if not high_corr_pairs.empty:
        high_corr_path = output_dir / "high_correlation_pairs.csv"
        high_corr_pairs.to_csv(high_corr_path, index=False)
        logger.info(f"Saved high correlation pairs: {high_corr_path}")
    
    # Create visualizations
    plot_correlation_heatmap(corr_matrix, output_dir / "spearman_correlation_heatmap.png")
    plot_correlation_clustermap(corr_matrix, output_dir / "spearman_correlation_clustermap.png")
    plot_feature_correlation_distribution(corr_matrix, output_dir / "correlation_distribution.png")
    
    # Generate report
    generate_report(corr_matrix, high_corr_pairs, stats, output_dir / "correlation_analysis_report.txt")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Features analyzed: {len(feature_cols)}")
    logger.info(f"Mean correlation: {stats['mean_correlation']:+.4f}")
    logger.info(f"High correlation pairs: {len(high_corr_pairs)}")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()