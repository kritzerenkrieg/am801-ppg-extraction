# HRV Feature Correlation Analysis - Insights and Interpretation

## Executive Summary

The Spearman correlation analysis of 11 HRV features (aggregated across 5 PPG channels) reveals **low overall redundancy** (mean r = +0.139) but identifies **13 highly correlated feature pairs** (|r| > 0.8) that may contain redundant information. The analysis is based on 1,502 complete windows from 12 subjects across baseline and deception tasks.

---

## Key Findings

### 1. Overall Feature Independence (Good News)

**Mean correlation: +0.139** (std: 0.613)

- Features are **relatively independent** on average
- Low average correlation indicates **good feature diversity**
- Most feature pairs provide **unique information**
- This is excellent for machine learning - minimal multicollinearity issues

### 2. High Correlation Pairs Identified (13 pairs with |r| > 0.8)

#### Perfect Correlations (r = 1.000) - Mathematically Related Features

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| **SD1 ↔ SDSD** | +1.000 | SD1 is derived from SDSD (Poincaré plot geometry) |
| **RMSSD ↔ SDSD** | +1.000 | RMSSD and SDSD are mathematically equivalent |
| **RMSSD ↔ SD1** | +1.000 | RMSSD = √2 × SD1 (theoretical relationship) |

**Action Required**: These three features are **completely redundant**. Keep only **RMSSD** (most commonly used in HRV literature) and remove SD1 and SDSD.

#### Very High Correlations (0.96 - 0.99) - Time-Domain Feature Cluster

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| **SD2 ↔ SDNN** | +0.998 | Both measure long-term HRV variability |
| **RMSSD ↔ SDNN** | +0.981 | Short-term and long-term variability measures |
| **SDNN ↔ SDSD** | +0.981 | Both capture overall HRV magnitude |
| **SD1 ↔ SDNN** | +0.981 | Short-term vs. overall variability |
| **RMSSD ↔ SD2** | +0.968 | Short-term and long-term Poincaré measures |
| **SD1 ↔ SD2** | +0.968 | Poincaré plot axes |
| **SD2 ↔ SDSD** | +0.968 | Long-term variability measures |

**Action Required**: This cluster of time-domain features is highly redundant. Consider keeping:
- **SDNN** (standard long-term HRV measure)
- **RMSSD** (standard short-term HRV measure)
- Remove: SDSD, SD1, SD2 (derived/less commonly used)

#### High Negative Correlation (-0.90) - Physiologically Expected

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| **Mean_HR ↔ Mean_NN** | -0.903 | Heart rate and NN intervals are inversely related (physiologically correct) |

**Action Required**: This is **not redundancy** - it's expected physiology. Keep both features as they provide complementary information.

#### High Correlation (-0.89) - Related Measures

| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| **Mean_NN ↔ NN_Count** | -0.890 | Longer intervals → fewer beats in window |
| **Mean_HR ↔ NN_Count** | +0.844 | Higher HR → more beats in window |

**Action Required**: These relationships are physiologically logical. Consider keeping **NN_Count** as a proxy for heart rate, or keep **Mean_HR** and remove NN_Count to reduce redundancy.

---

## Feature Clusters Identified

### Cluster 1: Short-Term Variability (RMSSD, SD1, SDSD)
- **Perfect correlation** (r = 1.0)
- Mathematically derived from the same underlying data
- **Recommendation**: Keep only **RMSSD**

### Cluster 2: Long-Term Variability (SDNN, SD2)
- **Very high correlation** (r = 0.968-0.998)
- Both measure overall HRV magnitude
- **Recommendation**: Keep only **SDNN** (standard metric)

### Cluster 3: Heart Rate Metrics (Mean_HR, Mean_NN, NN_Count, Heart_Rate, Median_NN)
- **Moderate to high correlations** (r = -0.90 to +0.84)
- All related to heart rate or NN intervals
- **Recommendation**: Keep **Mean_HR** and **Median_NN**, remove NN_Count (redundant with Mean_HR)

### Cluster 4: Independent Features (SD1_SD2 ratio)
- **Low correlations** with other features
- Unique information about autonomic balance
- **Recommendation**: **Keep** - provides unique information

---

## Recommended Feature Set

### Minimal Set (7 features - 36% reduction)
After removing redundant features:

1. **Mean_NN** - Average NN interval
2. **Median_NN** - Median NN interval (robust to outliers)
3. **Heart_Rate** - Instantaneous heart rate
4. **RMSSD** - Short-term HRV (replaces SD1, SDSD)
5. **SDNN** - Long-term HRV (replaces SD2)
6. **SD1_SD2** - Poincaré ratio (autonomic balance)
7. **NN_Count** - Number of beats (optional, or use Mean_HR instead)

### Optimal Set (6 features - 45% reduction)
For maximum efficiency:

1. **Mean_HR** - Replaces Mean_NN, NN_Count, Heart_Rate, Median_NN
2. **RMSSD** - Replaces SD1, SDSD
3. **SDNN** - Replaces SD2
4. **SD1_SD2** - Unique autonomic balance metric

**Result**: 11 features → 4 features (64% reduction) with minimal information loss

---

## Physiological Interpretation

### Why These Correlations Exist

1. **RMSSD = SDSD = √2 × SD1**
   - Mathematical identity in Poincaré plot analysis
   - All three measure short-term HRV variability
   - Keeping all three provides no additional information

2. **SDNN ≈ SD2**
   - SDNN measures total HRV variability
   - SD2 measures long-term variability in Poincaré plot
   - Highly correlated because both capture slow HRV components

3. **Mean_HR ↔ Mean_NN (r = -0.90)**
   - HR (bpm) = 60,000 / NN (ms)
   - Perfectly inverse relationship by definition
   - Not redundancy - complementary perspectives

4. **Time-domain features cluster together**
   - SDNN, RMSSD, SDSD, SD1, SD2 all measure HRV magnitude
   - Natural correlation due to shared underlying physiology
   - Different time scales provide some unique information

---

## Impact on Machine Learning

### Benefits of Feature Reduction

1. **Faster training**: 64% fewer features → significantly faster model training
2. **Reduced overfitting**: Fewer features → less chance of learning noise
3. **Better interpretability**: Simpler feature set → easier to explain results
4. **Improved generalization**: Less redundancy → models focus on signal, not noise

### Expected Performance Impact

- **Minimal accuracy loss**: Removing redundant features unlikely to hurt performance
- **Potential improvement**: Reduced overfitting may actually improve LOSOCV results
- **Faster inference**: Deployed models will be more efficient

---

## Visualization Insights

### Correlation Heatmap (spearman_correlation_heatmap.png)
- Shows all 55 original features (11 features × 5 channels)
- Clear block structure showing channel consistency
- Red blocks indicate high positive correlations
- Blue blocks indicate high negative correlations

### Clustered Heatmap (spearman_correlation_clustermap.png)
- Groups similar features together
- Clearly shows the three main clusters:
  - Short-term variability cluster (RMSSD, SD1, SDSD)
  - Long-term variability cluster (SDNN, SD2)
  - Heart rate cluster (Mean_HR, Mean_NN, NN_Count)

### Distribution Plot (correlation_distribution.png)
- Most correlations cluster near zero (mean: +0.139)
- Heavy tails indicate some highly correlated pairs
- Symmetric distribution around zero

---

## Recommendations

### Immediate Actions

1. **Remove perfectly correlated features**:
   - Delete: **SDSD**, **SD1** (keep RMSSD)
   - Expected improvement: 2 features removed, no information loss

2. **Remove highly correlated features**:
   - Delete: **SD2** (keep SDNN)
   - Expected improvement: 1 feature removed, minimal information loss

3. **Consolidate heart rate metrics**:
   - Keep: **Mean_HR** (most interpretable)
   - Delete: **Mean_NN**, **NN_Count**, **Heart_Rate**, **Median_NN**
   - Expected improvement: 4 features removed, redundant information eliminated

### Final Recommended Feature Set

**Option A - Minimal (7 features)**:
- Mean_NN, Median_NN, Heart_Rate, RMSSD, SDNN, SD1_SD2, NN_Count

**Option B - Optimal (4 features)**:
- Mean_HR, RMSSD, SDNN, SD1_SD2

**Option C - Balanced (6 features)**:
- Mean_HR, Median_NN, RMSSD, SDNN, SD1_SD2, NN_Count

### Next Steps

1. **Retrain models** with reduced feature set
2. **Compare performance** (LOSOCV accuracy, ROC-AUC)
3. **Validate** that feature reduction doesn't significantly hurt performance
4. **Update** `train_hrv_models.py` to use optimized feature set
5. **Document** final feature set in research paper/thesis

---

## Conclusion

The HRV feature set shows **good overall independence** (mean r = 0.139) with **expected physiological correlations**. The main redundancy comes from:

1. **Mathematically equivalent features** (RMSSD, SD1, SDSD)
2. **Highly correlated time-domain measures** (SDNN, SD2)
3. **Redundant heart rate representations** (Mean_HR, Mean_NN, NN_Count, Heart_Rate)

**Potential feature reduction: 64%** (from 11 to 4 features) with minimal information loss, leading to faster training, better generalization, and improved model interpretability.