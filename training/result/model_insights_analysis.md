# Model Performance Insights Analysis

## ⚠️ CRITICAL METHODOLOGICAL WARNING

**The "SubjectSpecific" evaluation uses a SINGLE 80/20 train/test split per subject, NOT cross-validation.**

This is a significant limitation that affects the interpretation of all "SubjectSpecific" results:
- Each subject's data is split once (random_state=42) into 80% train / 20% test
- Results are highly dependent on this single random split
- No cross-validation is performed within subjects
- Standard deviations reflect variance **across subjects**, not across different splits
- High accuracies (90-98%) may be inflated due to lucky splits and are not as reliable as they appear

**Only the LOSOCV results provide robust cross-validation estimates.**

---

## Overview
This analysis compares three machine learning models (SVM, RandomForest, MLP) across three experimental conditions (card-only, interview-only, mix) using two feature sets (2_hrv and 3_hrv). Each model is evaluated under two scenarios:
- **SubjectSpecific**: Single 80/20 stratified split per subject (NOT cross-validated)
- **LOSOCV**: Leave-One-Subject-Out cross-validation (proper cross-validation)

---

## 1. SVM (Support Vector Machine)

### Performance Summary

#### Subject-Specific Evaluation
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.920 (±0.071) | 0.963 (±0.048) | 0.920 (±0.052) | 0.987 (±0.017) |
| Interview-only | 0.964 (±0.044) | 0.997 (±0.007) | 0.944 (±0.038) | 0.998 (±0.002) |
| Mix | 0.896 (±0.053) | 0.952 (±0.039) | 0.869 (±0.051) | 0.972 (±0.015) |

**Key Insights:**
- **Best performance**: Interview-only condition with 2_hrv features (96.4% accuracy, 99.7% ROC-AUC)
- **Most stable**: Interview-only condition shows lowest standard deviation
- **Feature impact**: 3_hrv features improve ROC-AUC significantly in card-only (+0.024) and mix (+0.020) conditions
- **Generalizability**: Excellent ROC-AUC scores (>0.95) across all subject-specific evaluations

#### LOSOCV Evaluation (Generalization Test)
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.534 (±0.121) | 0.531 (±0.155) | 0.420 (±0.111) | 0.586 (±0.138) |
| Interview-only | 0.676 (±0.089) | 0.419 (±0.242) | 0.517 (±0.195) | 0.670 (±0.099) |
| Mix | 0.574 (±0.081) | 0.450 (±0.146) | 0.424 (±0.277) | 0.564 (±0.186) |

**Key Insights:**
- **Best generalization**: Interview-only with 2_hrv (67.6% accuracy) - significantly better than other conditions
- **ROC-AUC anomaly**: Interview-only 2_hrv shows low ROC-AUC (0.419) despite highest accuracy, suggesting imbalanced predictions
- **Feature impact**: 3_hrv improves ROC-AUC in card-only (+0.055) and interview-only (+0.251), but with high variance
- **Performance drop**: 26-50% accuracy decrease from subject-specific to LOSOCV, indicating overfitting to individual subjects
- **Most stable**: Mix condition with 2_hrv shows lowest std (0.081)

---

## 2. RandomForest

### Performance Summary

#### Subject-Specific Evaluation
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.920 (±0.042) | 0.941 (±0.084) | 0.943 (±0.050) | 0.989 (±0.016) |
| Interview-only | 0.971 (±0.030) | 0.997 (±0.007) | 0.980 (±0.014) | 0.999 (±0.002) |
| Mix | 0.889 (±0.020) | 0.975 (±0.009) | 0.941 (±0.019) | 0.994 (±0.002) |

**Key Insights:**
- **Best performance**: Interview-only with 3_hrv (98.0% accuracy, 99.9% ROC-AUC) - nearly perfect
- **Most consistent**: Mix condition shows lowest standard deviations across all metrics
- **Feature impact**: 3_hrv consistently improves performance, especially in mix condition (+0.052 accuracy, +0.019 ROC-AUC)
- **Stability**: Lowest std in accuracy among all models (0.014-0.050 range)
- **ROC-AUC excellence**: >0.975 across all subject-specific evaluations

#### LOSOCV Evaluation (Generalization Test)
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.529 (±0.117) | 0.566 (±0.084) | 0.379 (±0.189) | 0.562 (±0.312) |
| Interview-only | 0.566 (±0.226) | 0.407 (±0.281) | 0.599 (±0.227) | 0.753 (±0.190) |
| Mix | 0.610 (±0.062) | 0.480 (±0.126) | 0.515 (±0.351) | 0.708 (±0.253) |

**Key Insights:**
- **Best generalization**: Mix with 2_hrv (61.0% accuracy) - most stable (std 0.062)
- **Highest ROC-AUC**: Interview-only 3_hrv (0.753) - best discriminative ability across subjects
- **Feature impact**: 3_hrv improves ROC-AUC significantly in interview-only (+0.346) and mix (+0.228)
- **High variance**: Interview-only and mix 3_hrv show very high std (>0.190), indicating inconsistent performance
- **Performance drop**: 35-45% accuracy decrease from subject-specific to LOSOCV
- **Overfitting indicator**: Large gap between subject-specific (~95%) and LOSOCV (~50-60%)

---

## 3. MLP (Multi-Layer Perceptron)

### Performance Summary

#### Subject-Specific Evaluation
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.840 (±0.113) | 0.882 (±0.134) | 0.830 (±0.104) | 0.922 (±0.109) |
| Interview-only | 0.857 (±0.056) | 0.819 (±0.177) | 0.884 (±0.033) | 0.971 (±0.026) |
| Mix | 0.704 (±0.086) | 0.640 (±0.132) | 0.833 (±0.084) | 0.938 (±0.055) |

**Key Insights:**
- **Best performance**: Interview-only with 3_hrv (88.4% accuracy, 97.1% ROC-AUC)
- **Worst performance**: Mix with 2_hrv (70.4% accuracy, 64.0% ROC-AUC)
- **Feature impact**: 3_hrv dramatically improves performance in interview-only (+0.027 accuracy, +0.152 ROC-AUC) and mix (+0.129 accuracy, +0.298 ROC-AUC)
- **Highest variance**: Card-only 2_hrv shows highest std in accuracy (±0.113)
- **ROC-AUC improvement**: 3_hrv features consistently improve ROC-AUC by 0.040-0.298

#### LOSOCV Evaluation (Generalization Test)
| Condition | 2_hrv Accuracy | 2_hrv ROC-AUC | 3_hrv Accuracy | 3_hrv ROC-AUC |
|-----------|----------------|---------------|----------------|---------------|
| Card-only | 0.492 (±0.098) | 0.436 (±0.099) | 0.487 (±0.151) | 0.589 (±0.171) |
| Interview-only | 0.627 (±0.231) | 0.451 (±0.292) | 0.616 (±0.126) | 0.634 (±0.130) |
| Mix | 0.641 (±0.023) | 0.463 (±0.148) | 0.547 (±0.311) | 0.605 (±0.182) |

**Key Insights:**
- **Best generalization**: Mix with 2_hrv (64.1% accuracy) - most stable (std 0.023)
- **Best ROC-AUC**: Interview-only 3_hrv (0.634) and mix 3_hrv (0.605)
- **Feature impact**: 3_hrv improves ROC-AUC in all conditions (+0.153 to +0.183)
- **Most stable model**: Mix 2_hrv shows lowest std across all LOSOCV results (0.023)
- **Performance drop**: 25-35% accuracy decrease from subject-specific to LOSOCV (smallest gap among models)
- **Best generalizer**: MLP shows smallest performance degradation, suggesting better generalization capability

---

## Comparative Analysis

### ⚠️ Subject-Specific Performance Ranking (Single Split - Unreliable)
1. **RandomForest** - Most consistent high performance (but from single split)
2. **SVM** - Strong performance, especially with interview data (but from single split)
3. **MLP** - Variable performance, highly dependent on feature set (but from single split)

**⚠️ WARNING**: These rankings are based on single 80/20 splits and should NOT be considered reliable estimates of model performance. The high accuracies are likely inflated.

### LOSOCV Performance Ranking (Generalization - Reliable)
1. **MLP** - Best generalization, smallest performance drop (64.1% best accuracy)
2. **RandomForest** - Good with mix data, high variance otherwise (61.0% best accuracy)
3. **SVM** - Struggles with cross-subject generalization (67.6% best accuracy but with ROC-AUC issues)

### Best Condition per Model
- **SVM**: Interview-only (2_hrv) - 96.4% subject-specific ⚠️, 67.6% LOSOCV ✓
- **RandomForest**: Interview-only (3_hrv) - 98.0% subject-specific ⚠️, 75.3% ROC-AUC LOSOCV ✓
- **MLP**: Interview-only (3_hrv) - 88.4% subject-specific ⚠️, 63.4% ROC-AUC LOSOCV ✓

**Note**: Subject-specific values marked with ⚠️ are from single splits and unreliable. LOSOCV values marked with ✓ are the only reliable performance estimates.

### Feature Set Impact
- **2_hrv**: More stable, lower variance across all models
- **3_hrv**: Higher peak performance but increased variance, especially in LOSOCV
- **Recommendation**: Use 3_hrv for subject-specific tasks, 2_hrv for generalization tasks

### Condition Impact
- **Interview-only**: Best for subject-specific training (highest accuracy/ROC-AUC)
- **Mix**: Best for LOSOCV generalization (most stable results)
- **Card-only**: Moderate performance, highest variance in LOSOCV

### Overfitting Analysis
All models show significant overfitting (comparing unreliable subject-specific to LOSOCV):
- **SVM**: 26-50% accuracy drop (subject-specific unreliable)
- **RandomForest**: 35-45% accuracy drop (subject-specific unreliable)
- **MLP**: 25-35% accuracy drop (best, subject-specific unreliable)

**⚠️ CRITICAL CAVEAT**: The "overfitting" interpretation is problematic because:
1. Subject-specific results are from single splits (not cross-validated)
2. The comparison is between unreliable (single split) and reliable (LOSOCV) estimates
3. The true overfitting magnitude cannot be determined without proper cross-validation within subjects
4. LOSOCV results (50-67% accuracy) are the only trustworthy performance estimates

### Key Findings

1. **Interview data is most predictive**: Interview-only condition consistently achieves highest LOSOCV performance across all models (67.6% SVM, 75.3% ROC-AUC RF, 63.4% ROC-AUC MLP)

2. **⚠️ Subject-specific results are unreliable**: All 90-98% accuracies are from single 80/20 splits and likely inflated. Do NOT use these for model selection.

3. **MLP generalizes best**: Most stable LOSOCV results with lowest variance (std 0.023-0.231)

4. **RandomForest shows highest LOSOCV ROC-AUC**: 0.753 in interview-only 3_hrv, indicating good discriminative ability across subjects

5. **SVM has reliability issues**: High LOSOCV accuracy (67.6%) but very low ROC-AUC (0.419) in interview-only, suggesting prediction bias

6. **Feature complexity trade-off**: 3_hrv improves LOSOCV ROC-AUC but increases variance; 2_hrv more stable for accuracy

7. **Mix condition provides stable generalization**: Most consistent LOSOCV results across models, though not always the highest

8. **True performance range**: Based on LOSOCV only, realistic accuracy is 50-67% and ROC-AUC is 0.45-0.75

---

## Recommendations

### ⚠️ IMPORTANT: Ignore Subject-Specific Results
**Do NOT use subject-specific results for any conclusions.** They are based on single train/test splits and are unreliable. Only LOSOCV results should be used for model evaluation and selection.

### For Cross-Subject Generalization (Only Reliable Results)
- **Primary**: MLP with 2_hrv features on mix data (64.1% accuracy, std 0.023 - most stable)
- **Alternative 1**: RandomForest with 3_hrv on interview-only (75.3% ROC-AUC, best discrimination)
- **Alternative 2**: SVM with 2_hrv on interview-only (67.6% accuracy, but check ROC-AUC)
- **Expected Performance**: 50-67% accuracy, 0.45-0.75 ROC-AUC

### Feature Engineering
- Use 3_hrv when ROC-AUC is the primary metric (higher discrimination)
- Use 2_hrv when accuracy stability is prioritized (lower variance)
- Consider ensemble methods to combine both feature sets for robustness

### Model Selection Strategy
- **For deployment across different subjects**: MLP (most stable generalization)
- **If maximum discrimination needed**: RandomForest with 3_hrv on interview-only
- **If interview data available**: Prioritize interview-only condition
- **If only card data available**: Use 2_hrv features with MLP or RandomForest
- **If mix data available**: Use MLP with 2_hrv for most stable results

### Critical Next Steps
1. **Implement proper cross-validation within subjects** (e.g., 5-fold or 10-fold) to get reliable subject-specific estimates
2. **Investigate SVM's accuracy-ROC-AUC divergence** in interview-only LOSOCV
3. **Increase sample size** if possible - current LOSOCV results suggest limited predictive power
4. **Consider stratified LOSOCV** to ensure balanced class distribution in test folds

### Feature Engineering
- Use 3_hrv for maximum performance in controlled settings
- Use 2_hrv for robust, stable predictions across diverse conditions
- Consider ensemble methods to combine both feature sets

### Model Selection Strategy
- **If subject consistency is guaranteed**: RandomForest or SVM
- **If deployment across different subjects**: MLP
- **If interview data available**: Prioritize interview-only condition
- **If only card data available**: Use 3_hrv features with RandomForest