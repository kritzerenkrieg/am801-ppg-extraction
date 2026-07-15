# Deep Learning Model Performance Insights Analysis (Raw PPG)

## ⚠️ CRITICAL METHODOLOGICAL WARNING

**The "SubjectSpecific" evaluation uses a SINGLE 80/20 train/test split per subject, NOT cross-validation.**

This is a significant limitation that affects the interpretation of all "SubjectSpecific" results:
- Each subject's data is split once (random_state=42) into 80% train / 20% test
- Results are highly dependent on this single random split
- No cross-validation is performed within subjects
- Standard deviations reflect variance **across subjects**, not across different splits
- High accuracies (66-80%) may be inflated due to lucky splits and are not as reliable as they appear

**Only the LOSOCV results provide robust cross-validation estimates.**

---

## Overview

This analysis compares three deep learning architectures (CNN, CNN-LSTM, GRU) trained directly on raw PPG signals for deception detection. The models are evaluated across two classification tasks:
- **2-class**: Truth vs. Deception (baseline excluded)
- **3-class**: Baseline vs. Truth vs. Deception

Each model is evaluated under two scenarios:
- **SubjectSpecific**: Single 80/20 stratified split per subject (NOT cross-validated)
- **LOSOCV**: Leave-One-Subject-Out cross-validation (proper cross-validation)

**Data Processing Pipeline:**
- 5-channel input: red, infrared, ppg0, ppg1, ppg2
- 4th-order Butterworth band-pass filter (0.5-4.0 Hz)
- Z-score normalization per channel
- 10-second windows with 1-second stride
- ~100 Hz median sampling rate → ~1000 samples per window

---

## 1. CNN (1D Convolutional Neural Network)

### Architecture
- 2 Conv1D layers (64 and 128 filters)
- Batch normalization + ReLU + MaxPooling
- Global Average Pooling
- Dense(128) + Dropout(0.5)
- Softmax output

### Performance Summary

#### 2-Class Classification (Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.799 | 0.045 |
| Precision | 0.616 | 0.247 |
| Recall | 0.655 | 0.170 |
| Specificity | 0.655 | 0.170 |
| F1 | 0.622 | 0.200 |
| ROC-AUC | 0.855 | 0.218 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.513 | 0.230 |
| Precision | 0.308 | 0.169 |
| Recall | 0.507 | 0.017 |
| Specificity | 0.507 | 0.017 |
| F1 | 0.350 | 0.124 |
| ROC-AUC | 0.542 | 0.214 |

**Key Insights:**
- **Best deep learning model**: Highest subject-specific accuracy (79.9%) and LOSOCV accuracy (51.3%)
- **High variance in LOSOCV**: Std 0.230 indicates inconsistent performance across subjects
- **ROC-AUC vs Accuracy**: LOSOCV shows reasonable ROC-AUC (0.542) despite low accuracy (0.513), suggesting some discriminative ability
- **Precision issues**: Very low precision in LOSOCV (0.308) indicates many false positives
- **Performance drop**: 28.7% accuracy decrease from subject-specific to LOSOCV

#### 3-Class Classification (Baseline vs. Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.598 | 0.186 |
| Precision | 0.431 | 0.276 |
| Recall | 0.510 | 0.168 |
| Specificity | 0.778 | 0.105 |
| F1 | 0.430 | 0.216 |
| ROC-AUC | 0.839 | 0.089 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.429 | 0.027 |
| Precision | 0.221 | 0.139 |
| Recall | 0.341 | 0.034 |
| Specificity | 0.673 | 0.018 |
| F1 | 0.225 | 0.053 |
| ROC-AUC | 0.618 | 0.147 |

**Key Insights:**
- **Worse than 2-class**: Expected decrease in performance with 3 classes
- **High specificity**: 67.3% specificity in LOSOCV indicates good true negative rate
- **Low precision**: 22.1% precision suggests many false positives across all classes
- **Stable but low**: LOSOCV results show low variance (std 0.027-0.147) but poor performance
- **ROC-AUC reasonable**: 0.618 ROC-AUC suggests some discriminative ability despite low accuracy

---

## 2. CNN-LSTM (Hybrid Architecture)

### Architecture
- Conv1D(64) + BatchNorm + ReLU + MaxPooling
- LSTM(64)
- Dropout(0.5)
- Dense(64) + Softmax output

### Performance Summary

#### 2-Class Classification (Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.698 | 0.121 |
| Precision | 0.467 | 0.184 |
| Recall | 0.538 | 0.105 |
| Specificity | 0.538 | 0.105 |
| F1 | 0.468 | 0.124 |
| ROC-AUC | 0.602 | 0.231 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.483 | 0.192 |
| Precision | 0.516 | 0.080 |
| Recall | 0.506 | 0.024 |
| Specificity | 0.506 | 0.024 |
| F1 | 0.386 | 0.091 |
| ROC-AUC | 0.463 | 0.099 |

**Key Insights:**
- **Worst 2-class performer**: Lowest LOSOCV accuracy (48.3%) among all models
- **High variance**: Std 0.192 in LOSOCV accuracy indicates unstable performance
- **Precision anomaly**: Higher precision (0.516) than accuracy (0.483) in LOSOCV
- **Poor ROC-AUC**: 0.463 in LOSOCV is below random chance (0.5)
- **Performance drop**: 21.5% accuracy decrease from subject-specific to LOSOCV

#### 3-Class Classification (Baseline vs. Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.512 | 0.135 |
| Precision | 0.339 | 0.247 |
| Recall | 0.408 | 0.147 |
| Specificity | 0.714 | 0.089 |
| F1 | 0.314 | 0.181 |
| ROC-AUC | 0.602 | 0.150 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.411 | 0.072 |
| Precision | 0.147 | 0.002 |
| Recall | 0.322 | 0.028 |
| Specificity | 0.662 | 0.011 |
| F1 | 0.201 | 0.008 |
| ROC-AUC | 0.509 | 0.068 |

**Key Insights:**
- **Worst overall performer**: Lowest LOSOCV metrics across all models
- **Extremely low precision**: 0.147 in LOSOCV indicates severe false positive issues
- **Near-random ROC-AUC**: 0.509 is barely above chance level
- **Most stable but worst**: Lowest variance (std 0.002-0.072) but poorest performance
- **Performance drop**: 10.1% accuracy decrease (smallest drop, but from low baseline)

---

## 3. GRU (Gated Recurrent Unit)

### Architecture
- GRU(128)
- Dropout(0.5)
- Dense(64) + Softmax output

### Performance Summary

#### 2-Class Classification (Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.662 | 0.145 |
| Precision | 0.380 | 0.104 |
| Recall | 0.491 | 0.035 |
| Specificity | 0.491 | 0.035 |
| F1 | 0.423 | 0.074 |
| ROC-AUC | 0.513 | 0.094 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.391 | 0.114 |
| Precision | 0.354 | 0.168 |
| Recall | 0.484 | 0.027 |
| Specificity | 0.484 | 0.027 |
| F1 | 0.340 | 0.120 |
| ROC-AUC | 0.471 | 0.110 |

**Key Insights:**
- **Worst LOSOCV accuracy**: 39.1% accuracy is below random chance for binary classification
- **Below-random performance**: Accuracy and ROC-AUC (0.471) both below 0.5
- **High variance**: Std 0.114-0.168 in LOSOCV indicates unreliable predictions
- **Poor generalization**: 27.1% accuracy drop from subject-specific to LOSOCV
- **Not suitable for deployment**: Consistently poor cross-subject performance

#### 3-Class Classification (Baseline vs. Truth vs. Deception)

**Subject-Specific Evaluation**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.458 | 0.098 |
| Precision | 0.298 | 0.101 |
| Recall | 0.358 | 0.070 |
| Specificity | 0.686 | 0.056 |
| F1 | 0.307 | 0.095 |
| ROC-AUC | 0.561 | 0.123 |

**LOSOCV Evaluation (Generalization Test)**
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.439 | 0.003 |
| Precision | 0.146 | 0.001 |
| Recall | 0.332 | 0.002 |
| Specificity | 0.666 | 0.001 |
| F1 | 0.203 | 0.001 |
| ROC-AUC | 0.477 | 0.027 |

**Key Insights:**
- **Surprisingly stable**: Extremely low variance (std 0.001-0.003) in LOSOCV
- **Still poor performance**: Despite stability, accuracy (43.9%) and ROC-AUC (0.477) remain below random
- **Best 3-class LOSOCV**: Slightly better than CNN-LSTM (43.9% vs 41.1%) but still inadequate
- **Near-random ROC-AUC**: 0.477 indicates no discriminative ability
- **Consistently below chance**: All metrics below 0.5 except specificity

---

## Comparative Analysis

### Model Ranking (Based on LOSOCV - Only Reliable Results)

#### 2-Class Classification
1. **CNN** - Best overall (51.3% accuracy, 0.542 ROC-AUC)
2. **CNN-LSTM** - Moderate (48.3% accuracy, 0.463 ROC-AUC)
3. **GRU** - Worst (39.1% accuracy, 0.471 ROC-AUC)

#### 3-Class Classification
1. **GRU** - Most stable (43.9% accuracy, std 0.003)
2. **CNN** - Best ROC-AUC (42.9% accuracy, 0.618 ROC-AUC)
3. **CNN-LSTM** - Worst (41.1% accuracy, 0.509 ROC-AUC)

### Subject-Specific vs LOSOCV Comparison

| Model | 2-Class Subj-Spec | 2-Class LOSOCV | Drop | 3-Class Subj-Spec | 3-Class LOSOCV | Drop |
|-------|-------------------|----------------|------|-------------------|----------------|------|
| CNN | 79.9% | 51.3% | 28.7% | 59.8% | 42.9% | 16.9% |
| CNN-LSTM | 69.8% | 48.3% | 21.5% | 51.2% | 41.1% | 10.1% |
| GRU | 66.2% | 39.1% | 27.1% | 45.8% | 43.9% | -2.0% |

**Key Observations:**
- All models show significant overfitting (except GRU 3-class)
- CNN shows largest performance drop in 2-class
- GRU shows smallest variance in 3-class LOSOCV (but still poor performance)
- Subject-specific results are unreliable due to single split validation

### Architecture Comparison

**CNN Strengths:**
- Best overall LOSOCV performance
- Highest ROC-AUC in both 2-class (0.542) and 3-class (0.618)
- Most stable in 3-class LOSOCV (std 0.026-0.147)

**CNN-LSTM Weaknesses:**
- Worst LOSOCV performance in both tasks
- Below-random ROC-AUC in 2-class (0.463)
- Extremely low precision in 3-class (0.147)

**GRU Weaknesses:**
- Below-random performance in 2-class (39.1% accuracy, 0.471 ROC-AUC)
- Most unstable in 2-class LOSOCV (std 0.110-0.168)
- Poor discriminative ability across all metrics

### Binary vs. Ternary Classification

**2-class is easier than 3-class:**
- All models perform better in 2-class LOSOCV (39-51% vs 41-44%)
- However, even 2-class performance is below or near random chance
- The small gap suggests the models struggle to distinguish truth from deception regardless of class count

**ROC-AUC vs Accuracy:**
- CNN shows better ROC-AUC than accuracy in both tasks, suggesting ranking ability even if classification threshold is poor
- GRU shows near-random ROC-AUC, indicating no learned discriminative patterns
- CNN-LSTM shows inconsistent ROC-AUC (0.463 in 2-class, 0.509 in 3-class)

---

## Key Findings

### 1. **Raw PPG signals have limited deception detection capability**
- Best LOSOCV accuracy: 51.3% (CNN, 2-class) - barely above random
- Best LOSOCV ROC-AUC: 0.618 (CNN, 3-class) - weak discrimination
- All models struggle with cross-subject generalization

### 2. **CNN is the best architecture for raw PPG**
- Highest LOSOCV accuracy and ROC-AUC in both tasks
- Most stable performance across subjects
- Better feature extraction from raw signals than recurrent architectures

### 3. **⚠️ Subject-specific results are unreliable**
- All 66-80% accuracies are from single 80/20 splits and likely inflated
- Do NOT use these for model selection or conclusions
- Only LOSOCV results should be considered

### 4. **Significant overfitting in all models**
- 21-29% accuracy drop from subject-specific to LOSOCV
- Models learn subject-specific patterns that don't generalize
- Raw PPG signals may contain subject-specific artifacts

### 5. **Binary classification is preferred**
- 2-class performs better than 3-class across all models
- Simpler task with clearer decision boundary
- Baseline class may add noise without improving performance

### 6. **High variance in cross-subject performance**
- LOSOCV std ranges from 0.003 to 0.230
- Performance varies dramatically across subjects
- Some subjects may be easier to classify than others

### 7. **Precision is a major issue**
- Very low precision (14-35%) in LOSOCV indicates many false positives
- Models tend to predict the majority class or have biased decision boundaries
- Threshold tuning or class weighting may be needed

### 8. **True performance range**
- Based on LOSOCV only, realistic accuracy is 39-51% (2-class) and 41-44% (3-class)
- ROC-AUC ranges from 0.46 to 0.62
- These are near-random to weak performance levels

---

## Comparison with HRV-Based Models

| Model Type | Best LOSOCV Accuracy | Best LOSOCV ROC-AUC | Conclusion |
|------------|---------------------|---------------------|------------|
| **HRV + MLP** | 64.1% | 0.753 | **Better** |
| **HRV + RandomForest** | 61.0% | 0.753 | **Better** |
| **HRV + SVM** | 67.6% | 0.670 | **Better** |
| **Raw PPG + CNN** | 51.3% | 0.618 | Worse |
| **Raw PPG + CNN-LSTM** | 48.3% | 0.509 | Worse |
| **Raw PPG + GRU** | 39.1% | 0.471 | Worse |

**Key Conclusion:**
**HRV-based features significantly outperform raw PPG signals for deception detection.** The best HRV model (SVM) achieves 67.6% LOSOCV accuracy compared to 51.3% for the best raw PPG model (CNN). This suggests that:
1. HRV features capture more relevant physiological information
2. Raw PPG signals contain too much noise and subject-specific variability
3. Feature engineering (HRV extraction) is crucial for this task

---

## Recommendations

### ⚠️ IMPORTANT: Ignore Subject-Specific Results
**Do NOT use subject-specific results for any conclusions.** They are based on single train/test splits and are unreliable. Only LOSOCV results should be used for model evaluation and selection.

### For Raw PPG-Based Deception Detection

**Primary Recommendation: Use HRV features instead of raw PPG**
- HRV-based models achieve 50-67% accuracy vs 39-51% for raw PPG
- HRV models have better ROC-AUC (0.45-0.75 vs 0.46-0.62)
- Feature extraction provides better generalization

**If raw PPG must be used:**
- **Primary**: CNN with 2-class classification (51.3% accuracy, 0.542 ROC-AUC)
- **Alternative**: CNN with 3-class if baseline information is needed (42.9% accuracy, 0.618 ROC-AUC)
- **Avoid**: GRU architecture (below-random performance)
- **Expected Performance**: 39-51% accuracy, 0.46-0.62 ROC-AUC

### Model Selection Strategy

**For deployment:**
- **Do NOT deploy raw PPG models** - performance is too close to random
- Use HRV-based models instead (MLP or RandomForest)
- If raw PPG is the only option, use CNN with extensive threshold tuning

**For research purposes:**
- Investigate why raw PPG performs poorly (noise, artifacts, individual differences)
- Explore advanced preprocessing (artifact removal, quality assessment)
- Consider multimodal approaches combining raw PPG and HRV features
- Increase sample size - current results suggest limited predictive power

### Critical Next Steps

1. **Prioritize HRV features over raw PPG** for deception detection tasks
2. **Investigate CNN's superior performance** - analyze learned features to understand what patterns it detects
3. **Address precision issues** - implement class weighting, threshold optimization, or cost-sensitive learning
4. **Increase sample size** - current results indicate limited statistical power
5. **Explore subject-specific normalization** - reduce inter-subject variability in raw PPG signals
6. **Consider transfer learning** - pre-train on larger PPG datasets before fine-tuning
7. **Implement proper cross-validation within subjects** to get reliable subject-specific estimates
8. **Investigate data quality** - poor LOSOCV may be due to signal quality issues rather than model limitations

### Feature Engineering Recommendations

1. **Use HRV features** (time-domain, frequency-domain, nonlinear) instead of raw PPG
2. **If raw PPG is required:**
   - Implement artifact detection and removal
   - Try different window lengths (10s may be too short/long)
   - Experiment with different band-pass filter parameters
   - Consider additional channels or sensor fusion
3. **Combine both approaches:**
   - Use raw PPG for CNN feature extraction
   - Combine with HRV features for ensemble models
   - May capture both temporal patterns and physiological metrics

---

## Final Verdict

**Raw PPG signals are NOT suitable for deception detection** based on current results:
- Best model (CNN) achieves only 51.3% accuracy in cross-subject validation
- Performance is at or below random chance for most models
- High variance indicates unreliable predictions
- HRV-based models significantly outperform raw PPG approaches

**Recommendation:** Focus on HRV feature extraction and traditional ML models (MLP, RandomForest, SVM) for deception detection tasks. Raw PPG may be useful for other applications but not for this specific task with current methodology.