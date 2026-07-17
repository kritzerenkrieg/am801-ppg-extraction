# Experiment Configuration, Methods, and Parameters

## Overview

This document describes the experimental pipeline for deception detection using PPG (Photoplethysmography) signals, comprising three main stages: HRV feature extraction, traditional ML model training, and deep learning model training.

---

## 1. HRV Feature Extraction (`training/extract_hrv.py`)

### Purpose
Extract ultra-short Heart Rate Variability (HRV) features from raw PPG recordings using 10-second sliding windows.

### Input/Output
- **Input**: Raw PPG CSV files (`*.csv`) with columns: `host_timestamp_ns`, `red`, `infrared`, `ppg0`, `ppg1`, `ppg2`, `label`
- **Output**: `*_hrv.csv` files with 58 columns (2 timestamp + 5 channels × 11 features + 1 label)

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CHANNELS` | `["red", "infrared", "ppg0", "ppg1", "ppg2"]` | PPG signal channels processed |
| `WINDOW_SECONDS` | `10.0` | Sliding window duration (seconds) |
| `STRIDE_SECONDS` | `1.0` | Window stride (seconds) |
| `BANDPASS_LOW_HZ` | `0.5` | Band-pass filter lower cutoff (Hz) |
| `BANDPASS_HIGH_HZ` | `4.0` | Band-pass filter upper cutoff (Hz) |
| `BANDPASS_ORDER` | `4` | Butterworth filter order |
| `MIN_NN_FOR_STATS` | `2` | Minimum NN intervals for statistical features |
| `MIN_NN_FOR_POINCARE` | `3` | Minimum NN intervals for Poincaré features |

### Methods

#### Signal Preprocessing
1. **Band-pass Filtering**: 4th-order zero-phase Butterworth filter (0.5–4.0 Hz) using `scipy.signal.filtfilt`
2. **Normalization**: Z-score normalization per channel: `(x - mean) / std`

#### Peak Detection
- **Algorithm**: NeuroKit2 `ppg_findpeaks` for initial detection
- **Artifact Correction**: Kubios method via `signal_fixpeaks` (best-effort)
- **NN Interval Calculation**: Convert peak sample indices to milliseconds, filter to physiological range (300–2000 ms, ~30–200 bpm)

#### HRV Features Extracted (per channel)
| Feature | Description |
|---------|-------------|
| `Mean_NN` | Mean of NN intervals (ms) |
| `Median_NN` | Median of NN intervals (ms) |
| `Heart_Rate` | Instantaneous heart rate from last NN interval (bpm) |
| `Mean_HR` | Mean heart rate across window (bpm) |
| `SDNN` | Standard deviation of NN intervals (ms) |
| `SDSD` | Standard deviation of successive differences (ms) |
| `RMSSD` | Root mean square of successive differences (ms) |
| `NN_Count` | Number of valid NN intervals |
| `SD1` | Poincaré plot SD1 (ms) |
| `SD2` | Poincaré plot SD2 (ms) |
| `SD1_SD2` | SD1/SD2 ratio |

**Total features**: 5 channels × 11 features = 55 features + metadata columns

---

## 2. Traditional ML Model Training (`training/train_hrv_models.py`)

### Purpose
Train and evaluate classical machine learning models (SVM, Random Forest, MLP) on extracted HRV features for deception detection.

### Input/Output
- **Input**: `*_hrv.csv` files from Stage 1
- **Output**: 
  - `{class_count}_hrv_result/` directory containing:
    - `subject_specific_results.csv`
    - `losocv_results.csv`
    - `summary_results.csv`
    - Confusion matrices (PNG)
    - ROC curves (PNG)
    - Correlation/covariance matrices (CSV + PNG)

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RANDOM_STATE` | `42` | Random seed for reproducibility |
| `TEST_SIZE` | `0.2` | Test set proportion (subject-specific) |
| `CLASS_MODE` | `"three_class"` or `"binary"` | Classification target |
| `BASELINE_LABEL_NAMES` | `{"base", "baseline"}` | Labels treated as baseline |
| `DECEPTIVE_LABEL_NAMES` | `{"deceptive", "lie", "deception"}` | Labels treated as deceptive |
| `TRUTH_LABEL_NAMES` | `{"truth", "truthful"}` | Labels treated as truthful |
| `EXCLUDE_COLS` | `{"window_start_ns", "window_end_ns", "label", "subject", "task", "source_file"}` | Non-feature columns |

### Model Configurations

#### 1. Support Vector Machine (SVM)
```python
SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

#### 2. Random Forest
```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42
)
```

#### 3. Multi-Layer Perceptron (MLP)
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=3000,
    early_stopping=True,
    alpha=1e-4,
    random_state=42
)
```

### Preprocessing Pipeline
All models use a unified pipeline:
1. **Imputation**: Median imputation for missing values (`SimpleImputer`)
2. **Feature Selection**: Variance threshold (0.01) (`VarianceThreshold`)
3. **Scaling**: StandardScaler (zero mean, unit variance)
4. **Classification**: Model-specific estimator

### Methods

#### Evaluation Protocols

**1. Subject-Specific Evaluation**
- 80/20 stratified train/test split per subject
- Ensures each subject's data is evaluated independently
- Aggregates predictions across all subjects for overall metrics

**2. Leave-One-Subject-Out Cross-Validation (LOSOCV)**
- Iteratively holds out one subject as test set
- Trains on all remaining subjects
- Number of folds = number of unique subjects
- More rigorous generalization test than subject-specific

#### Metrics Computed
| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision** | Macro-averaged precision |
| **Recall** | Macro-averaged recall (sensitivity) |
| **Specificity** | Macro-averaged specificity (TN / (TN + FP)) |
| **F1-Score** | Macro-averaged F1 |
| **ROC-AUC** | One-vs-rest macro-averaged ROC-AUC |

#### Additional Analyses
- **Pearson Correlation Matrix**: Feature-feature relationships
- **Covariance Matrix**: Feature variance-covariance structure
- **Confusion Matrices**: Per-model, per-protocol
- **ROC Curves**: One-vs-rest for each class + macro-average

---

## 3. Deep Learning Model Training (`deep/deep_train.py`)

### Purpose
Train deep neural networks directly on raw multi-channel PPG sequences for deception detection.

### Input/Output
- **Input**: Raw PPG CSV files (`*.csv`) - same format as Stage 1, but `*_hrv.csv` files are excluded
- **Output**:
  - `subject_specific_results.csv`
  - `losocv_results.csv`
  - `summary_results.csv`
  - Final models: `cnn.keras`, `cnn_lstm.keras`, `gru.keras`
  - LOSOCV fold models: `{model}_LOSOCV_{subject}_best.keras`
  - Training curves (PNG)
  - Confusion matrices (PNG)
  - ROC curves (PNG)

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEED` | `42` | Global random seed for reproducibility |
| `CHANNELS` | `["red", "infrared", "ppg0", "ppg1", "ppg2"]` | PPG channels (multivariate input) |
| `WINDOW_SECONDS` | `10.0` | Sliding window duration (seconds) |
| `STRIDE_SECONDS` | `1.0` | Window stride (seconds) |
| `BANDPASS_LOW_HZ` | `0.5` | Band-pass filter lower cutoff (Hz) |
| `BANDPASS_HIGH_HZ` | `4.0` | Band-pass filter upper cutoff (Hz) |
| `BANDPASS_ORDER` | `4` | Butterworth filter order |
| `EPOCHS` | `100` | Maximum training epochs |
| `BATCH_SIZE` | `64` | Training batch size |
| `LEARNING_RATE` | `0.001` | Adam optimizer learning rate |
| `CLASS_MODE` | `"2class"` or `"3class"` | Classification target |
| `BASELINE_LABEL_KEYWORD` | `"base"` | Keyword to identify baseline windows |

### Signal Preprocessing
1. **Band-pass Filtering**: 4th-order zero-phase Butterworth (0.5–4.0 Hz) per channel
2. **Normalization**: Z-score normalization per channel
3. **Windowing**: Fixed-length timestamp-based sliding windows with linear interpolation resampling to uniform length

### Model Architectures

#### 1. 1D-CNN
```
Input (window_samples, 5)
  ↓ Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool
  ↓ Conv1D(128, kernel=3) + BatchNorm + ReLU
  ↓ GlobalAveragePooling1D
  ↓ Dense(128, ReLU) + Dropout(0.5)
  ↓ Dense(num_classes, Softmax)
```

#### 2. CNN-LSTM
```
Input (window_samples, 5)
  ↓ Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool
  ↓ LSTM(64)
  ↓ Dropout(0.5)
  ↓ Dense(64, ReLU)
  ↓ Dense(num_classes, Softmax)
```

#### 3. GRU
```
Input (window_samples, 5)
  ↓ GRU(128)
  ↓ Dropout(0.5)
  ↓ Dense(64, ReLU)
  ↓ Dense(num_classes, Softmax)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Cross-Entropy
- **Callbacks**:
  - EarlyStopping: monitor `val_loss`, patience=10, restore best weights
  - ReduceLROnPlateau: monitor `val_loss`, factor=0.5, patience=5

### Methods

#### Evaluation Protocols

**1. Subject-Specific Evaluation**
- 80/20 stratified train/validation/test split per subject
- Validation split: 20% of training data
- Trains separate model for each subject-model combination

**2. Leave-One-Subject-Out Cross-Validation (LOSOCV)**
- Holds out one subject as test set
- Trains on all other subjects
- Validation split: 10% of training data
- Saves best model per fold: `{model}_LOSOCV_{subject}_best.keras`

#### Metrics Computed
| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision** | Macro-averaged precision |
| **Recall** | Macro-averaged recall |
| **Specificity** | Macro-averaged specificity |
| **F1-Score** | Macro-averaged F1 |
| **ROC-AUC** | One-vs-rest ROC-AUC (binary) or multiclass OVR |

#### Reproducibility Measures
- Fixed random seeds: Python, NumPy, TensorFlow
- Deterministic operations enabled (when available)
- Seed = 42 across all experiments

---

## 4. Complete Experimental Pipeline

### Workflow

```
Raw PPG Data (*.csv)
    ↓
[Stage 1] extract_hrv.py
    ↓
HRV Features (*_hrv.csv)
    ↓
[Stage 2] train_hrv_models.py
    ↓
Traditional ML Results (SVM, RF, MLP)
    ↓
Raw PPG Data (*.csv)
    ↓
[Stage 3] deep_train.py
    ↓
Deep Learning Results (CNN, CNN-LSTM, GRU)
```

### Class Mode Configuration

**3-Class Mode** (default for deep learning):
- Classes: `baseline`, `truth`, `deception`
- All windows retained
- Suitable for baseline-aware analysis

**Binary Mode** (default for HRV):
- Classes: `truth`, `deception` only
- Baseline windows excluded
- Focuses on lie detection performance

### Cross-Validation Strategy
Both Stage 2 and Stage 3 use identical evaluation protocols:
1. **Subject-Specific**: 80/20 split per subject (tests within-subject generalization)
2. **LOSOCV**: Leave-one-subject-out (tests cross-subject generalization)

### Reproducibility
- Random seed: 42 (all stages)
- Deterministic TensorFlow operations (when available)
- Fixed train/test splits via `random_state` parameter
- Model checkpoints saved for LOSOCV folds

---

## 5. Dependencies

### Stage 1 (HRV Extraction)
- `pandas`, `numpy`, `scipy`, `neurokit2`

### Stage 2 (Traditional ML)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Stage 3 (Deep Learning)
- `tensorflow`, `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`

---

## 6. Key Hyperparameters Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Windowing** | Window size | 10 seconds |
| | Stride | 1 second |
| **Filtering** | Type | Butterworth band-pass |
| | Order | 4 |
| | Range | 0.5–4.0 Hz |
| **HRV Features** | Count | 11 per channel |
| | Channels | 5 |
| | Total | 55 features |
| **Traditional ML** | Test size | 20% |
| | SVM kernel | RBF |
| | RF estimators | 400 |
| | MLP hidden layers | (128, 64) |
| **Deep Learning** | Epochs | 100 (early stopping) |
| | Batch size | 64 |
| | Learning rate | 0.001 |
| | Dropout | 0.5 |
| | Validation split | 10–20% |

---

## 7. Output Files

### Stage 1 Outputs
- `{filename}_hrv.csv`: Extracted HRV features (58 columns)

### Stage 2 Outputs (`{class_count}_hrv_result/`)
- `subject_specific_results.csv`: Per-subject metrics
- `losocv_results.csv`: Per-fold metrics
- `summary_results.csv`: Aggregated mean/std metrics
- `correlation_matrix.csv`, `correlation_heatmap.png`
- `covariance_matrix.csv`, `covariance_heatmap.png`
- `{SVM,RF,MLP}_{SubjectSpecific,LOSOCV}_confusion.png`
- `{SVM,RF,MLP}_{SubjectSpecific,LOSOCV}_roc.png`

### Stage 3 Outputs
- `subject_specific_results.csv`: Per-subject metrics
- `losocv_results.csv`: Per-fold metrics
- `summary_results.csv`: Aggregated mean/std metrics
- `{CNN,CNNLSTM,GRU}_training_curve.png`
- `{CNN,CNNLSTM,GRU}_{SubjectSpecific,LOSOCV}_confusion.png`
- `{CNN,CNNLSTM,GRU}_ROC.png`
- `cnn.keras`, `cnn_lstm.keras`, `gru.keras`: Final models
- `{model}_LOSOCV_{subject}_best.keras`: LOSOCV fold models