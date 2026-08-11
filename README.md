# ML Classification - Breast Cancer Wisconsin

A complete Machine Learning classification project built around the
**Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository.

The project covers the full classification workflow, from exploratory data analysis and preprocessing
to the implementation of a **Gaussian Naive Bayes classifier from scratch**, model validation,
and comparison against established implementations from **scikit-learn** and **WEKA**.

The goal is not only to obtain good predictive performance, but also to make the complete
Machine Learning pipeline explicit, testable, reproducible, and easy to inspect.

---

## Project Overview

This repository covers the complete lifecycle of a binary classification experiment:

- exploratory data analysis;
- data quality checks;
- feature correlation analysis;
- train/test splitting;
- feature filtering;
- custom Gaussian Naive Bayes implementation;
- automated unit testing;
- stratified 5-fold cross-validation;
- model selection;
- final evaluation on a held-out test set;
- comparison with scikit-learn;
- comparison with WEKA.

The custom implementation is intentionally kept separate from the framework-based implementations,
so that the underlying behavior of Gaussian Naive Bayes can be inspected directly.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**  
UCI Machine Learning Repository

The dataset contains numerical measurements computed from digitized images of breast mass cell nuclei.

### Target classes

- `B` — Benign
- `M` — Malignant

### Dataset size

- 569 total observations
- 30 numerical predictive features
- 1 target variable: `diagnosis`
- 1 identifier column removed during preprocessing

The exploratory analysis found:

- no missing values;
- no duplicated observations.

### Class distribution

- Benign: 357 observations — approximately 62.7%
- Malignant: 212 observations — approximately 37.3%

---

## Project Structure

```text
ML_Classification_BreastCancer/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── full_features/
│       └── correlation_filtered/
│
├── results/
│   ├── custom/
│   ├── eda/
│   ├── sklearn/
│   └── weka/
│
├── scripts/
│   ├── analyze_features.py
│   ├── evaluate_custom_model.py
│   ├── explore_dataset.py
│   ├── prepare_dataset.py
│   ├── run_custom_model.py
│   └── run_sklearn_model.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   └── models/
│
└── tests/
```

---

## Exploratory Data Analysis

The initial analysis examines:

- dataset dimensions;
- column types;
- class distribution;
- missing values;
- duplicated observations;
- numerical feature ranges;
- descriptive statistics;
- relationships between features;
- differences between benign and malignant observations.

Several features show very strong Pearson correlations.

Examples include:

- `radius_mean` and `perimeter_mean`;
- `radius_mean` and `area_mean`;
- `radius_worst` and `perimeter_worst`;
- `radius_worst` and `area_worst`;
- `radius_se` and `perimeter_se`.

This motivated the evaluation of an alternative feature configuration in which highly
correlated features are removed.

EDA artifacts are stored under:

```text
results/eda/
```

---

## Preprocessing

The `id` column is removed because it has no predictive meaning.

The remaining data is divided into:

- **398 training observations**
- **171 test observations**

using:

- a 70/30 train/test split;
- `random_state=42`;
- stratification by `diagnosis`.

The final test set is kept separate from model-selection decisions.

Two feature configurations are generated.

### Full configuration

The complete set of:

```text
30 features
```

### Correlation-filtered configuration

Features with an absolute Pearson correlation greater than or equal to:

```text
0.95
```

are considered strongly correlated.

The filtering process reduces the feature set from:

```text
30 → 23 features
```

The features removed during preprocessing are:

- `perimeter_mean`
- `area_mean`
- `perimeter_se`
- `area_se`
- `radius_worst`
- `perimeter_worst`
- `area_worst`

The correlation filtering rule is learned from training data and then applied to unseen data.

This avoids using information from the final test set during preprocessing decisions.

### Feature scaling

No feature scaling is applied.

This is a deliberate choice because Gaussian Naive Bayes does not depend on Euclidean distances
or gradient-based optimization.

Instead, each numerical feature is modeled through a class-specific Gaussian distribution
defined by its mean and variance.

---

## Custom Gaussian Naive Bayes

The core classifier is implemented from scratch in:

```text
src/models/gaussian_naive_bayes.py
```

The implementation does not rely on scikit-learn for the classification algorithm itself.

### Training

During `fit()`, the model learns:

- the available classes;
- the prior probability of each class;
- the mean of every feature for each class;
- the variance of every feature for each class.

For this dataset, the training set contains:

```text
250 Benign observations
148 Malignant observations
```

which produces class priors of approximately:

```text
P(B) = 0.6281
P(M) = 0.3719
```

### Prediction

For each unseen observation, the classifier calculates how compatible every feature value is
with the Gaussian distribution learned for each class.

Conceptually:

```text
new observation
      ↓
evaluate all features for B
      ↓
score B

new observation
      ↓
evaluate all features for M
      ↓
score M
```

The class with the highest score is selected.

The calculation is performed in logarithmic space:

```text
log(class prior)
+
sum of per-feature Gaussian log-likelihoods
```

Using logarithms avoids numerical underflow caused by multiplying many very small probabilities.

### Variance smoothing

A small variance-smoothing value is included to avoid division-by-zero and numerical instability
when a feature has zero or extremely small variance.

The implementation uses:

```text
var_smoothing = 1e-9
```

---

## Unit Testing

The custom classifier is covered by automated tests using `pytest`.

The tests verify:

- learned classes;
- class priors;
- class-specific feature means;
- positive feature variances;
- predictions on clearly separable synthetic data;
- prediction of multiple observations;
- variance smoothing for constant features;
- protection against calling `predict()` before `fit()`;
- validation of incompatible input sizes.

Run the test suite with:

```bash
pytest -v
```

Current result:

```text
8 passed
```

These tests verify the behavior of the custom implementation on small controlled examples.

They are separate from model evaluation: unit tests verify whether the implementation behaves
as expected, while the final evaluation measures how well the model generalizes to unseen
Breast Cancer observations.

---

## Model Selection

The project evaluates two alternative feature configurations:

```text
FULL
30 features
```

and:

```text
FILTERED
23 features
```

The 171-observation final test set is not used to choose between the two.

Model selection is performed exclusively on the 398 training observations using
**stratified 5-fold cross-validation**.

### Cross-validation process

The training set is divided into five parts.

For each iteration:

1. four parts are used to train the model;
2. the remaining part is used as validation data;
3. the model performs predictions on the validation observations;
4. the predicted classes are compared with the real classes;
5. accuracy, precision, recall, and F1-score are calculated.

This process is repeated five times so that every part of the training set is used once
for validation.

For the filtered configuration, correlation-based feature selection is recomputed using only
the training portion of each fold and then applied to its validation portion.

The primary model-selection metric is the **F1-score for the malignant class (`M`)**.

### Cross-validation results

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Full — 30 features | 0.9396 | 0.9319 | 0.9048 | **0.9171** |
| Correlation filtered — 23 features | 0.9170 | 0.8927 | 0.8846 | 0.8875 |

The full 30-feature configuration achieved the highest average F1-score.

The final model therefore uses:

```text
30 features
```

This result also shows that removing strongly correlated features did not improve performance
for Gaussian Naive Bayes in this specific experiment.

---

## Final Custom Model Evaluation

After model selection, the custom Gaussian Naive Bayes classifier is retrained using:

```text
398 training observations
30 features
```

It is then evaluated once on the previously untouched:

```text
171 test observations
```

### Final results

| Metric | Score |
|---|---:|
| Accuracy | **0.9357** |
| Precision — Malignant | **0.9818** |
| Recall — Malignant | **0.8438** |
| F1-score — Malignant | **0.9076** |

### Confusion Matrix

```text
                 Predicted

                 B      M

Actual B        106      1
Actual M         10     54
```

Therefore:

```text
True Positives  = 54
True Negatives  = 106
False Positives = 1
False Negatives = 10
```

The four counters cover all test observations:

```text
54 + 106 + 1 + 10 = 171
```

### Interpretation

The custom model correctly classifies:

```text
160 / 171 observations
```

corresponding to an accuracy of approximately:

```text
93.57%
```

The precision for malignant predictions is very high:

```text
98.18%
```

meaning that when the model predicts `M`, it is almost always correct.

The recall is lower:

```text
84.38%
```

because 10 of the 64 truly malignant observations are classified as benign.

The final F1-score for the malignant class is:

```text
90.76%
```

The main limitation of the model is therefore the number of false negatives.

---

## Comparison with scikit-learn

The same experiment is repeated using:

sklearn.naive_bayes.GaussianNB

The comparison uses exactly the same:

- 398 training observations;
- 171 test observations;
- 30 predictive features;
- target variable;
- variance-smoothing value.

The scikit-learn model follows the same high-level interface:

```text
fit()
↓
predict()
```

and learns class priors, class-specific means, and class-specific variances.

### Results

| Metric | Custom | scikit-learn |
|---|---:|---:|
| Accuracy | 0.9357 | 0.9357 |
| Precision — M | 0.9818 | 0.9818 |
| Recall — M | 0.8438 | 0.8438 |
| F1 — M | 0.9076 | 0.9076 |
| TP | 54 | 54 |
| TN | 106 | 106 |
| FP | 1 | 1 |
| FN | 10 | 10 |

The two implementations produce identical final results.

This provides strong experimental evidence that the custom implementation behaves consistently
with the reference `GaussianNB` implementation provided by scikit-learn.

The comparison complements the unit tests:

```text
Unit tests
→ verify controlled implementation behavior

scikit-learn comparison
→ verifies consistency on the real dataset
```

---

## Comparison with WEKA

The selected 30-feature configuration is also evaluated using:

```text
weka.classifiers.bayes.NaiveBayes
```

with **WEKA 3.8.7**.

The same original train/test split is preserved:

```text
398 training observations
171 test observations
```

The processed CSV files are converted to WEKA's ARFF format and supplied directly to
WEKA Explorer.

### WEKA results

WEKA correctly classifies:

```text
160 / 171 observations
```

corresponding to the same overall accuracy:

```text
93.57%
```

For the malignant class:

```text
Precision = 0.949
Recall    = 0.875
F1        = 0.911
```

Its confusion matrix is:

```text
                 Predicted

                 M      B

Actual M         56      8
Actual B          3    104
```

Therefore:

```text
TP = 56
TN = 104
FP = 3
FN = 8
```

---

## Final Comparison

| Metric | Custom | scikit-learn | WEKA |
|---|---:|---:|---:|
| Accuracy | **0.9357** | **0.9357** | **0.9357** |
| Precision — M | 0.9818 | 0.9818 | 0.949 |
| Recall — M | 0.8438 | 0.8438 | 0.875 |
| F1 — M | 0.9076 | 0.9076 | 0.911 |
| TP | 54 | 54 | 56 |
| TN | 106 | 106 | 104 |
| FP | 1 | 1 | 3 |
| FN | 10 | 10 | 8 |

All three implementations achieve the same overall classification accuracy:

```text
93.57%
```

The custom implementation and scikit-learn produce exactly the same confusion matrix
and evaluation metrics.

WEKA produces the same number of total errors:

```text
11 incorrect predictions
```

but distributes those errors differently.

Compared with the custom and scikit-learn models, WEKA:

- identifies two additional malignant observations;
- reduces false negatives from 10 to 8;
- increases false positives from 1 to 3;
- achieves higher recall;
- achieves lower precision;
- produces a very similar F1-score.

These differences show why accuracy alone is not sufficient to fully describe the behavior
of a classifier.

Models with identical accuracy can still make different types of errors.

---

## Key Findings

### Feature removal is not automatically beneficial

Several dataset features are strongly correlated.

However, the cross-validation experiment showed:

```text
FULL F1     = 0.9171
FILTERED F1 = 0.8875
```

Therefore, the complete 30-feature representation performed better for this specific
Gaussian Naive Bayes experiment.

### Accuracy alone does not describe model behavior

All three final implementations achieve:

```text
93.57% accuracy
```

but WEKA produces fewer false negatives and more false positives.

Looking at the confusion matrix, precision, and recall provides a more complete interpretation
of model behavior.

### Precision and recall answer different questions

For the malignant class:

```text
Precision
→ when the model predicts malignant, how often is it correct?

Recall
→ among all truly malignant observations, how many are detected?
```

The custom and scikit-learn models prioritize very high precision, while WEKA achieves
slightly higher recall.

### Independent test data is essential

The final 171-observation test set is not used during feature-configuration selection.

The experiment follows the separation:

```text
training data
↓
model selection through cross-validation
↓
final configuration
↓
held-out test data
↓
final evaluation
```

This prevents the final test set from influencing model-selection decisions.

### The custom implementation is consistent with a reference library

The custom Gaussian Naive Bayes implementation produces exactly the same final results as
scikit-learn's `GaussianNB`.

Combined with the unit tests, this provides strong evidence of implementation correctness.

---

## Technologies

- Python 3
- NumPy
- pandas
- scikit-learn
- matplotlib
- pytest
- WEKA 3.8.7
- Git

---

## Running the Project

### Activate the virtual environment

Example:

```bash
source .venv/bin/activate
```

### Explore the dataset

```bash
python -m scripts.explore_dataset
```

### Run feature analysis

```bash
python -m scripts.analyze_features
```

### Prepare train/test datasets

```bash
python -m scripts.prepare_dataset
```

### Run the custom Gaussian Naive Bayes model

```bash
python -m scripts.run_custom_model
```

### Run automated tests

```bash
pytest -v
```

### Run cross-validation and final custom evaluation

```bash
python -m scripts.evaluate_custom_model
```

### Run the scikit-learn reference implementation

```bash
python -m scripts.run_sklearn_model
```

---

## WEKA Experiment

The final full-feature datasets used by WEKA are stored as:

```text
data/processed/full_features/train.arff
data/processed/full_features/test.arff
```

The WEKA experiment uses:

```text
Classifier:
weka.classifiers.bayes.NaiveBayes

Training set:
398 observations

Supplied test set:
171 observations

Class:
diagnosis
```

The saved WEKA artifacts are available under:

```text
results/weka/
```

including:

```text
evaluation.txt
naive_bayes.model
```

This preserves both the complete evaluation output and the trained WEKA model.

---

## Results Artifacts

Generated experiment results are stored under:

```text
results/
├── custom/
│   ├── cross_validation_results.csv
│   ├── cross_validation_summary.csv
│   ├── final_metrics.json
│   ├── selection_metadata.json
│   └── test_predictions.csv
│
├── sklearn/
│   ├── final_metrics.json
│   ├── model_metadata.json
│   └── test_predictions.csv
│
└── weka/
    ├── evaluation.txt
    └── naive_bayes.model
```

These artifacts make the experiments inspectable and reproducible without relying only
on terminal or GUI output.

---

## Background

This repository was originally developed as part of a university Machine Learning assignment
and extended with a software-engineering-oriented approach.

The project focuses on both sides of Machine Learning development:

### Machine Learning

- exploratory analysis;
- preprocessing;
- feature analysis;
- custom algorithm implementation;
- model selection;
- cross-validation;
- held-out test evaluation;
- interpretation of classification metrics;
- comparison with reference implementations.

### Software Engineering

- modular project structure;
- separation of responsibilities;
- automated testing;
- reproducible experiments;
- explicit result artifacts;
- version control;
- comparison against external reference implementations.

The repository is therefore intended both as an academic Machine Learning experiment and as a
portfolio project demonstrating the implementation and validation of a classification pipeline
from first principles.

---

## Final Conclusion

A Gaussian Naive Bayes classifier was successfully implemented from scratch and evaluated on
the Breast Cancer Wisconsin Diagnostic dataset.

Cross-validation showed that retaining all 30 predictive features produced better results than
removing highly correlated features.

On the final held-out test set, the custom implementation achieved:

```text
Accuracy  = 93.57%
Precision = 98.18%
Recall    = 84.38%
F1        = 90.76%
```

The implementation produced exactly the same final results as scikit-learn's `GaussianNB`.

WEKA achieved the same overall accuracy while producing a slightly different balance between
false positives and false negatives.

Overall, the experiment demonstrates that a relatively simple probabilistic classifier,
implemented directly from its mathematical foundations, can achieve strong performance while
remaining interpretable, testable, and comparable with established Machine Learning tools.