## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the full experiment pipeline with:

```bash
python heart_project_harmonized.py
```

The script will run the main UCI experiment, the Kaggle supplementary experiment, true external validation, class-weight analysis, top-k feature analysis, and threshold analysis.


# Heart Disease Prediction Using Standard Supervised Learning Models

This repository contains the code and experiment artifacts for the EE559 course project, **Heart Disease Prediction Using Standard Supervised Learning Models**.

The project treats heart disease prediction as a supervised binary classification task. It uses the UCI Heart Disease dataset as the main dataset and a related Kaggle heart disease dataset as a supplementary dataset for additional testing and external validation.

## Project Overview

The goal of this project is to build a reproducible machine learning pipeline that includes:

- data loading and cleaning
- target conversion to binary labels
- missing-value handling
- feature scaling
- UCI/Kaggle feature-name alignment
- selected categorical encoding harmonization
- hyperparameter tuning with cross-validation
- model evaluation with multiple metrics
- feature analysis
- threshold analysis
- true external validation from UCI to Kaggle

The project compares three standard supervised learning models:

- Logistic Regression
- Decision Tree
- Random Forest

## Repository Structure

```text
project/
├── heart_project_harmonized.py
├── heart_kaggle.csv
├── requirements.txt
├── README.md
├── uci_model_summary.csv
├── kaggle_supplementary_model_summary.csv
├── true_external_validation.csv
├── class_weight_comparison.csv
├── topk_feature_comparison.csv
├── threshold_analysis.csv
├── uci_class_distribution.png
├── uci_roc_curve.png
├── uci_confusion_matrices.png
├── uci_logistic_top_features.png
└── other generated CSV/PNG experiment outputs
```

The main experiment script is:

```text
heart_project_harmonized.py
```

This script contains the full experiment workflow, including data loading, preprocessing, model tuning, model evaluation, external validation, and output generation.

## Dataset Information

### UCI Heart Disease Dataset

The UCI Heart Disease dataset is fetched using `ucimlrepo`.

The original UCI target contains multiple disease-severity labels. In this project, it is converted into a binary label:

- `0`: no heart disease
- `1`: heart disease present, converted from any original target value greater than 0

### Kaggle Heart Disease Dataset

The Kaggle dataset is loaded locally from:

```text
heart_kaggle.csv
```

The Kaggle dataset is used for:

- supplementary train/test split evaluation
- true external validation after feature-name alignment and selected categorical encoding harmonization

Because the UCI and Kaggle datasets are not fully identical, the code includes harmonization steps before cross-dataset testing.


## Expected Outputs

After running the script, the project generates CSV result tables and PNG figures such as:

```text
uci_model_summary.csv
kaggle_supplementary_model_summary.csv
true_external_validation.csv
class_weight_comparison.csv
topk_feature_comparison.csv
threshold_analysis.csv
uci_class_distribution.png
uci_roc_curve.png
uci_confusion_matrices.png
uci_logistic_top_features.png
```

These files are used as experimental evidence in the final report and presentation slides.

## Evaluation Metrics

The project reports multiple metrics instead of relying only on accuracy:

- training accuracy
- test accuracy
- F1-score
- precision
- recall
- ROC-AUC
- false positives
- false negatives
- confusion matrices
- ROC curves

This is important because heart disease prediction is a medical-style classification task where false negatives and false positives have different practical meanings.

## Reproducibility Notes

The code uses fixed random seeds where applicable to make the experiment results more reproducible.

The UCI dataset is fetched online through `ucimlrepo`, so internet access may be required when running the script for the first time.

The Kaggle dataset must be present locally as:

```text
heart_kaggle.csv
```

If this file is missing, the Kaggle supplementary experiment and true external validation cannot be reproduced.

## Main Findings

The main UCI experiment shows that Logistic Regression and Random Forest perform best among the tested models. Logistic Regression achieves the strongest ROC-AUC on the UCI test split, while Random Forest achieves high recall and F1-score but has a larger train-test gap.

The true external validation experiment shows that models trained on UCI do not transfer well to the Kaggle dataset. This suggests that internal test performance alone is not sufficient evidence of robust generalization, especially when the data source changes.


