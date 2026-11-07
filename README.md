# Project 02 — Customer Purchase Prediction

## Overview

This project implements an end-to-end Machine Learning classification pipeline to predict customer conversion within the next 7 days (`purchased_next_7d`). The pipeline explores, benchmarks, and optimizes both **Logistic Regression** and **Random Forest Classifier** models to identify high-intent purchasing patterns from behavioral analytics.

---

## Key Features

- **Automated Feature Engineering:** Extracts engagement and intent metrics from raw behavior via `add_features()`, including:
  - `pages_per_visit`: Content exploration depth per session (`pages_viewed_last_30d / (visits_last_30d + 1)`).
  - `total_session_minutes`: Aggregate engagement time across visits (`visits_last_30d * avg_session_minutes`).
  - `cart_add_rate`: Micro-conversion ratio indicating purchase intent (`cart_adds_last_30d / (visits_last_30d + 1)`).
  - `is_repeat_buyer`: Behavioral segmentation indicator based on order history (`(previous_orders_last_90d > 0).astype(int)`).
- **Leak-Free Preprocessing Pipeline:** Uses Scikit-Learn `Pipeline` combined with `ColumnTransformer` to enforce strict isolation between training and validation folds during cross-validation (`StandardScaler` for 10 numeric features, `OneHotEncoder` for 2 categorical features).
- **Stratified Data Splitting:** Preserves the underlying minority-to-majority class ratio (80/20 split: 560 train / 140 test records) via `stratify=y`.
- **Baseline Benchmarking:** Establishes a foundational performance baseline using `DummyClassifier` (`strategy='most_frequent'`).
- **Rigorous Hyperparameter Optimization:** Uses 5-fold `StratifiedKFold` with `GridSearchCV` (for regularized Logistic Regression solvers `liblinear` and `saga`) and `RandomizedSearchCV` (for Random Forest hyperparameter exploration).
- **Leak-Free Threshold Calibration:** Uses Out-of-Fold (OOF) cross-validation predictions (`cross_val_predict`) on the training set to calibrate the classification decision threshold, maximizing the F1-Score for the positive class without touching the test set.
- **Reusable Model Bundle:** Serializes the complete model bundle (`model`, `threshold`, and `features`) via `joblib` to `models/purchase_bundle.joblib` and automates batch predictions on incoming customer records (`data/new_customers.csv`).

---

## Tech Stack

- **Core Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Evaluation:** Scikit-Learn
- **Visualization:** Matplotlib
- **Model Serialization:** Joblib

---

## Model Selection & Evaluation

### 1. Selection Criteria (Validation-First Approach)

To prevent **Test Set Leakage (Data Snooping)**, model selection was conducted strictly on the training set (`X_train`) using **5-Fold Stratified Cross-Validation** prior to evaluating on the hold-out test set:

- **Primary Metric (F1-Score):** Chosen to balance **Precision** (minimizing wasted marketing spend on false positives) and **Recall** (capturing actual buyers) on an imbalanced target distribution (~40% positive class).
- **Cross-Validation Results (Train Set Only):**
  - **Logistic Regression:** Mean CV F1 = `0.6146` (SD = `0.0371`)
  - **Random Forest Classifier:** Mean CV F1 = `0.6090` (SD = `0.0374`)
- **Selection Decision & Business Rationale:** **Logistic Regression** achieved a higher Cross-Validation F1-score with lower fold-to-fold variance. Following **Occam's Razor (parsimony)** on a dataset of 700 records, the simpler linear model is selected when it outperforms complex ensembles in CV, offering faster inference, lower overfitting risk, and direct feature interpretability.

### 2. Threshold Optimization & Unseen Test Evaluation

- **Out-of-Fold Threshold Tuning:** Decision threshold optimization was performed exclusively using out-of-fold probability predictions on `X_train`, yielding an optimal threshold of **`0.4236`**.
- **Hold-out Test Set Performance:** The unseen test set (`X_test`, 140 records) was evaluated only once using the locked **Logistic Regression** pipeline (`Test ROC-AUC = 0.7246`, compared against Dummy Baseline `Accuracy = 0.5929`, `ROC-AUC = 0.5000`):

| Metric (Purchase Class = 1) | Default Threshold (`0.5000`) | CV-Tuned Threshold (`0.4236`) |
| :--- | :---: | :---: |
| **Precision (Purchase)** | **0.55** | 0.52 |
| **Recall (Purchase)** | 0.61 | **0.79** |
| **F1-Score (Purchase)** | 0.58 | **0.62** |
| **Overall Accuracy** | **0.64** | 0.61 |
| **Test ROC-AUC** | **0.7246** | **0.7246** |

> **Key Observation:** Applying the CV-tuned threshold (`0.4236`) increased **Purchase Recall from 61% to 79%** and boosted the positive-class **F1-Score from 0.58 to 0.62**, capturing significantly more actual buyers on the unseen test set.

---

## Repository Structure

    ├── data/
    │   ├── customer_history.csv      # Historical training dataset (700 records)
    │   └── new_customers.csv         # Unseen inference batch (30 records)
    ├── models/
    │   └── purchase_bundle.joblib    # Serialized dict (pipeline, threshold, features)
    ├── output/
    │   └── predictions.csv           # Exported batch prediction results
    ├── main.ipynb                    # Development and modeling notebook
    ├── requirements.txt              # Project dependencies
    └── README.md                     # Project documentation

---

## How to Run

### 1. Environment Setup

Clone the repository and install the dependencies:

    pip install -r requirements.txt

### 2. Execute Modeling Pipeline

Open `main.ipynb` in your preferred environment (VS Code, JupyterLab) and run all cells sequentially.

The workflow will automatically:

1. Load `data/customer_history.csv` and engineer behavioral features.
2. Run 5-fold Stratified Cross-Validation hyperparameter sweeps and select the best model.
3. Calibrate the optimal classification threshold via out-of-fold predictions on the training set.
4. Evaluate final unbiased metrics on the unseen test set.
5. Save the trained bundle to `models/purchase_bundle.joblib`.
6. Perform inference on `data/new_customers.csv` and export results to `output/predictions.csv`.

---

## Output Specification

The generated file `output/predictions.csv` contains 30 records structured as follows:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `customer_ID` | `string` | Unique customer reference identifier (`CUST-0701` to `CUST-0730`) |
| `purchase_probability` | `float` | Predicted probability of purchase (0.0000 to 1.0000) |
| `predicted_purchase` | `integer` | Binary prediction label using tuned threshold `0.4236` (`0` = Not Purchase, `1` = Purchase) |

---

## Limitations & Future Improvements

- **Sample Size Constraints:** The model is trained on 700 records, which carries risks of variance when scaling to broad, diverse customer bases. Expanding data ingestion to larger sample sizes will improve generalizability.
- **Absence of Temporal & Seasonal Signals:** The current snapshot does not reflect seasonal shopping cycles, monthly paydays, or flash-sale promotional events.
- **Cold-Start Sensitivity:** The model depends on past session interactions (`visits_last_30d`, `cart_adds_last_30d`) and cannot reliably score brand-new visitors with zero activity.
- **Next Steps:** Benchmark gradient boosted decision trees (LightGBM / XGBoost) and integrate SHAP (SHapley Additive exPlanations) for global and local model interpretability.

This project uses synthetic customer data for educational and portfolio purposes. Model performance on real-world customer data has not been validated. The decision threshold is optimized for F1-score and may need adjustment based on business costs and campaign budgets.
