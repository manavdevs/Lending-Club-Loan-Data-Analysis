# Lending Club Loan Default Prediction

## 📌 Problem Statement
This project predicts loan default risk using historical Lending Club data.
The focus is on building an end-to-end machine learning pipeline using
Jupyter notebooks, from data exploration to model evaluation.

Due to class imbalance, Recall and ROC–AUC are emphasized over accuracy.

---

## 🎯 Objective
To identify borrowers likely to default on loans while minimizing false negatives
(missed defaulters).

---

## 📂 Dataset
**Lending Club Loan Data (2007–2015)**

### Target Variable
`not.fully.paid`
- `0` → Loan fully paid
- `1` → Loan defaulted / not fully paid

The target variable is already binary and is **not one-hot encoded**.

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## 📁 Repository Structure
```
lending-club-loan-default-prediction/
│
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Scaled & encoded data
│
├── notebooks/
│ ├── 01_data_overview.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_modeling.ipynb
│ ├── 05_evaluation.ipynb
│
├── models/
│ └── loan_default_model.h5
│
│
├── requirements.txt
└── README.md
```

---


---

## 🔄 Workflow Summary

### 01 – Data Overview
- Dataset inspection
- Target distribution analysis
- Missing value and duplicate checks

### 02 – Exploratory Data Analysis
- Default vs non-default comparisons
- Analysis of credit score, interest rate, DTI, and loan purpose
- Correlation analysis

### 03 – Feature Engineering
- One-hot encoding
- Feature scaling
- Stratified train-test split

### 04 – Modeling
- Deep learning model using Keras
- Class-weighted training
- Early stopping

### 05 – Evaluation
- Confusion matrix
- Recall calculation
- ROC curve and AUC analysis

---

## 📊 Key Results
- Recall (default class): ~62%
- ROC–AUC: ~0.69
- Model prioritizes default detection over overall accuracy

---
