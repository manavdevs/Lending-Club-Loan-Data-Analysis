# Lending Club Loan Default Prediction

## 📌 Problem Statement
Financial institutions must accurately assess the risk of loan default.
This project uses historical Lending Club data to predict whether a loan
will default based on borrower and loan characteristics.

The dataset is naturally imbalanced, reflecting real-world lending behavior.

---

## 🎯 Objective
Build a deep learning model to predict loan default risk while prioritizing
Recall (Sensitivity) and ROC-AUC over accuracy.

---

## 📂 Dataset
Lending Club loan data (2007–2015)

### Target Variable
`not.fully.paid`
- 0 → Loan fully paid
- 1 → Loan defaulted / not fully paid

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
│ ├── raw/
│ ├── processed/
│
├── notebooks/
│ ├── 01_data_overview.ipynb
│ ├── 02_eda.ipynb
│
├── src/
├── models/
├── reports/
├── requirements.txt
└── README.md
```
---

## 🔍 Project Workflow

### 1️⃣ Data Overview
- Loaded and inspected dataset
- Verified no missing values or duplicates
- Analyzed target distribution
- Identified strong class imbalance (~84% vs ~16%)

### 2️⃣ Exploratory Data Analysis (EDA)
- Compared default vs non-default loans
- Analyzed impact of:
  - Loan purpose
  - Interest rate
  - FICO score
  - Debt-to-income ratio
  - Revolving credit utilization
- Identified patterns indicating higher default risk
- Examined feature correlations for future feature selection

---

## 📊 Key Insights
- Lower FICO scores significantly increase default risk
- Higher interest rates correlate with higher defaults
- High debt-to-income ratios indicate financial stress
- Certain loan purposes carry higher default risk
- Class imbalance requires careful metric selection

---

## 🔜 Next Steps
- Feature engineering and encoding
- Handling class imbalance
- Train-test split
- Deep learning model development
- Evaluation using Recall and ROC-AUC

---

## ⚠️ Notes
- The target variable is not one-hot encoded, as it is already binary.
- Accuracy is not used as the primary evaluation metric due to class imbalance.

