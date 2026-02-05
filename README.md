# Lending Club Loan Default Prediction

## 📌 Problem Statement
Financial institutions must accurately assess the risk of loan default before
approving loans. This project uses historical Lending Club data to predict
whether a borrower is likely to default based on loan and borrower attributes.

The dataset reflects real-world lending behavior and is naturally imbalanced,
with far fewer default cases than successful repayments.

---

## 🎯 Objective
Build a deep learning model to predict loan default risk while prioritizing:
- Recall (Sensitivity) – to correctly identify defaulters
- ROC-AUC – to measure class separation performance

Accuracy is not used as the primary metric due to class imbalance.

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
│
├── models/
│ └── loan_default_model.h5
│
├── reports/
│
├── requirements.txt
└── README.md
```

---

## 🔄 Project Workflow

### 1️⃣ Data Overview
- Loaded and inspected the dataset
- Verified no missing values or duplicate records
- Analyzed target distribution
- Identified strong class imbalance (~84% paid vs ~16% default)

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Compared defaulted vs non-defaulted loans
- Analyzed relationships between default and:
  - Loan purpose
  - Interest rate
  - FICO credit score
  - Debt-to-income ratio (DTI)
  - Revolving credit utilization
- Identified risk patterns and feature correlations

---

### 3️⃣ Feature Engineering
- Separated features (`X`) and target (`y`)
- One-hot encoded categorical feature (`purpose`)
- Performed stratified train-test split
- Scaled numerical features using `StandardScaler`
- Saved processed datasets for modeling

---

### 4️⃣ Modeling
- Built a deep learning binary classification model using Keras
- Architecture:
  - Dense layers with ReLU activation
  - Dropout layers to prevent overfitting
  - Sigmoid output layer for probability prediction
- Addressed class imbalance using **class weights**
- Applied early stopping based on validation loss
- Saved the trained model for evaluation

---

## 📊 Key Insights So Far
- Lower FICO scores significantly increase default risk
- Higher interest rates correlate with higher defaults
- High debt-to-income ratios indicate financial stress
- Certain loan purposes carry higher risk
- Class imbalance requires recall-focused evaluation

---

## 🔜 Next Step
- Model evaluation
- Confusion matrix
- Recall (Sensitivity) calculation
- ROC curve and AUC score
- Threshold analysis

---

## ⚠️ Notes
- Accuracy alone is misleading for imbalanced datasets
- Feature engineering and modeling are kept separate for clarity
- Evaluation is performed in a dedicated notebook
