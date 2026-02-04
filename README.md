# Lending Club Loan Default Prediction

## 📌 Problem Statement
Predict whether a loan will default using historical Lending Club data.
The dataset is highly imbalanced, making recall and ROC-AUC critical metrics.

## 🎯 Objective
Build a deep learning model to classify loan default risk and minimize false negatives.

## 📂 Dataset
Lending Club loan data (2007–2015)

Target variable:
- `not.fully.paid`  
  - 0 → Fully paid
  - 1 → Defaulted

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

## 🔄 Project Workflow
1. Data loading & overview
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Handling class imbalance
5. Deep learning model building
6. Model training
7. Evaluation using Recall & ROC-AUC

## 📊 Key Evaluation Metrics
- Recall (Sensitivity)
- ROC-AUC

## 📁 Repository Structure
```
lending-club-loan-default-prediction/
│
├── data/
│   ├── raw/                
│   ├── processed/          
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py          
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── models/
│   ├── model.h5            
│
├── reports/
│   ├── figures/            
│   ├── results.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Results
- Achieved high recall on default class
- ROC-AUC demonstrates strong class separation

## 🔮 Future Improvements
- Hyperparameter tuning
- Try XGBoost for comparison
- Model explainability (SHAP)
