# 🚀 Generative AI for Class Imbalance using CTGAN vs SMOTE

## 📌 Overview
Class imbalance is a critical issue in machine learning, especially in fraud detection where minority cases are rare but highly important.

This project presents a comparative analysis of:
- Baseline Random Forest  
- SMOTE (Synthetic Minority Oversampling Technique)  
- CTGAN (Generative AI-based augmentation)  

The entire workflow is implemented in a single pipeline script for simplicity and reproducibility.

---

## 🎯 Objective
- Improve detection of minority class (fraud transactions)  
- Compare traditional oversampling with generative augmentation  
- Evaluate models using precision-recall based metrics  

---

## 🧠 Key Concepts
- Imbalanced Learning  
- Synthetic Data Augmentation  
- Generative Adversarial Networks (CTGAN)  
- Precision-Recall AUC (PR-AUC)  

---

## ⚙️ Tech Stack
- Python  
- Scikit-learn  
- Pandas, NumPy  
- Imbalanced-learn (SMOTE)  
- SDV (CTGAN)  
- Matplotlib  

---

## 📊 Dataset
- Credit Card Fraud Detection Dataset  
- Highly imbalanced (<1% fraud cases)  
- PCA-transformed numerical features  

Dataset Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

---

## 🧪 Methodology
The complete workflow is implemented in `train_all.py`:

1. Train baseline model on imbalanced data  
2. Apply SMOTE to balance training data  
3. Generate synthetic data using CTGAN  
4. Train Random Forest on all three datasets  
5. Evaluate performance using:
   - Recall  
   - F1-score  
   - PR-AUC  

---

## 📈 Results

| Method   | Recall | F1 Score | PR-AUC |
|----------|--------|----------|--------|
| Baseline | 0.7449 | 0.8391   | 0.8542 |
| SMOTE    | 0.8265 | 0.8265   | 0.8741 |
| CTGAN    | 0.8571 | 0.8442   | 0.8394 |

---

## 🔍 Key Insights
- CTGAN improves minority class detection (higher recall)  
- SMOTE provides better precision-recall balance (higher PR-AUC)  
- Choice of method depends on application priorities (sensitivity vs precision)  

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Place dataset
Put dataset file here:
data/creditcard.csv

### 3. Run the project
python src/train_all.py

---

## 📁 Project Structure

CTGAN-vs-SMOTE-Imbalanced-Learning/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   └── train_all.py
│
├── results/
│   └── pr_comparison.png
│
├── notebooks/
│   └── RPimplementation.ipynb
│
├── paper/
│   └── UPD_RP.pdf
│
├── README.md
└── requirements.txt

---

## 🚀 Future Work
- Apply diffusion-based models for tabular data  
- Improve CTGAN stability through hyperparameter tuning  
- Evaluate performance across multiple datasets  

---

## ✅ Conclusion
This project demonstrates that handling class imbalance requires a balance between sensitivity and precision. While CTGAN enhances detection of rare events, SMOTE remains a strong and efficient baseline for maintaining overall model stability.

Selecting the right approach depends on domain requirements, especially in high-risk applications such as fraud detection.

---
