# 🚀 Generative AI for Class Imbalance using CTGAN vs SMOTE

## 📌 Overview

Class imbalance is a critical issue in machine learning, especially in fraud detection where minority cases are rare but important.

This project presents a comparative analysis of:

* Baseline Random Forest
* SMOTE (Oversampling)
* CTGAN (Generative AI)

All models are implemented in a **single unified pipeline**.

---

## 🎯 Objective

* Improve detection of minority class (fraud transactions)
* Compare traditional oversampling vs generative augmentation
* Evaluate models using precision-recall metrics

---

## 🧠 Key Concepts

* Imbalanced Learning
* Synthetic Data Augmentation
* Generative Adversarial Networks (CTGAN)
* Precision-Recall AUC

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Imbalanced-learn (SMOTE)
* SDV (CTGAN)
* Matplotlib

---

## 📊 Dataset

* Credit Card Fraud Detection Dataset
* Highly imbalanced (<1% fraud cases)
* PCA-transformed numerical features

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧪 Methodology

The complete workflow is implemented in a single script:

1. Train baseline model on imbalanced data
2. Apply SMOTE to balance training data
3. Generate synthetic samples using CTGAN
4. Train Random Forest on all datasets
5. Evaluate using:

   * Recall
   * F1-score
   * PR-AUC

---

## 📈 Results

| Method   | Recall | F1 Score | PR-AUC |
| -------- | ------ | -------- | ------ |
| Baseline | 0.7449 | 0.8391   | 0.8542 |
| SMOTE    | 0.8265 | 0.8265   | 0.8741 |
| CTGAN    | 0.8571 | 0.8442   | 0.8394 |

---

## 🔍 Key Insights

* CTGAN improves minority class detection (higher recall)
* SMOTE provides better precision-recall stability (higher PR-AUC)
* Choice of method depends on application priorities (sensitivity vs precision)

---

## ▶️ How to Run

### 1. Install dependencies

```bash id="p1n0gq"
pip install -r requirements.txt
```

### 2. Run the complete pipeline

```bash id="0s6c2k"
python src/train_all.py
```

---

## 📁 Project Structure

```id="y3q4mw"
CTGAN-vs-SMOTE-Imbalanced-Learning/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   └── train_all.py
│
├── notebooks/
│   └── implementation.ipynb
│
├── paper/
│   └── research_paper.pdf
│
├── report/
│   └── project_report.pdf
│
├── presentation/
│   └── ppt.pptx
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Work

* Explore diffusion-based models for tabular data
* Improve CTGAN stability with hyperparameter tuning
* Evaluate performance across multiple real-world datasets

---

## ✅ Conclusion

This project demonstrates that handling class imbalance is not a one-size-fits-all problem. While generative models like CTGAN improve sensitivity toward rare events, traditional techniques like SMOTE still offer strong and reliable performance.

A careful balance between recall and precision is essential when deploying machine learning models in high-risk domains such as fraud detection.

---
