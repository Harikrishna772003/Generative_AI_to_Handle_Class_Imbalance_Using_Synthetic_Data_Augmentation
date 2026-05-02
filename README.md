# 🚀 Generative AI for Class Imbalance using CTGAN vs SMOTE

## 📌 Overview

Class imbalance is a major challenge in machine learning, especially in **fraud detection**, where rare events are critical.

This project presents a **comparative study** between:

* Baseline Random Forest
* SMOTE (Interpolation-based Oversampling)
* CTGAN (Generative AI-based Augmentation)

using the Credit Card Fraud Detection dataset.

---

## 🎯 Objective

* Improve detection of minority class (fraud cases)
* Compare traditional vs generative augmentation
* Analyze trade-offs using precision-recall metrics

---

## 🧠 Key Concepts

* Imbalanced Learning
* Synthetic Data Generation
* Generative Adversarial Networks (CTGAN)
* Precision-Recall AUC (PR-AUC)

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* CTGAN (SDV Library)
* Matplotlib

---

## 📊 Dataset

* Credit Card Fraud Detection Dataset
* Extremely imbalanced (<1% fraud)
* PCA-transformed features

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧪 Methodology

1. Train baseline model on imbalanced data
2. Apply SMOTE for oversampling
3. Generate synthetic data using CTGAN
4. Train Random Forest on all configurations
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

* ✅ CTGAN achieves **highest recall (85.71%)** → better fraud detection
* ✅ SMOTE achieves **highest PR-AUC (0.8741)** → better precision-recall balance
* ⚠️ Generative models improve sensitivity but may increase false positives

---

## 📊 Visualizations

### Precision-Recall Curve

(Add image here: `/results/pr_curve.png`)

### Model Comparison

(Add graph here: `/results/comparison.png`)

---

## 📁 Project Structure

```
CTGAN-vs-SMOTE-Imbalanced-Learning/
│── notebooks/
│── src/
│── results/
│── paper/
│── report/
│── presentation/
│── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_baseline.py
python train_smote.py
python train_ctgan.py
```

---

## 🚀 Future Work

* Apply diffusion models for tabular data
* Optimize CTGAN hyperparameters
* Test on multiple datasets

---


