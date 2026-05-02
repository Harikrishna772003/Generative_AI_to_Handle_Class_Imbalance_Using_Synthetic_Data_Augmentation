# Generative AI for Class Imbalance using CTGAN vs SMOTE

## 📌 Overview

Class imbalance is a critical problem in machine learning, especially in high-risk domains like fraud detection. Traditional models often fail to detect minority class instances.

This project compares:

* Baseline Random Forest
* SMOTE (Oversampling)
* CTGAN (Generative AI)

on the **Credit Card Fraud Detection dataset**.

---

## 🎯 Objective

* Improve minority class detection (fraud cases)
* Compare interpolation vs generative augmentation
* Analyze trade-offs using precision-recall metrics

---

## 🧠 Key Concepts

* Class Imbalance Learning
* Synthetic Data Augmentation
* Generative Adversarial Networks (CTGAN)
* Precision-Recall AUC

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* CTGAN (SDV)
* Matplotlib

---

## 📊 Dataset

* Credit Card Fraud Detection Dataset
* Highly imbalanced (<1% fraud cases)
* PCA-transformed features

Dataset Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧪 Methodology

1. Train baseline model on imbalanced data
2. Apply SMOTE for oversampling
3. Generate synthetic data using CTGAN
4. Train Random Forest on all setups
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

* CTGAN improves **minority detection (Recall)**
* SMOTE provides better **precision-recall balance**
* Generative models are powerful but not always superior

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_baseline.py
python train_smote.py
python train_ctgan.py
```

---

## 📊 Visualizations

* Precision-Recall Curve
* Performance comparison graphs

(Add images in /results folder)

---

## 📁 Project Structure

```
(mention folder tree here)
```

---

## 🚀 Future Work

* Try Diffusion Models for tabular data
* Hyperparameter tuning for CTGAN
* Multi-dataset evaluation

---

## 👨‍💻 Authors

* Harikrishna
* Team Members

---

## ⭐ If you found this useful, star the repo!
