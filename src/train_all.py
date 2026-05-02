import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
from sdv.tabular import CTGAN


# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ==============================
# Helper Function
# ==============================
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc


# ==============================
# 1. Baseline Model
# ==============================
baseline_model = RandomForestClassifier(class_weight="balanced", random_state=42)
baseline_model.fit(X_train, y_train)

p_base, r_base, auc_base = evaluate_model(
    "Baseline Model", baseline_model, X_test, y_test
)


# ==============================
# 2. SMOTE Model
# ==============================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_sm, y_train_sm)

p_sm, r_sm, auc_sm = evaluate_model(
    "SMOTE Model", smote_model, X_test, y_test
)


# ==============================
# 3. CTGAN Model
# ==============================
train_data = pd.concat([X_train, y_train], axis=1)

minority_data = train_data[train_data["Class"] == 1]

ctgan = CTGAN(epochs=10)
ctgan.fit(minority_data)

synthetic_data = ctgan.sample(len(minority_data))

augmented_data = pd.concat([train_data, synthetic_data])

X_train_ct = augmented_data.drop("Class", axis=1)
y_train_ct = augmented_data["Class"]

ctgan_model = RandomForestClassifier(random_state=42)
ctgan_model.fit(X_train_ct, y_train_ct)

p_ct, r_ct, auc_ct = evaluate_model(
    "CTGAN Model", ctgan_model, X_test, y_test
)


# ==============================
# Plot Comparison
# ==============================
plt.figure()

plt.plot(r_base, p_base, label=f"Baseline (AUC={auc_base:.4f})")
plt.plot(r_sm, p_sm, label=f"SMOTE (AUC={auc_sm:.4f})")
plt.plot(r_ct, p_ct, label=f"CTGAN (AUC={auc_ct:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()

plt.savefig("results/pr_comparison.png")
plt.show()
