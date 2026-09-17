import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. Synthetic Dataset Simulation (Simulating Kaggle CC Fraud)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 5000
n_features = 10

# Generate legitimate transactions (class 0) and fraudulent transactions (class 1)
# Class imbalance: ~1% Fraud
X_legit = np.random.normal(loc=0, scale=1, size=(4950, n_features))
X_fraud = np.random.normal(loc=1.5, scale=1, size=(50, n_features))

X = np.vstack((X_legit, X_fraud))
y = np.hstack((np.zeros(4950), np.ones(50)))

feature_names = [f'V{i+1}' for i in range(n_features - 1)] + ['Amount']
df = pd.DataFrame(X, columns=feature_names)

# ---------------------------------------------------------
# 2. Train-Test Split & Scaling
# ---------------------------------------------------------
# Stratified split ensures the ~1% fraud ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Handle Imbalance: SMOTE (Applied ONLY to Training Set)
# ---------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Original Training Distribution: {np.bincount(y_train.astype(int))}")
print(f"SMOTE Training Distribution: {np.bincount(y_train_resampled.astype(int))}\n")

# ---------------------------------------------------------
# 4. Baseline Model: Support Vector Machine (SVM)
# ---------------------------------------------------------
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

y_probs_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
svm_auc = roc_auc_score(y_test, y_probs_svm)

# ---------------------------------------------------------
# 5. Advanced Model: XGBoost Classifier
# ---------------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=1,  # Balanced via SMOTE
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_probs_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, y_probs_xgb)

# ---------------------------------------------------------
# 6. Decision Threshold Tuning for XGBoost
# ---------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_xgb)

# Optimize threshold based on F1-Score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_preds_default = (y_probs_xgb >= 0.5).astype(int)
y_preds_tuned = (y_probs_xgb >= best_threshold).astype(int)

print(f"SVM Baseline ROC-AUC: {svm_auc:.4f}")
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
print(f"Optimal Decision Threshold for XGBoost: {best_threshold:.4f}\n")

print("--- Default Threshold (0.5) Classification Report ---")
print(classification_report(y_test, y_preds_default))

print("--- Tuned Threshold Classification Report ---")
print(classification_report(y_test, y_preds_tuned))

# ---------------------------------------------------------
# 7. Interpretability: Feature Importance Plot
# ---------------------------------------------------------
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.title("XGBoost Feature Importance Scores")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()