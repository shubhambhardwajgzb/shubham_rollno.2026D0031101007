import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Synthetic Dataset Creation (For demonstration)
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 90, size=n_samples),
    'num_prior_visits': np.random.poisson(lam=2, size=n_samples),
    'blood_pressure_systolic': np.random.normal(loc=125, scale=15, size=n_samples),
    'primary_diagnosis': np.random.choice(['Cardiology', 'Endocrinology', 'Pulmonology', 'General'], size=n_samples),
    'readmitted_30d': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
})

X = data.drop(columns=['readmitted_30d'])
y = data['readmitted_30d']

# ---------------------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. Preprocessing & Model Pipeline
# ---------------------------------------------------------
numeric_features = ['age', 'num_prior_visits', 'blood_pressure_systolic']
categorical_features = ['primary_diagnosis']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Logistic Regression with L2 Regularization (Ridge Penalty)
# Note: 'penalty=l2' and solver='lbfgs' is the default in scikit-learn
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42))
])

# ---------------------------------------------------------
# 4. Model Training & Prediction
# ---------------------------------------------------------
model_pipeline.fit(X_train, y_train)

# Probabilities for positive class (readmitted = 1) needed for ROC-AUC
y_probs = model_pipeline.predict_proba(X_test)[:, 1]
y_preds = model_pipeline.predict(X_test)

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_preds))

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.title('ROC Curve - Hospital Readmission Prediction')
plt.legend()
plt.grid(True)
plt.show()