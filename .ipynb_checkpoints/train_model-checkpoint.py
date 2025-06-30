# train_model.py
"""
Train multiple classifiers on the Pima Indians Diabetes dataset,
select the best by ROC‑AUC, and save:
  • best_model.pkl      (model + scaler)
  • metrics.json        (accuracy & ROC‑AUC for each model)
  • shap_explainer.pkl  (Tree or Linear SHAP explainer, if supported)
Run this only when you want to (re)train.
"""
import json, joblib, warnings, shap, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------
# 1. Load and split the dataset
# ------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")          # same folder
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 2. Build candidate models (with scaling where needed)
# ------------------------------------------------------------------
candidates = {
    "Logistic Regression": Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    ),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

metrics = {}

best_auc = -1
best_name = None
best_model = None

for name, model in candidates.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    metrics[name] = {"accuracy": acc, "roc_auc": auc}
    print(f"{name:20s}  ACC={acc:.3f}  AUC={auc:.3f}")

    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, model

print(f"\nBest model → {best_name}  (AUC={best_auc:.3f})")

# ------------------------------------------------------------------
# 3. Persist best model & metrics
# ------------------------------------------------------------------
joblib.dump(best_model, "best_model.pkl")
with open("metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)
print("Saved   best_model.pkl   &   metrics.json")

# ------------------------------------------------------------------
# 4. SHAP Explainer (optional but nice)
# ------------------------------------------------------------------
try:
    if best_name == "Logistic Regression":
        explainer = shap.LinearExplainer(best_model.named_steps["clf"], X_train, feature_perturbation="interventional")
    else:
        explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, "shap_explainer.pkl")
    print("Saved   shap_explainer.pkl")
except Exception as e:
    print("SHAP explainer not created:", e)
