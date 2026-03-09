"""
train.py
Loads train.csv, trains a Random Forest classifier, logs to MLflow,
and saves the trained model.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

# ── Paths ──────────────────────────────────────────────────────────────────
PROC_DIR   = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
N_ESTIMATORS = 100
MAX_DEPTH     = None
RANDOM_STATE  = 42

# ── Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

X_train = train_df.drop("Species", axis=1)
y_train = train_df["Species"]
X_test  = test_df.drop("Species", axis=1)
y_test  = test_df["Species"]

# ── MLflow experiment ──────────────────────────────────────────────────────
mlflow.set_experiment("iris-classification")

with mlflow.start_run(run_name="random-forest-baseline"):

    # ── Train ──────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── Log to MLflow ──────────────────────────────────────────────────────
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth",    MAX_DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("model_type",   "RandomForest")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(clf, "random_forest_model")

    # ── Save model locally ─────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    mlflow.log_artifact(model_path)
    print(f"\nModel saved to {model_path}")
    print("Training complete!")
