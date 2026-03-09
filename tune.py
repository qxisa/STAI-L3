
import os
import pickle
import itertools
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

PROC_DIR   = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"))

X_train = train_df.drop("Species", axis=1)
y_train = train_df["Species"]
X_test  = test_df.drop("Species", axis=1)
y_test  = test_df["Species"]

PARAM_GRID = {
    "C":      [0.1, 1.0, 10.0],
    "kernel": ["linear", "rbf", "poly"],
    "gamma":  ["scale", "auto"],
}

mlflow.set_experiment("iris-classification")

best_accuracy  = 0.0
best_params    = {}
best_model     = None

combinations = list(itertools.product(
    PARAM_GRID["C"],
    PARAM_GRID["kernel"],
    PARAM_GRID["gamma"],
))

print(f"Running {len(combinations)} hyperparameter combinations …\n")

with mlflow.start_run(run_name="svm-hyperparameter-tuning") as parent_run:

    mlflow.log_param("model_type", "SVM")
    mlflow.log_param("search_strategy", "grid_search")
    mlflow.log_param("total_combinations", len(combinations))

    for C, kernel, gamma in combinations:

        run_name = f"svm-C={C}-kernel={kernel}-gamma={gamma}"

        with mlflow.start_run(run_name=run_name, nested=True):

            svc = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            svc.fit(X_train, y_train)

            y_pred   = svc.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1       = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_param("C",      C)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("gamma",  gamma)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(svc, "svm_model")

            print(f"  C={C:5}  kernel={kernel:6}  gamma={gamma:5}  "
                  f"acc={accuracy:.4f}  f1={f1:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params   = {"C": C, "kernel": kernel, "gamma": gamma}
                best_model    = svc

    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_param("best_C",      best_params["C"])
    mlflow.log_param("best_kernel", best_params["kernel"])
    mlflow.log_param("best_gamma",  best_params["gamma"])

    best_model_path = os.path.join(MODELS_DIR, "best_svm.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    mlflow.log_artifact(best_model_path)

print(f"\n✓ Best params   : {best_params}")
print(f"✓ Best accuracy : {best_accuracy:.4f}")
print(f"✓ Model saved   : {best_model_path}")
print("Tuning complete!")
