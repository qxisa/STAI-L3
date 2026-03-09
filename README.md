# Iris Classification ML Pipeline

This project builds a three-stage machine learning pipeline to classify Iris flower species. It uses DVC to manage the pipeline stages, MLflow to track experiments, and DagsHub as the remote server. Everything here is run on Linux.

The dataset is the classic Iris CSV with 150 samples, 4 features, and 3 target classes.

## Project Files

- `preprocess.py` downloads the raw data, cleans it, scales the features, and saves train/test splits
- `train.py` trains a Random Forest classifier and logs the results to MLflow
- `tune.py` runs a grid search over SVM hyperparameters using MLflow nested runs
- `dvc.yaml` defines the pipeline order: preprocess, then train, then tune

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Initialise DVC and point it to your DagsHub storage:

```bash
dvc init
dvc remote add origin https://dagshub.com/<YOUR_USERNAME>/ml-pipeline.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <YOUR_USERNAME>
dvc remote modify origin --local password <YOUR_TOKEN>
dvc remote default origin
```

Set your MLflow tracking environment variables so runs are logged to DagsHub:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<YOUR_USERNAME>/ml-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>
export MLFLOW_TRACKING_PASSWORD=<YOUR_TOKEN>
```

## Running the Pipeline

```bash
dvc repro
```

This runs all three stages in order. Once done, push everything:

```bash
dvc push
git add dvc.lock
git commit -m "Run pipeline"
git push origin main
```

## Viewing Results

MLflow experiments and model comparisons are available on your DagsHub repository under the Experiments tab.
