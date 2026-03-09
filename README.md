# 🌸 Iris Classification ML Pipeline

A three-stage DVC + MLflow pipeline for Iris species classification,
tracked on DagsHub.

**Dataset:** [Iris CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/iris.csv) — 150 samples, 4 features, 3 classes.

---

## 📁 Project Structure

```
ml-pipeline/
├── preprocess.py       # Stage 1 – Download, clean, split & scale data
├── train.py            # Stage 2 – Train Random Forest, log to MLflow
├── tune.py             # Stage 3 – SVM grid-search with nested MLflow runs
├── dvc.yaml            # DVC pipeline definition
├── requirements.txt    # Python dependencies
├── .gitignore
└── .dvcignore
```

---

## 🚀 Full Setup Guide (Linux)

### Step 1 — Clone / initialise repository

```bash
# If starting fresh:
mkdir ml-pipeline && cd ml-pipeline
git init
git checkout -b main     # or: git branch -M main

# Copy all project files into this directory, then:
git add .
git commit -m "Initial project structure"
```

### Step 2 — Create virtual environment & install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Create a GitHub repository

1. Go to https://github.com/new
2. Create a **public** repository called `ml-pipeline` (no README/licence/gitignore)
3. Push your local repo:

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/ml-pipeline.git
git push -u origin main
```

### Step 4 — Create a DagsHub account & repository

1. Sign up at https://dagshub.com (free — you can "Continue with GitHub")
2. Click **New Repository → Connect a repository**
3. Select your GitHub `ml-pipeline` repo and connect it
4. DagsHub will show you the repository page; note your remote URL, which looks like:
   `https://dagshub.com/<DAGSHUB_USERNAME>/ml-pipeline.git`

### Step 5 — Initialise DVC and add DagsHub as remote

```bash
# Inside your ml-pipeline directory with .venv active:
dvc init
git add .dvc .dvcignore
git commit -m "Initialise DVC"

# Add DagsHub as the DVC remote storage
# (replace <DAGSHUB_USERNAME> and <DAGSHUB_TOKEN> with your details)
dvc remote add origin https://dagshub.com/<DAGSHUB_USERNAME>/ml-pipeline.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DAGSHUB_USERNAME>
dvc remote modify origin --local password <DAGSHUB_TOKEN>

# Make it the default remote
dvc remote default origin
git add .dvc/config
git commit -m "Configure DagsHub DVC remote"
```

> **Get your DagsHub token:** Profile → Settings → Access Tokens → Generate new token

### Step 6 — Configure MLflow to log to DagsHub

DagsHub provides a hosted MLflow tracking server for every repo.
Set these environment variables **before** running any Python script:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<DAGSHUB_USERNAME>/ml-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=<DAGSHUB_USERNAME>
export MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
```

> **Tip:** Add these three lines to `.venv/bin/activate` or a `.env` file
> (sourced via `set -a && source .env && set +a`) so you don't have to
> re-export them every session.

### Step 7 — Run the pipeline

```bash
dvc repro
```

DVC will execute the three stages in order:

| Stage | Script | What it does |
|-------|--------|-------------|
| `preprocess` | `preprocess.py` | Downloads Iris CSV, scales features, saves `train.csv` & `test.csv` |
| `train` | `train.py` | Trains Random Forest; logs params + metrics to MLflow |
| `tune` | `tune.py` | Grid-searches SVM with **nested MLflow runs** (C × kernel × gamma) |

You should see MLflow runs appear live on:
`https://dagshub.com/<DAGSHUB_USERNAME>/ml-pipeline/experiments`

### Step 8 — Push data & pipeline state to DagsHub

```bash
# Push DVC-tracked data/model files
dvc push

# Commit DVC lock file & push to GitHub (DagsHub mirrors it automatically)
git add dvc.lock
git commit -m "Run pipeline and push artifacts"
git push origin main
```

---

## 📊 Viewing Results on DagsHub

* **MLflow experiments:** `https://dagshub.com/<DAGSHUB_USERNAME>/ml-pipeline/experiments`
* **Model comparison:** click **Experiments → iris-classification** → select runs → **Compare**
* **Data / model artifacts:** DVC tab in your DagsHub repo page

---

## 🧠 Pipeline Details

### preprocess.py
- Downloads the public Iris CSV dataset
- Drops nulls and duplicates
- Label-encodes the target (`Species`)
- Stratified 80/20 train-test split
- StandardScaler normalisation
- Saves `data/processed/train.csv` and `data/processed/test.csv`

### train.py
- Model: **Random Forest** (n_estimators=100)
- MLflow run name: `random-forest-baseline`
- Logs: `n_estimators`, `max_depth`, `accuracy`, `f1_score`
- Saves: `models/random_forest.pkl`

### tune.py
- Model: **Support Vector Machine (SVC)**
- Hyperparameters tuned: `C` × `kernel` × `gamma` = 18 combinations
- Uses **MLflow nested runs** (one child run per combination)
- Parent run logs best params and best accuracy
- Saves: `models/best_svm.pkl`

---

## 📋 Submission Checklist

- [ ] Public GitHub repository URL: `https://github.com/<USERNAME>/ml-pipeline`
- [ ] Screenshot of model comparison on DagsHub
- [ ] Public MLflow experiment URL: `https://dagshub.com/<USERNAME>/ml-pipeline/experiments`
