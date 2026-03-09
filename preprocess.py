"""
preprocess.py
Loads raw Iris dataset, cleans/splits it, and saves preprocessed data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import urllib.request

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
RAW_FILE = os.path.join(RAW_DIR, "iris.csv")

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── 1. Download dataset ────────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/"
    "master/iris.csv"
)

print(f"Downloading dataset from {URL} …")
urllib.request.urlretrieve(URL, RAW_FILE)
print(f"Saved raw data to {RAW_FILE}")

# ── 2. Load & inspect ──────────────────────────────────────────────────────
df = pd.read_csv(RAW_FILE)
print(f"Shape: {df.shape}")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ── 3. Clean ───────────────────────────────────────────────────────────────
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ── 4. Encode target ───────────────────────────────────────────────────────
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# ── 5. Split features / target ─────────────────────────────────────────────
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Scale ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. Save ────────────────────────────────────────────────────────────────
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df["Species"] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df["Species"] = y_test.values

train_df.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
test_df.to_csv( os.path.join(PROC_DIR, "test.csv"),  index=False)

print(f"\nTrain samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}")
print("Preprocessing complete!")
