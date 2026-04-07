import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

DATA_FILE = "modified_air_quality.csv"
MODEL_FILE = "aqi_model.pkl"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# -----------------------------
# 1. Handle Datetime
# -----------------------------
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# Extract useful time features (improves model 🔥)
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month

# Drop rows where datetime is invalid
df = df.dropna(subset=["Datetime"])

# -----------------------------
# 2. Define Features & Target
# -----------------------------
features = ["PM2.5","PM10","NO2","CO","SO2","O3","hour","day","month"]
target = "AQI"

X = df[features]
y = df[target]

# -----------------------------
# 3. Handle Missing Values (IMPORTANT)
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Also clean target
y = y.fillna(y.mean())

# -----------------------------
# 4. Handle Anomalies (Outliers)
# -----------------------------
# Clip extreme values (keeps data realistic)
X = np.clip(X, np.percentile(X, 1), np.percentile(X, 99))

# -----------------------------
# 5. Train Model
# -----------------------------
print("Training model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# 6. Save Model + Imputer
# -----------------------------
print("Saving model...")

with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        "model": model,
        "imputer": imputer
    }, f)

print("✅ Model saved as aqi_model.pkl")