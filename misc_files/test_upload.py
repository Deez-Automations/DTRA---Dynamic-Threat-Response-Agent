# Test if detector works with uploaded file
import pandas as pd
import numpy as np
from detector import ThreatDetector
from config import NUMERIC_FEATURES

# Load model
detector = ThreatDetector()
detector.load()

# Read the test file
df = pd.read_csv(r"d:\GIKI\CS 351\DTRA\CICIDS2017\test_10k_1.csv", encoding='latin1', low_memory=False)
df.columns = df.columns.str.strip()

# EXACT same process as test.py
feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
X_test = df[feature_cols].copy()

# Add missing columns
for col in NUMERIC_FEATURES:
    if col not in X_test.columns:
        X_test[col] = 0

# Reorder
X_test = X_test[[c for c in NUMERIC_FEATURES if c in X_test.columns]]

# Clean - EXACT same as test.py line 62
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Data shape:", X_test.shape)
print("Data types:", X_test.dtypes.unique())
print("Has inf:", np.isinf(X_test.values).any())
print("Has nan:", np.isnan(X_test.values).any())

# Try prediction
try:
    y_pred_prob = detector.predict(X_test.values)
    print("✅ SUCCESS! Predictions:", y_pred_prob[:10])
except Exception as e:
    print("❌ ERROR:", e)
    import traceback
    traceback.print_exc()
