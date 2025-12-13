import joblib
import numpy as np

# Load the scaler
scaler = joblib.load('models/dtra_scaler.pkl')

print("Scaler type:", type(scaler))
print("Scaler mean shape:", scaler.mean_.shape)
print("Has inf in mean?", np.isinf(scaler.mean_).any())
print("Has nan in mean?", np.isnan(scaler.mean_).any())
print("Mean values (first 5):", scaler.mean_[:5])

print("\nScaler scale shape:", scaler.scale_.shape)
print("Has inf in scale?", np.isinf(scaler.scale_).any())
print("Has nan in scale?", np.isnan(scaler.scale_).any())
print("Scale values (first 5):", scaler.scale_[:5])

# Check for zero or very small scale values (would cause inf during transform)
print("\nMin scale value:", scaler.scale_.min())
print("Any zero scales?", (scaler.scale_ == 0).any())
print("Any very small scales?", (scaler.scale_ < 1e-10).any())
