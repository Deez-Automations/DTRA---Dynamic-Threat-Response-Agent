# -*- coding: utf-8 -*-
"""
DTRA Detector Module
Machine Learning threat detection using Deep Neural Networks.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from config import (
    NUMERIC_FEATURES, DETECTOR_MODEL_PATH, 
    IMPUTER_PATH, SCALER_PATH, MODELS_DIR
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class ThreatDetector:
    """
    Deep Neural Network-based threat detector.
    Outputs probability scores (0.0 - 1.0) representing danger level.
    """
    
    def __init__(self):
        self.model = None
        self.imputer = None
        self.scaler = None
        self.input_dim = len(NUMERIC_FEATURES)
        
    def build_model(self, input_dim=None):
        """Build the DNN architecture."""
        if input_dim is None:
            input_dim = self.input_dim
            
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        """Train the detector on labeled data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_data: Optional tuple (X_val, y_val) for validation
        """
        print("🧠 Preprocessing data...")
        
        # Fit preprocessors
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        X_imputed = self.imputer.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Build and train model
        print("🔧 Building neural network...")
        self.model = self.build_model(X_scaled.shape[1])
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_imputed = self.imputer.transform(X_val)
            X_val_scaled = self.scaler.transform(X_val_imputed)
            val_data = (X_val_scaled, y_val)
        
        print("🚀 Training model...")
        history = self.model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            validation_split=0.1 if val_data is None else 0.0,
            callbacks=[early_stop],
            verbose=1
        )
        
        print("✅ Training complete!")
        return history
    
    def save(self):
        """Save model and preprocessors to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        self.model.save(DETECTOR_MODEL_PATH)
        joblib.dump(self.imputer, IMPUTER_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        
        print(f"✅ Saved model to {MODELS_DIR}")
    
    def load(self):
        """Load pre-trained model and preprocessors."""
        if not os.path.exists(DETECTOR_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {DETECTOR_MODEL_PATH}. "
                "Please train the model first using train.py"
            )
        
        self.model = load_model(DETECTOR_MODEL_PATH)
        self.imputer = joblib.load(IMPUTER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        
        print("✅ Loaded pre-trained model")
        return self
    
    def predict(self, X):
        """
        Predict danger score for input data.
        
        Args:
            X: numpy array or DataFrame with network features
            
        Returns:
            numpy array of danger scores (0.0 - 1.0)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert to float64 safely - this can itself throw error if data is bad
        try:
            X = np.asarray(X, dtype=np.float64)
        except (ValueError, OverflowError):
            # If conversion fails, force it and clean
            X = np.asarray(X, dtype=object)
            X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            
        # Handle dimension mismatch
        target_dim = self.input_dim
        if X.shape[-1] < target_dim:
            # Pad with zeros
            padding = np.zeros((X.shape[0], target_dim - X.shape[-1]))
            X = np.hstack([X, padding])
        elif X.shape[-1] > target_dim:
            # Trim
            X = X[:, :target_dim]
        
        # --- ROBUST DATA CLEANING ---
        # Replace infinity with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip to safe range
        X = np.clip(X, -1e10, 1e10)
        
        # Clean again to be absolutely sure
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Skip imputer since we already cleaned - go straight to scaler
        # The imputer.transform() validates for inf and throws error
        X_scaled = self.scaler.transform(X)
        
        # Final cleanup after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = np.clip(X_scaled, -10, 10)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def predict_single(self, row_values):
        """Predict danger score for a single packet/row."""
        row_array = np.array(row_values).reshape(1, -1)
        return float(self.predict(row_array)[0])


# Convenience function for fast inference
@tf.function(reduce_retracing=True)
def fast_inference(model, input_tensor):
    """TensorFlow optimized inference."""
    return model(input_tensor, training=False)
