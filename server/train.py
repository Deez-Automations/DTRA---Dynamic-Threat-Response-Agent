# -*- coding: utf-8 -*-
"""
DTRA Training Script
Train the Threat Detector (DNN) on the pre-split 700k training dataset.
Uses all rows for training - no internal split.
The 300k test set is used via the dashboard for evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NUMERIC_FEATURES, DATA_DIR, MODELS_DIR
from detector import ThreatDetector
from decider import QLearningAgent


def load_training_data():
    """Load the 700k training CSV (pre-split)."""
    
    print("\n" + "="*60)
    print("   DTRA - Training Pipeline")
    print("   Using Pre-Split 700k Training Dataset")
    print("="*60 + "\n")
    
    # Look for pre-split training CSV first
    train_csv = os.path.join(DATA_DIR, 'TRAIN_700k.csv')
    combined_csv = os.path.join(DATA_DIR, 'COMBINED_CICIDS2017.csv')
    
    if os.path.exists(train_csv):
        print(f"📂 Loading training set: {train_csv}")
        df = pd.read_csv(train_csv, encoding='latin1', low_memory=False)
        print(f"   ✅ Using pre-split training data: {len(df):,} rows")
    elif os.path.exists(combined_csv):
        print(f"⚠️  Pre-split train CSV not found: {train_csv}")
        print(f"📂 Falling back to combined dataset: {combined_csv}")
        print(f"   Run 'python split_data.py' first for proper 70/30 split!")
        df = pd.read_csv(combined_csv, encoding='latin1', low_memory=False)
        
        # Take 70% for training
        train_size = int(len(df) * 0.7)
        df = df.head(train_size)
        print(f"   Using first {len(df):,} rows for training")
    else:
        print(f"❌ No training data found!")
        print(f"   Expected: {train_csv}")
        print(f"   Or: {combined_csv}")
        print(f"\n   Please run: python split_data.py")
        sys.exit(1)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Select only needed columns
    columns_to_use = [c for c in NUMERIC_FEATURES if c in df.columns] + ['Label']
    missing_cols = [c for c in NUMERIC_FEATURES if c not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing {len(missing_cols)} feature columns, will use zeros")
    
    df = df[[c for c in columns_to_use if c in df.columns]]
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create binary labels
    df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    df = df.drop('Label', axis=1)
    
    print(f"\n✅ Training data loaded: {len(df):,} samples")
    print(f"   Benign: {(df['label'] == 0).sum():,}")
    print(f"   Attack: {(df['label'] == 1).sum():,}")
    
    return df


def train_detector(df):
    """Train the Deep Neural Network threat detector on ALL training data."""
    
    print("\n" + "-"*60)
    print("   Phase 1: Training Threat Detector (DNN)")
    print("   Training on 100% of data (700k rows)")
    print("-"*60 + "\n")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols]
    y = df['label']
    
    # Add missing columns with zeros
    for col in NUMERIC_FEATURES:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder columns to match config
    X = X[[c for c in NUMERIC_FEATURES if c in X.columns]]
    
    # NO SPLIT - use all data for training
    # Test data will be uploaded separately via dashboard
    print(f"📊 Training on: {len(X):,} samples (100% of training set)")
    print(f"   Note: Use TEST_300k.csv via dashboard for evaluation\n")
    
    # Use 10% of training data for validation during training
    val_split = 0.1
    val_size = int(len(X) * val_split)
    
    X_train = X.iloc[:-val_size]
    y_train = y.iloc[:-val_size]
    X_val = X.iloc[-val_size:]
    y_val = y.iloc[-val_size:]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,} (for early stopping)\n")
    
    # Create and train detector
    detector = ThreatDetector()
    history = detector.train(
        X_train, y_train, 
        epochs=30, 
        batch_size=64,
        validation_data=(X_val.values, y_val.values)
    )
    
    # Evaluate on validation set
    print("\n📈 Evaluating on validation set...")
    y_pred = (detector.predict(X_val.values) > 0.5).astype(int)
    
    print("\n" + "="*40)
    print("VALIDATION REPORT (10% of training data)")
    print("="*40)
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Attack']))
    
    print("\nCONFUSION MATRIX")
    print("-"*40)
    cm = confusion_matrix(y_val, y_pred)
    print(f"                Predicted")
    print(f"              Benign  Attack")
    print(f"Actual Benign  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"Actual Attack  {cm[1][0]:>6}  {cm[1][1]:>6}")
    
    # Save model
    detector.save()
    
    return detector


def train_rl_agent():
    """Train the Q-Learning decision agent."""
    
    print("\n" + "-"*60)
    print("   Phase 2: Training RL Decision Agent")
    print("-"*60 + "\n")
    
    agent = QLearningAgent()
    agent.train(num_episodes=10000, verbose=True)
    agent.save()
    
    return agent


def main():
    """Main training pipeline."""
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load pre-split training data (700k)
    df = load_training_data()
    
    # Train detector
    detector = train_detector(df)
    
    # Train RL agent
    agent = train_rl_agent()
    
    print("\n" + "="*60)
    print("   ✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print("\n📋 Next Steps:")
    print("  1. Start the dashboard: python api.py")
    print("  2. Open http://localhost:5000 in your browser")
    print("  3. Upload TEST_300k.csv to evaluate on unseen data!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
