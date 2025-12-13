# -*- coding: utf-8 -*-
"""
DTRA Test Script
Evaluate the trained model on the 300k test dataset.
Runs entirely in terminal - no dashboard needed.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NUMERIC_FEATURES, DATA_DIR, MODELS_DIR
from detector import ThreatDetector
from decider import HybridDecider


def main():
    """Main test pipeline."""
    
    print("\n" + "="*60)
    print("   DTRA - Model Evaluation")
    print("   Testing on 300k Test Dataset")
    print("="*60 + "\n")
    
    # ===== LOAD TEST DATA =====
    test_csv = os.path.join(DATA_DIR, 'TEST_300k.csv')
    
    if not os.path.exists(test_csv):
        print(f"❌ Test file not found: {test_csv}")
        print("   Run split_data.py first!")
        sys.exit(1)
    
    print(f"📂 Loading test data: {test_csv}")
    df = pd.read_csv(test_csv, encoding='latin1', low_memory=False)
    print(f"   Loaded {len(df):,} rows")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Create binary labels
    df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Prepare features
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X_test = df[feature_cols].copy()
    y_test = df['label']
    
    # Add missing columns with zeros
    for col in NUMERIC_FEATURES:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Reorder columns
    X_test = X_test[[c for c in NUMERIC_FEATURES if c in X_test.columns]]
    
    # Clean data
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"\n📊 Test set statistics:")
    print(f"   Total samples: {len(y_test):,}")
    print(f"   Benign: {(y_test == 0).sum():,}")
    print(f"   Attack: {(y_test == 1).sum():,}")
    
    # ===== LOAD MODEL =====
    print("\n" + "-"*60)
    print("   Loading Trained Model")
    print("-"*60 + "\n")
    
    detector = ThreatDetector()
    detector.load()
    print("✅ Model loaded successfully")
    
    # ===== RUN PREDICTIONS =====
    print("\n" + "-"*60)
    print("   Running Predictions on 300k Test Set")
    print("-"*60 + "\n")
    
    print("🔮 Predicting...")
    
    # Get raw predictions
    y_pred_prob = detector.predict(X_test.values)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("✅ Predictions complete!")
    
    # ===== EVALUATION RESULTS =====
    print("\n" + "="*60)
    print("   CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Confusion Matrix
    print("\n" + "="*60)
    print("   CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n                  Predicted")
    print(f"                Benign    Attack")
    print(f"Actual Benign   {cm[0][0]:>7,}   {cm[0][1]:>7,}")
    print(f"Actual Attack   {cm[1][0]:>7,}   {cm[1][1]:>7,}")
    
    # Summary metrics
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*60)
    print("   SUMMARY METRICS")
    print("="*60)
    print(f"\n   Accuracy:        {accuracy*100:.2f}%")
    print(f"   True Positives:  {tp:,} (attacks correctly detected)")
    print(f"   True Negatives:  {tn:,} (benign correctly identified)")
    print(f"   False Positives: {fp:,} (false alarms)")
    print(f"   False Negatives: {fn:,} (missed attacks)")
    print(f"\n   Detection Rate:  {tp/(tp+fn)*100:.2f}% of attacks caught")
    print(f"   False Alarm Rate: {fp/(fp+tn)*100:.2f}%")
    
    # ===== AI DECISION DISTRIBUTION =====
    print("\n" + "-"*60)
    print("   AI Decision Distribution (using HybridDecider)")
    print("-"*60 + "\n")
    
    decider = HybridDecider()
    decider.load()  # Load the trained Q-table!
    
    # Separate analysis for benign and attack traffic
    y_test_array = y_test.values
    
    # Get indices for benign and attack samples
    benign_indices = np.where(y_test_array == 0)[0]
    attack_indices = np.where(y_test_array == 1)[0]
    
    # Sample from each category
    sample_size = 5000
    benign_sample = np.random.choice(benign_indices, min(sample_size, len(benign_indices)), replace=False)
    attack_sample = np.random.choice(attack_indices, min(sample_size, len(attack_indices)), replace=False)
    
    # Analyze benign traffic
    benign_actions = {'Ignore': 0, 'Log': 0, 'Block': 0, 'Isolate': 0}
    for idx in benign_sample:
        danger_score = float(y_pred_prob[idx])
        action = decider.decide(danger_score)
        benign_actions[action] += 1
    
    print("   📗 BENIGN Traffic Response (5k sample):")
    for action, count in benign_actions.items():
        pct = count / len(benign_sample) * 100
        bar = "█" * int(pct / 2)
        print(f"      {action:>8}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Analyze attack traffic
    attack_actions = {'Ignore': 0, 'Log': 0, 'Block': 0, 'Isolate': 0}
    for idx in attack_sample:
        danger_score = float(y_pred_prob[idx])
        action = decider.decide(danger_score)
        attack_actions[action] += 1
    
    print("\n   📕 ATTACK Traffic Response (5k sample):")
    for action, count in attack_actions.items():
        pct = count / len(attack_sample) * 100
        bar = "█" * int(pct / 2)
        print(f"      {action:>8}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Show danger score distribution
    print("\n   📊 Danger Score Statistics:")
    benign_scores = [float(y_pred_prob[i]) for i in benign_sample]
    attack_scores = [float(y_pred_prob[i]) for i in attack_sample]
    
    print(f"      Benign traffic:  avg={np.mean(benign_scores):.4f}, max={np.max(benign_scores):.4f}")
    print(f"      Attack traffic:  avg={np.mean(attack_scores):.4f}, min={np.min(attack_scores):.4f}")
    
    print("\n" + "="*60)
    print("   ✅ EVALUATION COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
