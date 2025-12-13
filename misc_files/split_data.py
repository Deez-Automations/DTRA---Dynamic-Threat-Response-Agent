# -*- coding: utf-8 -*-
"""
DTRA Data Splitter
Split combined CSV into train (700k) and test (300k) sets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CICIDS2017')
COMBINED_CSV = os.path.join(DATA_DIR, 'COMBINED_CICIDS2017.csv')
TRAIN_CSV = os.path.join(DATA_DIR, 'TRAIN_700k.csv')
TEST_CSV = os.path.join(DATA_DIR, 'TEST_300k.csv')

# Split ratio: 70% train (700k), 30% test (300k)
TRAIN_RATIO = 0.7
RANDOM_SEED = 42


def split_data():
    """Split the combined CSV into train and test sets."""
    
    print("\n" + "="*60)
    print("   DTRA Data Splitter")
    print("="*60 + "\n")
    
    # Check if combined CSV exists
    if not os.path.exists(COMBINED_CSV):
        print(f"❌ Combined CSV not found: {COMBINED_CSV}")
        print("   Please ensure COMBINED_CICIDS2017.csv exists in CICIDS2017/ folder")
        return False
    
    # Load data
    print(f"📂 Loading: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV, encoding='latin1', low_memory=False)
    
    total_rows = len(df)
    print(f"   Total rows: {total_rows:,}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Calculate split sizes
    train_size = int(total_rows * TRAIN_RATIO)
    test_size = total_rows - train_size
    
    print(f"\n📊 Splitting data:")
    print(f"   Train: {train_size:,} rows ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Test:  {test_size:,} rows ({(1-TRAIN_RATIO)*100:.0f}%)")
    
    # Stratified split to maintain label distribution
    if 'Label' in df.columns:
        print("\n🔄 Using stratified split to preserve label distribution...")
        train_df, test_df = train_test_split(
            df, 
            train_size=TRAIN_RATIO, 
            random_state=RANDOM_SEED,
            stratify=df['Label']
        )
    else:
        print("\n🔄 Using random split...")
        train_df, test_df = train_test_split(
            df, 
            train_size=TRAIN_RATIO, 
            random_state=RANDOM_SEED
        )
    
    # Save train set in chunks to avoid memory issues
    print(f"\n💾 Saving train set: {TRAIN_CSV}")
    chunk_size = 50000
    for i, start in enumerate(range(0, len(train_df), chunk_size)):
        chunk = train_df.iloc[start:start + chunk_size]
        mode = 'w' if start == 0 else 'a'
        header = (start == 0)
        chunk.to_csv(TRAIN_CSV, index=False, mode=mode, header=header)
        if (i + 1) % 5 == 0:
            print(f"   Progress: {min(start + chunk_size, len(train_df)):,}/{len(train_df):,}")
    print(f"   ✅ Saved {len(train_df):,} rows")
    
    # Save test set in chunks
    print(f"\n💾 Saving test set: {TEST_CSV}")
    for i, start in enumerate(range(0, len(test_df), chunk_size)):
        chunk = test_df.iloc[start:start + chunk_size]
        mode = 'w' if start == 0 else 'a'
        header = (start == 0)
        chunk.to_csv(TEST_CSV, index=False, mode=mode, header=header)
        if (i + 1) % 5 == 0:
            print(f"   Progress: {min(start + chunk_size, len(test_df)):,}/{len(test_df):,}")
    print(f"   ✅ Saved {len(test_df):,} rows")
    
    # Print label distribution
    if 'Label' in df.columns:
        print("\n📈 Label Distribution:")
        print("-" * 40)
        
        train_benign = (train_df['Label'] == 'BENIGN').sum()
        train_attack = len(train_df) - train_benign
        test_benign = (test_df['Label'] == 'BENIGN').sum()
        test_attack = len(test_df) - test_benign
        
        print(f"   TRAIN: {train_benign:,} Benign, {train_attack:,} Attack")
        print(f"   TEST:  {test_benign:,} Benign, {test_attack:,} Attack")
    
    print("\n" + "="*60)
    print("   ✅ DATA SPLIT COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  1. {TRAIN_CSV}")
    print(f"  2. {TEST_CSV}")
    print(f"\nNext steps:")
    print(f"  1. Run: python train.py     (uses TRAIN_700k.csv)")
    print(f"  2. Run: python api.py")
    print(f"  3. Upload TEST_300k.csv to dashboard for testing")
    print("="*60 + "\n")
    
    return True


if __name__ == '__main__':
    split_data()
