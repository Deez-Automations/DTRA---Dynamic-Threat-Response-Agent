# -*- coding: utf-8 -*-
"""
Split TEST_300k.csv into smaller test files for dashboard testing.
Creates 5 files of 10k samples each.
"""

import pandas as pd
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CICIDS2017")
SOURCE_FILE = os.path.join(DATA_DIR, "TEST_300k.csv")
OUTPUT_DIR = DATA_DIR

# Number of files and samples per file
NUM_FILES = 5
SAMPLES_PER_FILE = 10000

def main():
    print("=" * 60)
    print("   Splitting TEST_300k.csv into smaller test files")
    print("=" * 60)
    
    # Load source file
    print(f"\n📂 Loading: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE, encoding='latin1', low_memory=False)
    print(f"   Total rows: {len(df):,}")
    
    # Check class distribution
    df.columns = df.columns.str.strip()
    benign_count = (df['Label'] == 'BENIGN').sum()
    attack_count = (df['Label'] != 'BENIGN').sum()
    print(f"   Benign: {benign_count:,} | Attack: {attack_count:,}")
    
    # Create stratified samples to maintain class balance
    print(f"\n🔀 Creating {NUM_FILES} files with {SAMPLES_PER_FILE:,} samples each...")
    
    for i in range(1, NUM_FILES + 1):
        # Sample maintaining approximate class ratios
        sample = df.sample(n=SAMPLES_PER_FILE, random_state=42 + i)
        
        # Count classes in sample
        sample_benign = (sample['Label'] == 'BENIGN').sum()
        sample_attack = (sample['Label'] != 'BENIGN').sum()
        
        # Save to file
        output_file = os.path.join(OUTPUT_DIR, f"test_10k_{i}.csv")
        sample.to_csv(output_file, index=False)
        
        print(f"   ✅ test_10k_{i}.csv - Benign: {sample_benign:,} | Attack: {sample_attack:,}")
    
    print(f"\n🎉 Done! Files saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for i in range(1, NUM_FILES + 1):
        print(f"   - test_10k_{i}.csv")

if __name__ == '__main__':
    main()
