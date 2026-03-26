#!/usr/bin/env python3
"""
02_data_split.py
================
Create stratified train/test splits for the synthesis+correction pipeline.

This script splits the cleaned dataset into:
- Seed set (m samples): Used to train the correction model
- Holdout set (n samples): Used as ground truth for validation

Stratified sampling ensures the coffee preference distribution is preserved
in both splits, simulating a well-designed pilot study.

Usage:
    python scripts/02_data_split.py --seed_size 100 --holdout_size 700

Output:
    - data/splits/seed_m{seed_size}.csv
    - data/splits/holdout_n{holdout_size}.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "coffee_data_cleaned.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

RANDOM_SEED = 42  # For reproducibility

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def create_splits(df, seed_size, holdout_size):
    """
    Create stratified splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full cleaned dataset
    seed_size : int
        Number of samples for the seed (training) set
    holdout_size : int
        Number of samples for the holdout (validation) set
    
    Returns:
    --------
    df_seed, df_holdout : tuple of DataFrames
    """
    print(f"Creating splits: seed_size={seed_size}, holdout_size={holdout_size}")
    print(f"  Full dataset size: {len(df)}")
    
    # First, create holdout set (stratified by coffee choice)
    holdout_fraction = min(holdout_size / len(df), 0.3)
    df_rest, df_holdout = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=df['overall_favorite'],
        random_state=RANDOM_SEED
    )
    
    # Trim holdout to exact size if needed
    if len(df_holdout) > holdout_size:
        df_holdout = df_holdout.sample(n=holdout_size, random_state=RANDOM_SEED)
    
    # Create seed set from remaining data (also stratified)
    seed_fraction = min(seed_size / len(df_rest), 0.5)
    df_seed, _ = train_test_split(
        df_rest,
        test_size=1 - seed_fraction,
        stratify=df_rest['overall_favorite'],
        random_state=RANDOM_SEED
    )
    
    # Trim seed to exact size if needed
    if len(df_seed) > seed_size:
        df_seed = df_seed.sample(n=seed_size, random_state=RANDOM_SEED)
    
    print(f"  Seed size: {len(df_seed)}")
    print(f"  Holdout size: {len(df_holdout)}")
    
    return df_seed, df_holdout


def verify_distributions(df_full, df_seed, df_holdout):
    """Verify that distributions are preserved."""
    print("\nDistribution Verification:")
    print("-" * 50)
    
    full_dist = df_full['overall_favorite'].value_counts(normalize=True).sort_index()
    seed_dist = df_seed['overall_favorite'].value_counts(normalize=True).sort_index()
    holdout_dist = df_holdout['overall_favorite'].value_counts(normalize=True).sort_index()
    
    comparison = pd.DataFrame({
        'Full': full_dist * 100,
        'Seed': seed_dist * 100,
        'Holdout': holdout_dist * 100
    }).round(1)
    
    print(comparison.to_string())
    print("-" * 50)
    print("Note: Stratified sampling preserves the distribution.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Create data splits for synthesis pipeline')
    parser.add_argument('--seed_size', type=int, default=100, help='Size of seed set (default: 100)')
    parser.add_argument('--holdout_size', type=int, default=700, help='Size of holdout set (default: 700)')
    args = parser.parse_args()
    
    print("="*60)
    print("Data Splitting for Synthesis Pipeline")
    print("="*60)
    
    # Load cleaned data
    if not CLEANED_DATA_PATH.exists():
        print(f"Error: Cleaned data not found at {CLEANED_DATA_PATH}")
        print("Run 01_data_analysis.py first.")
        return
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded cleaned data: {len(df)} rows")
    
    # Create splits
    df_seed, df_holdout = create_splits(df, args.seed_size, args.holdout_size)
    
    # Verify distributions
    verify_distributions(df, df_seed, df_holdout)
    
    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    seed_path = SPLITS_DIR / f"seed_m{args.seed_size}.csv"
    holdout_path = SPLITS_DIR / f"holdout_n{args.holdout_size}.csv"
    
    df_seed.to_csv(seed_path, index=False)
    df_holdout.to_csv(holdout_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Seed: {seed_path}")
    print(f"  Holdout: {holdout_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
