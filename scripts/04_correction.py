#!/usr/bin/env python3
"""
04_correction.py
================
Apply confusion matrix correction to synthetic predictions.

This script:
1. Loads the seed data with ground truth + LLM predictions
2. Learns a confusion matrix P(true | llm)
3. Applies the correction to the synthetic distribution
4. Computes validation metrics (TVD, KL divergence)

Usage:
    python scripts/04_correction.py \
        --seed data/splits/seed_m100.csv \
        --synthetic results/synthetic/synthetic_predictions.csv

Output:
    - results/correction/correction_matrix.csv
    - results/correction/corrected_distribution.csv
    - results/correction/metrics.json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "correction"

COFFEE_CHOICES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]

# -----------------------------------------------------------------------------
# Correction Functions
# -----------------------------------------------------------------------------

def learn_correction_matrix(df_seed):
    """
    Learn the confusion matrix P(true | llm) from seed data.
    
    Parameters:
    -----------
    df_seed : pd.DataFrame
        Must have columns: ground_truth, llm_choice
    
    Returns:
    --------
    correction_matrix : np.ndarray
        Shape (4, 4) where entry [i, j] = P(true=i | llm=j)
    """
    # Initialize counts
    n_classes = len(COFFEE_CHOICES)
    counts = np.zeros((n_classes, n_classes))
    
    # Count co-occurrences
    for _, row in df_seed.iterrows():
        true_idx = COFFEE_CHOICES.index(row['ground_truth'])
        if pd.notna(row['llm_choice']) and row['llm_choice'] in COFFEE_CHOICES:
            llm_idx = COFFEE_CHOICES.index(row['llm_choice'])
            counts[true_idx, llm_idx] += 1
    
    # Normalize columns to get P(true | llm)
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    correction_matrix = counts / col_sums
    
    return correction_matrix


def apply_correction(synthetic_dist, correction_matrix):
    """
    Apply the correction matrix to the synthetic distribution.
    
    P(true) = sum_z P(true | llm=z) * P(llm=z)
    
    Parameters:
    -----------
    synthetic_dist : np.ndarray
        Shape (4,) - the raw LLM prediction distribution
    correction_matrix : np.ndarray
        Shape (4, 4) - P(true | llm)
    
    Returns:
    --------
    corrected_dist : np.ndarray
        Shape (4,) - the corrected distribution
    """
    corrected_dist = correction_matrix @ synthetic_dist
    
    # Normalize to ensure it sums to 1
    corrected_dist = corrected_dist / corrected_dist.sum()
    
    return corrected_dist


def compute_distribution(df, column):
    """Compute the distribution of a categorical column."""
    counts = df[column].value_counts()
    dist = np.array([counts.get(c, 0) for c in COFFEE_CHOICES], dtype=float)
    return dist / dist.sum()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def total_variation_distance(p, q):
    """Compute Total Variation Distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))


def kl_divergence(p, q, epsilon=1e-10):
    """Compute KL divergence D(p || q)."""
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Apply correction to synthetic predictions')
    parser.add_argument('--seed', type=str, required=True, help='Path to seed CSV with predictions')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic predictions CSV')
    parser.add_argument('--holdout', type=str, default=None, help='Path to holdout CSV for ground truth')
    args = parser.parse_args()
    
    print("="*60)
    print("Confusion Matrix Correction")
    print("="*60)
    
    # Load data
    df_seed = pd.read_csv(args.seed)
    df_synthetic = pd.read_csv(args.synthetic)
    
    print(f"Seed data: {len(df_seed)} rows")
    print(f"Synthetic data: {len(df_synthetic)} rows")
    
    # Check if seed has LLM predictions (need to generate them if not)
    if 'llm_choice' not in df_seed.columns:
        print("\nWarning: Seed data doesn't have llm_choice column.")
        print("You need to run LLM synthesis on seed data first.")
        print("For now, using synthetic data to estimate correction matrix.")
        df_for_correction = df_synthetic[df_synthetic['llm_choice'].notna()].copy()
    else:
        df_for_correction = df_seed[df_seed['llm_choice'].notna()].copy()
    
    # Learn correction matrix
    print("\nLearning correction matrix...")
    correction_matrix = learn_correction_matrix(df_for_correction)
    
    print("\nCorrection Matrix P(true | llm):")
    cm_df = pd.DataFrame(correction_matrix, index=COFFEE_CHOICES, columns=COFFEE_CHOICES)
    print(cm_df.round(3).to_string())
    
    # Compute distributions
    synthetic_dist = compute_distribution(df_synthetic, 'llm_choice')
    ground_truth_dist = compute_distribution(df_synthetic, 'ground_truth')
    
    print(f"\nSynthetic (uncorrected) distribution: {dict(zip(COFFEE_CHOICES, synthetic_dist.round(3)))}")
    print(f"Ground truth distribution: {dict(zip(COFFEE_CHOICES, ground_truth_dist.round(3)))}")
    
    # Apply correction
    corrected_dist = apply_correction(synthetic_dist, correction_matrix)
    print(f"Corrected distribution: {dict(zip(COFFEE_CHOICES, corrected_dist.round(3)))}")
    
    # Compute metrics
    tvd_uncorrected = total_variation_distance(synthetic_dist, ground_truth_dist)
    tvd_corrected = total_variation_distance(corrected_dist, ground_truth_dist)
    kl_uncorrected = kl_divergence(ground_truth_dist, synthetic_dist)
    kl_corrected = kl_divergence(ground_truth_dist, corrected_dist)
    
    improvement_tvd = (tvd_uncorrected - tvd_corrected) / tvd_uncorrected * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"TVD (uncorrected): {tvd_uncorrected:.4f}")
    print(f"TVD (corrected):   {tvd_corrected:.4f}")
    print(f"Improvement:       {improvement_tvd:.1f}%")
    print(f"\nKL Divergence (uncorrected): {kl_uncorrected:.4f}")
    print(f"KL Divergence (corrected):   {kl_corrected:.4f}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save correction matrix
    cm_df.to_csv(OUTPUT_DIR / "correction_matrix.csv")
    
    # Save distributions
    dist_df = pd.DataFrame({
        'coffee': COFFEE_CHOICES,
        'ground_truth': ground_truth_dist,
        'synthetic_uncorrected': synthetic_dist,
        'synthetic_corrected': corrected_dist
    })
    dist_df.to_csv(OUTPUT_DIR / "distributions.csv", index=False)
    
    # Save metrics
    metrics = {
        'tvd_uncorrected': float(tvd_uncorrected),
        'tvd_corrected': float(tvd_corrected),
        'improvement_tvd_pct': float(improvement_tvd),
        'kl_uncorrected': float(kl_uncorrected),
        'kl_corrected': float(kl_corrected),
        'seed_size': len(df_for_correction),
        'synthetic_size': len(df_synthetic)
    }
    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
