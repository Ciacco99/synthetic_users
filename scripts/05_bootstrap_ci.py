#!/usr/bin/env python3
"""
05_bootstrap_ci.py
==================
Compute bootstrap confidence intervals for the corrected distribution.

This script resamples the seed data many times, re-computes the correction
matrix each time, and produces confidence intervals for the final estimates.

Usage:
    python scripts/05_bootstrap_ci.py \
        --seed data/splits/seed_m100.csv \
        --synthetic results/synthetic/synthetic_predictions.csv \
        --n_bootstrap 1000

Output:
    - results/bootstrap/bootstrap_results.csv
    - results/bootstrap/confidence_intervals.json
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "bootstrap"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

COFFEE_CHOICES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]
COLORS = ['#4080FF', '#57A9FB', '#37D4CF', '#23C343']

# -----------------------------------------------------------------------------
# Bootstrap Functions
# -----------------------------------------------------------------------------

def learn_correction_matrix(df):
    """Learn P(true | llm) from data."""
    n_classes = len(COFFEE_CHOICES)
    counts = np.zeros((n_classes, n_classes))
    
    for _, row in df.iterrows():
        if row['ground_truth'] in COFFEE_CHOICES and row['llm_choice'] in COFFEE_CHOICES:
            true_idx = COFFEE_CHOICES.index(row['ground_truth'])
            llm_idx = COFFEE_CHOICES.index(row['llm_choice'])
            counts[true_idx, llm_idx] += 1
    
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return counts / col_sums


def apply_correction(synthetic_dist, correction_matrix):
    """Apply correction matrix to distribution."""
    corrected = correction_matrix @ synthetic_dist
    return corrected / corrected.sum()


def compute_distribution(df, column):
    """Compute distribution from dataframe."""
    counts = df[column].value_counts()
    dist = np.array([counts.get(c, 0) for c in COFFEE_CHOICES], dtype=float)
    return dist / dist.sum()


def bootstrap_correction(df_seed, synthetic_dist, n_bootstrap=500, random_state=42):
    """
    Run bootstrap resampling to get confidence intervals.
    
    Parameters:
    -----------
    df_seed : pd.DataFrame
        Seed data with ground_truth and llm_choice columns
    synthetic_dist : np.ndarray
        The synthetic distribution to correct
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns:
    --------
    bootstrap_results : np.ndarray
        Shape (n_bootstrap, 4) - corrected distributions for each iteration
    """
    np.random.seed(random_state)
    n = len(df_seed)
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Resample seed data with replacement
        indices = np.random.choice(n, size=n, replace=True)
        df_resampled = df_seed.iloc[indices]
        
        # Learn correction matrix on resampled data
        correction_matrix = learn_correction_matrix(df_resampled)
        
        # Apply correction
        corrected_dist = apply_correction(synthetic_dist, correction_matrix)
        bootstrap_results.append(corrected_dist)
        
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
    
    return np.array(bootstrap_results)


def compute_confidence_intervals(bootstrap_results, alpha=0.05):
    """Compute confidence intervals from bootstrap results."""
    lower = np.percentile(bootstrap_results, 100 * alpha / 2, axis=0)
    upper = np.percentile(bootstrap_results, 100 * (1 - alpha / 2), axis=0)
    mean = np.mean(bootstrap_results, axis=0)
    std = np.std(bootstrap_results, axis=0)
    
    return {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_confidence_intervals(ground_truth, point_estimate, ci, output_path):
    """Plot the corrected distribution with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(COFFEE_CHOICES))
    width = 0.35
    
    # Ground truth bars
    bars1 = ax.bar(x - width/2, ground_truth * 100, width, label='Ground Truth', 
                   color='#A9AEB8', edgecolor='white', linewidth=1.5)
    
    # Corrected estimate with error bars
    bars2 = ax.bar(x + width/2, point_estimate * 100, width, label='Corrected (with 95% CI)',
                   color='#37D4CF', edgecolor='white', linewidth=1.5)
    
    # Add error bars
    yerr_lower = (point_estimate - ci['lower']) * 100
    yerr_upper = (ci['upper'] - point_estimate) * 100
    ax.errorbar(x + width/2, point_estimate * 100, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_xlabel('Coffee', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Corrected Distribution with 95% Bootstrap Confidence Intervals', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(COFFEE_CHOICES)
    ax.legend()
    ax.set_ylim(0, 50)
    
    # Add CI text annotations
    for i, coffee in enumerate(COFFEE_CHOICES):
        ci_text = f"[{ci['lower'][i]*100:.1f}, {ci['upper'][i]*100:.1f}]"
        ax.annotate(ci_text, (x[i] + width/2, point_estimate[i]*100 + yerr_upper[i] + 2),
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute bootstrap confidence intervals')
    parser.add_argument('--seed', type=str, required=True, help='Path to seed CSV with predictions')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic predictions CSV')
    parser.add_argument('--n_bootstrap', type=int, default=500, help='Number of bootstrap iterations')
    args = parser.parse_args()
    
    print("="*60)
    print("Bootstrap Confidence Intervals")
    print("="*60)
    
    # Load data
    df_seed = pd.read_csv(args.seed)
    df_synthetic = pd.read_csv(args.synthetic)
    
    # Filter to valid predictions
    df_seed = df_seed[df_seed['llm_choice'].isin(COFFEE_CHOICES)]
    df_synthetic = df_synthetic[df_synthetic['llm_choice'].isin(COFFEE_CHOICES)]
    
    print(f"Seed data (valid): {len(df_seed)} rows")
    print(f"Synthetic data (valid): {len(df_synthetic)} rows")
    
    # Compute distributions
    synthetic_dist = compute_distribution(df_synthetic, 'llm_choice')
    ground_truth_dist = compute_distribution(df_synthetic, 'ground_truth')
    
    # Run bootstrap
    print(f"\nRunning {args.n_bootstrap} bootstrap iterations...")
    bootstrap_results = bootstrap_correction(df_seed, synthetic_dist, args.n_bootstrap)
    
    # Compute confidence intervals
    ci = compute_confidence_intervals(bootstrap_results)
    
    # Point estimate (using full seed data)
    correction_matrix = learn_correction_matrix(df_seed)
    point_estimate = apply_correction(synthetic_dist, correction_matrix)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS (95% Confidence Intervals)")
    print("="*60)
    for i, coffee in enumerate(COFFEE_CHOICES):
        print(f"{coffee}: {point_estimate[i]*100:.1f}% [{ci['lower'][i]*100:.1f}%, {ci['upper'][i]*100:.1f}%]")
        print(f"  Ground truth: {ground_truth_dist[i]*100:.1f}%")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save bootstrap results
    bootstrap_df = pd.DataFrame(bootstrap_results, columns=COFFEE_CHOICES)
    bootstrap_df.to_csv(OUTPUT_DIR / "bootstrap_results.csv", index=False)
    
    # Save confidence intervals
    ci_output = {
        coffee: {
            'point_estimate': float(point_estimate[i]),
            'mean': float(ci['mean'][i]),
            'std': float(ci['std'][i]),
            'ci_lower': float(ci['lower'][i]),
            'ci_upper': float(ci['upper'][i]),
            'ground_truth': float(ground_truth_dist[i])
        }
        for i, coffee in enumerate(COFFEE_CHOICES)
    }
    with open(OUTPUT_DIR / "confidence_intervals.json", 'w') as f:
        json.dump(ci_output, f, indent=2)
    
    # Plot
    plot_confidence_intervals(ground_truth_dist, point_estimate, ci, 
                              FIGURES_DIR / "bootstrap_ci_plot.png")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
