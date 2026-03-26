#!/usr/bin/env python3
"""
01_data_analysis.py
===================
Initial exploration and visualization of the GACTT coffee tasting dataset.

This script loads the raw survey data, cleans it, and produces exploratory
visualizations of the key variables (demographics, preferences, ratings).

Usage:
    python scripts/01_data_analysis.py

Output:
    - Cleaned dataset saved to data/coffee_data_cleaned.csv
    - Figures saved to results/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "GACTT_RESULTS_ANONYMIZED_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "eda"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "coffee_data_cleaned.csv"

# Visual style
COLORS = ['#4080FF', '#57A9FB', '#37D4CF', '#23C343', '#FBE842', '#FF9A2E', '#A9AEB8']
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading and Cleaning
# -----------------------------------------------------------------------------

def load_and_clean_data():
    """Load raw data and perform basic cleaning."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")
    
    # Column renaming for convenience
    column_mapping = {
        'What is your age?': 'age',
        'Gender': 'gender',
        'Lastly, what was your favorite overall coffee?': 'overall_favorite',
        "Before today's tasting, which of the following best described what kind of coffee you like?": 'flavor_preference',
        "Before today's tasting, which roast level of coffee do you prefer?": 'roast_preference',
        'How many cups of coffee do you typically drink per day?': 'cups_per_day',
    }
    
    # Find expertise column
    for col in df.columns:
        if 'expertise' in col.lower():
            column_mapping[col] = 'expertise'
            break
    
    # Find rating columns
    for col in df.columns:
        if 'Coffee A' in col and 'rate' in col.lower():
            column_mapping[col] = 'rating_A'
        elif 'Coffee B' in col and 'rate' in col.lower():
            column_mapping[col] = 'rating_B'
        elif 'Coffee C' in col and 'rate' in col.lower():
            column_mapping[col] = 'rating_C'
        elif 'Coffee D' in col and 'rate' in col.lower():
            column_mapping[col] = 'rating_D'
    
    df = df.rename(columns=column_mapping)
    
    # Filter to valid coffee choices
    valid_choices = ['Coffee A', 'Coffee B', 'Coffee C', 'Coffee D']
    df = df[df['overall_favorite'].isin(valid_choices)]
    
    # Drop rows with missing key fields
    df = df.dropna(subset=['age', 'gender', 'overall_favorite'])
    
    print(f"  Cleaned shape: {df.shape}")
    
    return df


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def plot_coffee_preference(df):
    """Plot overall coffee preference distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df['overall_favorite'].value_counts().sort_index()
    percentages = counts / len(df) * 100
    
    bars = ax.bar(counts.index, percentages, color=COLORS[:4], edgecolor='white', linewidth=1.5)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Coffee', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Overall Coffee Preference Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(percentages) + 10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_coffee_preference.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_coffee_preference.png")


def plot_age_distribution(df):
    """Plot age distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    age_order = ['<18 years old', '18-24 years old', '25-34 years old', 
                 '35-44 years old', '45-54 years old', '55-64 years old', '>65 years old']
    
    counts = df['age'].value_counts()
    counts = counts.reindex([a for a in age_order if a in counts.index])
    percentages = counts / len(df) * 100
    
    bars = ax.bar(range(len(counts)), percentages, color=COLORS[:len(counts)], 
                  edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([a.replace(' years old', '') for a in counts.index], rotation=45, ha='right')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Age Distribution of Respondents', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_age_distribution.png")


def plot_expertise_distribution(df):
    """Plot self-reported expertise distribution."""
    if 'expertise' not in df.columns:
        print("  Skipped: expertise column not found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    expertise_counts = df['expertise'].value_counts().sort_index()
    
    ax.bar(expertise_counts.index, expertise_counts.values, color=COLORS[0], 
           edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Self-Reported Expertise (1-10)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Coffee Expertise Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_expertise_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_expertise_distribution.png")


def plot_preference_by_gender(df):
    """Plot coffee preference by gender."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Cross-tabulation
    ct = pd.crosstab(df['gender'], df['overall_favorite'], normalize='index') * 100
    ct = ct[['Coffee A', 'Coffee B', 'Coffee C', 'Coffee D']]
    
    x = np.arange(len(ct.index))
    width = 0.2
    
    for i, coffee in enumerate(ct.columns):
        ax.bar(x + i*width, ct[coffee], width, label=coffee, color=COLORS[i], 
               edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Coffee Preference by Gender', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ct.index)
    ax.legend(title='Coffee', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_preference_by_gender.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_preference_by_gender.png")


def plot_coffee_d_by_expertise(df):
    """Plot Coffee D preference by expertise level."""
    if 'expertise' not in df.columns:
        print("  Skipped: expertise column not found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate Coffee D percentage by expertise
    expertise_pref = df.groupby('expertise')['overall_favorite'].apply(
        lambda x: (x == 'Coffee D').sum() / len(x) * 100
    )
    
    ax.plot(expertise_pref.index, expertise_pref.values, 'o-', 
            color=COLORS[0], linewidth=2, markersize=8)
    
    ax.set_xlabel('Self-Reported Expertise (1-10)', fontsize=12)
    ax.set_ylabel('Coffee D Preference (%)', fontsize=12)
    ax.set_title('Coffee D Preference by Expertise Level', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_coffee_d_by_expertise.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_coffee_d_by_expertise.png")


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total respondents: {len(df)}")
    print(f"\nCoffee Preference Distribution:")
    print(df['overall_favorite'].value_counts().to_string())
    print(f"\nGender Distribution:")
    print(df['gender'].value_counts().to_string())
    print(f"\nAge Distribution:")
    print(df['age'].value_counts().to_string())
    if 'expertise' in df.columns:
        print(f"\nExpertise: mean={df['expertise'].mean():.1f}, median={df['expertise'].median():.0f}")
    print("="*60)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("="*60)
    print("GACTT Coffee Dataset Analysis")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Save cleaned data
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned data to: {CLEANED_DATA_PATH}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_coffee_preference(df)
    plot_age_distribution(df)
    plot_expertise_distribution(df)
    plot_preference_by_gender(df)
    plot_coffee_d_by_expertise(df)
    
    # Print summary
    print_summary_statistics(df)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
