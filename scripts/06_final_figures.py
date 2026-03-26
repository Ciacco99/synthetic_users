#!/usr/bin/env python3
"""
06_final_figures.py
===================
Generate the definitive, publication-quality figures for the thesis.
Uses the final results from the full pipeline run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "final"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

COFFEES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]
GT = np.array([0.217, 0.207, 0.203, 0.373])
SYNTH_RAW = np.array([0.123, 0.361, 0.183, 0.333])

# Corrected point estimates
CORRECTED = {
    50:  np.array([0.197, 0.164, 0.236, 0.402]),
    100: np.array([0.222, 0.194, 0.205, 0.379]),
    150: np.array([0.221, 0.203, 0.203, 0.374]),
}

# Bootstrap 95% CIs
CI_LOWER = {
    50:  np.array([0.098, 0.078, 0.101, 0.246]),
    100: np.array([0.137, 0.121, 0.136, 0.289]),
    150: np.array([0.153, 0.143, 0.143, 0.295]),
}
CI_UPPER = {
    50:  np.array([0.312, 0.270, 0.373, 0.556]),
    100: np.array([0.307, 0.266, 0.288, 0.468]),
    150: np.array([0.294, 0.270, 0.269, 0.447]),
}

TVD_UNCORRECTED = 0.154
TVD_CORRECTED = {50: 0.063, 100: 0.013, 150: 0.005}
IMPROVEMENT = {50: 59.4, 100: 91.4, 150: 97.0}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Four-distribution comparison (m=100)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_four_distribution():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(COFFEES))
    w = 0.2

    colors = ['#A9AEB8', '#FF9A2E', '#37D4CF', '#4080FF']
    labels = ['Ground Truth (n=700)', 'Synthetic Raw (LLM)', 'Corrected (m=100)', 'Seed (m=100)']
    
    seed_dist = np.array([0.22, 0.21, 0.20, 0.37])  # from split verification
    dists = [GT, SYNTH_RAW, CORRECTED[100], seed_dist]

    for i, (dist, color, label) in enumerate(zip(dists, colors, labels)):
        bars = ax.bar(x + i*w, dist*100, w, label=label, color=color, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, dist):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(COFFEES)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Distribution Comparison: Ground Truth vs. Synthetic vs. Corrected', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_four_distribution_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: fig1_four_distribution_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: TVD improvement by seed size
# ─────────────────────────────────────────────────────────────────────────────
def fig2_tvd_by_seed():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seeds = [50, 100, 150]
    tvd_vals = [TVD_CORRECTED[s] for s in seeds]
    imp_vals = [IMPROVEMENT[s] for s in seeds]

    # Left: TVD values
    ax1.bar(['Uncorrected'] + [f'm={s}' for s in seeds],
            [TVD_UNCORRECTED] + tvd_vals,
            color=['#FF9A2E'] + ['#37D4CF']*3,
            edgecolor='white', linewidth=1.5)
    for i, v in enumerate([TVD_UNCORRECTED] + tvd_vals):
        ax1.text(i, v + 0.003, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Total Variation Distance')
    ax1.set_title('TVD: Uncorrected vs. Corrected', fontweight='bold')
    ax1.set_ylim(0, 0.20)

    # Right: Improvement %
    bars = ax2.bar([f'm={s}' for s in seeds], imp_vals, color=['#4080FF', '#37D4CF', '#23C343'],
                   edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, imp_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax2.set_ylabel('TVD Improvement (%)')
    ax2.set_title('Correction Effectiveness by Seed Size', fontweight='bold')
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_tvd_by_seed_size.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_tvd_by_seed_size.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Bootstrap CIs across seed sizes
# ─────────────────────────────────────────────────────────────────────────────
def fig3_bootstrap_ci_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    for ax, m in zip(axes, [50, 100, 150]):
        x = np.arange(len(COFFEES))
        w = 0.35

        ax.bar(x - w/2, GT*100, w, label='Ground Truth', color='#A9AEB8', edgecolor='white')
        ax.bar(x + w/2, CORRECTED[m]*100, w, label=f'Corrected (m={m})', color='#37D4CF', edgecolor='white')

        yerr_lower = (CORRECTED[m] - CI_LOWER[m]) * 100
        yerr_upper = (CI_UPPER[m] - CORRECTED[m]) * 100
        ax.errorbar(x + w/2, CORRECTED[m]*100, yerr=[yerr_lower, yerr_upper],
                    fmt='none', color='black', capsize=4, capthick=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(['A', 'B', 'C', 'D'])
        ax.set_title(f'Seed m={m} (TVD={TVD_CORRECTED[m]:.3f})', fontweight='bold')
        ax.set_ylim(0, 60)
        if m == 50:
            ax.set_ylabel('Percentage (%)')
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Bootstrap 95% Confidence Intervals by Seed Size (B=1000)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_bootstrap_ci_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: fig3_bootstrap_ci_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: LLM Bias visualization
# ─────────────────────────────────────────────────────────────────────────────
def fig4_bias_visualization():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bias = (SYNTH_RAW - GT) * 100
    colors = ['#FF4444' if b > 0 else '#4080FF' for b in bias]
    
    bars = ax.barh(COFFEES, bias, color=colors, edgecolor='white', linewidth=1.5, height=0.5)
    
    for bar, b in zip(bars, bias):
        offset = 0.5 if b > 0 else -0.5
        ha = 'left' if b > 0 else 'right'
        ax.text(b + offset, bar.get_y() + bar.get_height()/2,
                f'{b:+.1f}pp', ha=ha, va='center', fontweight='bold', fontsize=12)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Bias (percentage points)')
    ax.set_title('LLM Systematic Bias: Synthetic vs. Ground Truth', fontweight='bold')
    ax.set_xlim(-12, 18)
    
    # Add legend
    over = mpatches.Patch(color='#FF4444', label='Over-prediction')
    under = mpatches.Patch(color='#4080FF', label='Under-prediction')
    ax.legend(handles=[over, under], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_llm_bias.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_llm_bias.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: CI width narrowing
# ─────────────────────────────────────────────────────────────────────────────
def fig5_ci_width():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seeds = [50, 100, 150]
    for i, coffee in enumerate(COFFEES):
        widths = [(CI_UPPER[m][i] - CI_LOWER[m][i]) * 100 for m in seeds]
        ax.plot(seeds, widths, 'o-', label=coffee, linewidth=2, markersize=8)
    
    ax.set_xlabel('Seed Size (m)')
    ax.set_ylabel('95% CI Width (percentage points)')
    ax.set_title('Confidence Interval Width Narrows with Seed Size', fontweight='bold')
    ax.set_xticks(seeds)
    ax.legend()
    ax.set_ylim(0, 35)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_ci_width_by_seed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_ci_width_by_seed.png")


if __name__ == "__main__":
    print("Generating final thesis figures...")
    fig1_four_distribution()
    fig2_tvd_by_seed()
    fig3_bootstrap_ci_comparison()
    fig4_bias_visualization()
    fig5_ci_width()
    print("\nAll figures saved to:", FIGURES_DIR)
