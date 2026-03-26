#!/usr/bin/env python3
"""
extra_exploration.py
====================
Standalone exploration and visualization script.

This script is separate from the main pipeline. It produces additional
figures that may be useful for the thesis appendix or defense presentation.
It reads from the existing data and results but does not modify anything.

Usage:
    python scripts/extra_exploration.py

Output:
    - All figures saved to results/figures/exploration/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "coffee_data_cleaned.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "splits" / "holdout_n700.csv"
SYNTH_PATH = PROJECT_ROOT / "results" / "synthetic" / "synthetic_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "exploration"

COFFEES = ["Coffee A", "Coffee B", "Coffee C", "Coffee D"]
COFFEE_SHORT = ["A", "B", "C", "D"]

# Consistent palette across all figures
PAL = {
    "Coffee A": "#4080FF",
    "Coffee B": "#57A9FB",
    "Coffee C": "#37D4CF",
    "Coffee D": "#23C343",
}
GREY = "#A9AEB8"
ORANGE = "#FF9A2E"
RED = "#FF4444"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load all datasets needed for exploration."""
    df = pd.read_csv(DATA_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    synth = pd.read_csv(SYNTH_PATH)

    # Merge holdout demographics with LLM predictions
    holdout_indexed = holdout.reset_index()
    merged = synth.merge(
        holdout_indexed, left_on="index", right_on="index",
        how="left", suffixes=("_synth", "_holdout"),
    )
    return df, holdout, synth, merged


# ---------------------------------------------------------------------------
# Figure 1: Sensory profile radar — what do the four coffees taste like?
# ---------------------------------------------------------------------------

def fig_sensory_profiles(df):
    """Radar chart of mean bitterness, acidity, and preference for each coffee."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw=dict(polar=True))

    attributes = ["Bitterness", "Acidity", "Personal Preference"]
    angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
    angles += angles[:1]

    for ax, coffee, short in zip(axes, COFFEES, COFFEE_SHORT):
        vals = []
        for attr in attributes:
            col = f"Coffee {short} - {attr}"
            vals.append(df[col].mean())
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=2, color=PAL[coffee])
        ax.fill(angles, vals, alpha=0.2, color=PAL[coffee])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(attributes, fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
        ax.set_title(coffee, fontweight="bold", pad=15)

    fig.suptitle(
        "Mean Sensory Ratings by Coffee (1-5 scale, all respondents)",
        fontweight="bold", fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_sensory_profiles.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_sensory_profiles.png")


# ---------------------------------------------------------------------------
# Figure 2: Cross-rating heatmap — how do fans of each coffee rate the others?
# ---------------------------------------------------------------------------

def fig_cross_rating_heatmap(df):
    """Heatmap: mean personal preference rating of each coffee, grouped by
    the respondent's overall favorite."""
    matrix = np.zeros((4, 4))
    for i, fav in enumerate(COFFEES):
        sub = df[df["overall_favorite"] == fav]
        for j, short in enumerate(COFFEE_SHORT):
            matrix[i, j] = sub[f"Coffee {short} - Personal Preference"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=2, vmax=5, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(COFFEES, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Chose {c}" for c in COFFEES], fontsize=10)
    ax.set_xlabel("Coffee Rated")
    ax.set_ylabel("Respondent Group")

    for i in range(4):
        for j in range(4):
            color = "white" if matrix[i, j] > 3.8 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Mean Personal Preference (1-5)")
    ax.set_title(
        "How Each Group Rates All Four Coffees",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_cross_rating_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_cross_rating_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3: Coffee D preference by expertise — the polarization gradient
# ---------------------------------------------------------------------------

def fig_coffee_d_by_expertise(df):
    """Coffee D preference rate increases with self-reported expertise."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rates = []
    sizes = []
    for exp in range(1, 11):
        sub = df[df["expertise"] == exp]
        if len(sub) > 0:
            rates.append((sub["overall_favorite"] == "Coffee D").mean() * 100)
            sizes.append(len(sub))
        else:
            rates.append(0)
            sizes.append(0)

    x = range(1, 11)
    bars = ax.bar(x, rates, color=PAL["Coffee D"], edgecolor="white", linewidth=1.2)

    # Add sample size labels
    for bar, n in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="#666",
        )

    ax.axhline(y=37.3, color=RED, linestyle="--", linewidth=1.5, label="Population avg (37.3%)")
    ax.set_xlabel("Self-Reported Coffee Expertise (1-10)")
    ax.set_ylabel("Coffee D Preference (%)")
    ax.set_title(
        "Coffee D Preference Rises Sharply with Expertise",
        fontweight="bold",
    )
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 60)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_coffee_d_expertise_gradient.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_coffee_d_expertise_gradient.png")


# ---------------------------------------------------------------------------
# Figure 4: Coffee D preference by roast and flavor preference
# ---------------------------------------------------------------------------

def fig_coffee_d_by_taste_profile(df):
    """Two-panel figure: Coffee D rate by roast preference and flavor preference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: By roast preference
    roast_col = "What roast level of coffee do you prefer?"
    roast_order = ["Light", "Nordic", "Blonde", "Medium", "Dark"]
    roast_rates = []
    roast_ns = []
    roast_labels = []
    for r in roast_order:
        sub = df[df[roast_col] == r]
        if len(sub) > 15:
            roast_rates.append((sub["overall_favorite"] == "Coffee D").mean() * 100)
            roast_ns.append(len(sub))
            roast_labels.append(r)

    bars1 = ax1.barh(roast_labels[::-1], roast_rates[::-1],
                     color=PAL["Coffee D"], edgecolor="white", linewidth=1.2, height=0.6)
    for bar, n in zip(bars1, roast_ns[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}% (n={n})", va="center", fontsize=10)
    ax1.axvline(x=37.3, color=RED, linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Coffee D Preference (%)")
    ax1.set_title("By Roast Preference", fontweight="bold")
    ax1.set_xlim(0, 65)

    # Panel 2: By flavor preference
    flav_data = []
    for flav in df["flavor_preference"].value_counts().index:
        sub = df[df["flavor_preference"] == flav]
        if len(sub) > 30:
            rate = (sub["overall_favorite"] == "Coffee D").mean() * 100
            flav_data.append((flav, rate, len(sub)))

    flav_data.sort(key=lambda x: x[1])
    flav_labels = [f[0] for f in flav_data]
    flav_rates = [f[1] for f in flav_data]
    flav_ns = [f[2] for f in flav_data]

    colors2 = [PAL["Coffee D"] if r > 37.3 else GREY for r in flav_rates]
    bars2 = ax2.barh(flav_labels, flav_rates, color=colors2,
                     edgecolor="white", linewidth=1.2, height=0.6)
    for bar, n in zip(bars2, flav_ns):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}% (n={n})", va="center", fontsize=9)
    ax2.axvline(x=37.3, color=RED, linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Coffee D Preference (%)")
    ax2.set_title("By Stated Flavor Preference", fontweight="bold")
    ax2.set_xlim(0, 80)

    fig.suptitle(
        "Who Likes Coffee D? Preference by Taste Profile",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_coffee_d_taste_profile.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_coffee_d_taste_profile.png")


# ---------------------------------------------------------------------------
# Figure 5: Full preference mosaic — all 4 coffees by all demographics
# ---------------------------------------------------------------------------

def fig_preference_mosaic(df):
    """Stacked bar chart showing full preference distribution across key
    demographic cuts: age, gender, expertise bucket."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    def stacked_bar(ax, groupby_col, order, title, short_labels=None):
        ct = pd.crosstab(df[groupby_col], df["overall_favorite"], normalize="index") * 100
        ct = ct.reindex(index=[o for o in order if o in ct.index])
        ct = ct[COFFEES]
        labels = short_labels if short_labels else ct.index.tolist()

        bottom = np.zeros(len(ct))
        for coffee in COFFEES:
            ax.barh(labels, ct[coffee], left=bottom, label=coffee,
                    color=PAL[coffee], edgecolor="white", linewidth=0.5, height=0.6)
            bottom += ct[coffee].values

        ax.set_xlabel("Preference (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 100)

    # Age
    age_order = [
        "<18 years old", "18-24 years old", "25-34 years old",
        "35-44 years old", "45-54 years old", "55-64 years old", ">65 years old",
    ]
    age_short = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">65"]
    stacked_bar(axes[0], "age", age_order, "By Age Group", age_short)

    # Gender (top 3 only)
    gender_order = ["Male", "Female", "Non-binary"]
    stacked_bar(axes[1], "gender", gender_order, "By Gender")

    # Expertise bucket
    df["expertise_bucket"] = pd.cut(
        df["expertise"], bins=[0, 3, 6, 10],
        labels=["Low (1-3)", "Mid (4-6)", "High (7-10)"],
    )
    exp_order = ["Low (1-3)", "Mid (4-6)", "High (7-10)"]
    stacked_bar(axes[2], "expertise_bucket", exp_order, "By Expertise")
    axes[2].legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Coffee Preference Distribution Across Demographics",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_preference_mosaic.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_preference_mosaic.png")


# ---------------------------------------------------------------------------
# Figure 6: LLM confusion matrix heatmap (holdout, n=700)
# ---------------------------------------------------------------------------

def fig_confusion_heatmap(synth):
    """Heatmap of the LLM's prediction confusion on the full holdout set."""
    valid = synth[synth["llm_choice"].isin(COFFEES) & synth["ground_truth"].isin(COFFEES)]
    cm = np.zeros((4, 4))
    for _, row in valid.iterrows():
        true_idx = COFFEES.index(row["ground_truth"])
        pred_idx = COFFEES.index(row["llm_choice"])
        cm[true_idx, pred_idx] += 1

    # Normalize by row (true class) to get recall-style rates
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > 50 else "black"
            ax1.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                     fontsize=13, fontweight="bold", color=color)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"Pred {s}" for s in COFFEE_SHORT])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"True {s}" for s in COFFEE_SHORT])
    ax1.set_title("Raw Counts (n=700)", fontweight="bold")

    # Normalized (row = true class)
    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
    for i in range(4):
        for j in range(4):
            color = "white" if cm_norm[i, j] > 0.4 else "black"
            ax2.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"Pred {s}" for s in COFFEE_SHORT])
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([f"True {s}" for s in COFFEE_SHORT])
    ax2.set_title("Row-Normalized (P(pred | true))", fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="Rate")

    fig.suptitle(
        "LLM Prediction Confusion Matrix on Holdout (n=700)",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_llm_confusion_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_llm_confusion_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7: LLM accuracy by subgroup — where does the model succeed/fail?
# ---------------------------------------------------------------------------

def fig_llm_accuracy_by_subgroup(merged):
    """Grouped bar chart of LLM accuracy by expertise and roast preference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # By expertise bucket
    buckets = [("Low (1-3)", 1, 3), ("Mid (4-6)", 4, 6), ("High (7-10)", 7, 10)]
    labels_exp = []
    accs_exp = []
    ns_exp = []
    for label, lo, hi in buckets:
        sub = merged[(merged["expertise"] >= lo) & (merged["expertise"] <= hi)]
        acc = (sub["ground_truth"] == sub["llm_choice"]).mean() * 100
        labels_exp.append(label)
        accs_exp.append(acc)
        ns_exp.append(len(sub))

    bars1 = ax1.bar(labels_exp, accs_exp, color=[GREY, PAL["Coffee B"], PAL["Coffee D"]],
                    edgecolor="white", linewidth=1.5)
    for bar, acc, n in zip(bars1, accs_exp, ns_exp):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("LLM Accuracy (%)")
    ax1.set_title("By Expertise Level", fontweight="bold")
    ax1.set_ylim(0, 55)

    # By roast preference
    roast_col = "What roast level of coffee do you prefer?"
    roast_order = ["Light", "Medium", "Dark"]
    labels_roast = []
    accs_roast = []
    ns_roast = []
    for r in roast_order:
        sub = merged[merged[roast_col] == r]
        if len(sub) > 20:
            acc = (sub["ground_truth"] == sub["llm_choice"]).mean() * 100
            labels_roast.append(r)
            accs_roast.append(acc)
            ns_roast.append(len(sub))

    bars2 = ax2.bar(labels_roast, accs_roast,
                    color=[PAL["Coffee A"], PAL["Coffee B"], GREY],
                    edgecolor="white", linewidth=1.5)
    for bar, acc, n in zip(bars2, accs_roast, ns_roast):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("LLM Accuracy (%)")
    ax2.set_title("By Roast Preference", fontweight="bold")
    ax2.set_ylim(0, 55)

    fig.suptitle(
        "LLM Individual-Level Accuracy by Subgroup",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_llm_accuracy_subgroups.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_llm_accuracy_subgroups.png")


# ---------------------------------------------------------------------------
# Figure 8: The A-vs-D split — the dataset's hidden structure
# ---------------------------------------------------------------------------

def fig_a_vs_d_structure(df):
    """Visualize the A-vs-D head-to-head and how it maps to overall preference."""
    ad_col = "Between Coffee A and Coffee D, which did you prefer?"
    df_ad = df[df[ad_col].isin(["Coffee A", "Coffee D"])].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Head-to-head split
    counts = df_ad[ad_col].value_counts()
    total = counts.sum()
    bars = ax1.bar(
        ["Preferred A", "Preferred D"],
        [counts.get("Coffee A", 0), counts.get("Coffee D", 0)],
        color=[PAL["Coffee A"], PAL["Coffee D"]],
        edgecolor="white", linewidth=1.5,
    )
    for bar in bars:
        pct = bar.get_height() / total * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"{int(bar.get_height())}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_title("A vs D Head-to-Head", fontweight="bold")
    ax1.set_ylim(0, max(counts) * 1.2)

    # Panel 2: Where do A-preferers and D-preferers end up?
    x = np.arange(4)
    w = 0.35

    a_fans = df_ad[df_ad[ad_col] == "Coffee A"]
    d_fans = df_ad[df_ad[ad_col] == "Coffee D"]

    a_dist = np.array([(a_fans["overall_favorite"] == c).mean() * 100 for c in COFFEES])
    d_dist = np.array([(d_fans["overall_favorite"] == c).mean() * 100 for c in COFFEES])

    ax2.bar(x - w / 2, a_dist, w, label="Preferred A in head-to-head",
            color=PAL["Coffee A"], edgecolor="white", linewidth=1)
    ax2.bar(x + w / 2, d_dist, w, label="Preferred D in head-to-head",
            color=PAL["Coffee D"], edgecolor="white", linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(COFFEES)
    ax2.set_ylabel("Overall Favorite (%)")
    ax2.set_title("Overall Favorite by Head-to-Head Preference", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 75)

    fig.suptitle(
        "The A-vs-D Divide: A Structural Split in the Dataset",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_a_vs_d_structure.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_a_vs_d_structure.png")


# ---------------------------------------------------------------------------
# Figure 9: Correction matrix heatmaps for all three seed sizes
# ---------------------------------------------------------------------------

def fig_correction_matrices(seed_dir):
    """Side-by-side heatmaps of the learned correction matrices P(true|llm)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)

    for ax, m in zip(axes, [50, 100, 150]):
        df_seed = pd.read_csv(
            PROJECT_ROOT / "results" / "synthetic" / f"seed_predictions_m{m}.csv"
        )
        df_seed = df_seed[
            df_seed["llm_choice"].isin(COFFEES) & df_seed["ground_truth"].isin(COFFEES)
        ]

        counts = np.zeros((4, 4))
        for _, row in df_seed.iterrows():
            true_idx = COFFEES.index(row["ground_truth"])
            llm_idx = COFFEES.index(row["llm_choice"])
            counts[true_idx, llm_idx] += 1

        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm = counts / col_sums

        im = ax.imshow(cm, cmap="YlOrRd", vmin=0, vmax=0.7, aspect="auto")
        for i in range(4):
            for j in range(4):
                color = "white" if cm[i, j] > 0.45 else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

        ax.set_xticks(range(4))
        ax.set_xticklabels([f"LLM: {s}" for s in COFFEE_SHORT], fontsize=10)
        if m == 50:
            ax.set_yticks(range(4))
            ax.set_yticklabels([f"True: {s}" for s in COFFEE_SHORT], fontsize=10)
        ax.set_title(f"m = {m}", fontweight="bold")

    cbar = fig.colorbar(im, ax=axes.tolist(), label="P(true | LLM prediction)", shrink=0.75, pad=0.02)
    fig.suptitle(
        "Learned Correction Matrices P(true | LLM) by Seed Size",
        fontweight="bold", fontsize=14,
    )
    fig.subplots_adjust(left=0.05, right=0.88, wspace=0.15)
    plt.savefig(OUTPUT_DIR / "09_correction_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_correction_matrices.png")


# ---------------------------------------------------------------------------
# Figure 10: Full pipeline summary — one figure that tells the whole story
# ---------------------------------------------------------------------------

def fig_pipeline_summary():
    """Three-panel figure: raw bias, correction effect, CI narrowing.
    Uses the same hardcoded values as 06_final_figures.py for consistency."""
    GT = np.array([0.217, 0.207, 0.203, 0.373])
    SYNTH = np.array([0.123, 0.361, 0.183, 0.333])
    CORRECTED = {
        50:  np.array([0.197, 0.164, 0.236, 0.402]),
        100: np.array([0.222, 0.194, 0.205, 0.379]),
        150: np.array([0.221, 0.203, 0.203, 0.374]),
    }
    TVD_UNCORR = 0.154
    TVD_CORR = {50: 0.063, 100: 0.013, 150: 0.005}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Bias
    bias = (SYNTH - GT) * 100
    colors = [RED if b > 0 else PAL["Coffee A"] for b in bias]
    ax1.barh(COFFEE_SHORT, bias, color=colors, edgecolor="white", height=0.5)
    for i, b in enumerate(bias):
        offset = 0.5 if b > 0 else -0.5
        ha = "left" if b > 0 else "right"
        ax1.text(b + offset, i, f"{b:+.1f}pp", ha=ha, va="center",
                 fontweight="bold", fontsize=11)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("Bias (percentage points)")
    ax1.set_title("1. LLM Systematic Bias", fontweight="bold")
    ax1.set_xlim(-12, 18)

    # Panel 2: TVD reduction
    seeds = [50, 100, 150]
    tvd_vals = [TVD_CORR[s] for s in seeds]
    bars = ax2.bar(
        ["Raw"] + [f"m={s}" for s in seeds],
        [TVD_UNCORR] + tvd_vals,
        color=[ORANGE] + [PAL["Coffee D"]] * 3,
        edgecolor="white", linewidth=1.5,
    )
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Total Variation Distance")
    ax2.set_title("2. Correction Reduces Error", fontweight="bold")
    ax2.set_ylim(0, 0.20)

    # Panel 3: Corrected vs GT for m=100
    x = np.arange(4)
    w = 0.35
    ax3.bar(x - w / 2, GT * 100, w, label="Ground Truth", color=GREY, edgecolor="white")
    ax3.bar(x + w / 2, CORRECTED[100] * 100, w, label="Corrected (m=100)",
            color=PAL["Coffee D"], edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(COFFEE_SHORT)
    ax3.set_ylabel("Preference (%)")
    ax3.set_title("3. Recovered Distribution", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 50)

    fig.suptitle(
        "Pipeline Summary: Bias, Correction, Recovery",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_pipeline_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: 10_pipeline_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Extra Exploration: Generating additional figures")
    print("=" * 60)

    df, holdout, synth, merged = load_data()

    print(f"\nDatasets loaded:")
    print(f"  Full cleaned: {len(df)} rows")
    print(f"  Holdout: {len(holdout)} rows")
    print(f"  Synthetic predictions: {len(synth)} rows")
    print(f"  Merged (holdout + predictions): {len(merged)} rows")

    print(f"\nGenerating figures to {OUTPUT_DIR}/\n")

    fig_sensory_profiles(df)
    fig_cross_rating_heatmap(df)
    fig_coffee_d_by_expertise(df)
    fig_coffee_d_by_taste_profile(df)
    fig_preference_mosaic(df)
    fig_confusion_heatmap(synth)
    fig_llm_accuracy_by_subgroup(merged)
    fig_a_vs_d_structure(df)
    fig_correction_matrices(PROJECT_ROOT / "data" / "splits")
    fig_pipeline_summary()

    print(f"\nDone! {len(list(OUTPUT_DIR.glob('*.png')))} figures saved.")


if __name__ == "__main__":
    main()
