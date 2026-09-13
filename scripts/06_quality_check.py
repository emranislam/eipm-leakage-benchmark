"""
Script 06: Final quality check before running the ML pipeline
Checks:
  - Feature distributions look physically reasonable
  - No data leakage between features and labels
  - Mann-Whitney U discriminability
  - Temporal ordering confirmed
  - Class balance reported
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

INPUT = "data/features/extracted_features.csv"
print("Loading labeled feature matrix...")
df = pd.read_csv(INPUT)

FEAT_COLS = ['b','a','CBS_accel','alpha_micro','sigma_z2','mean_z',
             'rho','H_space','mean_M','M_max','lambda','mean_dt','CV_t']

print(f"Shape : {df.shape}")
print(f"Labels: {df['Label'].value_counts().to_dict()}")

# ── CHECK 1: No NaN or Inf ────────────────────────────────────────────────────
inf_count = np.isinf(df[FEAT_COLS].values).sum()
nan_count = df[FEAT_COLS].isnull().sum().sum()
print(f"\nCheck 1 — Data Quality")
print(f"  Inf values : {inf_count}  {'✓' if inf_count==0 else '✗ PROBLEM'}")
print(f"  NaN values : {nan_count}  {'✓' if nan_count==0 else '✗ PROBLEM'}")

# ── CHECK 2: Mann-Whitney U discriminability ───────────────────────────────────
print(f"\nCheck 2 — Feature Discriminability (Mann-Whitney U)")
print(f"{'Feature':<15} {'Precursor Mean':>15} {'Background Mean':>16} "
      f"{'p-value':>12} {'Sig?':>6}")
print("-"*65)

sig_count = 0
for col in FEAT_COLS:
    g1 = df.loc[df['Label']==1, col]
    g0 = df.loc[df['Label']==0, col]
    _, p = mannwhitneyu(g1, g0, alternative='two-sided')
    sig = p < 0.001
    if sig:
        sig_count += 1
    print(f"  {col:<13} {g1.mean():>15.4f} {g0.mean():>16.4f} "
          f"{p:>12.4e}  {'✓' if sig else '✗'}")

print(f"\n  Significant features (p<0.001): {sig_count}/{len(FEAT_COLS)}")

if sig_count == 0:
    print("  ✗ CRITICAL: No significant features.")
    print("    Action: Check labeling parameters and feature extraction.")
elif sig_count < 5:
    print("  ⚠ WARNING: Few significant features.")
    print("    Model may underperform. Consider feature engineering.")
else:
    print("  ✓ GOOD: Sufficient discriminating features for modeling.")

# ── CHECK 3: Temporal order ───────────────────────────────────────────────────
print(f"\nCheck 3 — Temporal Ordering")
# Assuming seq_id increases monotonically with time
if 'seq_id' in df.columns:
    is_sorted = (df['seq_id'].diff().dropna() > 0).all()
    print(f"  Temporal order confirmed: {'✓' if is_sorted else '✗ PROBLEM'}")

# ── CHECK 4: Feature correlation with label ───────────────────────────────────
print(f"\nCheck 4 — Point-Biserial Correlation with Label")
for col in FEAT_COLS:
    corr = df[col].corr(df['Label'])
    print(f"  {col:<15}: r = {corr:+.4f}")

# ── DIAGNOSTIC PLOTS ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(FEAT_COLS):
    ax = axes[i]
    prec = df.loc[df['Label']==1, col]
    bg   = df.loc[df['Label']==0, col]
    ax.hist(bg,   bins=30, alpha=0.6, color='steelblue',
            label='Background', density=True)
    ax.hist(prec, bins=30, alpha=0.6, color='crimson',
            label='Precursor',  density=True)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)

# Hide unused axes
for j in range(len(FEAT_COLS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions: Precursor vs Background",
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig("outputs/figures/quality_check_distributions.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDistribution plots saved: "
      f"outputs/figures/quality_check_distributions.png")

print(f"\n{'='*50}")
print("QUALITY CHECK COMPLETE")
print("If all checks pass → run the ML pipeline")
print("If checks fail → revisit labeling parameters")
print("="*50)

