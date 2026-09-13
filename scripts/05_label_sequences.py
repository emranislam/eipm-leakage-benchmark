"""
Script 05: Assign binary labels to each micro-sequence

Prediction target (formally defined):
  Label = 1 (Precursor) if a Mw >= 4.5 earthquake occurs
               WITHIN 15 days AFTER the sequence end time
               AND WITHIN 100 km of the sequence centroid

  Label = 0 (Background) otherwise

This script implements the labeling algorithm that was
missing from the original manuscript.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

CATALOG_FILE  = "data/processed/catalog_clean.csv"
FEATURE_FILE  = "data/features/extracted_features_v2.csv"
OUTPUT        = "data/features/extracted_features_labeled.csv"

# ── PREDICTION TARGET PARAMETERS ──────────────────────────────────────────────
TARGET_MAG   = 4.5    # Mw threshold for target earthquakes
LEAD_DAYS    = 15     # Forecasting horizon in days
RADIUS_KM    = 100    # Spatial radius around sequence centroid

# ── HAVERSINE DISTANCE ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_cat  = pd.read_csv(CATALOG_FILE)
df_feat = pd.read_csv(FEATURE_FILE)

df_cat['time'] = pd.to_datetime(df_cat['time'], format='mixed', utc=True)
df_feat['end_time'] = pd.to_datetime(df_feat['end_time'], format='mixed', utc=True)

# Filter target events (Mw >= 4.5 only)
df_targets = df_cat[df_cat['mag'] >= TARGET_MAG].copy()
df_targets = df_targets.reset_index(drop=True)

print(f"Feature sequences   : {len(df_feat)}")
print(f"Target events (M≥{TARGET_MAG}): {len(df_targets)}")
print(f"\nLabeling with parameters:")
print(f"  Target Mw  >= {TARGET_MAG}")
print(f"  Lead time  <= {LEAD_DAYS} days")
print(f"  Radius     <= {RADIUS_KM} km")

# ── LABELING ALGORITHM ────────────────────────────────────────────────────────
labels    = []
lead_mags = []  # magnitude of the triggering event (for analysis)

for _, seq in tqdm(df_feat.iterrows(), total=len(df_feat),
                   desc="Labeling sequences"):
    t_end     = seq['end_time']
    t_horizon = t_end + pd.Timedelta(days=LEAD_DAYS)
    c_lat     = seq['centroid_lat']
    c_lon     = seq['centroid_lon']

    # Step 1: Filter targets within time window
    future = df_targets[
        (df_targets['time'] > t_end) &
        (df_targets['time'] <= t_horizon)
    ]

    # Step 2: Check spatial radius for each candidate
    label    = 0
    trig_mag = np.nan

    for _, ev in future.iterrows():
        dist = haversine_km(c_lat, c_lon,
                            ev['latitude'], ev['longitude'])
        if dist <= RADIUS_KM:
            label    = 1
            trig_mag = ev['mag']
            break  # label=1 confirmed, no need to check further

    labels.append(label)
    lead_mags.append(trig_mag)

# ── ATTACH LABELS ─────────────────────────────────────────────────────────────
df_feat['Label']    = labels
df_feat['trig_mag'] = lead_mags  # keep for error analysis

# ── LABEL STATISTICS ──────────────────────────────────────────────────────────
n_pos = (df_feat['Label'] == 1).sum()
n_neg = (df_feat['Label'] == 0).sum()
ratio = n_neg / n_pos if n_pos > 0 else float('inf')

print(f"\n{'='*50}")
print(f"LABELING SUMMARY")
print(f"{'='*50}")
print(f"Precursor sequences (Label=1) : {n_pos}")
print(f"Background sequences (Label=0): {n_neg}")
print(f"Imbalance ratio               : {ratio:.1f}:1")

if n_pos < 50:
    print("\n  ⚠ WARNING: Very few precursor sequences (<50).")
    print("    Consider increasing LEAD_DAYS to 21 or RADIUS_KM to 150.")
elif ratio > 50:
    print("\n  ⚠ WARNING: Extreme class imbalance (>50:1).")
    print("    Consider increasing LEAD_DAYS or RADIUS_KM.")
else:
    print("\n  ✓ Class balance is acceptable for modeling.")

# Magnitude distribution of triggering events
print(f"\nTriggering event magnitudes:")
print(df_feat['trig_mag'].dropna().describe().round(3))

# ── SAVE ──────────────────────────────────────────────────────────────────────
# Save full file (with metadata)
df_feat.to_csv(OUTPUT, index=False)

# Save model-ready file (features + label only)
feat_cols = ['b','a','CBS_accel','alpha_micro','sigma_z2','mean_z',
             'rho','H_space','mean_M','M_max','lambda','mean_dt',
             'CV_t','Label']
df_feat[feat_cols].to_csv(
    "data/features/extracted_features.csv", index=False
)

print(f"\nFull file saved    : {OUTPUT}")
print(f"Model-ready saved  : data/features/extracted_features.csv")

