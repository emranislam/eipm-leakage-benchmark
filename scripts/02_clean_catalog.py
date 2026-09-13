"""
Script 02: Clean and standardize the raw USGS catalog
Actions:
  - Select and rename relevant columns
  - Parse datetime
  - Remove duplicates
  - Remove events with missing depth or magnitude
  - Sort chronologically
  - Save reviewed events only
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT  = "data/raw/usgs_catalog_raw.csv"
OUTPUT = "data/processed/catalog_clean.csv"

print("Loading raw catalog...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw events: {len(df)}")
print(f"Columns available: {df.columns.tolist()}")

# ── STEP 2.1: Select essential columns ───────────────────────────────────────
essential = ['time', 'latitude', 'longitude', 'depth', 'mag',
             'magType', 'type', 'status']

# Keep only columns that exist
keep = [c for c in essential if c in df.columns]
df   = df[keep].copy()

# ── STEP 2.2: Keep only earthquakes (not explosions, quarry blasts) ───────────
if 'type' in df.columns:
    before = len(df)
    df = df[df['type'] == 'earthquake']
    print(f"Non-earthquake events removed: {before - len(df)}")

# ── STEP 2.3: Keep only reviewed events ──────────────────────────────────────
# "reviewed" status ensures higher quality locations and magnitudes
if 'status' in df.columns:
    before = len(df)
    df_reviewed = df[df['status'] == 'reviewed']
    print(f"Reviewed events: {len(df_reviewed)} "
          f"(removed {before - len(df_reviewed)} automatic)")
    # Only use reviewed if sufficient — otherwise keep all
    if len(df_reviewed) >= 3000:
        df = df_reviewed
    else:
        print("  WARNING: Too few reviewed events. Keeping all.")

# ── STEP 2.4: Parse and sort by time ─────────────────────────────────────────
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df = df.dropna(subset=['time'])
df = df.sort_values('time').reset_index(drop=True)

# ── STEP 2.5: Remove missing values ──────────────────────────────────────────
before = len(df)
df = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])
print(f"Removed {before - len(df)} events with missing lat/lon/depth/mag")

# ── STEP 2.6: Remove physically impossible values ─────────────────────────────
before = len(df)
df = df[
    (df['depth'] >= 0)    & (df['depth'] <= 700) &   # km
    (df['mag']   >= 1.0)  & (df['mag']   <= 10.0) &  # Richter
    (df['latitude']  >= 10) & (df['latitude']  <= 30) &
    (df['longitude'] >= 85) & (df['longitude'] <= 98)
]
print(f"Removed {before - len(df)} physically impossible events")

# ── STEP 2.7: Remove duplicate events ────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])
print(f"Removed {before - len(df)} duplicate events")

# ── STEP 2.8: Reset index and save ───────────────────────────────────────────
df = df.reset_index(drop=True)
df.to_csv(OUTPUT, index=False)

# ── STEP 2.9: Summary statistics ─────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"CLEAN CATALOG SUMMARY")
print(f"{'='*50}")
print(f"Total events    : {len(df)}")
print(f"Date range      : {df['time'].min()} → {df['time'].max()}")
print(f"Magnitude range : {df['mag'].min():.1f} → {df['mag'].max():.1f}")
print(f"Depth range     : {df['depth'].min():.1f} → {df['depth'].max():.1f} km")
print(f"Lat range       : {df['latitude'].min():.2f} → {df['latitude'].max():.2f}")
print(f"Lon range       : {df['longitude'].min():.2f} → {df['longitude'].max():.2f}")
print(f"\nMagnitude distribution:")
print(df['mag'].describe())

# ── STEP 2.10: Diagnostic plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Magnitude histogram
axes[0].hist(df['mag'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_xlabel("Magnitude"); axes[0].set_ylabel("Count")
axes[0].set_title("Magnitude Distribution")
axes[0].axvline(4.5, color='red', linestyle='--', label='Target Mw≥4.5')
axes[0].legend()

# Depth histogram
axes[1].hist(df['depth'], bins=30, color='darkorange', edgecolor='white')
axes[1].set_xlabel("Depth (km)"); axes[1].set_ylabel("Count")
axes[1].set_title("Depth Distribution")

# Events per year
df['year'] = df['time'].dt.year
yearly = df.groupby('year').size()
axes[2].bar(yearly.index, yearly.values, color='green', edgecolor='white')
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Event Count")
axes[2].set_title("Events Per Year")

plt.tight_layout()
fig.savefig("outputs/figures/catalog_diagnostics.png", dpi=150)
plt.close()
print(f"\nDiagnostic plot saved: outputs/figures/catalog_diagnostics.png")
print(f"Clean catalog saved  : {OUTPUT}")

