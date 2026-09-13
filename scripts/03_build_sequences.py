"""
Script 03: Build event-driven micro-sequences.

Window size : N = 20 consecutive catalog events
Stride      : 1 event (unit stride, maximum overlap)

A catalog of n events yields n - N + 1 windows. The previous version of this
script looped over `range(len(df) - N)`, which produced n - N windows and
silently discarded the final complete window, so the last event in the
catalog never appeared in any sequence. That is fixed here, and asserted.

Each sequence records:
  seq_id              unique identifier, ordered by end time
  start_idx, end_idx  row indices into the clean catalog (end_idx inclusive)
  start_time, end_time
  centroid_lat/lon    arithmetic mean of the 20 epicenters
  n_events            always N (asserted)
  mean_mag, max_mag
  duration_days       wall-clock span of the window
  footprint_km2       bounding-box area of the 20 epicenters
  centroid_to_nearest_km
                      great-circle distance from the centroid to the nearest
                      constituent event

The last three are diagnostics, not features. They exist because the
manuscript must report them:

  * duration_days quantifies the mismatch between an event-counted window
    and a wall-clock label horizon, which is what makes a count-based
    embargo inadequate during bursts. It is also the quantity a time-based
    purge needs.
  * footprint_km2 is the denominator of the spatial-density feature rho,
    which the manuscript must define explicitly.
  * centroid_to_nearest_km measures how far the centroid can sit from any
    real epicenter. For a diffuse window the centroid may lie in a place
    with no seismicity at all, which bears directly on what the 100 km
    label radius means.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT = "data/processed/catalog_clean.csv"
OUTPUT = "data/processed/sequences_meta.csv"

N = 20        # window length, in events
EARTH_R = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalars or arrays, degrees in."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def main():
    print("Loading clean catalog ...")
    df = pd.read_csv(INPUT)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    n = len(df)
    n_seq = n - N + 1
    if n_seq < 1:
        raise SystemExit(f"FATAL: catalog has {n} events, fewer than N = {N}")

    print(f"Catalog size      : {n} events")
    print(f"Window size       : N = {N}")
    print(f"Expected sequences: {n_seq}   (n - N + 1)")

    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    mag = df["mag"].to_numpy(float)
    tim = df["time"].to_numpy()

    rows = []
    for i in tqdm(range(n_seq), desc="Building sequences"):
        j = i + N                      # exclusive end
        wlat, wlon = lat[i:j], lon[i:j]
        clat, clon = wlat.mean(), wlon.mean()

        # bounding-box footprint, the denominator used by rho in script 04
        lat_km = (wlat.max() - wlat.min()) * 111.0
        lon_km = (wlon.max() - wlon.min()) * 111.0 * np.cos(np.radians(clat))
        footprint = max(lat_km * lon_km, 0.01)

        d_near = haversine_km(clat, clon, wlat, wlon).min()
        dur = (tim[j - 1] - tim[i]) / np.timedelta64(1, "D")

        rows.append({
            "seq_id": i,
            "start_idx": i,
            "end_idx": j - 1,
            "start_time": df["time"].iloc[i],
            "end_time": df["time"].iloc[j - 1],
            "centroid_lat": round(float(clat), 4),
            "centroid_lon": round(float(clon), 4),
            "n_events": N,
            "mean_mag": round(float(mag[i:j].mean()), 3),
            "max_mag": round(float(mag[i:j].max()), 3),
            "duration_days": round(float(dur), 6),
            "footprint_km2": round(float(footprint), 4),
            "centroid_to_nearest_km": round(float(d_near), 3),
        })

    seq = pd.DataFrame(rows)

    # ---- assertions: fail loudly rather than write a wrong file ------------
    assert len(seq) == n_seq, f"built {len(seq)} sequences, expected {n_seq}"
    assert (seq["n_events"] == N).all(), "a window does not contain N events"
    assert seq["end_idx"].iloc[-1] == n - 1, (
        "the last catalog event is not in any sequence -- off-by-one")
    assert seq["start_idx"].iloc[0] == 0, "the first catalog event is missing"
    assert seq["end_time"].is_monotonic_increasing, (
        "sequence end times are not ordered; the catalog was not sorted")
    assert seq["seq_id"].is_unique, "duplicate seq_id"
    assert (seq["duration_days"] >= 0).all(), "negative window duration"

    seq.to_csv(OUTPUT, index=False)

    # ---- summary -----------------------------------------------------------
    dur = seq["duration_days"]
    near = seq["centroid_to_nearest_km"]
    print(f"\n{'=' * 62}")
    print("SEQUENCE BUILD SUMMARY")
    print(f"{'=' * 62}")
    print(f"Total sequences        : {len(seq)}")
    print(f"First sequence ends    : {seq['end_time'].iloc[0]}")
    print(f"Last sequence ends     : {seq['end_time'].iloc[-1]}")
    print(f"Last event in catalog  : {df['time'].iloc[-1]}   "
          f"(must equal the line above)")
    print()
    print("Window duration (days), the event-count vs wall-clock mismatch:")
    print(f"  min {dur.min():.4f}   median {dur.median():.2f}   "
          f"mean {dur.mean():.2f}   max {dur.max():.1f}")
    for q in (1, 5, 50, 95, 99):
        print(f"    p{q:<3} = {np.percentile(dur, q):10.4f} d")
    short = int((dur < 15).sum())
    print(f"  windows spanning less than the 15-day label horizon: "
          f"{short} ({100 * short / len(seq):.1f}%)")
    print()
    print("Centroid to nearest constituent event (km):")
    print(f"  median {near.median():.1f}   mean {near.mean():.1f}   "
          f"max {near.max():.1f}")
    for q in (50, 90, 99):
        print(f"    p{q:<3} = {np.percentile(near, q):8.1f} km")
    far = int((near > 100).sum())
    print(f"  centroids further than the 100 km label radius from every "
          f"constituent event: {far} ({100 * far / len(seq):.1f}%)")
    print()
    print(f"Footprint area (km^2): median {seq['footprint_km2'].median():.0f}"
          f"   p99 {np.percentile(seq['footprint_km2'], 99):.0f}")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()

