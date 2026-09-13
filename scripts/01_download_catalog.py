"""
Script 01: Download the raw seismic catalog from the USGS ComCat FDSN service.

Region : Eastern Indian Plate margin (10-30 N, 85-98 E)
Period : 1973-01-01 to 2026-01-01  (half-open; see note on chunking below)
Minimum magnitude requested: 2.0

Why the record starts in 1973
-----------------------------
The start year is not a seismological choice. ComCat is a composite catalog:
it aggregates products from many contributing networks and reanalyzed global
catalogs and reports a preferred solution for each event. It holds no U.S.
Geological Survey solution for this region before 1973, because the National
Earthquake Information Center became part of the Survey in 1973 and its
global bulletin, the Preliminary Determination of Epicenters, became a Survey
product at that time. The interval 1970-1972 therefore consists solely of
ISC-GEM entries, all reported as moment magnitude with a completeness floor
near M 5.0. Including them would splice a differently scaled and differently
complete catalog onto the front of the record.

Those 38 events are nonetheless retrieved into a separate file, so that the
question "what if you had kept them?" can be answered from the archive rather
than from a fresh ComCat query, whose results will differ over time.

Outputs
-------
data/raw/usgs_catalog_raw.csv          the analyzed catalog, 1973 onward
data/raw/iscgem_1970_1972.csv          the excluded interval, for reference
data/raw/usgs_catalog_manifest.json    retrieval date, query, counts, SHA-256

Why this script is stricter than a plain loop
---------------------------------------------
1. Chunk boundaries are HALF-OPEN. FDSNWS interprets a bare date as
   T00:00:00, so `endtime=YYYY-12-31` silently discards the whole of
   31 December. Using `endtime` = the following 1 January leaves no gaps and
   creates no duplicates. A boundary-overlap check below asserts this.
2. Failures are RETRIED and then FATAL. A transient HTTP error must never
   produce a silently short catalog; a missing year would shift every
   downstream number with nothing in the output to show it.
3. HTTP 204 is success, not failure -- but the status code is checked BEFORE
   the response body, because an error response also has an empty body and
   must not be mistaken for a genuinely empty window.
4. A manifest records the retrieval date, the start-year rationale, and a
   checksum, because ComCat is a living database: magnitudes are revised and
   events are added, so this script alone does not make the analysis
   reproducible. Archive the CSVs together with the manifest and cite the
   archived copy.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

# ---- CONFIGURATION ---------------------------------------------------------
REGION = {
    "minlatitude": 10.0,
    "maxlatitude": 30.0,
    "minlongitude": 85.0,
    "maxlongitude": 98.0,
}
MIN_MAG = 2.0

START_YEAR = 1973          # ComCat holds no USGS (PDE) solution for this
                           # region before 1973; 1970-1972 is ISC-GEM only,
                           # moment magnitude, M>=5.0 floor. See module
                           # docstring and the Data section of the paper.
END_YEAR = 2025            # inclusive: chunks run to 2026-01-01

# The excluded pre-1973 interval, retrieved separately for reference only.
# Set FETCH_EXCLUDED = False to skip it.
FETCH_EXCLUDED = True
EXCLUDED_START_YEAR = 1970
EXCLUDED_END_YEAR = 1972

OUT_DIR = "data/raw"
OUTPUT = os.path.join(OUT_DIR, "usgs_catalog_raw.csv")
EXCLUDED_OUTPUT = os.path.join(OUT_DIR, "iscgem_1970_1972.csv")
MANIFEST = os.path.join(OUT_DIR, "usgs_catalog_manifest.json")

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MAX_RETRIES = 5
BACKOFF = 5                # seconds; doubles each retry
POLITE_PAUSE = 1.0         # seconds between successful requests
TIMEOUT = 120


def query_params(start, end):
    """FDSNWS parameters for one half-open [start, end) window."""
    return {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minlatitude": REGION["minlatitude"],
        "maxlatitude": REGION["maxlatitude"],
        "minlongitude": REGION["minlongitude"],
        "maxlongitude": REGION["maxlongitude"],
        "minmagnitude": MIN_MAG,
        "orderby": "time-asc",
        "limit": 20000,
    }


def download_window(start, end, label):
    """Fetch one window, retrying transient failures. Fatal on exhaustion."""
    params = query_params(start, end)
    delay = BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(USGS_URL, params=params, timeout=TIMEOUT)

            # Check the status code BEFORE the body: an error response also
            # has an empty body and must not be read as zero events.
            if r.status_code == 204:
                print(f"  {label}: 0 events (HTTP 204)")
                return pd.DataFrame()

            if r.status_code == 200:
                if not r.text.strip():
                    print(f"  {label}: 0 events (empty 200 body)")
                    return pd.DataFrame()
                df = pd.read_csv(StringIO(r.text))
                if len(df) >= 20000:
                    sys.exit(f"FATAL {label}: hit the 20,000-event API limit. "
                             f"Split this window into months and re-run.")
                print(f"  {label}: {len(df)} events")
                return df

            # 400 is a malformed query: retrying will not help.
            if r.status_code == 400:
                sys.exit(f"FATAL {label}: HTTP 400 from FDSNWS. "
                         f"Query was: {r.url}")

            print(f"  {label}: HTTP {r.status_code}, "
                  f"attempt {attempt}/{MAX_RETRIES}, waiting {delay}s")

        except requests.RequestException as e:
            print(f"  {label}: {type(e).__name__}, "
                  f"attempt {attempt}/{MAX_RETRIES}, waiting {delay}s")

        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    sys.exit(f"FATAL {label}: failed after {MAX_RETRIES} attempts. "
             f"Aborting rather than writing an incomplete catalog.")


def download_years(years):
    """Fetch a list of calendar years as half-open annual chunks."""
    frames, per_year = [], {}
    for year in years:
        df = download_window(f"{year}-01-01", f"{year + 1}-01-01", str(year))
        per_year[year] = len(df)
        if len(df):
            frames.append(df)
        time.sleep(POLITE_PAUSE)
    assert len(per_year) == len(years), "a year was skipped"
    combined = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame())
    return combined, per_year


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    years = list(range(START_YEAR, END_YEAR + 1))
    print(f"Downloading USGS ComCat catalog, {START_YEAR}-01-01 to "
          f"{END_YEAR + 1}-01-01 (half-open), M >= {MIN_MAG}")
    print("=" * 62)

    df_all, per_year = download_years(years)
    if df_all.empty:
        sys.exit("FATAL: no events retrieved at all.")

    # Half-open windows must not have produced duplicate event ids.
    if "id" in df_all.columns:
        dupes = int(df_all["id"].duplicated().sum())
        if dupes:
            sys.exit(f"FATAL: {dupes} duplicate event ids across chunk "
                     f"boundaries. Chunking is wrong; do not use this file.")
        print("\nBoundary check: no duplicate event ids.")

    # New Year's Eve coverage check. A bare endtime=YYYY-12-31 would drop
    # 31 December of every year; if the half-open fix is working, some
    # 31 December events should be present.
    t = pd.to_datetime(df_all["time"], errors="coerce", utc=True,
                       format="mixed")
    nye = int(((t.dt.month == 12) & (t.dt.day == 31)).sum())
    print(f"Coverage check: {nye} events on 31 December across all years.")

    df_all.to_csv(OUTPUT, index=False)
    digest = sha256(OUTPUT)

    # ---- the excluded pre-1973 interval, for reference only ---------------
    excluded = {"fetched": False}
    if FETCH_EXCLUDED:
        print("\nRetrieving the excluded pre-1973 interval for reference "
              "(not part of the analysis)")
        ex_years = list(range(EXCLUDED_START_YEAR, EXCLUDED_END_YEAR + 1))
        df_ex, ex_per_year = download_years(ex_years)
        if not df_ex.empty:
            df_ex.to_csv(EXCLUDED_OUTPUT, index=False)
            src = (dict(df_ex["net"].value_counts())
                   if "net" in df_ex.columns else {})
            excluded = {
                "fetched": True,
                "file": EXCLUDED_OUTPUT,
                "sha256": sha256(EXCLUDED_OUTPUT),
                "n_events": int(len(df_ex)),
                "n_events_by_year": ex_per_year,
                "contributing_sources": {k: int(v) for k, v in src.items()},
                "mag_min": float(df_ex["mag"].min()),
                "mag_max": float(df_ex["mag"].max()),
                "magnitude_types": (
                    {k: int(v) for k, v in
                     df_ex["magType"].value_counts().items()}
                    if "magType" in df_ex.columns else {}),
                "why_excluded": "No USGS (PDE) solution exists for this "
                                "region before 1973; these events are "
                                "ISC-GEM contributions on the moment "
                                "magnitude scale with a completeness floor "
                                "near M 5.0. Retained for reference so that "
                                "their effect can be tested from the archive.",
            }
            print(f"  wrote {len(df_ex)} events to {EXCLUDED_OUTPUT}")

    manifest = {
        "script": os.path.basename(__file__),
        "retrieved_utc": retrieved,
        "service": USGS_URL,
        "region": REGION,
        "min_magnitude": MIN_MAG,
        "window_start": f"{START_YEAR}-01-01",
        "window_end": f"{END_YEAR + 1}-01-01",
        "window_convention": "half-open [start, end); FDSNWS reads a bare "
                             "date as T00:00:00",
        "start_year_rationale": (
            "ComCat holds no USGS (PDE) solution for this region before "
            "1973. The NEIC became part of the USGS in 1973 and the PDE "
            "bulletin became a Survey product at that time. 1970-1972 is "
            "ISC-GEM only, moment magnitude, completeness floor near M 5.0; "
            "it is retrieved separately rather than analyzed."),
        "n_events": int(len(df_all)),
        "n_events_by_year": per_year,
        "events_on_31_december": nye,
        "contributing_sources": (
            {k: int(v) for k, v in df_all["net"].value_counts().items()}
            if "net" in df_all.columns else {}),
        "magnitude_types": (
            {k: int(v) for k, v in df_all["magType"].value_counts().items()}
            if "magType" in df_all.columns else {}),
        "time_min": str(t.min()),
        "time_max": str(t.max()),
        "mag_min": float(df_all["mag"].min()),
        "mag_max": float(df_all["mag"].max()),
        "output_file": OUTPUT.replace("\\", "/"),
        "output_sha256": digest,
        "excluded_interval": excluded,
        "pandas": pd.__version__,
        "requests": requests.__version__,
        "python": sys.version.split()[0],
        "note": "ComCat is revised over time. Cite the archived copy of "
                "these CSVs and this manifest, not a re-run of the script.",
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 62)
    print(f"Analyzed catalog: {len(df_all)} events")
    print(f"Date range      : {t.min()}  to  {t.max()}")
    print(f"Magnitude range : {df_all['mag'].min():.1f} to "
          f"{df_all['mag'].max():.1f}")
    print(f"SHA-256         : {digest}")
    print(f"Retrieved (UTC) : {retrieved}")
    print(f"Saved           : {OUTPUT}")
    if excluded["fetched"]:
        print(f"Excluded interval: {excluded['n_events']} events -> "
              f"{EXCLUDED_OUTPUT}")
    print(f"Manifest        : {MANIFEST}")
    print("\nFor the Data and Resources section, quote the retrieval date, "
          "the half-open window, the 1973 start year with its rationale, "
          "and the SHA-256 above.")


if __name__ == "__main__":
    main()

