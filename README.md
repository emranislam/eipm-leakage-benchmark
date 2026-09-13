# eipm-leakage-benchmark

## Clustering Persistence Dominates Apparent Precursory Skill: A Leakage-Controlled Benchmark for Short-Term Earthquake Forecasting in the Eastern Indian Plate Margin

### Overview

This project tests whether the apparent short-term precursory skill reported in the earthquake-forecasting work on the Eastern Indian Plate Margin survives a leakage-controlled benchmark, or whether it is instead explained by ordinary aftershock/cluster persistence. It follows directly from open issues flagged in paper one (embargo design defined in sequence rather than calendar time, bootstrap underestimation from stride-1 window overlap, and related reproducibility gaps).

### Repository layout

- `data/` — raw and processed earthquake catalogs used for the benchmark (large CSVs are kept local-only; `data/raw/usgs_catalog_manifest.json` here records retrieval date, query parameters, and SHA-256 checksums for provenance)
- `scripts/` — data pipeline, benchmark, and analysis code
- `manuscript/` — drafts, figures, and manuscript source files
- `output/` — generated tables, figures, and results

### Status

Catalog download script (`scripts/01_download_catalog.py`) written and run: 7,175 events retrieved for the Eastern Indian Plate Margin (10-30°N, 85-98°E), 1973-2025, M≥2.0, via USGS ComCat. No benchmark analysis yet.
