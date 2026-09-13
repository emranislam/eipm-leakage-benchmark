# scripts/

Data download/cleaning, clustering-persistence baselines, embargo/leakage-control logic, and benchmark evaluation code.

- `01_download_catalog.py` — downloads the USGS ComCat catalog for the Eastern Indian Plate Margin (10-30°N, 85-98°E), 1973-2025, M≥2.0, using half-open annual chunks with a boundary-duplicate check; writes `data/raw/usgs_catalog_raw.csv`, the excluded pre-1973 ISC-GEM interval, and a manifest with SHA-256 checksums.

