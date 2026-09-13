# data/

Raw and processed earthquake catalogs (e.g. USGS ComCat pulls) for the leakage-controlled benchmark.

Large CSVs are kept local-only (see `.gitignore`). `data/raw/usgs_catalog_manifest.json` records retrieval date, query parameters, event counts, and SHA-256 checksums so results stay reproducible without committing the raw files.

