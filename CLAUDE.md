# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Phishing URL classifier: a FastAPI backend serving a pre-trained scikit-learn model (currently `HistGradientBoostingClassifier`), plus a single static HTML page as the frontend. Predictions use lexical/structural features of the URL string plus an offline Tranco domain-popularity lookup. The backend never fetches the target site.

## Commands

Run everything from the project root (model paths and the `src.` package import depend on it). Use the Python 3.12 `.venv/` (`.venv/Scripts/python.exe` on Windows). The system Python has different, unpinned package versions.

```bash
python -m pip install -r requirements.txt          # install deps (into .venv)
python -m uvicorn src.app:app --reload             # API + page on http://127.0.0.1:8000 (Swagger at /docs)
python -m http.server 5500 --directory frontend    # optional standalone frontend on http://127.0.0.1:5500
python -m src.train                                # retrain; overwrites everything in models/
```

Quick check without the server:

```bash
python -c "from src.model import predict_url; print(predict_url('https://example.com/login'))"
```

Deployment: `Dockerfile` + `render.yaml` (Render, free Docker web service, health check `/health`). The image installs `requirements-deploy.txt`, a runtime-only subset of `requirements.txt`. Keep its scikit-learn/shap/numpy/pandas pins equal to `requirements.txt`, or the pickled artifacts may fail to load.

There is no test suite, linter, formatter config, or frontend build step. The closest thing to tests is `python -m src.train`, which prints held-out metrics and sanity suites (`KNOWN_GOOD_URLS` in `src/train.py`, per-source phishing recall, homepage-only subset) and records them in `models/model_meta.json`.

## Architecture

Request flow: `frontend/index.html` → `POST /predict` (`src/app.py`) → `predict_url` (`src/model.py`) → `extract_url_features` (`src/features.py`) → model + SHAP explainer.

- **`src/features.py`** is the single feature extractor for both training and serving. It measures a **canonicalized** URL (scheme, leading `www.` and trailing `/` stripped, host lowercased) so that dataset-collection artifacts don't become features. `domain_rank_log10` reads `models/domain_ranks.joblib`, which `src/train.py` builds from `data/raw/tranco.csv`. Free-hosting, user-content hosts and shorteners (`FREE_HOSTING`, `SHORTENERS`) are deliberately treated as unranked.
- **`src/model.py`** loads `phishing_model.joblib`, `feature_columns.joblib`, `shap_explainer.joblib` and `model_meta.json` at **import time**, so a missing or incompatible artifact breaks app startup, not the first request. It labels a URL Phishing when `phishing_probability >= meta["threshold"]` (tuned on CV, not 0.5), and returns `contributions` (SHAP values toward phishing). It handles both the RF `(n, f, 2)` and HGB `(n, f)` SHAP output shapes. Label encoding: `1` = Phishing, `0` = Legitimate.
- **`src/app.py`** defines `GET /` (serves `frontend/index.html`), `GET /health` (model name and training date from `model_meta.json`) and `POST /predict` (`{"url": str}`). The CORS middleware (allow all origins) is added at the bottom of the file, after the routes.
- **`frontend/index.html`** is self-contained vanilla HTML/CSS/JS. `DEFAULT_API` is the page's own origin, or `http://127.0.0.1:8000` when opened as a file or from port 5500. The `GROUPS`/`BOOLEAN_KEYS` tables give each feature key a display name. Unknown keys fall into an "Other" group, and the "Effect" column renders when `contributions` is present.

### Training (`src/data.py`, `src/train.py`)

- **Data:** `data/` is gitignored, so download sources manually (links in README §5). Phishing comes from `phishtank.csv` (verified + online) and the Hannousse 2020 phishing rows. Legitimate comes from Hannousse 2020 rows, plus ISCX-URL-2016 benign URLs if `data/raw/iscx/Benign_list_big_final.csv` exists. Phishing is downsampled to ≤2× legit with an equal quota per source. Hosts that appear in both classes are dropped.
- **Do not use homepage-only legit sources** (Tranco `https://<domain>`, UCI PhiUSIIL). That is what made the original model learn "has a path ⇒ phishing". `leakage_report()` prints per-class path/query/www/https shares and warns on large path or query gaps.
- **Splits:** train/test and CV are **grouped by registered domain** (`GroupShuffleSplit`, `StratifiedGroupKFold`). Random splits leak same-domain URLs and inflate scores.
- The script compares RF vs HGB on CV PR-AUC, picks the winner, and writes all artifacts and `model_meta.json`.

Any change to `src/features.py` (adding, removing or renaming a feature, or changing how one is computed) requires `python -m src.train` so the model, `feature_columns.joblib` and explainer match. Then update the frontend's `GROUPS`. The notebook (`notebooks/01_dataset_exploration.ipynb`) is exploration only. Its early cells make ~1,000 live HTTP requests, and its final cells load the current artifacts from `src` for SHAP plots.
