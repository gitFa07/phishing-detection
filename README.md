# Phishing URL Detection

A machine-learning project that classifies a submitted URL as **Phishing** or **Legitimate**. It includes a FastAPI backend, a trained gradient-boosted classifier, URL feature extraction, SHAP explanations, and a lightweight browser-based frontend.

> **Important:** This project analyzes URL-level features. It does not currently visit the website or inspect its page content. Treat predictions as indicators, not as a guarantee that a website is safe or malicious.

## Contents

- [Features](#features)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Clone the repository](#1-clone-the-repository)
- [Set up and run the backend](#2-set-up-and-run-the-backend)
- [Run the frontend](#3-run-the-frontend)
- [Use the application](#4-use-the-application)
- [Retrain the model](#5-retrain-the-model-optional)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Features

- URL classification using a trained gradient-boosted classifier (scikit-learn `HistGradientBoostingClassifier`).
- Extraction of 24 lexical, structural and domain-popularity URL features, with no network access at prediction time.
- Phishing and legitimate class-probability values, using a decision threshold tuned during training.
- Per-feature SHAP contributions explaining each prediction.
- Browser-based interface for submitting URLs and viewing results.
- FastAPI interactive API documentation.
- A reproducible training script with grouped (by domain) evaluation and leakage checks.

## Technology stack

**Backend**

- Python
- FastAPI and Uvicorn
- Pandas and scikit-learn
- Joblib for loading saved model artifacts

**Frontend**

- HTML
- CSS
- Vanilla JavaScript
- Browser Fetch API

The current frontend is a static page in `frontend/index.html`; it does **not** require Node.js, npm, React, or a frontend build step.

## Project structure

```text
phishing-detection/
├── frontend/
│   └── index.html
├── models/
│   ├── domain_ranks.joblib
│   ├── feature_columns.joblib
│   ├── model_meta.json
│   ├── phishing_model.joblib
│   └── shap_explainer.joblib
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

- `frontend/index.html` — scanner UI and browser-side API requests.
- `src/app.py` — FastAPI application and API endpoints.
- `src/features.py` — extracts features from a submitted URL (used by both training and the API).
- `src/model.py` — loads the saved artifacts and returns predictions with SHAP contributions.
- `src/data.py` — loads and combines the raw datasets for training.
- `src/train.py` — trains, evaluates and saves the model.
- `models/phishing_model.joblib` — trained classifier.
- `models/feature_columns.joblib` — expected model feature order.
- `models/shap_explainer.joblib` — SHAP explainer used to compute per-feature contributions.
- `models/domain_ranks.joblib` — Tranco top-100,000 domain ranks for the popularity feature.
- `models/model_meta.json` — decision threshold, evaluation metrics and dataset summary from the last training run.
- `notebooks/01_dataset_exploration.ipynb` — raw-data exploration and SHAP plots for the current model.

## Prerequisites

Install the following:

- **Git** — to clone the repository.
- **Python 3.12** — recommended for the pinned dependencies in `requirements.txt`.
- A modern web browser.

Check your Python installation:

```bash
python --version
```

On Windows, if `python` does not point to the intended version, try:

```powershell
py -3.12 --version
```

The model artifacts are already included in the repository, so you do not need to train the model to run the application.

## 1. Clone the repository

Replace `<YOUR_REPOSITORY_URL>` with the HTTPS or SSH URL of your Git repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd backend-phishing-detection-main
```

The directory name after cloning depends on the repository name. If your cloned folder has a different name, `cd` into that folder instead.

If you downloaded the project as a ZIP, extract it and open a terminal in the extracted project root—the directory containing `requirements.txt`, `src/`, `models/`, and `frontend/`.

## 2. Set up and run the backend

Run these commands from the **project root**, where `requirements.txt` is located.

### Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails, confirm that the virtual environment is activated and that you are using Python 3.12. Do not install packages globally just to work around a virtual-environment problem.

### Start the API

From the same project root, run:

```bash
python -m uvicorn src.app:app --reload
```

The API should be available at:

- API base URL: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Keep this terminal open while using the frontend.

The health check should return:

```json
{
  "message": "Phishing Detection API is running",
  "model": "hist_gradient_boosting",
  "trained_at": "2026-10-24T09:33:55+00:00"
}
```

## 3. Run the frontend

Open a **second terminal** in the project root. Do not stop the backend terminal.

Run:

```bash
python -m http.server 5500 --directory frontend
```

Then open:

**http://127.0.0.1:5500**

This serves `frontend/index.html` locally. No `npm install` or frontend build is needed.

If your Python version does not support `--directory`, use:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://127.0.0.1:5500`.

The frontend's API address field defaults to `http://127.0.0.1:8000`. If your backend is running at another address, enter that backend base URL in the field at the top of the page.

## 4. Use the application

1. Start the backend and frontend using the instructions above.
2. Open `http://127.0.0.1:5500`.
3. Confirm that the page reports **API connected**.
4. Enter a full URL, such as `https://example.com/login`, or choose one of the example URLs.
5. Select **Check URL**.
6. Review the classification, probability scores, and extracted URL features.

A “Legitimate” result is not proof that a site is safe. Avoid opening suspicious links just to test them.

## 5. Retrain the model (optional)

The trained artifacts are included, so this is only needed to change features or data. Raw data lives in `data/raw/`, which is not committed. Download these files:

| File | Source | Used for |
| --- | --- | --- |
| `data/raw/phishtank.csv` | [PhishTank](https://phishtank.org/developer_info.php) verified online feed (CSV) | Phishing URLs |
| `data/raw/hannousse/dataset_B_05_2020.csv` | Hannousse & Yahiouche (2021), [Mendeley Data `c2gw7fy2j4`](https://data.mendeley.com/datasets/c2gw7fy2j4/3) | Legitimate and phishing URLs |
| `data/raw/tranco.csv` | [Tranco](https://tranco-list.eu/) top-1M list (`rank,domain`, no header) | Domain popularity feature |
| `data/raw/iscx/Benign_list_big_final.csv` *(optional)* | [ISCX-URL-2016](https://www.unb.ca/cic/datasets/url-2016.html) benign URLs (registration required) | Extra legitimate URLs, used automatically when present |

Then, from the project root with the virtual environment active:

```bash
python -m src.train
```

The script prints a leakage report, cross-validation results for a random forest and a gradient-boosting model, held-out test metrics, and sanity checks. It then overwrites the files in `models/`. Restart the API afterwards.

Evaluation is grouped by registered domain, so no domain appears in both the training and test sets. Last run (see `models/model_meta.json`): test PR-AUC 0.976, F1 0.925, legitimate recall 0.83, phishing recall 0.94, and 27 of 30 hand-picked real deep links (GitHub, Wikipedia, PayPal, Amazon and others) classified as legitimate.

Legitimate URLs come from a dataset whose URLs include real paths. The previous version used `https://<domain>` homepages only, so the model learned that any URL with a path was phishing.

## API reference

### `GET /`

Checks whether the backend is running.

```bash
curl http://127.0.0.1:8000/
```

### `POST /predict`

Accepts a URL and returns the prediction, class probabilities, the decision threshold, extracted features, and SHAP contributions. A positive contribution pushes the score toward phishing and a negative one toward legitimate, in log-odds units.

**Request:**

```http
POST http://127.0.0.1:8000/predict
Content-Type: application/json
```

```json
{
  "url": "https://example.com/login"
}
```

**Example response** (`features` and `contributions` shortened; the real response has all 24 features):

```json
{
  "url": "https://example.com/login",
  "prediction": "Legitimate",
  "phishing_probability": 0.18,
  "legitimate_probability": 0.82,
  "threshold": 0.2901,
  "features": {
    "url_length": 17,
    "path_length": 6,
    "num_subdomains": 0,
    "domain_rank_log10": 2.2601,
    "is_free_hosting": 0,
    "has_suspicious_keyword": 1
  },
  "contributions": {
    "url_length": 0.16,
    "path_length": 0.61,
    "num_subdomains": -0.44,
    "domain_rank_log10": -3.19,
    "is_free_hosting": -0.08,
    "has_suspicious_keyword": 1.77
  }
}
```

`prediction` is `Phishing` when `phishing_probability` is at least `threshold`. Lengths and counts are measured on a normalized URL, without the scheme, a leading `www.` or a trailing `/`. The real response values depend on the URL and the trained model. You can also send requests through the interactive Swagger UI at `http://127.0.0.1:8000/docs`.

## Troubleshooting

### The frontend says the API is unavailable

- Confirm the backend terminal is still running.
- Open `http://127.0.0.1:8000/` directly in your browser.
- Check the API address field in the frontend.
- Check the browser developer tools' Console and Network tabs for errors.

### A CORS error appears

The backend currently enables CORS in `src/app.py`. If you change the frontend origin or deployment setup, update the backend CORS configuration to allow the intended frontend origin. For production, avoid allowing every origin; configure only the origins you control.

### Model files cannot be found

Run Uvicorn from the project root, not from inside `src/`. Verify that these files exist:

```text
models/phishing_model.joblib
models/feature_columns.joblib
models/shap_explainer.joblib
models/domain_ranks.joblib
models/model_meta.json
```

### Dependency installation fails

- Check `python --version` and use Python 3.12.
- Activate `.venv` before installing dependencies.
- Upgrade pip inside the virtual environment.
- Read the first package error in the installation output; later errors may be consequences of that first failure.

### Port 8000 or 5500 is already in use

Choose a free port. For example, start the API on another port with:

```bash
python -m uvicorn src.app:app --reload --port 8001
```

Then update the frontend API address to `http://127.0.0.1:8001`. To change the frontend port, pass a different port to `python -m http.server`.

## Limitations

- Classification is based on URL-level lexical and structural features plus an offline domain-popularity lookup; the backend does not fetch the target webpage.
- DNS reputation, certificate information, redirects, and page content are not currently analyzed.
- Popular domains are strongly trusted, so phishing on a compromised popular site can be missed. Pages on user-content platforms (Google Docs/Sites, SharePoint, Weebly and similar) get no popularity credit, so legitimate shared documents may be flagged.
- About 1 in 6 legitimate URLs in the held-out test set is flagged as phishing. Long authentication URLs (for example OAuth redirects) are the most common false positive.
- The legitimate training URLs date from 2020, while most phishing URLs are recent, so some patterns may reflect when the data was collected.
- Model probabilities are estimates from the trained classifier, not absolute certainty.
- The current backend does not include authentication, rate limiting, or production-grade monitoring.
- The current CORS configuration is permissive and should be restricted before production deployment.

## Future improvements

- Add more recent legitimate URLs (for example ISCX-URL-2016 or a crawl of popular sites' internal links).
- Add stronger URL validation and automated tests.
- Add domain-reputation, DNS, certificate, or webpage-content signals.
- Restrict CORS and add rate limiting, logging, and monitoring.
- Add deployment instructions for a hosted frontend and backend.

## License

No license is specified yet. Add a `LICENSE` file before distributing the project under an open-source license.

## Author

**Enoch**

Built as a minor project on **Phishing Website Detection Using Explainable AI**.
