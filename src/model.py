import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from .features import extract_url_features


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Artifacts written by src/train.py
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.joblib"
EXPLAINER_PATH = BASE_DIR / "models" / "shap_explainer.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"



# Load saved artifacts
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
explainer = joblib.load(EXPLAINER_PATH)
meta = json.loads(META_PATH.read_text())

# Decision threshold chosen on cross-validation, not a fixed 0.5
threshold = meta["threshold"]



def _phishing_contributions(X):
    """
    SHAP value of each feature toward the phishing class.
    Random forests return one array per class; gradient boosting returns
    a single array for the positive class.
    """
    values = np.asarray(explainer.shap_values(X))

    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    return {
        name: round(float(v), 4)
        for name, v in zip(feature_columns, values)
    }


def predict_url(url):
    """
    Extract features from a URL and return
    the model prediction, probabilities and SHAP contributions.
    """

    # Extract features
    features = extract_url_features(url)

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # Make sure feature order matches training
    X = X[feature_columns]

    # Probabilities
    probabilities = model.predict_proba(X)[0]

    phishing_probability = float(probabilities[1])
    legitimate_probability = float(probabilities[0])

    # Label
    label = "Phishing" if phishing_probability >= threshold else "Legitimate"

    return {
        "url": url,
        "prediction": label,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "threshold": threshold,
        "features": features,
        "contributions": _phishing_contributions(X),
    }
