"""
Train the phishing URL classifier and write the artifacts in models/.

Run from the project root:

    python -m src.train
"""

import json
import math
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict

from .data import BASE_DIR, RAW_DIR, build_dataset, leakage_report
from .features import TOP_RANKS, UNRANKED, build_domain_ranks, extract_url_features


MODELS_DIR = BASE_DIR / "models"
RANDOM_STATE = 42

# Real pages on their own domains, many with deep paths and login words.
# A model that learned "has a path => phishing" fails most of these.
KNOWN_GOOD_URLS = [
    "https://github.com/psf/requests",
    "https://github.com/login",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://en.wikipedia.org/wiki/Random_forest",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.amazon.com/gp/css/order-history",
    "https://docs.google.com/document/d/1A2b3C4d5E6f7G8h9I0j/edit",
    "https://accounts.google.com/signin",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.paypal.com/signin",
    "https://www.paypal.com/myaccount/summary",
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=abc&response_type=code",
    "https://outlook.live.com/mail/0/",
    "https://www.chase.com/personal/checking",
    "https://secure.chase.com/web/auth/dashboard",
    "https://www.bankofamerica.com/online-banking/sign-in/",
    "https://appleid.apple.com/account/manage",
    "https://www.apple.com/iphone/",
    "https://www.netflix.com/browse",
    "https://www.linkedin.com/in/someone-123456/",
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.reddit.com/r/Python/comments/abc123/some_post_title/",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://scikit-learn.org/stable/modules/ensemble.html#random-forests",
    "https://www.bbc.co.uk/news/technology-12345678",
    "https://www.nytimes.com/2024/05/01/technology/ai-security.html",
    "https://medium.com/@author/how-to-build-a-classifier-1a2b3c4d5e6f",
    "https://www.ebay.com/itm/123456789012",
    "https://store.steampowered.com/app/570/Dota_2/",
]


def featurize(urls):
    return pd.DataFrame([extract_url_features(u) for u in urls])


def candidate_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.08,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def scores(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "legit_recall": round(float(((pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)), 4),
        "phishing_recall": round(float(((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)), 4),
    }


def main():
    df = build_dataset(random_state=RANDOM_STATE)
    print("Dataset:", len(df), "rows")
    print(df.groupby(["label", "source"]).size().rename("rows").to_string())
    leakage = leakage_report(df)

    ranks = build_domain_ranks(RAW_DIR / "tranco.csv")
    print(f"\nDomain ranks: {len(ranks)} registered domains from Tranco top {TOP_RANKS}")

    X = featurize(df["url"])
    ranked_share = (X["domain_rank_log10"] <= math.log10(TOP_RANKS)).groupby(df["label"]).mean().round(3)
    print("Share in Tranco top list by label (0=legit, 1=phishing):", ranked_share.to_dict())
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()

    # No registered domain appears in both train and test
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nTrain: {len(train_idx)}  Test: {len(test_idx)}  (grouped by registered domain)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nCross-validation on train (5-fold, grouped)")
    cv_results = {}
    for name, model in candidate_models().items():
        oof = cross_val_predict(
            model, X_train, y_train, groups=groups[train_idx], cv=cv, method="predict_proba"
        )[:, 1]
        threshold = best_f1_threshold(y_train, oof)
        cv_results[name] = {"threshold": round(threshold, 4), **scores(y_train, oof, threshold)}
        print(f"  {name:<24}{cv_results[name]}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["pr_auc"])
    threshold = cv_results[best_name]["threshold"]
    print(f"\nSelected: {best_name} (threshold {threshold})")

    model = candidate_models()[best_name]
    model.fit(X_train, y_train)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_scores = scores(y_test, test_proba, threshold)
    print("\nHeld-out test")
    print(" ", test_scores)
    print(classification_report(y_test, test_pred, target_names=["Legitimate", "Phishing"], digits=3))
    print("Confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_test, test_pred).tolist())

    # Sanity suite (a): known-good deep links
    good_proba = model.predict_proba(featurize(KNOWN_GOOD_URLS)[X.columns])[:, 1]
    good_flagged = [(u, round(float(p), 3)) for u, p in zip(KNOWN_GOOD_URLS, good_proba) if p >= threshold]
    print(f"\nSanity (a) known-good deep links: {len(KNOWN_GOOD_URLS) - len(good_flagged)}/{len(KNOWN_GOOD_URLS)} legit")
    for u, p in good_flagged:
        print(f"    flagged {p}  {u}")

    # Sanity (b): phishing recall per source on the test set
    test_df = df.iloc[test_idx].assign(pred=test_pred)
    per_source = (
        test_df[test_df["label"] == 1].groupby("source")["pred"].mean().round(4).to_dict()
    )
    print("Sanity (b) test phishing recall by source:", per_source)

    # Sanity (c): homepages only. With path leakage gone, the model has to
    # tell legit from phishing homepages by the hostname alone.
    homepage = (X_test["path_length"] == 0) & (X_test["has_query"] == 0)
    hp_scores = {
        "n_legit": int(((y_test == 0) & homepage).sum()),
        "n_phishing": int(((y_test == 1) & homepage).sum()),
    }
    if hp_scores["n_legit"] and hp_scores["n_phishing"]:
        hp_scores.update(scores(y_test[homepage.to_numpy()], test_proba[homepage.to_numpy()], threshold))
    print("Sanity (c) homepage-only test subset:", hp_scores)

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = (
            pd.Series(model.feature_importances_, index=X.columns)
            .sort_values(ascending=False).round(4).to_dict()
        )
        print("\nFeature importance:", importance)

    explainer = shap.TreeExplainer(model)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "phishing_model.joblib")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_columns.joblib")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    meta = {
        "model": best_name,
        "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "by_label_source": {f"{k[0]}:{k[1]}": int(v) for k, v in df.groupby(["label", "source"]).size().items()},
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "split": "GroupShuffleSplit by registered domain, 80/20",
        },
        "leakage_report": leakage,
        "tranco_ranked_share_by_label": {str(k): float(v) for k, v in ranked_share.items()},
        "cv": cv_results,
        "test": test_scores,
        "sanity": {
            "known_good_legit": f"{len(KNOWN_GOOD_URLS) - len(good_flagged)}/{len(KNOWN_GOOD_URLS)}",
            "known_good_flagged": good_flagged,
            "test_phishing_recall_by_source": per_source,
            "homepage_only": hp_scores,
        },
        "feature_importance": importance,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved artifacts to", MODELS_DIR)


if __name__ == "__main__":
    main()
