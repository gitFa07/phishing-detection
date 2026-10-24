"""
Build the labeled URL dataset used by src/train.py.

Sources (all under data/raw/, which is gitignored):

- phishtank.csv: PhishTank feed, label 1 (verified + online rows only).
- hannousse/dataset_B_05_2020.csv: Hannousse & Yahiouche (2021), Mendeley
  Data c2gw7fy2j4. Legitimate URLs here include real paths, unlike
  homepage-only sources such as Tranco or PhiUSIIL. Both classes are used.
- iscx/Benign_list_big_final.csv (optional): ISCX-URL-2016 benign URLs.
  Used as extra legitimate data when present.
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .features import _tld, canonicalize


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

PHISHTANK_PATH = RAW_DIR / "phishtank.csv"
HANNOUSSE_PATH = RAW_DIR / "hannousse" / "dataset_B_05_2020.csv"
ISCX_BENIGN_PATH = RAW_DIR / "iscx" / "Benign_list_big_final.csv"

LEAKAGE_GAP = 0.3


def load_phishing():
    phish = pd.read_csv(PHISHTANK_PATH)
    phish = phish[(phish["verified"] == "yes") & (phish["online"] == "yes")]
    phish = phish[["url"]].assign(label=1, source="phishtank")

    hannousse = pd.read_csv(HANNOUSSE_PATH, usecols=["url", "status"])
    hannousse = hannousse[hannousse["status"] == "phishing"]
    hannousse = hannousse[["url"]].assign(label=1, source="hannousse")

    return pd.concat([phish, hannousse], ignore_index=True)


def load_legit():
    hannousse = pd.read_csv(HANNOUSSE_PATH, usecols=["url", "status"])
    hannousse = hannousse[hannousse["status"] == "legitimate"]
    frames = [hannousse[["url"]].assign(label=0, source="hannousse")]

    if ISCX_BENIGN_PATH.exists():
        iscx = pd.read_csv(ISCX_BENIGN_PATH, header=None, names=["url"])
        frames.append(iscx.assign(label=0, source="iscx"))

    return pd.concat(frames, ignore_index=True)


def registered_domain(url):
    host = urlparse("http://" + canonicalize(url)).hostname or ""
    return _tld(host).top_domain_under_public_suffix or host


def build_dataset(max_ratio=2, random_state=42):
    """
    Combine sources, remove duplicates and cross-class hosts, and cap the
    phishing:legit ratio. Adds a "group" column (registered domain) used
    for grouped train/test splitting.
    """
    df = pd.concat([load_phishing(), load_legit()], ignore_index=True)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]

    df["canonical"] = df["url"].map(canonicalize)
    df = df.drop_duplicates(subset="canonical")

    df["host"] = df["canonical"].map(lambda u: urlparse("http://" + u).hostname or "")
    hosts_by_label = df.groupby("host")["label"].nunique()
    df = df[~df["host"].isin(hosts_by_label[hosts_by_label > 1].index)]

    legit = df[df["label"] == 0]
    phish = df[df["label"] == 1]

    # Downsample phishing with an equal quota per source, so the older
    # Hannousse phishing URLs aren't swamped by the much larger PhishTank feed
    n_phish = min(len(phish), max_ratio * len(legit))
    quota = n_phish // phish["source"].nunique()
    picked = pd.concat([
        g.sample(n=min(len(g), quota), random_state=random_state)
        for _, g in phish.groupby("source")
    ])
    leftover = phish.drop(picked.index)
    shortfall = n_phish - len(picked)
    if shortfall > 0:
        picked = pd.concat([picked, leftover.sample(n=shortfall, random_state=random_state)])
    phish = picked

    df = pd.concat([legit, phish], ignore_index=True)
    df["group"] = df["url"].map(registered_domain)

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def leakage_report(df):
    """
    Per-class share of surface properties that often differ between data
    sources rather than between phishing and legitimate URLs.
    """
    checks = {
        "has_path": lambda u: urlparse(u if "://" in u else "http://" + u).path.strip("/") != "",
        "has_query": lambda u: bool(urlparse(u if "://" in u else "http://" + u).query),
        "has_www": lambda u: "://www." in u.lower() or u.lower().startswith("www."),
        "https": lambda u: u.lower().startswith("https://"),
        "trailing_slash": lambda u: u.endswith("/"),
    }

    rows = {}
    for name, check in checks.items():
        shares = df.groupby("label")["url"].apply(lambda s: s.map(check).mean())
        rows[name] = {
            "legit": round(float(shares.get(0, 0.0)), 3),
            "phishing": round(float(shares.get(1, 0.0)), 3),
        }
        rows[name]["gap"] = round(abs(rows[name]["legit"] - rows[name]["phishing"]), 3)

    # Raw-URL properties. Features are computed on canonical URLs, so
    # scheme / www / trailing slash gaps are neutralized; path and query are not.
    print("\nLeakage report (raw URLs; share per class)")
    print(f"  {'property':<16}{'legit':>8}{'phishing':>10}{'gap':>7}")
    for name, r in rows.items():
        flag = "  <-- WARNING" if r["gap"] > LEAKAGE_GAP and name in ("has_path", "has_query") else ""
        print(f"  {name:<16}{r['legit']:>8}{r['phishing']:>10}{r['gap']:>7}{flag}")

    return rows
