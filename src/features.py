from pathlib import Path
from urllib.parse import urlparse, parse_qsl
import re
import math
import ipaddress

import joblib
import tldextract


# Offline extractor: uses the public suffix list bundled with tldextract,
# so feature extraction never touches the network.
_tld = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "password",
    "bank",
    "confirm",
    "authenticate",
    "wallet",
    "payment"
]

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office", "outlook", "onedrive",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "google", "gmail",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "dhl", "fedex",
    "usps", "adobe", "dropbox", "docusign", "coinbase", "binance", "metamask",
    "linkedin", "yahoo", "ebay", "steam", "roblox"
]

# Registered domains the brands above actually own. A brand name on one of
# these (e.g. outlook.live.com, appleid.apple.com) is not impersonation.
OFFICIAL_BRAND_DOMAINS = {
    "paypal.com", "apple.com", "icloud.com", "microsoft.com", "microsoftonline.com",
    "live.com", "office.com", "office365.com", "outlook.com", "onedrive.com",
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.in", "netflix.com",
    "facebook.com", "instagram.com", "whatsapp.com", "google.com", "gmail.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com", "citi.com",
    "hsbc.com", "hsbc.co.uk", "dhl.com", "fedex.com", "usps.com", "adobe.com",
    "dropbox.com", "docusign.com", "docusign.net", "coinbase.com", "binance.com",
    "metamask.io", "linkedin.com", "yahoo.com", "ebay.com", "steampowered.com",
    "steamcommunity.com", "roblox.com"
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "q-r.to", "t.ly", "tiny.cc",
    "qrco.de", "ead.me", "forms.gle", "app.link", "wl.co"
}

# Free site builders and user-content platforms that phishing kits commonly
# sit on. Matched against the registered domain, or the exact hostname for
# entries like docs.google.com whose parent domain also serves its own pages.
FREE_HOSTING = {
    "weebly.com", "wixsite.com", "wixstudio.com", "web.app", "firebaseapp.com",
    "pages.dev", "workers.dev", "github.io", "netlify.app", "vercel.app",
    "glitch.me", "000webhostapp.com", "blogspot.com", "webflow.io",
    "square.site", "godaddysites.com", "ipfs.io", "infura-ipfs.io",
    "r2.dev", "repl.co", "herokuapp.com", "my-free.website", "webcindario.com",
    "appspot.com", "framer.app", "framer.website", "dweb.link",
    "campaign-archive.com", "hsforms.com", "jotform.com", "duckdns.org",
    "codesandbox.io", "csb.app", "sharepoint.com",
    "sites.google.com", "docs.google.com", "drive.google.com",
    "storage.googleapis.com", "express.adobe.com", "new.express.adobe.com"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "shop", "lat", "click", "online", "site", "live", "icu",
    "buzz", "rest", "cfd", "sbs", "monster", "support", "cyou", "tk", "ml",
    "ga", "cf", "gq", "zip", "mov"
}

DOMAIN_RANKS_PATH = Path(__file__).resolve().parent.parent / "models" / "domain_ranks.joblib"
TOP_RANKS = 100_000
UNRANKED = TOP_RANKS * 10  # log10 lands one full step past the last ranked value

_ranks = None


def _domain_ranks():
    """{registered_domain: Tranco rank} for the top TOP_RANKS, built by src/train.py."""
    global _ranks
    if _ranks is None:
        _ranks = joblib.load(DOMAIN_RANKS_PATH) if DOMAIN_RANKS_PATH.exists() else {}
    return _ranks


def build_domain_ranks(tranco_csv):
    """Map registered domain -> best Tranco rank, keeping the top TOP_RANKS entries."""
    global _ranks
    ranks = {}
    with open(tranco_csv, encoding="utf-8") as f:
        for line in f:
            rank, domain = line.strip().split(",", 1)
            if int(rank) > TOP_RANKS:
                break
            key = _tld(domain.lower()).top_domain_under_public_suffix or domain.lower()
            ranks.setdefault(key, int(rank))
    joblib.dump(ranks, DOMAIN_RANKS_PATH, compress=3)
    _ranks = ranks
    return ranks


_HEX_TOKEN = re.compile(r"[0-9a-f]{16,}")
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_=-]{30,}")


def canonicalize(url):
    """
    Normalize away differences that come from how a dataset was collected
    rather than from the URL itself: scheme, a leading "www.", host case
    and a trailing slash.
    """
    url = url.strip()
    if "://" not in url:
        url = "http://" + url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    rest = (parsed.path or "").rstrip("/")
    if parsed.params:
        rest += ";" + parsed.params
    if parsed.query:
        rest += "?" + parsed.query
    if parsed.fragment:
        rest += "#" + parsed.fragment

    userinfo = ""
    if "@" in parsed.netloc:
        userinfo = parsed.netloc.rsplit("@", 1)[0] + "@"

    port = ""
    try:
        if parsed.port:
            port = ":" + str(parsed.port)
    except ValueError:
        pass

    return userinfo + hostname + port + rest


def _entropy(text):
    if not text:
        return 0.0
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def extract_url_features(url):
    canonical = canonicalize(url)
    parsed = urlparse("http://" + canonical)

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    ext = _tld(hostname)
    registered_domain = ext.top_domain_under_public_suffix
    subdomain = ext.subdomain

    features = {}

    # Basic length features
    features["url_length"] = len(canonical)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)

    # Character counts
    features["num_dots"] = canonical.count(".")
    features["num_hyphens"] = canonical.count("-")
    features["num_slashes"] = canonical.count("/")
    features["num_digits"] = sum(c.isdigit() for c in canonical)
    features["num_special_chars"] = len(
        re.findall(r"[^a-zA-Z0-9]", canonical)
    )

    # Suspicious characters
    features["has_at_symbol"] = int("@" in canonical)
    features["has_query"] = int(bool(query))

    # IP address instead of domain
    try:
        ipaddress.ip_address(hostname)
        features["has_ip"] = 1
    except ValueError:
        features["has_ip"] = 0

    # Structure
    features["num_subdomains"] = len(subdomain.split(".")) if subdomain else 0
    features["path_depth"] = len([p for p in path.split("/") if p])
    features["num_query_params"] = len(parse_qsl(query, keep_blank_values=True))

    # Hostname randomness
    features["host_entropy"] = round(_entropy(hostname), 4)
    features["host_digit_ratio"] = round(
        sum(c.isdigit() for c in hostname) / len(hostname), 4
    ) if hostname else 0.0

    tokens = re.split(r"[^A-Za-z0-9]", canonical)
    features["longest_token_len"] = max((len(t) for t in tokens), default=0)

    # Domain reputation hints that need no network lookup
    features["has_punycode"] = int("xn--" in hostname)
    features["is_shortener"] = int(registered_domain in SHORTENERS)
    features["is_free_hosting"] = int(
        registered_domain in FREE_HOSTING or hostname in FREE_HOSTING
    )
    features["suspicious_tld"] = int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS)

    # A brand name outside the brand's own registered domain,
    # e.g. paypal.secure-login.com or apple-id-support.com
    outside = (subdomain + " " + path + " " + query).lower()
    features["brand_in_subdomain_or_path"] = int(
        registered_domain not in OFFICIAL_BRAND_DOMAINS
        and (
            any(b in outside for b in BRANDS)
            or any(b in ext.domain and ext.domain != b for b in BRANDS)
        )
    )

    # Popularity of the registered domain in the Tranco list (offline lookup).
    # Free hosting and shorteners are unranked: anyone can publish there.
    rank = UNRANKED
    if not (features["is_shortener"] or features["is_free_hosting"]):
        rank = _domain_ranks().get(registered_domain, UNRANKED)
    features["domain_rank_log10"] = round(math.log10(rank), 4)

    # Long hex or base64-like tokens (tracking ids, encoded victim emails)
    features["has_hex_or_base64_token"] = int(
        bool(_HEX_TOKEN.search((path + query).lower()))
        or bool(_LONG_TOKEN.search(path + query))
    )

    # Suspicious keywords
    url_lower = canonical.lower()

    features["has_suspicious_keyword"] = int(
        any(word in url_lower for word in SUSPICIOUS_WORDS)
    )

    return features
