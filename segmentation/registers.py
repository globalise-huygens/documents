"""
Handwritten registers ('Register der brieven en papieren ...'): where they
are, and (rows.py) what they list.

Detection is a page classifier (logistic regression) on
  - the words of the page (normalised tokens, unigrams and bigrams, tf-idf),
  - text counts: a register heading near the top, folio ranges ('115 a 122'),
    dittos, 'in dato', item numbers ('N:o 3'), 'ingenaaijt' / 'niet ontfangen',
  - layout from the PageXML: number of lines and rows, line length, share of
    lines starting or ending with a number, rows split into several line
    fragments (wide gaps between dittos), spread of line starts (columns),
  - the same counts and layout of the scans before and after it.
It is trained on the validated inventories: their 'Table of contents' pages
and the pages of validated documents whose OBP title is a register of papers
(OBP indexes list many registers as documents). Reviewed scans
(register_review table, /review/registers in the app) are added to the
training data when present.

Scans scoring >= threshold form register ranges (consecutive scans; a blank
scan in between does not break a range): register_ranges.csv.
"""

import glob
import logging
import os
import pickle
import re
import sqlite3
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from lxml import etree

from .texts import TEXT_DIR, normalize_tokens

logger = logging.getLogger("segmentation")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEATURE_DIR = os.path.join(DATA_DIR, "registers", "features")
MODEL_PATH = os.path.join(DATA_DIR, "registers", "model.pkl")
PAGEXML_DIR = os.environ.get("SEGMENTATION_PAGEXML_DIR", "/Volumes/HDE0090")
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
TYPE_RE = re.compile(r"type:([^;]+);")

HEAD = re.compile(r"regi[sz]t(er|re)|l[iy]j?st\s+(der|van)|inhoud|specificatie|notitie\s+(der|van)\s+(de\s+)?(brieven|papieren|stukken)|"
                  r"(brieven|papieren)\s*(en|&|\+)\s*(papieren|brieven)|bladw[iy]", re.I)
RANGE = re.compile(r"\b(\d{1,4})\s*[„\"'aà,\-–.]{1,3}\s*(\d{1,4})\b")
DITTO = re.compile(r"\bdito\b|\bd[:\.]?o\b|„|_o\b|idem\b", re.I)
DATO = re.compile(r"\b(in|de|sub)\s*dato\b", re.I)
ITEM = re.compile(r"\bN[:\.]?\s*[o°]\W{0,2}\s*\d{1,3}\b")
SEWN = re.compile(r"inge?na[aeoi]{0,2}[iy]?[jy]?[dt]|niet\s+ontfangen", re.I)
START_NUM = re.compile(r"^\W{0,3}(f\W{0,3}|fo\W{0,2}|n\W{0,2}[o°]\W{0,2})?\d{1,4}\b", re.I)
END_NUM = re.compile(r"\b\d{1,4}\W{0,3}$")

COUNT_COLS = ["words", "head", "ranges", "dittos", "dato", "items", "sewn"]
LAYOUT_COLS = ["n_lines", "med_len", "start_num", "end_num", "ditto_lines", "rows", "frag_rows", "x_spread", "short_lines"]
ROW_GAP = 0.008  # line centres closer than this (share of page height) are one row


# ── features ──────────────────────────────────────────────────────────────────


def text_counts(text: str) -> dict:
    text = text or ""
    return {
        "words": len(text.split()),
        "head": bool(HEAD.search(text[:300])),
        "ranges": sum(1 for m in RANGE.finditer(text) if 0 < int(m.group(2)) - int(m.group(1)) <= 300),
        "dittos": len(DITTO.findall(text)),
        "dato": len(DATO.findall(text)),
        "items": len(ITEM.findall(text)),
        "sewn": len(SEWN.findall(text)),
    }


def page_lines(xml: bytes) -> tuple[list[dict], float, float]:
    """Text lines of a PageXML page: x0, x1, y (centre), height (shares of the page), text, region type."""
    root = etree.fromstring(xml)
    page = root.find("p:Page", NS)
    W, H = float(page.get("imageWidth") or 1), float(page.get("imageHeight") or 1)
    out = []
    for region in root.iterfind(".//p:TextRegion", NS):
        m = TYPE_RE.search(region.get("custom") or "")
        rtype = m.group(1) if m else ""
        for line in region.iterfind("p:TextLine", NS):
            te = line.find("p:TextEquiv/p:Unicode", NS)
            text = ((te.text if te is not None else "") or "").strip()
            coords = line.find("p:Coords", NS)
            if not text or coords is None:
                continue
            xy = np.array([[float(v) for v in p.split(",")] for p in coords.get("points").split()])
            base = line.find("p:Baseline", NS)
            by = np.array([[float(v) for v in p.split(",")] for p in base.get("points").split()])[:, 1].mean() if base is not None else xy[:, 1].mean()
            out.append({"x0": xy[:, 0].min() / W, "x1": xy[:, 0].max() / W, "y": by / H, "y0": xy[:, 1].min() / H, "y1": xy[:, 1].max() / H,
                        "text": text, "region": rtype})
    return out, W, H


def layout_features(lines: list[dict]) -> dict:
    if not lines:
        return {c: 0.0 for c in LAYOUT_COLS}
    n = len(lines)
    lens = [len(l["text"]) for l in lines]
    ys = np.sort([l["y"] for l in lines])
    rows = 1 + int((np.diff(ys) > ROW_GAP).sum())
    return {
        "n_lines": n,
        "med_len": float(np.median(lens)),
        "start_num": sum(bool(START_NUM.search(l["text"])) for l in lines) / n,
        "end_num": sum(bool(END_NUM.search(l["text"])) for l in lines) / n,
        "ditto_lines": sum(bool(DITTO.search(l["text"])) for l in lines) / n,
        "rows": rows,
        "frag_rows": (n - rows) / rows,
        "x_spread": float(np.std([l["x0"] for l in lines])),
        "short_lines": sum(x < 25 for x in lens) / n,
    }


def pagexml_names(inventory: str) -> tuple[zipfile.ZipFile | None, dict]:
    path = os.path.join(PAGEXML_DIR, f"{inventory}.zip")
    if not os.path.exists(path):
        return None, {}
    z = zipfile.ZipFile(path)
    return z, {os.path.basename(n)[:-4]: n for n in z.namelist() if n.endswith(".xml")}


def inventory_features(inventory: str, refresh: bool = False) -> pd.DataFrame:
    """Counts and layout per scan of an inventory (cached in FEATURE_DIR)."""
    path = os.path.join(FEATURE_DIR, f"{inventory}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    folder = os.path.join(TEXT_DIR, f"inv={inventory}")
    texts = pd.read_parquet(folder, columns=["filename", "text"]).sort_values("filename") if os.path.isdir(folder) else pd.DataFrame(columns=["filename", "text"])
    z, names = pagexml_names(inventory)
    rows = []
    for fn, text in zip(texts["filename"], texts["text"]):
        r = {"inventory": inventory, "filename": fn, **text_counts(text)}
        try:
            r.update(layout_features(page_lines(z.read(names[fn]))[0]) if z is not None and fn in names else {c: np.nan for c in LAYOUT_COLS})
        except Exception:
            r.update({c: np.nan for c in LAYOUT_COLS})
        rows.append(r)
    df = pd.DataFrame(rows, columns=["inventory", "filename"] + COUNT_COLS + LAYOUT_COLS)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _features_one(inv: str) -> int:
    return len(inventory_features(inv))


def cache_features(inventories: list[str], workers: int = 6):
    todo = [i for i in inventories if not os.path.exists(os.path.join(FEATURE_DIR, f"{i}.parquet"))]
    logger.info("Register features: %d of %d inventories to compute", len(todo), len(inventories))
    t0, scans = time.time(), 0
    with Pool(workers) as pool:
        for k, n in enumerate(pool.imap_unordered(_features_one, todo, chunksize=2), 1):
            scans += n
            if k % 100 == 0 or k == len(todo):
                logger.info("  %d/%d inventories, %d scans (%.0fs; about %.0f min to go)", k, len(todo), scans, time.time() - t0,
                            (time.time() - t0) / k * (len(todo) - k) / 60)


def load_features(inventories: list[str]) -> pd.DataFrame:
    """Features of these inventories with the tokens of each scan, in scan order."""
    parts = []
    for inv in inventories:
        f = inventory_features(inv)
        folder = os.path.join(TEXT_DIR, f"inv={inv}")
        tx = pd.read_parquet(folder, columns=["filename", "text"]).set_index("filename")["text"] if os.path.isdir(folder) else pd.Series(dtype=str)
        f["tokens"] = [" ".join(normalize_tokens(tx.get(fn) or "", prefix=6)) for fn in f["filename"]]
        parts.append(f)
    return pd.concat(parts, ignore_index=True).sort_values(["inventory", "filename"]).reset_index(drop=True)


def numeric(df: pd.DataFrame) -> np.ndarray:
    w = np.maximum(df["words"].to_numpy(dtype=float), 1)
    base = pd.DataFrame({
        "log_words": np.log1p(df["words"]), "head": df["head"].astype(float),
        **{c: df[c] / w * 100 for c in ("ranges", "dittos", "dato", "items", "sewn")},
        **{c: df[c].fillna(0).astype(float) for c in LAYOUT_COLS},
    })
    base["n_lines"], base["rows"] = np.log1p(base["n_lines"]), np.log1p(base["rows"])
    neighbours = [base.groupby(df["inventory"].to_numpy()).shift(k).fillna(0).add_suffix(f"_{'prev' if k == 1 else 'next'}") for k in (1, -1)]
    return pd.concat([base] + neighbours, axis=1).to_numpy(dtype=float)


# ── training data, model ──────────────────────────────────────────────────────


REGISTER_TITLE = re.compile(r"(?i)^\s*(register\s*\.?\s*$|register\s+(der|van)\s+(alle\s+)?(de\s+|sodanige\s+|soodanige\s+|zodanige\s+)?"
                            r"(brieven|papieren|stukken|missiven|pacquetten|bijlagen|documenten|diverse|boeken))")


def review_labels(conn: sqlite3.Connection) -> pd.DataFrame:
    """Reviewed scans: filename, inventory, label (1 register page, 0 not), stratum, weight (stratum size / labelled in it)."""
    try:
        rev = pd.read_sql("SELECT filename, inventory, label AS review, stratum, stratum_size FROM register_review WHERE label IN ('start', 'continuation', 'not')", conn)
    except Exception:
        return pd.DataFrame(columns=["filename", "inventory", "label", "stratum", "weight"])
    rev["label"] = rev["review"].isin(["start", "continuation"]).astype(int)
    rev["weight"] = rev["stratum_size"] / rev.groupby("stratum")["filename"].transform("size")
    return rev[["filename", "inventory", "label", "stratum", "weight"]]


def training_labels(conn: sqlite3.Connection, use_review: bool = True, exclude_inventories=()) -> pd.DataFrame:
    """filename, inventory, label (1 register, 0 not) of the validated inventories, plus reviewed scans
    (except those of exclude_inventories)."""
    rows = []
    for path in glob.glob(os.path.join(DATA_DIR, "* - Document Segmentation.csv")):
        inv = os.path.basename(path).split(" ")[0]
        d = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
        col = next(c for c in d.columns if "non-document" in c)
        toc = set(d[d[col].fillna("").str.lower().str.contains("content")].iloc[:, 0])
        rows += [(fn, inv, int(fn in toc)) for fn in d.iloc[:, 0]]
    lab = pd.DataFrame(rows, columns=["filename", "inventory", "label"]).drop_duplicates("filename")
    t = pd.read_sql("SELECT s.filename, d.title FROM page2document p2d JOIN page p ON p.id = p2d.page_id JOIN scan s ON s.id = p.scan_id "
                    "JOIN document d ON d.id = p2d.document_id WHERE p2d.source = 'SEGMENTATION' AND d.title IS NOT NULL", conn)
    register_pages = set(t[t["title"].map(lambda x: bool(REGISTER_TITLE.match(x)))]["filename"])
    lab.loc[lab["filename"].isin(register_pages), "label"] = 1
    if use_review:
        rev = review_labels(conn)
        rev = rev[~rev["inventory"].isin(set(exclude_inventories))]
        lab = pd.concat([lab[~lab["filename"].isin(rev["filename"])], rev[["filename", "inventory", "label"]]], ignore_index=True)
    return lab


def train(lab: pd.DataFrame) -> dict:
    """The classifier, trained on labelled scans (filename, inventory, label)."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X = load_features(sorted(lab["inventory"].unique())).merge(lab[["filename", "label"]], on="filename")
    X = X[X["words"] > 0]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=50000)
    N = numeric(X)
    mu, sd = N.mean(0), N.std(0) + 1e-9
    model = LogisticRegression(C=1, class_weight="balanced", max_iter=3000)
    model.fit(sparse.hstack([vec.fit_transform(X["tokens"]), (N - mu) / sd]).tocsr(), X["label"])
    return {"vectorizer": vec, "mu": mu, "sd": sd, "model": model, "n": len(X), "positives": int(X["label"].sum())}


def fit(conn: sqlite3.Connection, path: str = MODEL_PATH):
    m = train(training_labels(conn))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(m, f)
    logger.info("Register model: %d scans (%d registers) → %s", m["n"], m["positives"], path)


def score(inventories: list[str], path: str = MODEL_PATH, model: dict | None = None) -> pd.DataFrame:
    """Register probability per scan (0 for scans without text)."""
    from scipy import sparse

    if model is None:
        with open(path, "rb") as f:
            model = pickle.load(f)
    m = model
    out = []
    for k in range(0, len(inventories), 100):
        X = load_features(inventories[k : k + 100])
        Xs = sparse.hstack([m["vectorizer"].transform(X["tokens"]), (numeric(X) - m["mu"]) / m["sd"]]).tocsr()
        X["score"] = np.where(X["words"] > 0, m["model"].predict_proba(Xs)[:, 1], 0.0)
        out.append(X.drop(columns=["tokens"]))
        logger.info("  scored %d/%d inventories", min(k + 100, len(inventories)), len(inventories))
    return pd.concat(out, ignore_index=True)


CONTINUE_THRESHOLD = 0.3  # register-evaluate (2026-10): threshold 0.9 + continuation 0.3 → precision ~0.95, recall ~0.59
BACK = False


def ranges(scores: pd.DataFrame, threshold: float = 0.5, continue_threshold: float | None = CONTINUE_THRESHOLD, back: bool = BACK) -> pd.DataFrame:
    """
    Consecutive scans scoring >= threshold (a blank scan in between does not
    break a range). With continue_threshold, a range runs on into the following
    scans while they score >= continue_threshold (register continuation pages
    look less like registers than first pages); with back, also one scan back.
    """
    out = []
    low = threshold if continue_threshold is None else continue_threshold
    for inv, g in scores.sort_values(["inventory", "filename"]).groupby("inventory", sort=False):
        sc = g["score"].to_numpy()
        on, cont, blank, fns, n = sc >= threshold, sc >= low, (g["words"].to_numpy() == 0), g["filename"].to_numpy(), len(g)
        i = 0
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and (cont[j + 1] or (blank[j + 1] and j + 2 < n and cont[j + 2])):
                j += 1
            if back and i > 0 and cont[i - 1] and (not out or out[-1]["inventory"] != inv or out[-1]["last_scan"] < fns[i - 1]):
                i -= 1
            out.append({"inventory": inv, "first_scan": fns[i], "last_scan": fns[j], "n_scans": j - i + 1,
                        "max_score": round(float(sc[i : j + 1].max()), 3), "mean_score": round(float(sc[i : j + 1].mean()), 3),
                        "position": round(i / n, 3)})
            i = j + 1
    return pd.DataFrame(out)


# ── review sample ─────────────────────────────────────────────────────────────


def review_sample(scores: pd.DataFrame, n: int = 200, seed: int = 1) -> pd.DataFrame:
    """
    Scans to label by hand, stratified so that precision and recall can both be
    estimated: high, medium and low scores, scans near the start of a volume
    (where registers usually are) and scans next to high-scoring ones (range
    edges, continuation pages).
    """
    rng = np.random.default_rng(seed)
    s = scores[scores["words"] > 0].copy()
    s["pos"] = s.groupby("inventory").cumcount()
    s["prev_score"] = s.groupby("inventory")["score"].shift(1).fillna(0)
    strata = {
        "score >= 0.9": s["score"] >= 0.9,
        "0.5 <= score < 0.9": s["score"].between(0.5, 0.9, inclusive="left"),
        "0.1 <= score < 0.5": s["score"].between(0.1, 0.5, inclusive="left"),
        "score < 0.1, first 15 scans": (s["score"] < 0.1) & (s["pos"] < 15),
        "score < 0.1, after a register scan": (s["score"] < 0.1) & (s["prev_score"] >= 0.9),
        "score < 0.1, elsewhere": (s["score"] < 0.1) & (s["pos"] >= 15) & (s["prev_score"] < 0.9),
    }
    per = n // len(strata)
    parts = []
    for name, mask in strata.items():
        g = s[mask]
        take = g.iloc[rng.choice(len(g), size=min(per, len(g)), replace=False)] if len(g) else g
        parts.append(take.assign(stratum=name, stratum_size=int(mask.sum())))
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    return out[["filename", "inventory", "score", "stratum", "stratum_size"]].reset_index(drop=True)


# ── evaluation on the reviewed scans ──────────────────────────────────────────


def _in_ranges(rng: pd.DataFrame, filenames: pd.Series) -> np.ndarray:
    spans = {}
    for r in rng.itertuples():
        spans.setdefault(r.inventory, []).append((r.first_scan, r.last_scan))
    return np.array([any(a <= f <= b for a, b in spans.get(f.split("_")[2], [])) for f in filenames])


def evaluate(conn: sqlite3.Connection, folds: int = 5, thresholds=(0.5, 0.7, 0.9), continue_thresholds=(None, 0.3, 0.2, 0.1, 0.05)) -> pd.DataFrame:
    """
    Precision and recall of register ranges for all scans of the inventories
    without an index, estimated from the reviewed scans (weighted by stratum
    size). The reviewed inventories are split into folds; each fold is scored by
    a model trained on the validated inventories plus the reviewed scans of the
    other folds ('with review') or without any reviewed scans ('without review').
    """
    rev = review_labels(conn)
    invs = sorted(rev["inventory"].unique())
    fold_of = {inv: k % folds for k, inv in enumerate(np.random.default_rng(0).permutation(invs))}
    base = train(training_labels(conn, use_review=False))
    scored = {"without review": [], "with review": []}
    for k in range(folds):
        fold_invs = [i for i in invs if fold_of[i] == k]
        model = train(training_labels(conn, use_review=True, exclude_inventories=fold_invs))
        scored["with review"].append(score(fold_invs, model=model))
        scored["without review"].append(score(fold_invs, model=base))
        logger.info("  fold %d/%d: %d inventories", k + 1, folds, len(fold_invs))
    rows = []
    for variant, parts in scored.items():
        sc = pd.concat(parts, ignore_index=True)
        for thr in thresholds:
            for ct in continue_thresholds:
                for back in (False, True):
                    if ct is None and back:
                        continue
                    pred = _in_ranges(ranges(sc, thr, ct, back), rev["filename"])
                    y, w = rev["label"].to_numpy() == 1, rev["weight"].to_numpy()
                    tp, fp, fn = (w * (pred & y)).sum(), (w * (pred & ~y)).sum(), (w * (~pred & y)).sum()
                    rows.append({"model": variant, "threshold": thr, "continue": ct, "back": back,
                                 "precision": round(tp / max(tp + fp, 1e-9), 2), "recall": round(tp / max(tp + fn, 1e-9), 2),
                                 "est_register_scans": int(tp + fn), "est_predicted_scans": int(tp + fp),
                                 "sample_tp": int((pred & y).sum()), "sample_fp": int((pred & ~y).sum()), "sample_fn": int((~pred & y).sum())})
    return pd.DataFrame(rows)
