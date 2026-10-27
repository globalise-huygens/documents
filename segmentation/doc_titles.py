"""
English titles for documents that have none (subdocuments, unindexed
documents, the volume's own registers of papers), inferred from their text:

    <type>, <place>, <date>         e.g. "Letter, Cochin, 12 November 1748"
    Unknown document, <place>, <date>   when the type is uncertain
    Unknown document                     when nothing is known

  type   the GLOBALISE document type (data/concepts.trig, English label), from a
         classifier on the opening of the first page and the end of the last page,
         trained on documents with an index entry (their type comes from the
         Digitized Indexes); given only when it is at least TYPE_MIN sure.
  place  the place of the closing formula ("Bantam den 16 maij 1743"), named as
         in the VOC gazetteer (data/organisational-structure-voc.xlsx; English
         where it has an English form) or a ship ("on board the ...").
  date   the date of that closing; else the last full date on the last page;
         else the date in the running header of the first pages. A year outside
         the inventory's period is a misreading and left out.

Dates and places are read with the GLOBALISE entity annotations (persons,
places, dates per scan), fetched once and cached in data/entities/. Sender,
addressee and subject are not attempted: a translator or copyist signs too,
and the subject needs more than the first and last page.

  uv run python -m segmentation title-fit                 # train the type classifier (data/titles/)
  uv run python -m segmentation infer-titles [inv ...]    # → inferred_titles.csv, used by import
"""

import concurrent.futures as cf
import gzip
import json
import logging
import os
import pickle
import re
import sqlite3
import urllib.request

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .header_dates import parse_header_date
from .inventory import _parse_list
from .texts import load_texts, normalize_tokens

logger = logging.getLogger("segmentation")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENTITY_DIR = os.path.join(DATA_DIR, "entities")
TITLE_DIR = os.path.join(DATA_DIR, "titles")
ENTITY_URL = "https://objectstore.surf.nl/87435b768620494e8e911c83d1997f24:globalise-data/annotations/entities/{}.json"
INDEX_XLSX = os.path.join(DATA_DIR, "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799) - no linebreaks.xlsx")
CONCEPTS = os.path.join(DATA_DIR, "concepts.trig")
GAZETTEER = os.path.join(DATA_DIR, "organisational-structure-voc.xlsx")
OUT_CSV = "inferred_titles.csv"

TYPE_MIN = 0.5  # type given from this classifier confidence (cross-validated: 84% right for 78% of documents)
TYPE_MIN_DOCS = 150  # types with fewer training documents are not learned
YEAR_SLACK = 2  # years outside the inventory's period by more than this are misreadings
UNKNOWN = "Unknown document"
MONTHS = "January February March April May June July August September October November December".split()

# ── entity annotations ────────────────────────────────────────────────────────


def entity_path(filename: str) -> str:
    inv = filename.split("_")[2] if filename.count("_") >= 3 else "other"
    return os.path.join(ENTITY_DIR, inv, filename + ".json.gz")


def fetch_entities(filename: str) -> bool:
    """Download a scan's entity annotations into the cache (gzip as served). False when unavailable."""
    path = entity_path(filename)
    if os.path.exists(path):
        return True
    try:
        with urllib.request.urlopen(ENTITY_URL.format(filename), timeout=30) as r:
            data = r.read()
    except Exception:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data if data[:2] == b"\x1f\x8b" else gzip.compress(data))
    os.replace(tmp, path)
    return True


def fetch_all(filenames, workers: int = 16):
    todo = [f for f in set(filenames) if f and not os.path.exists(entity_path(f))]
    if todo:
        logger.info("Fetching entity annotations for %d scans", len(todo))
        with cf.ThreadPoolExecutor(workers) as ex:
            list(ex.map(fetch_entities, todo))


def entities(filename: str) -> list[tuple[str, str, int]]:
    """(type, text, position in the normalised transcription) per entity, in text order."""
    path = entity_path(filename) if filename else None
    if not path or not os.path.exists(path):
        return []
    try:
        page = json.loads(gzip.decompress(open(path, "rb").read()))
    except Exception:
        return []
    out = []
    for it in page.get("items", []):
        body = it["body"][0]
        cls = body.get("classified_as", {}).get("id", "")
        text = body.get("label") or (body.get("ascribes_appellation") or {}).get("content") or ""
        for t in it["target"]:
            if t.get("type") == "SpecificResource" and "page-normalized" in t["source"]["id"]:
                pos = next((s["start"] for s in t["selector"] if s["type"] == "TextPositionSelector"), None)
                if pos is not None and text:
                    out.append((cls, text, pos))
    return sorted(out, key=lambda e: e[2])


# ── dates ─────────────────────────────────────────────────────────────────────

YEAR_RE = re.compile(r"\b1[5-8]\d\d\b")


def page_dates(ents):
    """Parsed dates on a page: adjacent date entities joined ("Ao. 1700" + "den 7 Septemb:"); a date
    without a year takes the year of another date on the page. [(HeaderDate, start, end)]"""
    spans = []
    for cls, text, pos in ents:
        if cls != "ner:date":
            continue
        if spans and pos - spans[-1][2] <= 6:
            t, s, _ = spans[-1]
            spans[-1] = (t + " " + text, s, pos + len(text))
        else:
            spans.append((text, pos, pos + len(text)))
    years = [m.group(0) for t, _, _ in spans for m in YEAR_RE.finditer(t)]
    out = []
    for text, s, e in spans:
        d = parse_header_date(text)
        if d is None and not YEAR_RE.search(text) and years:
            d = parse_header_date(f"{text} {years[0]}")
        if d is not None:
            out.append((d, s, e))
    return out


def english_date(d) -> str:
    if d.precision == "day":
        return f"{d.date.day} {MONTHS[d.date.month - 1]} {d.date.year}"
    if d.precision == "month":
        return f"{MONTHS[d.date.month - 1]} {d.date.year}"
    return str(d.date.year)


# ── closing formula: place and date ───────────────────────────────────────────

CLOSING_GAP = 25  # characters between a place and the date that follows it
BETWEEN_OK = re.compile(r"^[\s,.;:'„=\-]*((in\s*'?t\s*)?casteel\s*|adij|ady|desen|dezen|den|in\s*dato|op\s*'?t\s*comptoir|opt\s*compt\W*r)?[\s,.;:'„=\-]*$", re.I)
NOT_CLOSING = re.compile(r"accord\w*|ont[fv]angen|gecollationeert|\bvan\W*$", re.I)  # collation and receipt notes, the next document's dateline
SHIP_RE = re.compile(r"(?:op|aan\s+boord\s+van|in)\s+(?:'?t|het|de)\s+(?:schip|fluit|fluijt|hoeker|chialoup|jagt|pink)\s+((?:'t\s+)?[A-Z][\w']+(?:\s+[A-Z][\w']+){0,2})")


def closing(ents, text):
    """(place as written, date, position of the date) of the last place-then-date on the page, or None."""
    dates = page_dates(ents)
    locs = [(t, p) for c, t, p in ents if c == "ner:loc_name"]
    best = None
    for d, s, _ in dates:
        before = [(t, p) for t, p in locs if p + len(t) <= s and s - (p + len(t)) <= CLOSING_GAP]
        if before:
            t, p = before[-1]
            if not NOT_CLOSING.search(text[max(0, p - 30):p]) and BETWEEN_OK.match(text[p + len(t):s]):
                best = (t, d, s)
                continue
        ship = SHIP_RE.search(text[max(0, s - 120):s])  # "op 't schip 't Huijs Rhensburg den 28 maart 1736"
        if ship:
            best = ("on board the " + ship.group(1).replace("'t ", "").strip(), d, s)
    return best


# ── places: the VOC gazetteer ─────────────────────────────────────────────────

PLACE_NOISE = re.compile(r"^(in\s*'?\s*t|int|in het|ten|op\s*'?\s*t|opt|in de)?\s*(casteel|comptoire?|compt\W*r|fort|gouvernement|gouvernem\W*|commandement|directie|negotie)?\s*(van|de)?\s*", re.I)
EN_OFFICE = re.compile(r"^the\s+(government|directorate|commandement|commandment|governorate|office|factory|residency|settlement)\s+of\s+", re.I)
ALIASES = {"victoria": "Ambon", "nieuwvictoria": "Ambon", "rotterdam": "Makassar", "nassau": "Banda", "nassauw": "Banda",
           "speelwijk": "Bantam", "orange": "Ternate", "oranje": "Ternate", "goedehoop": "Kaap de Goede Hoop",
           "cabodegoedehoop": "Kaap de Goede Hoop", "zeelandia": "Formosa", "geldria": "Pulicat", "cochim": "Cochin",
           "kolombo": "Colombo", "gustavus": "Bengalen", "houglij": "Bengalen", "hougly": "Bengalen",
           "drakenstein": "Kaap de Goede Hoop", "stellenbosch": "Kaap de Goede Hoop", "bescherming": "Cirebon",
           "bata": "Batavia", "batav": "Batavia", "bat": "Batavia"}
SKIP_ORGS = re.compile(r"Raad|Council|Weeskamer|Orphan|Kamer|Chamber|Commissie|Commission|Justitie|Justice", re.I)


class Gazetteer:
    """Place names in closings → the VOC organisation's name, in English where the gazetteer has one."""

    def __init__(self, path: str = GAZETTEER):
        lab = pd.read_excel(path, sheet_name="labels")
        org = pd.read_excel(path, sheet_name="organization_overview")
        lab = lab[~lab["Refers To"].astype(str).str.contains(SKIP_ORGS)]
        name = {}
        for r in org.itertuples():
            name[r.ORG_ID] = re.sub(r"\s*\((area|settlement)\)", "", str(r.organisation_label)).strip()
        for r in lab[lab.Language == "EN"].itertuples():  # "The Government of Malacca" → Malacca
            label = str(r.Label).strip()
            if EN_OFFICE.match(label):
                name[r.ORG_ID] = EN_OFFICE.sub("", label).strip()
        self.keys = {}
        for r in lab.itertuples():
            k = self._key(str(r.Label))
            if len(k) >= 4:
                self.keys.setdefault(k, name.get(r.ORG_ID, str(r.Label)))
        self.english = {self._key(org_label): name[oid] for oid, org_label in zip(org.ORG_ID, org.organisation_label.astype(str)) if oid in name}

    @staticmethod
    def _key(s: str) -> str:
        s = PLACE_NOISE.sub("", s.strip()).lower()
        return re.sub(r"[^a-z]", "", re.sub(r"^fort\s+", "", s))

    def name(self, raw: str) -> str | None:
        if raw.startswith("on board the "):
            return raw
        k = self._key(raw)
        if not k:
            return None
        for alias, target in ALIASES.items():
            if k == alias or (len(alias) >= 6 and alias in k):
                return self.english.get(self._key(target), target)
        for cand in (k, "g" + k[1:] if k.startswith("s") else None):  # HTR reads G as S: Sale → Gale
            if cand and len(cand) >= 4:
                hit = process.extractOne(cand, self.keys.keys(), scorer=fuzz.ratio, score_cutoff=85)
                if hit:
                    return self.keys[hit[0]]
        return None  # not a VOC place the gazetteer knows: mostly HTR noise


# ── document types ────────────────────────────────────────────────────────────


def english_types(concepts: str = CONCEPTS, index: str = INDEX_XLSX) -> dict[str, str]:
    """Dutch GLOBALISE type → its English label, through the type URIs of the Digitized Indexes (cached)."""
    cache = os.path.join(TITLE_DIR, "type_labels.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    from rdflib import Dataset, Namespace

    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    g = Dataset(default_union=True)  # the thesaurus is a named graph
    g.parse(concepts, format="trig")
    en = {str(s): str(o) for s, o in g.subject_objects(skos.prefLabel) if o.language == "en"}
    idx = pd.read_excel(index, usecols=["DOCUMENT TYPE (GLOBALISE)", "DOCUMENT TYPE URI (GLOBALISE)"]).dropna()
    idx = idx[~idx.iloc[:, 0].astype(str).str.contains(r"[;|]")]
    uri = idx.groupby("DOCUMENT TYPE (GLOBALISE)")["DOCUMENT TYPE URI (GLOBALISE)"].agg(lambda s: s.mode().iat[0])
    labels = {nl: en[u] for nl, u in uri.items() if u in en}
    os.makedirs(TITLE_DIR, exist_ok=True)
    json.dump(labels, open(cache, "w"), ensure_ascii=False, indent=1)
    return labels


def type_text(first: str, last: str) -> str:
    """The classifier's input: the opening of the first page and the end of the last page, as marked word keys."""
    return " ".join("A_" + w for w in normalize_tokens(first[:600])) + " " + " ".join("Z_" + w for w in normalize_tokens(last[-500:]))


def fit_types(segments_csv: str = "segments.csv", n: int = 40000):
    """Train the type classifier on segmented documents with one index entry (type from the Digitized Indexes)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    idx = pd.read_excel(INDEX_XLSX, usecols=["ID", "DOCUMENT TYPE (GLOBALISE)"])
    idx["type"] = idx["DOCUMENT TYPE (GLOBALISE)"].astype(str).str.split(r"[;|]").str[0].str.strip()
    typ = {str(k): v for k, v in zip(idx.ID, idx.type) if v not in ("nan", "None", "")}
    seg = pd.read_csv(segments_csv, usecols=["inventory", "kind", "start_scan", "end_scan", "toc_csv_ids"], dtype=str)
    seg = seg[(seg.kind == "toc") & seg.toc_csv_ids.isin(typ.keys())]
    seg = seg.sample(min(n, len(seg)), random_state=5).assign(type=lambda d: d.toc_csv_ids.map(typ))
    keep = seg.type.value_counts()
    seg = seg[seg.type.isin(keep[keep >= TYPE_MIN_DOCS].index)]
    texts, labels = [], []
    for inv, g in seg.groupby("inventory"):
        files = pd.Series(sorted(set(g.start_scan) | set(g.end_scan)))
        t = dict(zip(files, load_texts(inv, files)))
        for r in g.itertuples():
            texts.append(type_text(t.get(r.start_scan, ""), t.get(r.end_scan, "")))
            labels.append(r.type)
    model = make_pipeline(TfidfVectorizer(min_df=3, sublinear_tf=True, ngram_range=(1, 2)), LogisticRegression(C=5, max_iter=3000))
    model.fit(texts, labels)
    os.makedirs(TITLE_DIR, exist_ok=True)
    with open(os.path.join(TITLE_DIR, "doctype.pkl"), "wb") as f:
        pickle.dump(model, f)
    english_types()
    logger.info("Type classifier: %d documents, %d types → %s", len(labels), len(set(labels)), TITLE_DIR)


# ── titles ────────────────────────────────────────────────────────────────────


def title(type_en: str | None, place: str | None, date: str | None) -> str:
    head = type_en[:1].upper() + type_en[1:] if type_en else UNKNOWN
    return ", ".join(x for x in (head, place, date) if x)


def infer(doc: dict, texts: dict, headers: dict, model, labels: dict, gazetteer: Gazetteer, years: tuple) -> dict:
    """Type, place, date and title of one document: doc has first, last, before_last, second (scan filenames)."""
    first, last = texts.get(doc["first"], ""), texts.get(doc["last"], "")
    proba = model.predict_proba([type_text(first, last)])[0]
    k = int(proba.argmax())
    type_nl, conf = model.classes_[k], float(proba[k])
    type_en = labels.get(type_nl) if conf >= TYPE_MIN else None
    lo, hi = years

    def plausible(d):
        return lo is None or lo - YEAR_SLACK <= d.date.year <= hi + YEAR_SLACK

    place = date = None
    source = None
    for f in (doc["last"], doc.get("before_last")):  # the closing, on the last page or the one before
        if not f:
            continue
        found = closing(entities(f), texts.get(f, ""))
        if found:
            raw, d, _ = found
            place = gazetteer.name(raw)
            date = d if plausible(d) else None
            source = "closing"
            break
    if date is None:  # the last full date on the last page
        full = [d for d, s, _ in page_dates(entities(doc["last"])) if d.precision == "day" and plausible(d)]
        if full:
            date, source = full[-1], source or "last date"
    if date is None:  # the running header of the first pages
        for f in (doc["first"], doc.get("second")):
            d = parse_header_date(headers.get(f, "")) if f else None
            if d is not None and plausible(d):
                date, source = d, source or "header"
                break
    date_text = english_date(date) if date is not None else None
    return {"title": title(type_en, place, date_text), "type": type_en, "type_nl": type_nl, "type_conf": round(conf, 2),
            "place": place, "date": date.date.isoformat() if date is not None else None,
            "date_precision": date.precision if date is not None else None, "date_text": date_text, "found_in": source}


def infer_titles(segments_csv: str = "segments.csv", inventories: list[str] | None = None, out: str = OUT_CSV, conn=None):
    """Titles for every segment without one (no index entry, court case or General Missive) → out CSV."""
    with open(os.path.join(TITLE_DIR, "doctype.pkl"), "rb") as f:
        model = pickle.load(f)
    labels = english_types()
    gazetteer = Gazetteer()
    seg = pd.read_csv(segments_csv, dtype=str)
    if inventories:
        seg = seg[seg.inventory.isin(inventories)]
    todo = seg[(seg.kind != "non-document") & seg.title.isna() & seg.toc_csv_ids.isna()]
    years = {r[0]: (int(r[1][:4]) if r[1] else None, int(r[2][:4]) if r[2] else None)
             for r in conn.execute("SELECT inventory_number, date_start, date_end FROM inventory")}
    rows = []
    groups = list(todo.groupby("inventory", sort=False))
    for n, (inv, g) in enumerate(groups, 1):
        inv_id = conn.execute("SELECT id FROM inventory WHERE inventory_number = ?", (inv,)).fetchone()[0]
        scans = conn.execute(
            "SELECT s.filename, p.header FROM scan s LEFT JOIN page p ON p.scan_id = s.id WHERE s.inventory_id = ? ORDER BY s.scan_order, s.filename",
            (inv_id,)).fetchall()
        names = list(dict.fromkeys(f for f, _ in scans))
        headers = {f: _parse_list(h) for f, h in scans if h}
        pos = {f: i for i, f in enumerate(names)}
        docs = []
        for r in g.itertuples():
            a, b = pos[r.start_scan], pos[r.end_scan]
            docs.append((r, {"first": r.start_scan, "last": r.end_scan, "before_last": names[b - 1] if b - 1 >= a else None,
                             "second": names[a + 1] if a + 1 <= b else None}))
        files = {d[k] for _, d in docs for k in ("first", "last", "before_last", "second") if d[k]}
        fetch_all(f for _, d in docs for f in (d["last"], d["before_last"]) if f)
        texts = dict(zip(sorted(files), load_texts(inv, pd.Series(sorted(files)))))
        for r, d in docs:
            rows.append({"inventory": inv, "segment": r.segment, "kind": r.kind, "start_scan": r.start_scan, "end_scan": r.end_scan,
                         **infer(d, texts, headers, model, labels, gazetteer, years.get(inv, (None, None)))})
        if n % 100 == 0 or n == len(groups):
            logger.info("%d/%d inventories, %d titles", n, len(groups), len(rows))
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)
