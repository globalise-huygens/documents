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
  place  the place of the closing formula ("Bantam den 16 maij 1743"): found
         with the VOC gazetteer (data/organisational-structure-voc.xlsx) by
         its exact labels, else with the GLOBALISE places dataset (exact
         matches of its names and spelling variants), else with the gazetteer
         again by the closest label (fuzzy, for misreadings), or a ship ("on
         board the ..."). An exact name goes before a close one: Manaar is a
         variant of Mannar Island, though closer to the gazetteer's Manapaar.
         A name of several places in the dataset (Hollandia: forts on Banda,
         Saparua and Ternate) is the one nearest the inventory's settlements,
         when it clearly is; else none. The dataset's GAVOC atlas names are
         not used (islands named like forts and ships: Leijden, Samar).
         Written as in the text with its Dutch (VOC) name in brackets where they
         differ ("Gale (Galle)", "Rotterdam (Makassar)"): the settlement's name
         where the place is one of the settlements ("Kaap de Goede Hoop", "Java's
         (Noord-) Oostkust"), else the gazetteer's name of the VOC office; a
         place only the places dataset knows is given as written. "Goede Hoop"
         alone is not a name: the castle ("Casteel de Goede Hoop") or the Cape.
         The GLOB id is that of the places dataset, made current: retired ids
         (deprecated_uris), the settlements' chosen ids (SETTLEMENT_IDS), and
         ids whose old meaning moved to a new record (updated_uris).
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
PLACES = os.path.join(DATA_DIR, "places_v3_complete_20261125.xlsx")
# The GLOB id of each settlement whose id changed: the team's choices (data/settlement-place-check.xlsx,
# column Choice), Sumatra's Westkust as its own VOC area (not the island), and Java's Oostkust merged into
# Java's (Noord-) Oostkust. The chosen ids map to themselves: they go before updated_uris (Siam, Jambi).
SETTLEMENT_IDS = {
    "GLOB2_894": "GLOB2_894_HUM_1",   # Batavia
    "GLOB_800": "GLOB_800_HUM_1",     # Ceylon
    "GLOB2_899": "GLOB2_899_HUM_1",   # Bantam
    "GLOB_796": "GLOB_796_HUM_1",     # Bengalen
    "GLOB2_900": "GLOB2_900_HUM_3",   # Ternate
    "GLOB_593": "GLOB_593_HUM_1",     # Cheribon
    "GLOB2_947": "GLOB_434",          # Sumatra's Westkust
    "GLOB_123": "GLOB2_1848",         # Java's Oostkust -> Java's (Noord-) Oostkust
    "GLOB_539": "GLOB_539_HUM_2",     # Siam (kingdom), as in updated_uris (the team chose the general record GLOB_539 at first)
    "GLOB_533": "GLOB_533",           # Jambi: the general record, not updated_uris' Kota Jambi
}
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
PLACE_PREFIX = re.compile(r"^((in\s*'?\s*t|int|in het|ten|op\s*'?\s*t|opt|in de)\b\W*)?((casteel|comptoire?|compt\W*r|fort|gouvernement|gouvernem\W*|commandement|directie|negotie)\b\W*)?((van|de)\b\s*)?", re.I)  # PLACE_NOISE whole words only: Deshima keeps its De
ALIASES = {"victoria": "Ambon", "nieuwvictoria": "Ambon", "rotterdam": "Makassar", "nassau": "Banda", "nassauw": "Banda",
           "speelwijk": "Bantam", "orange": "Ternate", "oranje": "Ternate", "goedehoop": "Kaap de Goede Hoop",
           "cabodegoedehoop": "Kaap de Goede Hoop", "zeelandia": "Formosa", "geldria": "Pulicat", "cochim": "Cochin",
           "cochinchina": "Quinam", "cochimchina": "Quinam", "kolombo": "Colombo", "gustavus": "Bengalen", "houglij": "Bengalen", "hougly": "Bengalen",
           "drakenstein": "Kaap de Goede Hoop", "stellenbosch": "Kaap de Goede Hoop", "bescherming": "Cirebon",
           "bata": "Batavia", "batav": "Batavia", "bat": "Batavia"}
SKIP_ORGS = re.compile(r"Raad|Council|Weeskamer|Orphan|Kamer|Chamber|Commissie|Commission|Justitie|Justice", re.I)


class Gazetteer:
    """Place names in closings → the VOC organisation: its name (in English where the
    gazetteer has one) and the GLOB id of its place at a given year."""

    def __init__(self, path: str = GAZETTEER, places: "Places | None" = None):
        self.exact = set(places.by_key) if places is not None else set()  # names in the places dataset: not fuzzy-matched here (Tegenepatnam is not Tengapatnam)
        lab = pd.read_excel(path, sheet_name="labels")
        org = pd.read_excel(path, sheet_name="organization_overview")
        loc = pd.read_excel(path, sheet_name="locations")
        lab = lab[~lab["Refers To"].astype(str).str.contains(SKIP_ORGS)]
        lab = lab[[(o, str(lb).strip()) not in GAZETTEER_WRONG_LABELS for o, lb in zip(lab.ORG_ID, lab.Label)]]
        self.name_of = {}  # organisation → its Dutch name: "Malakka (Comptoir)" → Malakka
        for r in lab[lab.Language == "NL"].dropna(subset=["Label"]).itertuples():  # offices without a name of their own
            self.name_of.setdefault(r.ORG_ID, str(r.Label).strip())
        for r in org.dropna(subset=["organisation_label"]).itertuples():
            self.name_of[r.ORG_ID] = re.sub(r"\s*\((area|settlement|comptoir)\)", "", str(r.organisation_label), flags=re.I).strip()
        self.keys = {}  # label key → organisation
        for r in lab.itertuples():
            k = self._key(str(r.Label))
            if len(k) >= 4:
                self.keys.setdefault(k, r.ORG_ID)
        self.org_of_label = {self._key(str(lbl)): oid for oid, lbl in zip(org.ORG_ID, org.organisation_label)}
        self.places = {}  # organisation → [(from year, to year, GLOB id)]
        current = places.current if places is not None else (lambda g: g)  # the gazetteer still has some retired ids (GLOB2_E_17)
        for r in loc.dropna(subset=["place_id"]).itertuples():
            self.places.setdefault(r.ORG_ID, []).append((r.startDate, r.endDate, current(str(r.place_id))))

    @staticmethod
    def _key(s: str) -> str:
        s = PLACE_NOISE.sub("", s.strip()).lower()
        return re.sub(r"[^a-z]", "", re.sub(r"^fort\s+", "", s))

    def org(self, raw: str) -> str | None:
        """The VOC organisation a place name as written refers to, or None."""
        k = self._key(raw)
        if not k:
            return None
        exact = [t for a, t in ALIASES.items() if k == a]  # an exact alias before one inside the name: Cochim China is not Cochim
        for target in exact or [t for a, t in ALIASES.items() if len(a) >= 6 and a in k]:
            return self.org_of_label.get(self._key(target)) or self.keys.get(self._key(target))
        if k in self.keys:
            return self.keys[k]
        if k in self.exact:  # Manaar is Mannar in the places dataset, not the closer label Manapaar
            return None
        for cand in (k, "g" + k[1:] if k.startswith("s") else None):  # HTR reads G as S: Sale → Gale
            if cand and len(cand) >= 4:
                hit = process.extractOne(cand, self.keys.keys(), scorer=fuzz.ratio, score_cutoff=85)
                if hit:
                    return self.keys[hit[0]]
        return None  # not a VOC place the gazetteer knows: mostly HTR noise

    def name(self, raw: str) -> str | None:
        if raw.startswith("on board the "):
            return raw
        o = self.org(raw)
        return self.name_of.get(o) if o else None

    def glob_id(self, raw: str, year: int | None = None) -> str | None:
        """GLOB id of the organisation's place, the one valid in `year` where it moved."""
        o = None if raw.startswith("on board the ") else self.org(raw)
        spans = self.places.get(o, []) if o else []
        if not spans:
            return None
        if year is not None:
            for start, end, pid in spans:
                if (pd.isna(start) or start <= year) and (pd.isna(end) or year <= end):
                    return pid
        return spans[-1][2]


# Names the dataset gives to the wrong place: Fort St George is the English fort at Madras, not the church St: Jago near Cochin (NR_250)
NAME_FIXES = {re.compile(r"(^|fort)st?george$"): "GLOB_530"}
# Labels of the VOC gazetteer that name another place: Cochin China (Quinam) is not Cochin
GAZETTEER_WRONG_LABELS = {("ORG_68", "In Cochin China")}
FAMILY = re.compile(r"_(HUM|GEO)_\d+$")  # GLOB_38_HUM_1 is a record of the place GLOB_38
ATLAS = "GLOB_GAVOC_"  # names from the Atlas of Dutch colonial place names (GAVOC): islands and capes that share their names with forts, ships and misreadings
# a name of several places: the one nearest the inventory's settlements, if within NEAR_KM and NEAR_RATIO times nearer than the next;
# records within SAME_KM of each other are one place (the town before its fort: Maros, Larike); a record without coordinates
# stands in the way unless the nearest is within SURE_KM (Fort Amsterdam on Ambon, 10 km; Amsterdam, fort at the Cape: none)
NEAR_KM, NEAR_RATIO, SAME_KM, SURE_KM = 300, 3.0, 25, 100


def _km(a, b) -> float:
    (la1, lo1), (la2, lo2) = np.radians(a), np.radians(b)
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)))


class Places:
    """The GLOBALISE places dataset: names and spelling variants → GLOB id. For places in
    closings the VOC gazetteer does not know; exact matches only (no fuzzy matching), so that
    misread words do not become places. A name of several places (Hollandia: forts on Banda,
    Saparua and Ternate) is the one nearest the inventory's settlements, when one clearly is."""

    def __init__(self, path: str = PLACES):
        p = pd.read_excel(path, sheet_name="places").drop_duplicates("glob_id")
        labels = pd.read_excel(path, sheet_name="labels")
        dep = pd.read_excel(path, sheet_name="deprecated_uris")
        upd = pd.read_excel(path, sheet_name="updated_uris")
        types = pd.read_excel(path, sheet_name="types")
        self.forts = set(types[types.place_type.astype(str).str.contains("fort", case=False)].glob_id.astype(str))
        self.ids = set(p.glob_id.astype(str))
        self.label = dict(zip(p.glob_id.astype(str), p.pref_label.astype(str)))  # GLOB id → preferred label
        self.coords = {str(g): (la, lo) for g, la, lo in zip(p.glob_id, p.latitude, p.longitude) if pd.notna(la) and pd.notna(lo)}
        self.deprecated = dict(zip(dep.Old_ID.astype(str), dep.New_ID.astype(str)))  # retired GLOB id → its replacement
        # an id that became the general record of a place → the record that now has its old meaning (Colombo → the city)
        self.updated = dict(zip(upd.New_Parent_ID.astype(str), upd.New_ID.astype(str)))
        alt = labels.dropna(subset=["label"]).groupby(labels.glob_id.astype(str)).label.agg(list).to_dict()
        self.by_key = {}  # name → {place (family id): GLOB ids with that name}
        for r in p.itertuples():
            gid = str(r.glob_id)
            if gid.startswith(ATLAS):
                continue
            for label in [str(r.pref_label)] + [str(a) for a in alt.get(gid, [])]:
                k = Gazetteer._key(label)
                if len(k) >= 5:
                    self.by_key.setdefault(k, {}).setdefault(FAMILY.sub("", gid), set()).add(gid)
        self.ambiguous = {k for k, fam in self.by_key.items() if len(fam) > 1}  # names of more than one place (Patani: Halmahera and Thailand)

    def current(self, glob_id: str | None) -> str | None:
        """The id to use now: a retired id's replacement (deprecated_uris), then a settlement's chosen
        id (SETTLEMENT_IDS), else the record that took over the id's old meaning (updated_uris)."""
        if not glob_id:
            return glob_id
        g = self.deprecated.get(glob_id, glob_id)
        return SETTLEMENT_IDS[g] if g in SETTLEMENT_IDS else self.updated.get(g, g)

    def _resolve(self, family: str, members: set) -> str:
        """The current id of a place from the records that have the name: its general record if one of them, else the first."""
        return self.current(family if family in members else sorted(members)[0])

    def lookup(self, raw: str, near: list | None = None) -> str | None:
        """The current GLOB id of a place name; near: (lat, lon) of the inventory's settlements, for a name of several places."""
        if raw.startswith("on board the "):
            return None
        k = Gazetteer._key(raw)
        for pattern, gid in NAME_FIXES.items():
            if pattern.search(k):
                return self.current(gid)
        fams = self.by_key.get(k, {})
        ids = [self._resolve(f, m) for f, m in fams.items()]
        if len(ids) <= 1:
            return ids[0] if ids else None
        located = [g for g in ids if g in self.coords]
        if not near or not located:
            return None
        dist = sorted((min(_km(self.coords[g], pt) for pt in near), g) for g in located)
        d1, best = dist[0]
        same = [g for d, g in dist if _km(self.coords[g], self.coords[best]) <= SAME_KM]  # one place: the town before its fort
        best = min(same, key=lambda g: (g in self.forts, g.startswith("GLOB3_"), g))
        others = [d for d, g in dist if g not in same]
        if d1 > NEAR_KM or (others and d1 * NEAR_RATIO > others[0]) or (len(located) < len(ids) and d1 > SURE_KM):
            return None
        return best


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


CAPE = "Kaap de Goede Hoop"
CASTLE = re.compile(r"\b[CcKk]ast\w*")
CAPE_WORD = re.compile(r"cab|caab|kaap|caap|cap|cub")


def _clean(w: str, raw: str) -> str:
    w = re.sub(r"(?<=\w)(\s*[=„¬:])+\s*(?=\w)", "", w)  # a word broken over two lines
    w = re.sub(r"\s*,\s*(?=(de|d'|den|der|van|het)\b|[A-Z])", " ", w)  # no commas in a title's place: "Casteel, de Goede Hoop"
    w = re.sub(r"(?<=\w),\s*(?=[a-z])", "", w)  # "Amed, abaad", "Coc, him"
    w = re.sub(r"^[\W\d_]+|[\W\d_]+$", "", w) or raw.strip()  # "Manaar:", "Manaar 6"
    return w[:1].upper() + w[1:]


def written_place(raw: str) -> str:
    """The place name as written, without the office ("in 't Casteel Batavia") and transcription marks ("Ma= „kasser").
    "Goede Hoop" alone is not a name: "in 't Casteel de Goede Hoop" is the castle, else it is the Cape."""
    w = _clean(PLACE_PREFIX.sub("", raw.strip()) or raw.strip(), raw)
    letters = re.sub(r"[^a-z]", "", w.lower())
    if "goedeho" in letters and not CAPE_WORD.match(letters):
        castle = CASTLE.search(raw)
        return _clean(raw[castle.start():], raw) if castle else CAPE
    return w


def place_text(raw: str | None, place: str | None) -> str | None:
    """The place as written, with its Dutch name where that differs: "Gale (Galle)"."""
    if not place or not raw or raw.startswith("on board the "):
        return place
    written = written_place(raw)
    if Gazetteer._key(written) == Gazetteer._key(place):
        return place
    return f"{written} ({place})"


def title(type_en: str | None, place: str | None, date: str | None) -> str:
    head = type_en[:1].upper() + type_en[1:] if type_en else UNKNOWN
    return ", ".join(x for x in (head, place, date) if x)


def infer(doc: dict, texts: dict, headers: dict, model, labels: dict, gazetteer: Gazetteer, years: tuple, places: Places | None = None,
          settlement_names: dict | None = None, near: list | None = None) -> dict:
    """Type, place, date and title of one document: doc has first, last, before_last, second (scan filenames).
    settlement_names: GLOB id → the settlement's name (its first label), the place name for those ids;
    near: (lat, lon) of the inventory's settlements, to tell places of the same name apart."""
    first, last = texts.get(doc["first"], ""), texts.get(doc["last"], "")
    proba = model.predict_proba([type_text(first, last)])[0]
    k = int(proba.argmax())
    type_nl, conf = model.classes_[k], float(proba[k])
    type_en = labels.get(type_nl) if conf >= TYPE_MIN else None
    lo, hi = years

    def plausible(d):
        return lo is None or lo - YEAR_SLACK <= d.date.year <= hi + YEAR_SLACK

    place = date = raw_place = None
    source = None
    for f in (doc["last"], doc.get("before_last")):  # the closing, on the last page or the one before
        if not f:
            continue
        found = closing(entities(f), texts.get(f, ""))
        if found:
            raw_place, d, _ = found
            place = gazetteer.name(raw_place)
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
    year = date.date.year if date is not None else (lo if lo == hi else None)
    glob_id = None
    if place and raw_place:  # a VOC place: the GLOB id of its location, when the places dataset has it
        glob_id = gazetteer.glob_id(raw_place, year)
        if places is not None and glob_id not in places.ids:
            glob_id = None
    elif raw_place and places is not None:  # another place the places dataset knows: its modern name is not used, the name as written is
        hit = places.lookup(raw_place, near)
        if hit:
            glob_id, place = hit, written_place(raw_place)
    if glob_id is not None and settlement_names and glob_id in settlement_names:  # Kaap de Goede Hoop, not the office's name
        place = settlement_names[glob_id]
    shown = place_text(raw_place, place)
    return {"title": title(type_en, shown, date_text), "type": type_en, "type_nl": type_nl if type_en else None, "type_conf": round(conf, 2),
            "place": place, "place_text": written_place(raw_place) if place and raw_place and not raw_place.startswith("on board the ") else None, "place_glob_id": glob_id,
            "date": date.date.isoformat() if date is not None else None,
            "date_precision": date.precision if date is not None else None, "date_text": date_text, "found_in": source}


def inventory_places(conn, inv_id: str, places: Places) -> list:
    """(lat, lon) of the settlements of the inventory's index entries other than Batavia (letters go there from everywhere)."""
    globs = [g for (g,) in conn.execute(
        "SELECT DISTINCT s.glob_id FROM document d JOIN settlement s ON s.id = d.location_id "
        "JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' WHERE d.inventory_id = ?", (inv_id,))]
    globs = [g for g in globs if g != SETTLEMENT_IDS["GLOB2_894"]]  # Batavia: letters come there from everywhere, its inventories tell nothing
    return [places.coords[g] for g in globs if g in places.coords]


def infer_titles(segments_csv: str = "segments.csv", inventories: list[str] | None = None, out: str = OUT_CSV, conn=None):
    """Titles for every segment without one (no index entry, court case or General Missive) → out CSV."""
    with open(os.path.join(TITLE_DIR, "doctype.pkl"), "rb") as f:
        model = pickle.load(f)
    labels = english_types()
    places = Places()
    gazetteer = Gazetteer(places=places)
    settlement_names = dict(conn.execute(  # the name the app shows: the first label (step 27)
        "SELECT s.glob_id, (SELECT l.label FROM settlement_label l WHERE l.settlement_id = s.id ORDER BY l.rowid LIMIT 1) FROM settlement s"))
    stale = sorted(g for g in settlement_names if places.current(g) != g)
    if stale:
        raise SystemExit(f"settlements with an id that is no longer current: {', '.join(stale)}; run step 28 first")
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
        near = inventory_places(conn, inv_id, places)
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
                         **infer(d, texts, headers, model, labels, gazetteer, years.get(inv, (None, None)), places, settlement_names, near)})
        if n % 100 == 0 or n == len(groups):
            logger.info("%d/%d inventories, %d titles", n, len(groups), len(rows))
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)
