"""
Per-scan predictors for document boundaries.

Each Predictor turns an InventoryData into a DataFrame aligned with the scans.
Columns starting with "_" are auxiliary values used elsewhere (e.g. the cleaned
page/folio number for ToC alignment); all other columns are numeric features
for the boundary model. Feature values describe scan i as a possible document
start, so "prev_*" features look at scan i-1.

To add a predictor (e.g. the text-embedding first/last-page model), subclass
Predictor, return its feature columns, and add it to default_predictors() or
pass it explicitly. The boundary model picks up new feature names by itself;
refit the weights afterwards.
"""

import math
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .header_dates import compare_dates, parse_header_date
from .inventory import InventoryData
from .texts import TextIndex, load_texts


class Predictor(ABC):
    name: str

    @abstractmethod
    def compute(self, inv: InventoryData) -> pd.DataFrame: ...


def _prev(s: pd.Series, fill=0) -> pd.Series:
    return s.shift(1, fill_value=fill)


# ── blank pages ───────────────────────────────────────────────────────────────


class BlankPredictor(Predictor):
    name = "blank"

    def compute(self, inv):
        blank = inv.scans["is_blank"].astype(int)
        run = np.zeros(inv.n)  # number of blank scans directly before scan i
        for i in range(1, inv.n):
            run[i] = run[i - 1] + 1 if blank.iat[i - 1] else 0
        first_content = np.zeros(inv.n)
        nz = np.flatnonzero(blank.values == 0)
        if len(nz):
            first_content[nz[0]] = 1
        return pd.DataFrame(
            {
                "blank": blank,
                "after_blank": ((_prev(blank) == 1) & (blank == 0)).astype(int),
                "blank_run_before": np.log1p(run) * (blank == 0),
                "first_content": first_content,
            }
        )


# ── signature marks ───────────────────────────────────────────────────────────

COLLATION_RE = re.compile(r"ac+[ck]ord|akkord|concord|collat|geaccord", re.I)


def signature_kind(text: str) -> str:
    """collation | signed | quire | number | none"""
    if not text:
        return "none"
    if COLLATION_RE.search(text):
        return "collation"
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", text)
    if len(letters) == 0:
        return "number"  # page numbers, amounts
    if len(letters) <= 2:
        return "quire"  # quire signatures such as "E", "Gg"
    return "signed"  # names, closing place/date lines


class SignaturePredictor(Predictor):
    name = "signature"

    def compute(self, inv):
        kind = inv.scans["signatures"].map(signature_kind)
        out = {}
        for k in ("collation", "signed", "quire"):
            here = (kind == k).astype(int)
            out[f"sig_{k}_here"] = here
            out[f"prev_sig_{k}"] = _prev(here)
        return pd.DataFrame(out)


# ── page / folio numbers ──────────────────────────────────────────────────────

RESTART_MAX, RESTART_MIN_PREVIOUS = 5, 10


def clean_numbers(folio_lists: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick one plausible page/folio number per scan and split the scans into
    numbering runs (a new run starts where the numbering restarts).

    A value is kept when it is consistent with a nearby observation: numbers
    increase with scan position at no more than ~3 per scan (pagination on
    double scans is the fastest case). Misreads like 1691 for 169 fail this.
    Returns (number or nan per scan, run id per scan).
    """
    n = len(folio_lists)
    obs = [(i, v) for i, vals in enumerate(folio_lists) for v in vals]
    by_scan: dict[int, list[int]] = {}
    for i, v in obs:
        by_scan.setdefault(i, []).append(v)
    idx = sorted(by_scan)

    def consistent(i, v, j, w):
        di, dv = j - i, w - v
        return (di > 0 and 0 <= dv <= 3 * di + 3) or (di < 0 and 0 <= -dv <= 3 * -di + 3)

    value = np.full(n, np.nan)
    for k, i in enumerate(idx):
        neighbours = [(j, w) for j in idx[max(0, k - 3) : k + 4] if j != i for w in by_scan[j]]
        best, best_support = None, 0
        for v in by_scan[i]:
            support = sum(consistent(i, v, j, w) for j, w in neighbours)
            if support > best_support:
                best, best_support = v, support
        if best is not None:
            value[i] = best

    # restarts: a small number after the run reached RESTART_MIN_PREVIOUS,
    # confirmed by a consistent following observation
    run = np.zeros(n, dtype=int)
    current, running_max = 0, None
    kept = [i for i in range(n) if not np.isnan(value[i])]
    for k, i in enumerate(kept):
        v = value[i]
        if running_max is not None and running_max >= RESTART_MIN_PREVIOUS and v <= RESTART_MAX:
            nxt = kept[k + 1 : k + 3]
            if not nxt or any(consistent(i, v, j, value[j]) for j in nxt):
                current += 1
                running_max = None
                # the run starts after the previous numbered scan
                prev_i = kept[k - 1] if k else -1
                run[prev_i + 1 :] = current
        running_max = v if running_max is None else max(running_max, v)
    return value, run


class FolioPredictor(Predictor):
    name = "folio"

    def compute(self, inv):
        value, run = clean_numbers(inv.scans["folios"])
        restart_here = np.zeros(inv.n)
        restart_ahead = np.zeros(inv.n)
        numbered = ~np.isnan(value)
        for r in np.unique(run)[1:]:
            first_numbered = np.flatnonzero((run == r) & numbered)
            if len(first_numbered) == 0:
                continue
            i = first_numbered[0]
            restart_here[i] = 1
            for d in (1, 2, 3):  # section title pages often precede the first number
                if i - d >= 0:
                    restart_ahead[i - d] = max(restart_ahead[i - d], 1 / d)
        return pd.DataFrame(
            {
                "_number": value,
                "_run": run,
                "has_number": numbered.astype(int),
                "number_restart": restart_here,
                "number_restart_ahead": restart_ahead,
            }
        )


# ── running headers: text ─────────────────────────────────────────────────────


def normalize_header(text: str) -> str:
    """Letters only (dates are compared separately), lower case, noise tokens dropped."""
    text = text.lower().replace("ſ", "s").replace("ij", "y")
    return " ".join(t for t in re.findall(r"[a-zà-ÿ]+", text) if len(t) >= 3)


class HeaderTextPredictor(Predictor):
    """Change of running-header wording (place, addressee) against the last
    two headed scans within LOOKBACK scans, fuzzy to absorb HTR noise."""

    name = "header_text"
    LOOKBACK = 3
    AHEAD, BEHIND = 2, 4

    def compute(self, inv):
        norm = inv.scans["header"].map(normalize_header)
        present = (norm.str.len() > 0).astype(int)
        change = np.zeros(inv.n)
        first_after_gap = np.zeros(inv.n)
        for i in range(inv.n):
            if not present.iat[i]:
                continue
            prev = [norm.iat[j] for j in range(max(0, i - self.LOOKBACK), i) if present.iat[j]][-2:]
            if not prev:
                first_after_gap[i] = 1
                continue
            sim = max(fuzz.token_set_ratio(norm.iat[i], p) for p in prev) / 100
            change[i] = 1 - sim
        # Headers are often missing on a document's first page and appear from
        # the second page on: compare the first header on scans i..i+AHEAD with
        # the last header on scans i-BEHIND..i-1 ("across" a possible start).
        across = np.zeros(inv.n)
        heads = np.flatnonzero(present.to_numpy())
        for i in range(inv.n):
            after = heads[(heads >= i) & (heads <= i + self.AHEAD)]
            before = heads[(heads < i) & (heads >= i - self.BEHIND)]
            if len(after) and len(before):
                across[i] = 1 - fuzz.token_set_ratio(norm.iat[after[0]], norm.iat[before[-1]]) / 100
        return pd.DataFrame(
            {
                "_header_norm": norm,
                "header_present": present,
                "header_text_change": change,
                "header_after_gap": first_after_gap,
                "header_text_change_across": across,
            }
        )


# ── running headers: dates ────────────────────────────────────────────────────


class HeaderDatePredictor(Predictor):
    """Parsed header date, compared with the last two dated scans within
    LOOKBACK scans. 'uncertain' covers one-character day misreads (14/11)."""

    name = "header_date"
    LOOKBACK = 3
    AHEAD, BEHIND = 2, 4

    def compute(self, inv):
        dates = inv.scans["header"].map(parse_header_date)
        changed = np.zeros(inv.n)
        uncertain = np.zeros(inv.n)
        same = np.zeros(inv.n)
        for i in range(inv.n):
            if dates.iat[i] is None:
                continue
            prev = [dates.iat[j] for j in range(max(0, i - self.LOOKBACK), i) if dates.iat[j] is not None][-2:]
            if not prev:
                continue
            verdicts = {compare_dates(dates.iat[i], p) for p in prev}
            if "same" in verdicts:
                same[i] = 1
            elif "uncertain" in verdicts:
                uncertain[i] = 1
            else:
                changed[i] = 1
        # across a possible start (headers often begin on a document's 2nd page)
        across_changed = np.zeros(inv.n)
        across_same = np.zeros(inv.n)
        dated = np.flatnonzero(dates.notna().to_numpy())
        for i in range(inv.n):
            after = dated[(dated >= i) & (dated <= i + self.AHEAD)]
            before = dated[(dated < i) & (dated >= i - self.BEHIND)]
            if len(after) and len(before):
                v = compare_dates(dates.iat[after[0]], dates.iat[before[-1]])
                across_changed[i] = v == "changed"
                across_same[i] = v == "same"
        return pd.DataFrame(
            {
                "_header_date": dates,
                "header_date_changed_across": across_changed,
                "header_date_same_across": across_same,
                "has_header_date": dates.notna().astype(int),
                "header_date_changed": changed,
                "header_date_uncertain": uncertain,
                "header_date_same": same,
            }
        )


# ── languages, marginalia ─────────────────────────────────────────────────────


class LanguagePredictor(Predictor):
    name = "language"

    def compute(self, inv):
        langs = inv.scans["languages"].map(lambda s: frozenset(x for x in s.split(",") if x and x != "unknown"))
        changed = np.zeros(inv.n)
        last = None
        for i, l in enumerate(langs):
            if l:
                if last is not None and l != last:
                    changed[i] = 1
                last = l
        return pd.DataFrame(
            {"language_changed": changed, "non_dutch": langs.map(lambda l: int(bool(l) and "nld" not in l))}
        )


class MarginaliaPredictor(Predictor):
    name = "marginalia"

    def compute(self, inv):
        m = inv.scans["marginalia"].astype(int)
        return pd.DataFrame({"marginalia": m, "prev_marginalia": _prev(m)})


# ── text length, position ─────────────────────────────────────────────────────


class TextLengthPredictor(Predictor):
    """Characters of normalized text per scan (from the inventory text offsets).
    Covers, title pages, section titles and notes are short."""

    name = "text_length"

    def compute(self, inv):
        length = inv.scans["text_len"].to_numpy(dtype=float)
        log_len = np.log1p(length)
        nonzero = length[length > 0]
        typical = np.log1p(np.median(nonzero)) if len(nonzero) else 0.0
        rel = np.where(length > 0, log_len - typical, -typical)
        return pd.DataFrame(
            {
                "log_text_len": log_len,
                "rel_text_len": rel,
                "short_text": ((length > 0) & (rel < np.log(0.3))).astype(int),
                "text_len_change_prev": np.abs(rel - np.concatenate([[rel[0]], rel[:-1]])),
            }
        )


class PositionPredictor(Predictor):
    """Distance to the start and end of the inventory (covers, fly-leaves)."""

    name = "position"

    def compute(self, inv):
        i = np.arange(inv.n)
        return pd.DataFrame(
            {"near_inventory_start": np.exp(-i / 3.0), "near_inventory_end": np.exp(-(inv.n - 1 - i) / 3.0)}
        )


class ToCNumberPredictor(Predictor):
    """
    Page/folio numbers the ToC mentions, matched to the numbers seen on the
    scans (no alignment needed): an entry starts on this number, one ends on
    it, or one ends and the next starts on it (a shared page in paginated
    volumes, the same leaf in foliated ones). All zero without a ToC.
    """

    name = "toc_numbers"
    TOC_FEATURES = ("toc_start_number", "toc_end_number", "toc_end_start_number")

    def compute(self, inv):
        number = FolioPredictor().compute(inv)["_number"].to_numpy()
        toc = inv.toc[inv.toc["toc_order"].notna()]
        starts = set(toc["folio_start"].dropna().astype(int))
        ends = set(toc["end_eff"].dropna().astype(int))
        pairs = set()
        fs, fe = toc["folio_start"].to_numpy(dtype=float), toc["end_eff"].to_numpy(dtype=float)
        for k in range(1, len(toc)):
            if not np.isnan(fs[k]) and fs[k] == fe[k - 1]:
                pairs.add(int(fs[k]))
        num = pd.Series(number).fillna(-1).astype(int)
        return pd.DataFrame(
            {
                "toc_start_number": num.isin(starts).astype(int),
                "toc_end_number": num.isin(ends).astype(int),
                "toc_end_start_number": num.isin(pairs).astype(int),
            }
        )


# ── full text: opening and closing formulas ───────────────────────────────────


def scan_texts(inv: InventoryData) -> pd.Series:
    if inv.texts is None:
        inv.texts = load_texts(inv.inventory_number, inv.scans["filename"])
    return inv.texts


GENRE_RE = re.compile(
    r"\b(cop(ia|ie|y)|extract|transla[ae]?t|register|missive|memorie|instructie|rapport|l[iy]st|notitie|resolutie|"
    r"contract|rendement|e[iy]j?sch|factura|cognoiss?ement|journa[ae]?l|dagregister|verbaal|attestatie|verklaring|"
    r"declaratie|request|brief(je)?|acte|obligatie|inventaris|reekening|rekening|staat|opgave|generaal|generale)\b",
    re.I,
)
SALUTATION_RE = re.compile(
    r"(hoog\s*ed|edele?\s*(groot|gestr|heer)|groot\s*a[cg]h?tb|gestreng|e[ea]rntfest|eerbaere|wel\s*edele|"
    r"mijn\s*heer|edelheyd|edelhe[iy]j?t|seer\s*genereus|voorsienige|welwijse)",
    re.I,
)
CLOSING_RE = re.compile(
    r"(acc?[ck]ord|akkord|onderst(o|ae)n[dt]|was\s*get[ey]?[ck]ent|w[ae]s\s*geteek|getekent|geteekend|"
    r"in\s*margine|dienaren|dienaer|dienaar|ter\s*zijden|ter\s*sijden|verblijven|blijven)",
    re.I,
)
PLACE_DATE_RE = re.compile(r"(casteel|batavia|fort|comptoir|adij|den\s*\d{1,2}\W).{0,60}1[5-8]\d\d", re.I | re.S)
LEADER_RE = re.compile(r"(\.\s){3,}|\.{4,}|(-\s){3,}|(„\s?){3,}")


class TextFormulaPredictor(Predictor):
    """
    Opening and closing formulas in the full text of a scan. The top of a page
    (after the running header) often opens a document with a genre word
    (Copia, Extract, Register, Missive, ...) or a salutation (Hoog Edele Groot
    Achtbare ...); its end often closes one (Accordeert, onderstond, was
    geteekent, dienaren, place and date). A closing formula followed by an
    opening on the same page marks a boundary within the scan.
    """

    name = "text_formula"
    TOP, BOTTOM = 250, 300

    def compute(self, inv):
        texts = scan_texts(inv)
        rows = []
        for t in texts:
            t = t or ""
            top, bottom = t[: self.TOP], t[-self.BOTTOM :]
            close_positions = [m.end() for m in CLOSING_RE.finditer(t)]
            mid = 0.0
            if close_positions and len(t) > 400:
                after = t[close_positions[-1] :]
                if len(after) > 200 and (GENRE_RE.search(after[:250]) or SALUTATION_RE.search(after[:300])):
                    mid = 1.0
            n = max(len(t), 1)
            rows.append(
                (
                    int(bool(GENRE_RE.search(top))),
                    int(bool(SALUTATION_RE.search(t[:400]))),
                    int(bool(CLOSING_RE.search(bottom))),
                    int(bool(PLACE_DATE_RE.search(bottom))),
                    mid,
                    len(LEADER_RE.findall(t)) / n * 1000,
                    sum(c.isdigit() for c in t) / n,
                    int(t.strip() == ""),
                )
            )
        cols = ["open_genre", "open_salutation", "close_formula", "close_place_date", "close_then_open", "leaders_per_1000", "digit_share", "no_text"]
        return pd.DataFrame(rows, columns=cols)


class ToCTextPredictor(Predictor):
    """
    How well a scan's opening words match a ToC description (IDF-weighted
    word overlap; texts.TextIndex). A short title page right before a scan
    counts for that scan (title pages are outside documents). A scan matching
    many entries is probably one of the volume's own table-of-contents pages;
    its matches are discarded. All zero without a ToC. Also stores the entries
    × scans similarity for the ToC alignment.
    """

    name = "toc_text"
    TOC_FEATURES = ("toc_title_match", "toc_title_match_title_page", "toc_entries_matched")
    OPENING_CHARS = 400
    MATCH = 0.5  # similarity counted as a match when counting entries per scan
    TOC_PAGE_ENTRIES = 3

    def compute(self, inv):
        n = inv.n
        zeros = pd.DataFrame({c: np.zeros(n) for c in self.TOC_FEATURES})
        toc = inv.toc[inv.toc["toc_order"].notna()]
        texts = scan_texts(inv)
        if toc.empty or not (texts.str.len() > 0).any():
            inv.toc_text_sim = np.zeros((len(toc), n))
            return zeros
        passages = []
        for i, t in enumerate(texts):
            passages.append((i, t[: self.OPENING_CHARS]))
            for m in CLOSING_RE.finditer(t):  # a next document may start after a closing formula
                rest = t[m.end() :]
                if len(rest) > 200:
                    passages.append((i, rest[: self.OPENING_CHARS]))
        index = TextIndex(passages, n)
        sim = np.vstack([index.scores(str(d or "")) for d in toc["title"]])
        n_matched = (sim >= self.MATCH).sum(axis=0)
        toc_page = n_matched >= self.TOC_PAGE_ENTRIES
        sim[:, toc_page] = 0
        # A short page matching an entry is a title page: before the document
        # (front title) or on its last page (docket, end title). It is never a
        # start itself; it counts for the next scan only when that scan opens
        # a document (genre word or salutation at the top).
        tokens = texts.map(lambda t: len(t.split())).to_numpy()
        title_like = (tokens > 0) & (tokens < 60)
        opens = texts.map(lambda t: bool(GENRE_RE.search(t[:250]) or SALUTATION_RE.search(t[:400]))).to_numpy()
        blank = inv.scans["is_blank"].to_numpy()
        title_sim = sim[:, title_like].copy() if title_like.any() else None
        via_title = np.zeros_like(sim)
        title_pos = np.flatnonzero(title_like)
        for m, i in enumerate(title_pos):
            j = i + 1
            if j < n and blank[j]:
                j += 1  # blank verso after the title page
            if j < n and opens[j] and not title_like[j]:
                via_title[:, j] = np.maximum(via_title[:, j], title_sim[:, m])
        sim[:, title_like] = 0
        inv.toc_text_sim = np.maximum(sim, via_title)
        return pd.DataFrame(
            {
                "toc_title_match": sim.max(axis=0),
                "toc_title_match_title_page": via_title.max(axis=0),
                "toc_entries_matched": np.minimum(n_matched, 10),
            }
        )


# ── page layout (PageXML) ─────────────────────────────────────────────────────


class LayoutPredictor(Predictor):
    """Where text starts and stops on the page, gaps, closing/opening formulas
    halfway down, centred headings and catch-words (pagexml.py). Zero without
    PageXML."""

    name = "layout"

    def compute(self, inv):
        from .pagexml import load_layout

        return load_layout(inv.inventory_number, inv.scans["filename"]).reset_index(drop=True)


# ── external per-scan scores (e.g. the text-embedding model) ─────────────────


def _logit(p: pd.Series) -> pd.Series:
    p = p.clip(1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class ExternalScorePredictor(Predictor):
    """
    Per-scan probabilities from another model, read from a CSV or parquet file
    with columns: filename, p_first, p_last (p_last optional). Produces
    logit(p_first) for scan i, logit(p_last) for scan i-1, and a missing flag.
    """

    def __init__(self, path: str, name: str = "external"):
        self.path = path
        self.name = name
        self._table: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._table is None:
            t = pd.read_parquet(self.path) if self.path.endswith(".parquet") else pd.read_csv(self.path)
            self._table = t.set_index("filename")
        return self._table

    def compute(self, inv):
        t = self._load()
        first = inv.scans["filename"].map(t["p_first"]) if "p_first" in t else pd.Series(np.nan, index=inv.scans.index)
        last = inv.scans["filename"].map(t["p_last"]) if "p_last" in t else pd.Series(np.nan, index=inv.scans.index)
        missing = first.isna().astype(int)
        return pd.DataFrame(
            {
                f"{self.name}_first_logit": _logit(first).fillna(0),
                f"{self.name}_prev_last_logit": _prev(_logit(last).fillna(0)),
                f"{self.name}_missing": missing,
            }
        )


# ── assembly ──────────────────────────────────────────────────────────────────

EXTERNAL_SCORES_ENV = "SEGMENTATION_EXTERNAL_SCORES"


def default_predictors() -> list[Predictor]:
    preds: list[Predictor] = [
        BlankPredictor(),
        SignaturePredictor(),
        FolioPredictor(),
        HeaderTextPredictor(),
        HeaderDatePredictor(),
        LanguagePredictor(),
        MarginaliaPredictor(),
        TextLengthPredictor(),
        PositionPredictor(),
        ToCNumberPredictor(),
        TextFormulaPredictor(),
        ToCTextPredictor(),
        LayoutPredictor(),
    ]
    path = os.environ.get(EXTERNAL_SCORES_ENV)
    if path:
        preds.append(ExternalScorePredictor(path))
    return preds


def compute_features(inv: InventoryData, predictors: list[Predictor] | None = None) -> pd.DataFrame:
    frames = [p.compute(inv).reset_index(drop=True) for p in (predictors or default_predictors())]
    inv.features = pd.concat(frames, axis=1)
    return inv.features


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if not c.startswith("_")]
