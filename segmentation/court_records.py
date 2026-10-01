"""
Court records (Raad van Justitie of Batavia and the Cape) segmented from the
EMDCCR dataset (Early Modern Dutch Colonial Court Records, Resilient Diversity
project): data/EMDCCR*.xlsx, one row per accused person with the case id
(ID_Rechtszaak) and the case's first and last scan (Begin_/Eind_Zaak_Scannummer).

These inventories have no ToC; the dataset takes its place:

  - A case range with a start and an end scan is a fixed document (kind
    'case'). When the accused of one case have their own ranges, the case
    spans them all and each range is a subdocument ('subdoc') of it; when
    another case lies in between, each range is a 'case' of its own.
  - A case with only a start scan (the Cape volumes) runs to the scan before
    the next known case; a case that ends in the next inventory runs to the end
    of this one; a case that began in the previous inventory runs from the end
    of the previous known case. Covers and blank pages at the open end are left
    out (non-document log-odds > 0). A start scan with nothing after it is a
    forced start for the model, which also decides where that case ends.
  - Scan references in another inventory ('99 [9353]', '656 (9354)') are
    applied to that inventory. Starts and ends that fall inside another case's
    fixed range are dropped (conflicting data).
  - All scans outside the cases are segmented by the model as usual.

Titles are in Dutch: claimant contra defendants; court, place, date (charge).
The claimant is the prosecutor (advocaat-fiscaal van India at Batavia,
independent fiscaal at the Cape), named when the case's text names him;
in civil disputes the claimant is not in the dataset.
"""

import datetime
import difflib
import glob
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .inventory import InventoryData
from .model import SegmentationModel
from .predictors import compute_features, scan_texts
from .texts import TEXT_DIR
from .segmenter import Result, Segment, scan_scores, segment_scans

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COURT_RECORDS_PATH = os.environ.get("SEGMENTATION_COURT_RECORDS") or next(
    iter(sorted(p for p in glob.glob(os.path.join(DATA_DIR, "EMDCCR*.xlsx")) if not os.path.basename(p).startswith("~$"))), None
)

CIVIL = {"Overig (financieel conflict)", "Boedel", "Afhandeling schulden", "Eigendom"}
OFFICE = {"Batavia": "advocaat-fiscaal van India", "Kaap de Goede Hoop": "independent fiscaal"}
MONTHS = "januari februari maart april mei juni juli augustus september oktober november december".split()
PARTICLES = {"van", "de", "der", "den", "la", "le", "du", "ter", "ten", "des", "het", "in", "'t"}
MAX_NAMES = 3
MAX_NAME_LENGTH = 60


# ── the dataset ───────────────────────────────────────────────────────────────


@dataclass
class Range:
    """One case's start and/or end scan (scan numbers) in one inventory."""

    case_id: str
    start: int | None
    end: int | None
    persons: list[dict] = field(default_factory=list)
    open_end: str | None = None  # 'next_inventory' when the case continues in the next inventory
    open_start: str | None = None  # 'previous_inventory' when it began in the previous one; 'reversed' when its
    # end came before its start: then the end is used only when the start conflicts with another case


@dataclass
class Case:
    case_id: str
    court: str  # Raad van Justitie
    place: str  # Batavia | Kaap de Goede Hoop
    persons: list[dict]
    date_begin: datetime.date | None
    date_end: datetime.date | None
    date_text: str
    charges: list[str]
    civil: bool
    excel: dict  # the scan numbers as given


def _scan_ref(value, inventory: str) -> tuple[str, int] | None:
    """'99', 99, '99 [9353]' or '656 (9354)' → (inventory, scan number)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    m = re.fullmatch(r"\s*(\d+)\s*(?:[\[(]\s*(\d+)\s*[\])])?\s*", str(value))
    if not m:
        return None
    return (m.group(2) or inventory), int(m.group(1))


def _date(raw, year) -> tuple[datetime.date | None, datetime.date | None, str]:
    """Aanklacht_Datum (YYYY-MM-DD, a year, or a timestamp) or else Jaar → (earliest, latest, Dutch text)."""
    for v in (raw, year):
        if v is None or pd.isna(v):
            continue
        if isinstance(v, (datetime.date, pd.Timestamp)):
            d = pd.Timestamp(v).date()
            return d, d, f"{d.day} {MONTHS[d.month - 1]} {d.year}"
        s = str(v).strip()
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?: 00:00:00)?", s)
        if m:
            try:
                d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return d, d, f"{d.day} {MONTHS[d.month - 1]} {d.year}"
            except ValueError:
                pass
        m = re.match(r"(\d{4})(?:\.0)?\b", s)
        if m:
            y = int(m.group(1))
            return datetime.date(y, 1, 1), datetime.date(y, 12, 31), str(y)
    return None, None, ""


def _clean(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip().rstrip(":;,").strip()


def person_name(p: dict) -> str:
    """Voornaam [patroniem] [tussenvoegsel] achternaam [alias bijnaam]; a single
    name gets its origin ('Tamboe van Madagascar')."""
    first = _clean(p.get("Voornaam"))
    if first.lower() in ("anoniem", "onbekend", "nn", "n.n."):
        first = ""
    surname = " ".join(x for x in (_clean(p.get("Tussenvoegsel")), _clean(p.get("Achternaam"))) if x)
    parts = [x for x in (first, _clean(p.get("Patroniem")), surname) if x]
    name = " ".join(parts)
    if name and len(name.split()) == 1 and _clean(p.get("Herkomst")):
        name += f" van {_clean(p.get('Herkomst'))}"
    if _clean(p.get("Bijnaam")):
        name = f"{name} alias {_clean(p.get('Bijnaam'))}" if name else _clean(p.get("Bijnaam"))
    if len(name) > MAX_NAME_LENGTH:  # a description rather than a name
        name = name[:MAX_NAME_LENGTH].rsplit(" ", 1)[0] + " …"
    return name


def load_court_cases(path: str | None = COURT_RECORDS_PATH) -> tuple[dict[str, Case], dict[str, list[Range]]]:
    """Cases by id, and the case ranges per inventory (where its scans are)."""
    if not path or not os.path.exists(path):
        return {}, {}
    df = pd.read_excel(path, sheet_name=0)
    df = df[df["ID_Rechtszaak"].notna() & df["Inventarisnummer"].notna()].copy()
    df["inv"] = df["Inventarisnummer"].astype(int).astype(str)
    cases: dict[str, Case] = {}
    ranges: dict[str, dict[tuple, Range]] = defaultdict(dict)
    for case_id, rows in df.groupby("ID_Rechtszaak", sort=False):
        persons = rows.to_dict("records")
        place, court = (_clean(rows["COURT"].iloc[0]).split(" - ", 1) + [""])[:2]
        dates = [_date(r["Aanklacht_Datum"], r["Jaar"]) for r in persons]
        dates = [d for d in dates if d[0] is not None]
        first = min(dates, key=lambda d: d[0]) if dates else (None, None, "")
        charges = list(dict.fromkeys(_clean(r["Aanklacht_Standaard"]) for r in persons if _clean(r["Aanklacht_Standaard"])))
        cases[case_id] = Case(
            case_id=case_id,
            court=court or "Raad van Justitie",
            place=place,
            persons=persons,
            date_begin=first[0],
            date_end=max(d[1] for d in dates) if dates else None,
            date_text=first[2],
            charges=charges,
            civil=bool(charges) and all(c in CIVIL for c in charges),
            excel={k: sorted({str(v) for v in rows[k].dropna()}) for k in ("Begin_Zaak_Scannummer", "Eind_Zaak_Scannummer", "Eis_Scannummer", "Vonnis_Scannummer")},
        )
        for r in persons:
            b = _scan_ref(r["Begin_Zaak_Scannummer"], r["inv"])
            e = _scan_ref(r["Eind_Zaak_Scannummer"], r["inv"])
            if b is None and e is None:
                continue
            if b and e and b[0] == e[0]:
                if e[1] >= b[1]:
                    pieces = [(b[0], b[1], e[1], None, None)]
                else:  # reversed: usually the end is in the next inventory; else one of them is a typo
                    pieces = [(b[0], b[1], None, None, None), (e[0], None, e[1], None, "reversed")]
            else:
                pieces = []
                if b:
                    later = e is not None and int(e[0]) > int(b[0])
                    pieces.append((b[0], b[1], None, "next_inventory" if later else None, None))
                if e and (b is None or int(e[0]) > int(b[0])):
                    pieces.append((e[0], None, e[1], None, "previous_inventory" if b else None))
            for inv, s, en, oe, os_ in pieces:
                _court_inventories[cases[case_id].place].add(inv)
                key = (case_id, s, en, oe, os_)
                rg = ranges[inv].setdefault(key, Range(case_id, s, en, open_end=oe, open_start=os_))
                rg.persons.append(r)
    return cases, {inv: list(v.values()) for inv, v in ranges.items()}


# ── titles ────────────────────────────────────────────────────────────────────


def _title_case(name: str) -> str:
    words = name.split()
    return " ".join(w.lower() if k and w.lower() in PARTICLES else w[:1].upper() + w[1:] for k, w in enumerate(words))


FISCAL_NAME = r"((?:[^\W\d_]+\.?\s+){0,4}[^\W\d_]{3,})"
FISCAL_PATTERNS = {
    # 'mr Adrianus Bergsma, advocaat fiscaal van India'
    "Batavia": re.compile(r"\bm\W{0,3}r\.?\s+" + FISCAL_NAME + r"[\s,.;]{0,4}(?:advoca\w*|adv\W{0,2}t)\W{0,3}fisca", re.I),
    # 'independent fiscaal mr Daniel van den Henghel nomine officii Eijscher'
    "Kaap de Goede Hoop": re.compile(
        r"fisca\w*\W{0,3}m\W{0,3}r\.?\s+" + FISCAL_NAME + r"[\s,.;]{0,4}(?:nomine|nom\b|q\W?q|ei[jy]?s|als\b|r\W?o\b|rat)", re.I
    ),
}


def _norm_name(s: str) -> str:
    s = re.sub(r"pro\s*interim|\bp\.?\s*i\.?$", "", s, flags=re.I)
    return re.sub(r"[^a-z ]", "", s.lower().replace("ij", "y")).strip()


FORMER = re.compile(r"\b(voormaals|voormalig\w*|gewesen|geweest|oud|oudt|wijlen|overleden)\b", re.I)
MIN_MENTIONS = 10  # a prosecutor's name must recur in the court volumes (filters HTR and parsing noise)
_court_inventories: dict[str, set[str]] = defaultdict(set)  # place -> inventories (filled by load_court_cases)
_clusters: dict[str, list[tuple[str, str]]] = {}


def _fiscal_mentions(text: str, pattern: re.Pattern) -> list[str]:
    """Prosecutor names in one scan's text, without former office holders."""
    out = []
    for m in pattern.finditer(text or ""):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if FORMER.search(name) or FORMER.search(text[max(0, m.start() - 25) : m.start()]):
            continue
        if len(_norm_name(name).split()) >= 2:
            out.append(name)
    return out


def _display(raw: str) -> str:
    return _title_case(re.sub(r"\s+(pro\s*\w*|p\.?\s*i\.?)$", "", raw, flags=re.I))


def prosecutor_names(place: str) -> list[tuple[str, str]]:
    """(normalized key, display name) of the prosecutors named in all court
    volumes of `place`: spelling variants merged, shown in their most frequent
    spelling, and only names with at least MIN_MENTIONS mentions."""
    if place not in _clusters:
        pattern = FISCAL_PATTERNS.get(place)
        variants: Counter = Counter()
        for inv in sorted(_court_inventories.get(place, ())) if pattern else []:
            folder = os.path.join(TEXT_DIR, f"inv={inv}")
            if os.path.isdir(folder):
                for t in pd.read_parquet(folder, columns=["text"])["text"].fillna(""):
                    variants.update(_fiscal_mentions(t, pattern))
        clusters: list[list] = []  # [key, display, count]
        for raw, count in variants.most_common():
            key = _norm_name(raw)
            match = next((c for c in clusters if difflib.SequenceMatcher(None, key, c[0]).ratio() >= 0.8), None)
            if match is None:
                clusters.append([key, _display(raw), count])
            else:
                match[2] += count
        _clusters[place] = [(k, d) for k, d, c in clusters if c >= MIN_MENTIONS]
    return _clusters[place]


class FiscalNames:
    """Per case the prosecutor named most often in the case's own scans (one
    of prosecutor_names)."""

    def __init__(self, texts: pd.Series, place: str):
        pattern = FISCAL_PATTERNS.get(place)
        known = prosecutor_names(place)
        self.names = [d for _, d in known]
        self.hits: list[list[str]] = []
        for t in texts.fillna(""):
            found = []
            for raw in _fiscal_mentions(t, pattern) if pattern else []:
                key = _norm_name(raw)
                best = max(known, key=lambda c: difflib.SequenceMatcher(None, key, c[0]).ratio(), default=None)
                if best and difflib.SequenceMatcher(None, key, best[0]).ratio() >= 0.8:
                    found.append(best[1])
            self.hits.append(found)

    def for_span(self, a: int, e: int) -> str | None:
        c = Counter(n for hits in self.hits[a : e + 1] for n in hits)
        return c.most_common(1)[0][0] if c else None


def _is_prosecutor(p: dict, prosecutors: list[str]) -> bool:
    """A fiscaal listed among the accused without a charge (data entry of the claimant)."""
    if _clean(p.get("Aanklacht_Standaard")) or _clean(p.get("Aanklacht_Origineel_Individueel")):
        return False
    if re.search(r"fisca", _clean(p.get("Beroep")), re.I):
        return True
    key = _norm_name(person_name(p))
    return any(difflib.SequenceMatcher(None, key, _norm_name(f)).ratio() >= 0.8 for f in prosecutors)


def case_title(case: Case, persons: list[dict], prosecutor: str | None, prosecutors: list[str] = ()) -> str:
    """'Advocaat-fiscaal van India mr. Adrianus Bergsma contra Jacobus Bunnegam en
    Pieter Hildernisse; Raad van Justitie, Batavia, 20 oktober 1734 (geweld)'.
    `prosecutors`: fiscaals named in the inventory, left out of the defendants."""
    persons = [p for p in persons if not _is_prosecutor(p, prosecutors)] or persons
    if case.civil:
        claimant = "Onbekende eiser"
    else:
        office = OFFICE.get(case.place, "fiscaal")
        claimant = office[:1].upper() + office[1:] + (f" mr. {prosecutor}" if prosecutor else "")
    names = list(dict.fromkeys(n for n in (person_name(p) for p in persons) if n))
    unnamed = sum(1 for p in persons if not person_name(p))
    if not names:
        defendants = "onbekende gedaagde" if len(persons) <= 1 else f"{len(persons)} onbekende gedaagden"
    else:
        shown = names[:MAX_NAMES]
        rest = len(names) - len(shown) + unnamed
        if rest:
            defendants = ", ".join(shown) + f" en {rest} {'andere' if rest == 1 else 'anderen'}"
        else:
            defendants = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " en " + shown[-1]
    dates = [d for d in (_date(p.get("Aanklacht_Datum"), p.get("Jaar")) for p in persons) if d[0] is not None]
    date_text = min(dates, key=lambda d: d[0])[2] if dates else case.date_text
    where = ", ".join(x for x in (case.court, case.place, date_text) if x)
    own = list(dict.fromkeys(_clean(p.get("Aanklacht_Standaard")) for p in persons if _clean(p.get("Aanklacht_Standaard")))) or case.charges
    charges = [c[:1].lower() + c[1:] for c in own[:2]] + (["e.a."] if len(own) > 2 else [])
    return f"{claimant} contra {defendants}; {where}" + (f" ({'; '.join(charges)})" if charges else "")


# ── segmentation ──────────────────────────────────────────────────────────────


@dataclass
class _Doc:
    start: int  # positions
    end: int
    case_id: str
    persons: list[dict]
    how: str
    parent: int | None = None  # index of the case document (for subdocuments)


def segment_court_inventory(inv: InventoryData, model: SegmentationModel, cases: dict[str, Case], ranges: list[Range]) -> Result:
    """Fixed case documents from the dataset; the model segments the rest."""
    if inv.features is None:
        compute_features(inv)
    sc = scan_scores(inv, model, use_toc=False)
    n = inv.n
    number = inv.scans["filename"].str.extract(r"_(\d+)$")[0].astype(int)
    pos_of = {int(v): k for k, v in number.items()}

    def pos(scan: int | None) -> int | None:
        return None if scan is None else pos_of.get(max(int(scan), int(number.iat[0])))  # scan 0: from the first scan

    # 1. fixed ranges (start and end known in this inventory), grouped per case
    docs: list[_Doc] = []
    by_case: dict[str, list[Range]] = defaultdict(list)
    loose: list[Range] = []
    for rg in ranges:
        if pos(rg.start) is not None and pos(rg.end) is not None:
            by_case[rg.case_id].append(rg)
        elif pos(rg.start) is not None or pos(rg.end) is not None:
            loose.append(rg)
    fixed_spans = {cid: [(pos(r.start), pos(r.end)) for r in rgs] for cid, rgs in by_case.items()}
    for cid, rgs in by_case.items():
        spans = sorted({(pos(r.start), pos(r.end)) for r in rgs})
        persons = {s: [p for r in rgs if (pos(r.start), pos(r.end)) == s for p in r.persons] for s in spans}
        lo, hi = min(a for a, _ in spans), max(e for _, e in spans)
        crossed = any(lo < a < hi for other, sp in fixed_spans.items() if other != cid for a, _ in sp)
        if len(spans) == 1 or crossed:
            for s in spans:  # one document per range (another case lies in between)
                docs.append(_Doc(s[0], s[1], cid, persons[s], "excel"))
        else:
            parent = len(docs)
            docs.append(_Doc(lo, hi, cid, [p for r in rgs for p in r.persons], "excel"))
            for s in spans:
                if s != (lo, hi):
                    docs.append(_Doc(s[0], s[1], cid, persons[s], "excel (accused's own range)", parent))

    covered = np.zeros(n, dtype=bool)
    for d in docs:
        covered[d.start : d.end + 1] = True

    # 2. open ranges: one end known; the other from the neighbouring cases
    open_starts, open_ends = [], []  # (position, Range)
    for rg in loose:
        p = pos(rg.start) if rg.start is not None else pos(rg.end)
        if p is None or covered[p]:
            continue  # inside another case's fixed range (or its own): conflicting data
        (open_starts if rg.start is not None else open_ends).append((p, rg))
    kept_starts = {rg.case_id for _, rg in open_starts}
    open_ends = [(p, rg) for p, rg in open_ends if rg.open_start != "reversed" or rg.case_id not in kept_starts]
    fixed_starts = sorted({d.start for d in docs})
    fixed_ends = sorted({d.end for d in docs})
    starts = sorted({p for p, _ in open_starts} | set(fixed_starts))
    nondoc = sc.nondoc > 0
    trimmed = np.zeros(n, dtype=bool)  # covers/blanks at the open end of a case: non-document
    forced: dict[int, Range] = {}  # unbounded starts: the model decides where they end
    for p, rg in sorted(open_starts, key=lambda t: t[0]):
        nxt = [s for s in starts if s > p] + [q for q, _ in open_ends if q > p]
        if rg.open_end == "next_inventory" and not nxt:
            e, how = n - 1, "start from excel, continues in the next inventory"
        elif nxt:
            e, how = min(nxt) - 1, "start from excel, end before the next case"
            if any(p < q <= e for q, _ in open_ends):  # an open end in between belongs to another case
                e = min(q for q, _ in open_ends if p < q <= e) - 1
        else:
            forced[p] = rg
            continue
        while e > p and nondoc[e]:
            e -= 1
            trimmed[e + 1] = True
        docs.append(_Doc(p, e, rg.case_id, rg.persons, how))
    ends_before = sorted(set(fixed_ends) | {d.end for d in docs})
    for q, rg in sorted(open_ends, key=lambda t: t[0]):
        prev = [e for e in ends_before if e < q] + [s - 1 for s in starts if s <= q]
        a = max(prev) + 1 if prev else 0
        while a < q and nondoc[a]:
            trimmed[a] = True
            a += 1
        how = {"previous_inventory": "end from excel, began in the previous inventory", "reversed": "end from excel (start scan after the end)"}.get(
            rg.open_start, "end from excel, start after the previous case")
        docs.append(_Doc(a, q, rg.case_id, rg.persons, how))
        ends_before = sorted(set(ends_before) | {q})

    covered[:] = trimmed
    for d in docs:
        covered[d.start : d.end + 1] = True

    # 3. the model segments every region outside the cases
    model_docs: list[tuple[int, int, str | None]] = []  # (start, end, case_id when a forced start)
    k = 0
    while k < n:
        if covered[k]:
            k += 1
            continue
        lo = k
        while k < n and not covered[k]:
            k += 1
        hi = k - 1
        part = type(sc)(**{f: getattr(sc, f)[lo : hi + 1] for f in ("start", "end", "log_shared", "log_not_shared", "nondoc")})
        f_here = {p - lo for p in forced if lo <= p <= hi}
        raw, _ = segment_scans(part, model.length_prior, model.max_length, f_here)
        for a, e, _ in raw:
            model_docs.append((a + lo, e + lo, forced[a + lo].case_id if a + lo in forced else None))

    # 4. assemble, in scan order
    items = []  # (start, -end, order, Segment)
    titles: dict[int, str] = {}
    fiscal = FiscalNames(scan_texts(inv), cases[ranges[0].case_id].place) if ranges else None
    for i, d in enumerate(docs):
        case = cases[d.case_id]
        titles[i] = case_title(case, d.persons, fiscal.for_span(d.start, d.end) if fiscal else None, fiscal.names if fiscal else [])
    seg_of_doc: dict[int, Segment] = {}
    for i, d in enumerate(docs):
        case = cases[d.case_id]
        sg = Segment(d.start, d.end, "subdoc" if d.parent is not None else "case", "", float(sc.start[d.start]), float(sc.end[d.end]))
        sg.court = _court_info(case, d, titles[i], len(d.persons))
        seg_of_doc[i] = sg
        items.append((d.start, -d.end, 0 if d.parent is None else 1, sg))
    for i, d in enumerate(docs):
        if d.parent is not None:
            seg_of_doc[i].court["parent_case"] = docs[d.parent].case_id
    for a, e, cid in model_docs:
        sg = Segment(a, e, "unindexed", "", float(sc.start[a]), float(sc.end[e]))
        if cid is not None:
            case = cases[cid]
            d = _Doc(a, e, cid, forced[a].persons, "start from excel, end from the model")
            sg.kind = "case"
            sg.court = _court_info(case, d, case_title(case, d.persons, fiscal.for_span(a, e) if fiscal else None, fiscal.names if fiscal else []), len(d.persons))
        items.append((a, -e, 2, sg))
    items.sort(key=lambda t: (t[0], t[1], t[2]))

    segments: list[Segment] = []
    prev_end = -1
    for a, neg_e, _, sg in items:
        e = -neg_e
        if sg.kind != "subdoc":
            if a > prev_end + 1:
                segments.append(Segment(prev_end + 1, a - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[a - 1])))
            sg.boundary = "first" if prev_end < 0 else "shared" if a <= prev_end else "adjacent" if a == prev_end + 1 else "gap"
            prev_end = max(prev_end, e)
        else:
            sg.boundary = "subdocument"
        segments.append(sg)
    if prev_end < n - 1:
        segments.append(Segment(prev_end + 1, n - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[-1])))
    return Result(inv.inventory_number, segments, [], 0.0)


def _court_info(case: Case, d: _Doc, title: str, n_persons: int) -> dict:
    return {
        "case_id": case.case_id,
        "title": title,
        "court": case.court,
        "place": case.place,
        "date_begin": case.date_begin.isoformat() if case.date_begin else None,
        "date_end": case.date_end.isoformat() if case.date_end else None,
        "accused": n_persons,
        "charges": case.charges,
        "how": d.how,
        "excel_scans": case.excel,
    }
