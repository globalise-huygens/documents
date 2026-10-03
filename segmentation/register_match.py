"""
Register entries matched to the documents of their own volume, by date and
title words, with the order of the entries (by folio, or by register order
where that follows the binding) to choose between candidates.

Documents: the volume segmented by the model alone (no ToC, so independent of
the cross-references), plus scans with some start probability that the model
did not use as a start.

Score of entry e against a document starting on scan s:
  - date: e's date (day-month-year) among the dates in the document's text;
    found in its last scans (a letter's closing formula) or first scans counts
    most, elsewhere in it less. Month spellings such as '7ber', 'xber' and a
    year cut short in the register ('177') are completed from the other
    entries of the register.
  - words: the IDF-weighted share of e's title words (normalised, rare words
    weigh most; words of nearly every document weigh nothing) in the opening
    and closing text of the document.
  - order: a monotone dynamic programme over the entries in order (by folio
    when most entries have one) prefers assignments in increasing scan order;
    entries can be left out.
The result per entry: the best candidate and two alternatives, with evidence.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .header_dates import parse_header_date
from .rows import MONTH
from .texts import normalize_tokens

DATE_RE = re.compile(r"\b(\d{1,2})\W{0,3}\s*(?:en|e|ste|de|sten|stij)?\W{0,2}\s*(" + MONTH + r")\W{0,3}\s*(?:a[:\.]?o\s*|anno\s*)?(1[67]\d\d)\b", re.I)
SHORT_YEAR = re.compile(r"\b(1[67]\d)\b(?!\d)")
GENERIC = {w[:5] for w in ("missive brief briefje copie copia copye extract origineel origineele door aan den der des van ende met "
                          "gouverneur raad raden heer heeren edele hoog agtb dato datum gedateerd gerigt geschreven alhier "
                          "nevens benevens als vooren voorsz dito idem register papieren brieven een twee drie welke "
                          "directeur generaal generale januarij februarij maart maert april meij maij junij julij augustus "
                          "september october november december 7ber 8ber 9ber xber anno").split()}
MIN_WORD_WEIGHT = 8.0  # summed IDF of an entry's distinctive words for a full word score
OPEN_SCANS, CLOSE_SCANS = 2, 2
TOP_K = 3


def text_dates(text: str) -> set:
    out = set()
    for m in DATE_RE.finditer(text or ""):
        d = parse_header_date(m.group(0))
        if d is not None and d.precision == "day":
            out.add(d.date)
    return out


CLOSING_TAIL = 350  # characters at the end of a scan where a closing formula's date stands
OPENING_DATO = re.compile(r"\b(in|de|sub)\s*dato\b", re.I)


def own_dates(text: str) -> set:
    """Dates that are a document's own: in the closing formula at the end of the text
    ('Kolombo den 13 November 1777', signatures) or after 'dato' in the opening of a copy;
    not the dates of other letters referred to in the body ('uwe missive van den 30 December')."""
    text = text or ""
    out = text_dates(text[-CLOSING_TAIL:])
    m = OPENING_DATO.search(text[:250])
    if m:
        out |= text_dates(text[m.start(): m.start() + 80])
    return out


def entry_date(text, register_year: int | None):
    """The entry's date; a year cut short ('177') is completed from the register's year."""
    if not isinstance(text, str):
        return None
    d = parse_header_date(text)
    if d is None and register_year:
        m = SHORT_YEAR.search(text)
        if m and str(register_year).startswith(m.group(1)):
            d = parse_header_date(text[: m.start()] + str(register_year) + text[m.end():])
    return d.date if d is not None and d.precision == "day" else None


def _tokens(text: str) -> list[str]:
    text = text if isinstance(text, str) else ""
    return [t for t in normalize_tokens(text, prefix=5) if t not in GENERIC and len(t) >= 4]


def _near_year(a: int, b: int) -> bool:
    """Years that differ in one digit (a misread or miswritten year: 1771 / 1777)."""
    return a != b and len(str(a)) == len(str(b)) and sum(x != y for x, y in zip(str(a), str(b))) == 1


def candidate_documents(inv, model, min_p_start: float = 0.1) -> pd.DataFrame:
    """Starts of the model's documents (no ToC) and other scans with p(start) >= min_p_start; each runs to the next start."""
    from .segmenter import scan_scores, segment_inventory

    res = segment_inventory(inv, model, use_toc=False)
    starts = {sg.start for sg in res.segments if sg.kind != "non-document"}
    sc = scan_scores(inv, model, use_toc=False)
    p = 1 / (1 + np.exp(-sc.start))
    starts |= set(np.flatnonzero(p >= min_p_start))
    starts = sorted(starts)
    ends = [b - 1 for b in starts[1:]] + [inv.n - 1]
    return pd.DataFrame({"start": starts, "end": ends, "p_start": p[starts], "model_start": [s in {sg.start for sg in res.segments} for s in starts]})


def match_volume(inv, model, entries: pd.DataFrame) -> pd.DataFrame:
    """Top candidates per entry (one row per entry and candidate)."""
    from .predictors import scan_texts

    texts = scan_texts(inv).fillna("")
    fn = inv.scans["filename"]
    register_scans = set()
    for first, last in entries[["register_first_scan", "register_last_scan"]].drop_duplicates().itertuples(index=False):
        prefix, a, b = first.rsplit("_", 1)[0], int(first.rsplit("_", 1)[1]), int(last.rsplit("_", 1)[1])
        register_scans |= {f"{prefix}_{k:04d}" for k in range(a, b + 1)}
    # register-like scans (also those the range detection missed) are no candidates and no source of dates or words
    reg_score = _register_scores(inv.inventory_number)
    registerish = {f for f in fn if f in register_scans or reg_score.get(f, 0) >= REGISTERISH}
    docs = candidate_documents(inv, model)
    folio_hits = _folio_positions(inv, entries)
    extra = sorted({p for p, _ in folio_hits.values() if p not in set(docs["start"]) and not any(abs(p - s) <= 1 for s in docs["start"])})
    if extra:  # the folio's scan as a candidate start of its own when the model has no start near it
        docs = pd.concat([docs, pd.DataFrame({"start": extra, "end": extra, "p_start": 0.0, "model_start": False})]).sort_values("start")
        docs["end"] = list(docs["start"].iloc[1:] - 1) + [inv.n - 1]
        docs = docs.reset_index(drop=True)
    docs = docs[~docs["start"].map(lambda s: fn.iat[s] in registerish)].reset_index(drop=True)
    texts = pd.Series([("" if f in registerish else t) for f, t in zip(fn, texts)], index=texts.index)
    scan_dates = [own_dates(t) for t in texts]
    # per document: dates in its opening / closing scans and anywhere, tokens of its opening and closing
    open_dates, close_dates, all_dates, toks = [], [], [], []
    for d in docs.itertuples():
        span = range(d.start, d.end + 1)
        o, c = range(d.start, min(d.end, d.start + OPEN_SCANS - 1) + 1), range(max(d.start, d.end - CLOSE_SCANS + 1), d.end + 1)
        open_dates.append(set().union(*(scan_dates[k] for k in o)))
        close_dates.append(set().union(*(scan_dates[k] for k in c)))
        all_dates.append(set().union(*(scan_dates[k] for k in span)) if len(span) <= 60 else set())
        toks.append(set(_tokens(" ".join(texts.iat[k] for k in sorted(set(o) | set(c))))))
    df = Counter(t for ts in toks for t in ts)
    n_docs = max(len(docs), 1)
    idf = {t: math.log((n_docs + 1) / (c + 0.5)) for t, c in df.items()}

    # register year (for dates with a cut-short year): the most common year of its complete dates
    years = Counter(d.year for d in (entry_date(t, None) for t in entries["date"]) if d)
    reg_year = years.most_common(1)[0][0] if years else None
    rows = []
    scores = np.zeros((len(entries), len(docs)))
    why = {}
    for i, e in enumerate(entries.itertuples()):
        date = entry_date(e.date, reg_year)
        et = [t for t in dict.fromkeys(_tokens(e.title)) if t in idf]
        w_total = sum(idf[t] for t in et)
        for j in range(len(docs)):
            s, reasons = 0.0, []
            if date is not None:
                if date in close_dates[j] or date in open_dates[j]:
                    s += 3.0
                    reasons.append(f"date {date} at the {'end' if date in close_dates[j] else 'start'}")
                elif date in all_dates[j]:
                    s += 1.5
                    reasons.append(f"date {date} in the text")
                else:
                    near = [d for d in close_dates[j] | open_dates[j] if (d.month, d.day) == (date.month, date.day) and _near_year(d.year, date.year)]
                    if near:
                        s += 2.0
                        reasons.append(f"date {near[0]} (register: {date.year})")
            if w_total > 0:
                hit = [t for t in et if t in toks[j]]
                share = sum(idf[t] for t in hit) / w_total
                s += 3.0 * share * min(1.0, w_total / MIN_WORD_WEIGHT)
                if hit:
                    reasons.append("words: " + ", ".join(sorted(hit, key=lambda t: -idf[t])[:6]))
            hit = folio_hits.get(i)
            if hit is not None and abs(int(docs.at[j, "start"]) - hit[0]) <= 1:
                s += 4.0 if hit[1] else 3.0
                reasons.append(f"folio {int(e.folio_start)} {'on' if hit[1] else 'interpolated at'} scan {fn.iat[hit[0]].rsplit('_', 1)[1]}")
            scores[i, j] = s
            if s > 0:
                why[(i, j)] = "; ".join(reasons)
    # an entry's document follows its register: candidates after the register's last page and
    # before the next register (a scan carrying the entry's own folio number is the exception)
    spans = _register_spans(entries, fn)
    starts = docs["start"].to_numpy()
    for i, e in enumerate(entries.itertuples()):
        lo, hi = spans[e.register_first_scan]
        outside = (starts < lo) | (starts > hi)
        hit = folio_hits.get(i)
        if hit is not None and hit[1]:
            outside &= np.abs(starts - hit[0]) > 1
        scores[i, outside] = -1.0
    order = _order(entries)
    best = _monotone(scores, order)
    reg_entries = entries.groupby("register_first_scan")["entry"].agg(list).to_dict()
    for i, e in enumerate(entries.itertuples()):
        scored = [j for j in np.argsort(-scores[i]) if scores[i, j] > 0][:TOP_K]
        if best.get(i) is not None and best[i] not in scored:
            scored = [best[i]] + scored[: TOP_K - 1]
        top = sorted(scored, key=lambda j: (j != best.get(i), -scores[i, j]))
        # fill up with the model's starts where the entry's place in the register points (entry k of K → k/K into the span)
        lo, hi = spans[e.register_first_scan]
        ks = reg_entries[e.register_first_scan]
        frac = (ks.index(e.entry) + 0.5) / len(ks)
        target = lo + frac * max(hi - lo, 0)
        inside = [j for j in range(len(docs)) if lo <= starts[j] <= hi and j not in top]
        for j in sorted(inside, key=lambda j: abs(starts[j] - target))[: max(0, TOP_K - len(top))]:
            why[(i, j)] = f"position in the register (entry {ks.index(e.entry) + 1} of {len(ks)})"
            top.append(j)
        for j in range(len(docs)):  # still too few (a short span): the next candidate starts after the best one
            if len(top) >= TOP_K or not top:
                break
            if starts[j] > starts[top[0]] and j not in top:
                why[(i, j)] = "a later document start (in case the document starts a little later)"
                top.append(j)
        for rank, j in enumerate(top):
            rows.append({
                "inventory": inv.inventory_number, "register_first_scan": e.register_first_scan, "entry": int(e.entry), "entry_scan": e.scan,
                "x0": e.x0, "y0": e.y0, "x1": e.x1, "y1": e.y1,
                "title": e.title, "date": e.date, "folio_start": e.folio_start, "item": e.item,
                "rank": rank, "candidate_scan": fn.iat[int(docs.at[j, "start"])], "candidate_end_scan": fn.iat[int(docs.at[j, "end"])],
                "score": round(float(max(scores[i, j], 0)), 2), "in_order": best.get(i) == j, "model_start": bool(docs.at[j, "model_start"]),
                "p_start": round(float(docs.at[j, "p_start"]), 2), "why": why.get((i, j), ""),
            })
    return pd.DataFrame(rows)


REGISTERISH = 0.3
_SCORES = {}


def _register_scores(inventory: str) -> dict:
    """Register-page scores of a volume (data/registers/scores.parquet, register-ranges)."""
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "registers", "scores.parquet")
    if "all" not in _SCORES:
        _SCORES["all"] = pd.read_parquet(path, columns=["inventory", "filename", "score"]) if os.path.exists(path) else pd.DataFrame(columns=["inventory", "filename", "score"])
    s = _SCORES["all"]
    s = s[s["inventory"] == inventory]
    return dict(zip(s["filename"], s["score"]))


def _folio_positions(inv, entries: pd.DataFrame) -> dict[int, tuple[int, bool]]:
    """Scan of each entry's first folio from the volume's page numbers: (position, exact) where numbered, else interpolated."""
    from .predictors import FolioPredictor
    from .segmenter import NumberIndex

    f = FolioPredictor().compute(inv)
    number, run = f["_number"].to_numpy(dtype=float), f["_run"].to_numpy()
    if np.isfinite(number).sum() < 3:
        return {}
    idx = NumberIndex(number, run, inv.scans["layout"])
    out = {}
    for i, fs in enumerate(entries["folio_start"]):
        if pd.isna(fs):
            continue
        exact = np.flatnonzero(number == fs)
        if len(exact):
            out[i] = (int(exact[0]), True)
            continue
        pos = idx.positions(float(fs), margin=10)
        if len(pos) == 1:  # only when the number fits one numbering run
            out[i] = (int(round(pos[0][1])), False)
    return out


def _register_spans(entries: pd.DataFrame, fn: pd.Series, gap: int = 3) -> dict[str, tuple[int, int]]:
    """
    Per register (first scan): the scan positions its documents can be in. Registers
    at most `gap` scans apart form one group (one register over several detected
    ranges, or registers of one shipment); a group's documents follow its last page,
    up to the next group or the end of the volume. A group with no room after it
    (a register at the end of the volume) looks back to the previous group.
    """
    pos = {f: k for k, f in enumerate(fn)}
    regs = entries[["register_first_scan", "register_last_scan"]].drop_duplicates().copy()
    regs["a"], regs["b"] = regs["register_first_scan"].map(pos), regs["register_last_scan"].map(pos)
    regs = regs.dropna().sort_values("a")
    groups: list[list] = []
    for r in regs.itertuples():
        if groups and r.a - groups[-1][-1].b <= gap + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = {}
    last = len(fn) - 1
    for k, g in enumerate(groups):
        a, b = int(g[0].a), int(max(r.b for r in g))
        nxt = int(groups[k + 1][0].a) - 1 if k + 1 < len(groups) else last
        prev = int(max(r.b for r in groups[k - 1])) + 1 if k > 0 else 0
        span = (b + 1, nxt) if nxt - b >= 2 else (prev, a - 1)
        for r in g:
            out[r.register_first_scan] = span
    return out


def _order(entries: pd.DataFrame) -> list[int]:
    """Entry indices in the order their documents should follow in the volume."""
    if entries["folio_start"].notna().mean() >= 0.5:
        e = entries.reset_index(drop=True)
        return list(e.sort_values(["folio_start", "register_first_scan", "entry"], na_position="last").index)
    return list(range(len(entries)))  # register order (item numbers follow the binding)


def _monotone(scores: np.ndarray, order: list[int], min_score: float = 1.5) -> dict[int, int]:
    """
    Best assignment of the entries, in `order`, to documents with non-decreasing
    document index (several entries may share a document); an entry may be left
    out, and pairs scoring below min_score are not used. {entry: document}.
    """
    n_e, n_d = scores.shape
    if n_e == 0 or n_d == 0:
        return {}
    g = np.zeros(n_d)  # best total so far with the last used document <= j
    idx_hist, choose_hist = [np.arange(n_d)], []
    for k in order:
        s = np.where(scores[k] >= min_score, scores[k], -np.inf)
        assign = g + s  # this entry on document j, the previous ones on documents <= j
        h = np.maximum(g, assign)
        choose_hist.append(assign > g)
        idx = np.zeros(n_d, dtype=int)
        best_j = 0
        for j in range(n_d):
            if h[j] > h[best_j]:
                best_j = j
            idx[j] = best_j
        g = h[idx]
        idx_hist.append(idx)
    out, j = {}, int(idx_hist[-1][n_d - 1])
    for step in range(len(order) - 1, -1, -1):
        if choose_hist[step][j]:
            out[order[step]] = j
        j = int(idx_hist[step][j])
    return out


def match_all(inventories: list[str], entries: dict, out: str, workers: int = 6):
    """Candidates for the register entries of every inventory, with progress."""
    import logging
    import time
    from multiprocessing import Pool

    logger = logging.getLogger("segmentation")
    todo = [i for i in inventories if i in entries]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_match_one, [(i, entries[i]) for i in todo], chunksize=4), 1):
            rows.append(r)
            if k % 100 == 0 or k == len(todo):
                logger.info("  matched %d/%d inventories (%.0fs)", k, len(todo), time.time() - t0)
    res = pd.concat([r for r in rows if len(r)], ignore_index=True)
    res.to_csv(out, index=False)
    return res


def _match_one(args):
    import logging

    from .inventory import connect, load_inventory
    from .model import SegmentationModel

    inv_number, entries = args
    logging.disable(logging.INFO)
    try:
        conn, model = connect(), SegmentationModel.load()
        words_based = match_volume(load_inventory(conn, inv_number), model, entries).assign(method="words, dates, folios")
        numbered = match_numbered(load_inventory(conn, inv_number), model, entries)
        if len(numbered) and len(words_based):  # numbered registers: candidates from the document numbers replace the others
            key = set(zip(numbered["register_first_scan"], numbered["entry"]))
            words_based = words_based[[(a, b) not in key for a, b in zip(words_based["register_first_scan"], words_based["entry"])]]
        return pd.concat([words_based, numbered], ignore_index=True)
    except Exception:
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def _evidence(why: str, method: str = "", score: float = 0) -> str:
    if method == "document number":
        return "N°: start after the previous N°" if score >= 5 else "N°: between marks"
    if "folio" in why and " on scan" in why:
        return "folio on the scan"
    if "folio" in why:
        return "folio interpolated"
    if "date" in why:
        return "date"
    if "words" in why:
        return "words only"
    return "order only"


def link_sample(candidates: pd.DataFrame, n: int = 80, seed: int = 3) -> pd.DataFrame:
    """Entries to review, stratified by the evidence of their best candidate; one row per entry
    with its candidates (best first) as JSON."""
    import json

    c = candidates.copy()
    c["entry_key"] = c["register_first_scan"] + ":" + c["entry"].astype(str)
    best = c[c["rank"] == 0].copy()
    best["stratum"] = [_evidence(w, m, sc) for w, m, sc in zip(best["why"].fillna(""), best.get("method", pd.Series("", index=best.index)).fillna(""), best["score"])]
    sizes = best["stratum"].value_counts()
    per = max(1, n // len(sizes))
    pick = best.groupby("stratum", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
    rows = []
    for b in pick.itertuples():
        cs = c[c["entry_key"] == b.entry_key].sort_values("rank")
        rows.append({
            "entry_key": b.entry_key, "inventory": b.inventory, "stratum": b.stratum, "stratum_size": int(sizes[b.stratum]),
            "register_scan": b.entry_scan, "title": b.title,
            "entry_box": json.dumps([b.x0, b.y0, b.x1, b.y1]) if not pd.isna(b.x0) else None, "date": b.date if isinstance(b.date, str) else None,
            "folio": None if pd.isna(b.folio_start) else str(int(b.folio_start)),
            "candidates": json.dumps([{"scan": r.candidate_scan, "score": r.score, "why": r.why if isinstance(r.why, str) else ""} for r in cs.itertuples()]),
        })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── numbered registers: document numbers on the documents' last pages ─────────


MIN_NUMBERED = 3


def _items(entries: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(entries["item"], errors="coerce")


def numbered_registers(entries: pd.DataFrame) -> list[str]:
    """Registers (first scan) whose entries are mostly numbered."""
    it = _items(entries)
    g = pd.DataFrame({"r": entries["register_first_scan"], "has": it.notna()}).groupby("r")["has"].agg(["sum", "mean"])
    return list(g[(g["sum"] >= MIN_NUMBERED) & (g["mean"] >= 0.5)].index)


def match_numbered(inv, model, entries: pd.DataFrame) -> pd.DataFrame:
    """Candidates for the entries of numbered registers, from their document-number marks."""
    from .docnumbers import align, volume_marks
    from .predictors import scan_texts

    regs = numbered_registers(entries)
    if not regs:
        return pd.DataFrame()
    fn = inv.scans["filename"]
    pos = {f: k for k, f in enumerate(fn)}
    words = scan_texts(inv).fillna("").str.split().str.len().to_numpy()
    marks = volume_marks(inv.inventory_number)
    marks = marks.assign(pos=marks["filename"].map(pos)).dropna(subset=["pos"])
    marks["pos"] = marks["pos"].astype(int)
    marks = marks.sort_values(["pos", "left"], ascending=[True, False]).reset_index(drop=True)
    spans = _register_spans(entries, fn)
    model_starts = sorted(set(candidate_documents(inv, model)["start"]))
    rows = []

    def next_start(m) -> tuple[int, str]:
        """Where the document after mark m starts: the same scan (mark on the left page of a spread
        with text on the right) or the first scan with text after it."""
        if m.left and words[m.pos] >= 15:
            return m.pos, f"N° {m.n} on the left page of scan {fn.iat[m.pos].rsplit('_', 1)[1]}, the next document on the right"
        k = m.pos + 1
        while k < len(fn) and words[k] < 5:
            k += 1
        return min(k, len(fn) - 1), f"after N° {m.n} on scan {fn.iat[m.pos].rsplit('_', 1)[1]}"

    for reg in regs:
        lo, hi = spans[reg]
        e = entries[(entries["register_first_scan"] == reg)].copy()
        e["it"] = _items(e)
        e = e[e["it"].notna()].sort_values("entry")
        mk = marks[(marks["pos"] >= lo) & (marks["pos"] <= hi)].reset_index(drop=True)
        matched = align([int(x) for x in e["it"]], mk)
        ends = {i: mk.iloc[j] for i, j in matched.items()}
        for i, r in enumerate(e.itertuples()):
            prev = max((p for p in ends if p < i), default=None)
            nxt = min((p for p in ends if p >= i), default=None)
            win_lo, why_lo = (lo, "the first scan after the register") if prev is None else next_start(ends[prev])
            end = ends[i].pos if i in ends else (ends[nxt].pos if nxt is not None else hi)
            certain = (prev == i - 1) or (prev is None and i == 0)  # the mark of the entry just before it was found (or it is the first)
            own = f"; its own N° {int(r.it)} on scan {fn.iat[ends[i].pos].rsplit('_', 1)[1]}" if i in ends else ""
            if certain:
                cands = [(win_lo, 5.0 + (1.0 if own else 0.0), (why_lo if prev is not None else "first entry: " + why_lo) + own)]
                others = [s for s in model_starts if win_lo < s <= end]
            else:
                # entries without a mark between two marks: the model's starts in between, by the entry's place among them
                k_prev = prev if prev is not None else -1
                k_next = i + 1 if i in ends else (nxt if nxt is not None else len(e))
                frac = (i - k_prev) / max(k_next - k_prev, 1)
                inside = [s for s in model_starts if win_lo <= s <= end] or [win_lo]
                target = win_lo + frac * max(end - win_lo, 0)
                ranked = sorted(inside, key=lambda s: abs(s - target))
                gap = f"entry {i - k_prev} of {k_next - k_prev - 1} without the mark of the entry before it"
                cands = [(ranked[0], 2.0, f"{why_lo[0].upper() + why_lo[1:]}, before {('its own N° ' + str(int(r.it))) if own else 'the next mark'} ({gap}); a document start of the model" + own)]
                others = ranked[1:]
            for s in others[: TOP_K - 1]:
                cands.append((s, 1.0, f"a document start of the model before {'its own N°' if own else 'the next mark'}"))
            k = cands[0][0] + 1  # too few: the next scans with text after the best candidate
            while len(cands) < TOP_K and k < len(fn):
                if words[k] >= 5 and k not in {c[0] for c in cands}:
                    cands.append((k, 0.5, "a later scan with text (in case the document starts a little later)"))
                k += 1
            for rank, (s, sc, w) in enumerate(cands):
                rows.append({
                    "inventory": inv.inventory_number, "register_first_scan": r.register_first_scan, "entry": int(r.entry), "entry_scan": r.scan,
                    "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1, "title": r.title, "date": r.date, "folio_start": r.folio_start, "item": r.item,
                    "rank": rank, "candidate_scan": fn.iat[s], "candidate_end_scan": fn.iat[end], "score": sc, "in_order": rank == 0,
                    "model_start": s in model_starts, "p_start": None, "why": w, "method": "document number",
                })
    return pd.DataFrame(rows)
