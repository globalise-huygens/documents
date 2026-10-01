"""
Find the best segmentation of one inventory.

1. Per-scan start / shared-start log-odds from the boundary model.
2. ToC alignment: place the ToC entries (in toc_order) on start scans with a
   monotone dynamic program. Candidate start scans come from observed page/
   folio numbers (exact matches, or interpolation within a numbering run);
   the score adds the start log-odds, header-date agreement, and the fit
   between the gap to the next entry and the span implied by the numbers.
   Entries can be left out of this pass at a cost; a second pass then places
   every remaining entry between its placed neighbours, in index order, on the
   best start evidence, so every ToC entry is found.
3. Fine segmentation: a semi-Markov dynamic program over scans that scores
   document starts, document ends, a length prior, the type of every boundary
   (next document on the same scan / on the next scan / after non-document
   scans such as covers, blanks and ToC pages) and the non-document scans
   themselves. The ToC starts from step 2 are forced.
4. Every document is labelled: 'toc' (starts a ToC entry), 'subdoc' (inside
   the page range of the ToC entry it follows, e.g. an enclosure or appendix),
   or 'unindexed' (outside any entry's range, or no ToC at all); runs of
   non-document scans are reported as 'non-document'.
"""

import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .header_dates import HeaderDate
from .inventory import InventoryData
from .model import AlignParams, SegmentationModel
from .predictors import ToCNumberPredictor, ToCTextPredictor, compute_features

BIG = 1e6


# ── page/folio number index ───────────────────────────────────────────────────


class NumberIndex:
    """Observed page/folio numbers per numbering run, with interpolation."""

    def __init__(self, number: np.ndarray, run: np.ndarray, layout: pd.Series):
        self.n = len(number)
        self.run = run
        self.points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for r in np.unique(run):
            idx = np.flatnonzero((run == r) & ~np.isnan(number))
            if len(idx) >= 1:
                self.points[r] = (idx, number[idx])
        double_share = (layout == "double").mean() if len(layout) else 0
        self.default_rate = 1.0 if double_share < 0.5 else 0.5
        self._rates: dict[int, float] = {}

    def rate(self, r: int) -> float:
        """Median scans per number step in run r (pagination/foliation × layout)."""
        if r not in self._rates:
            self._rates[r] = self._rate(r)
        return self._rates[r]

    def _rate(self, r: int) -> float:
        if r not in self.points or len(self.points[r][0]) < 2:
            return self.default_rate
        idx, val = self.points[r]
        di, dv = np.diff(idx), np.diff(val)
        ok = (dv > 0) & (dv <= 20)
        if not ok.any():
            return self.default_rate
        return float(np.median(di[ok] / dv[ok]))

    def foliated(self, r: int, layout: str) -> bool:
        """Numbers count leaves (recto and verso share a number) rather than pages:
        ~2 scans per number on single scans, ~1 on double scans."""
        return self.rate(r) >= (1.5 if layout == "single" else 0.75)

    def positions(self, number: float, margin: int = 30) -> list[tuple[int, float]]:
        """Interpolated scan positions of `number`, one per run that covers it."""
        out = []
        for r, (idx, val) in self.points.items():
            lo, hi = val.min(), val.max()
            if not (lo - margin <= number <= hi + margin):
                continue
            order = np.argsort(val, kind="stable")
            v_sorted, i_sorted = val[order], idx[order]
            k = np.searchsorted(v_sorted, number)
            rate = self.rate(r)
            if k == 0:
                pos = i_sorted[0] - (v_sorted[0] - number) * rate
            elif k == len(v_sorted):
                pos = i_sorted[-1] + (number - v_sorted[-1]) * rate
            else:
                v1, v2, i1, i2 = v_sorted[k - 1], v_sorted[k], i_sorted[k - 1], i_sorted[k]
                pos = i1 + (number - v1) * (i2 - i1) / (v2 - v1) if v2 > v1 else i1
            run_scans = np.flatnonzero(self.run == r)
            pos = min(max(pos, run_scans[0]), run_scans[-1])
            out.append((r, float(pos)))
        return out

    def number_at(self, pos: int) -> float:
        r = self.run[pos]
        if r not in self.points:
            return np.nan
        idx, val = self.points[r]
        before, after = idx[idx <= pos], idx[idx >= pos]
        if len(before) and len(after):
            i1, i2 = before[-1], after[0]
            v1, v2 = val[idx == i1][0], val[idx == i2][0]
            return v1 if i1 == i2 else v1 + (pos - i1) * (v2 - v1) / (i2 - i1)
        rate = self.rate(r)
        if len(before):
            return val[idx == before[-1]][0] + (pos - before[-1]) / rate
        return val[idx == after[0]][0] - (after[0] - pos) / rate


# ── ToC alignment ─────────────────────────────────────────────────────────────


def scan_dates(dates: pd.Series, lookahead: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per scan: ordinal of the first day/month-precision header date on scans
    i..i+lookahead (nan if none), and whether that date is month-precision only."""
    n = len(dates)
    ordinal = np.full(n, np.nan)
    month_only = np.zeros(n, dtype=bool)
    for i in range(n):
        for d in dates.iloc[i : i + lookahead + 1]:
            if d is not None and d.precision != "year":
                ordinal[i], month_only[i] = d.date.toordinal(), d.precision == "month"
                break
    return ordinal, month_only


def date_terms(entry, sd: tuple[np.ndarray, np.ndarray], p: AlignParams) -> np.ndarray:
    """Date agreement score of `entry` starting at each scan."""
    ordinal, month_only = sd
    b, e = entry.date_begin, entry.date_end
    if b is None or pd.isna(b):
        return np.zeros(len(ordinal))
    e = b if e is None or pd.isna(e) else e
    lo = np.where(month_only, b.replace(day=1).toordinal(), b.toordinal())
    hi = np.where(month_only, e.replace(day=1).toordinal(), e.toordinal())
    diff = np.where(ordinal < lo, ordinal - lo, np.where(ordinal > hi, ordinal - hi, 0))
    out = np.where((diff >= -30) & (diff <= 365), p.date_match, np.where(np.abs(diff) > 730, p.date_mismatch, 0.0))
    return np.where(np.isnan(ordinal), 0.0, out)


@dataclass
class Placement:
    entry: int  # row in inv.toc
    scan: int
    score: float
    how: str  # 'exact' | 'interpolated' (page numbers) | 'text' (description matches the text) | 'evidence' | 'forced' (placed between neighbours)


def _candidates(entry, numbers: NumberIndex, number: np.ndarray, layout: np.ndarray, s, dterm: np.ndarray, sim: np.ndarray, p: AlignParams, n: int):
    cand: dict[int, tuple[float, str]] = {}

    def add(pos, score, how):
        if 0 <= pos < n and (pos not in cand or cand[pos][0] < score):
            cand[pos] = (score, how)

    # scans whose opening words (or title page) match the entry's description
    for j in np.argsort(-sim)[: p.text_candidates]:
        if sim[j] >= p.text_min:
            add(int(j), 0.0, "text")
    fs = entry.folio_start
    if fs is None or pd.isna(fs):
        return _score_candidates(cand, s, dterm, p)

    verso = entry.start_side == "Verso"
    for i in np.flatnonzero(number == fs):
        if numbers.foliated(numbers.run[i], layout[i]):
            # the numbered scan shows the recto; the verso of the same leaf is on the next scan
            add(i, p.exact_number + (p.verso_shift if verso else 0), "exact")
            add(i + 1, p.exact_number + (0 if verso else p.verso_shift), "exact")
        else:
            add(i, p.exact_number, "exact")
    for _, pos in numbers.positions(float(fs)):
        for c in range(int(round(pos)) - p.interp_window, int(round(pos)) + p.interp_window + 1):
            add(c, p.interp_base - p.interp_per_scan * abs(c - pos), "interpolated")
    return _score_candidates(cand, s, dterm, p)


def _score_candidates(cand, s, dterm, p: AlignParams):
    scored = [(c, base + p.w_start * s[c] + dterm[c], how) for c, (base, how) in cand.items()]
    scored.sort(key=lambda t: -t[1])
    return scored[: p.max_candidates]


def align_toc(inv: InventoryData, s: np.ndarray, p: AlignParams) -> list[Placement]:
    toc = inv.toc[inv.toc["toc_order"].notna()].reset_index()  # 'index' = row in inv.toc
    if toc.empty:
        return []
    f = inv.features
    number = f["_number"].to_numpy()
    numbers = NumberIndex(number, f["_run"].to_numpy(), inv.scans["layout"])
    sd = scan_dates(f["_header_date"])
    n = inv.n
    sims = inv.toc_text_sim if inv.toc_text_sim is not None and len(inv.toc_text_sim) == len(toc) else np.zeros((len(toc), n))
    # extra evidence per entry and scan: header date agreement + description matching the opening words
    dterms = [date_terms(e, sd, p) + p.text_weight * np.clip(sims[k] - p.text_floor, 0, None) for k, e in enumerate(toc.itertuples())]
    fs_arr = toc["folio_start"].to_numpy(dtype=float)

    layout = inv.scans["layout"].to_numpy()
    cands = [_candidates(e, numbers, number, layout, s, dterms[k], sims[k], p, n) for k, e in enumerate(toc.itertuples())]
    placeable = [k for k in range(len(toc)) if cands[k]]

    # DP over (placeable entry, candidate): best[t][c]
    best: list[np.ndarray] = []
    back: list[list[tuple[int, int] | None]] = []
    for t, k in enumerate(placeable):
        ck = cands[k]
        scores = np.full(len(ck), -np.inf)
        bp: list[tuple[int, int] | None] = [None] * len(ck)
        for ci, (pos, local, _) in enumerate(ck):
            scores[ci] = local + p.skip * t  # all earlier placeable entries skipped
            for tp in range(max(0, t - p.max_skip - 1), t):
                kp = placeable[tp]
                for cpi, (ppos, _, _) in enumerate(cands[kp]):
                    if ppos > pos or not np.isfinite(best[tp][cpi]):
                        continue
                    v = best[tp][cpi] + local + p.skip * (t - tp - 1)
                    if ppos == pos:
                        v += p.same_start
                    v += _length_term(fs_arr[kp], fs_arr[k], ppos, pos, numbers, p)
                    if v > scores[ci]:
                        scores[ci], bp[ci] = v, (tp, cpi)
        best.append(scores)
        back.append(bp)

    placements: list[Placement] = []
    if placeable:
        total = [best[t].max() + p.skip * (len(placeable) - 1 - t) if len(best[t]) else -np.inf for t in range(len(placeable))]
        t = int(np.argmax(total))
        ci = int(np.argmax(best[t]))
        while True:
            k = placeable[t]
            pos, local, how = cands[k][ci]
            placements.append(Placement(int(toc.at[k, "index"]), int(pos), float(local), how))
            if back[t][ci] is None:
                break
            t, ci = back[t][ci]
        placements.reverse()

    placements += _place_unnumbered(inv, toc, placements, s, dterms, p)
    placements.sort(key=lambda pl: (pl.scan, inv.toc.at[pl.entry, "toc_order"]))
    return placements


def _length_term(fs0: float, fs1: float, ppos: int, pos: int, numbers: NumberIndex, p: AlignParams) -> float:
    if np.isnan(fs0) or np.isnan(fs1) or fs1 < fs0 or numbers.run[ppos] != numbers.run[pos]:
        return 0.0
    expected = (fs1 - fs0) * numbers.rate(numbers.run[ppos])
    return -p.length_weight * abs(math.log((pos - ppos + 1) / (expected + 1)))


def _place_unnumbered(inv, toc, placed: list[Placement], s, dterms, p: AlignParams) -> list[Placement]:
    """
    Place every entry the number-based alignment left out, between its placed
    neighbours in index order, on the scans with the best start evidence
    (start log-odds + header-date agreement). Positions are non-decreasing;
    sharing a scan with a neighbour is allowed at the same_start penalty (for
    when there is no room). A placement is 'evidence' when its evidence clears
    unnumbered_threshold, 'forced' otherwise.
    """
    placed_rows = {pl.entry for pl in placed}
    row_to_k = {int(r): k for k, r in enumerate(toc["index"])}
    anchors = [(-1, -1)] + sorted((row_to_k[pl.entry], pl.scan) for pl in placed) + [(len(toc), inv.n)]
    out = []
    for (ka, ca), (kb, cb) in zip(anchors, anchors[1:]):
        todo = [k for k in range(ka + 1, kb) if int(toc.at[k, "index"]) not in placed_rows]
        if not todo:
            continue
        lo, hi = max(ca, 0), min(cb, inv.n - 1)  # inclusive; the ends share a scan with an anchor
        J = hi - lo + 1
        pos = np.arange(lo, hi + 1)
        edge_penalty = np.where((pos == ca) | (pos == cb), p.same_start, 0.0)
        gain = np.array([p.w_start * s[lo : hi + 1] + dterms[k][lo : hi + 1] + edge_penalty for k in todo])
        U = len(todo)
        best = np.full((U, J), -np.inf)
        from_same = np.zeros((U, J), dtype=bool)  # True: previous entry on the same scan
        best[0] = gain[0]
        for u in range(1, U):
            prefix = np.maximum.accumulate(best[u - 1])  # best previous position <= j
            earlier = np.concatenate([[-np.inf], prefix[:-1]])  # previous position < j
            same = best[u - 1] + p.same_start
            from_same[u] = same > earlier
            best[u] = gain[u] + np.maximum(earlier, same)
        j = int(np.argmax(best[U - 1]))
        for u in range(U - 1, -1, -1):
            g = float(gain[u][j])
            out.append(Placement(int(toc.at[todo[u], "index"]), lo + j, g, "evidence" if g >= p.unnumbered_threshold else "forced"))
            if u == 0:
                break
            if from_same[u][j]:
                continue
            j = int(np.argmax(best[u - 1][:j]))
    return out


# ── fine segmentation ─────────────────────────────────────────────────────────


@dataclass
class ScanScores:
    """Per-scan log-odds / log-probabilities used by the segmentation DP."""

    start: np.ndarray  # a document starts on scan i
    end: np.ndarray  # a document ends on scan i
    log_shared: np.ndarray  # log P(start shares its scan with the previous end | start)
    log_not_shared: np.ndarray
    nondoc: np.ndarray  # scan i belongs to no document (cover, blank, ToC page, ...)


def segment_scans(sc: ScanScores, length_prior, max_length: int, forced=frozenset(), use_prior: np.ndarray | None = None):
    """
    Semi-Markov DP over scans. A segmentation is a sequence of documents
    [a, e]; between two documents the boundary is 'shared' (the next starts on
    the previous one's last scan), 'adjacent' (it starts on the next scan) or
    'gap' (non-document scans in between). Score = sum over documents of
    start[a] + end[e] + length prior, plus log P(shared)/log P(not shared) per
    boundary, plus nondoc[i] for every scan in a gap (including before the
    first and after the last document).

    Returns ([(start, end, boundary_type)], score); boundary_type of the first
    document is 'first'.
    """
    n = len(sc.start)
    L = min(max_length, n)
    lp = np.concatenate([[0.0], length_prior(np.arange(1, L + 1))])
    prior_on = np.ones(n) if use_prior is None else use_prior.astype(float)
    start = sc.start.astype(float).copy()
    for f_ in forced:
        start[f_] += BIG
    C = np.concatenate([[0.0], np.cumsum(sc.nondoc)])  # C[x+1] = sum nondoc[0..x]; gap(a..b) = C[b+1]-C[a]

    NEG = -np.inf
    B = np.full(n, NEG)  # best score up to and including the start of a document at a
    btype = np.zeros(n, dtype=np.int8)  # 0 first, 1 shared, 2 adjacent, 3 gap
    bprev = np.full(n, -1)  # end of the previous document (gap case)
    D = np.full(n, NEG)  # best score with a document ending at e
    D2 = np.full(n, NEG)  # ... of length >= 2
    d_single = np.zeros(n, dtype=bool)
    bp2 = np.zeros(n, dtype=int)
    M = np.full(n, NEG)  # max over e' <= x of D[e'] - C[e'+1]  (for gaps)
    Marg = np.full(n, -1)
    for x in range(n):
        # documents [a, x] with a <= x-1
        a0 = max(0, x - L + 1)
        if x - a0 >= 1:
            vals = B[a0:x] + lp[x - np.arange(a0, x) + 1] * prior_on[a0:x]
            k = int(np.argmax(vals))
            D2[x], bp2[x] = vals[k] + sc.end[x], a0 + k
        # a document starting at x
        opts = [(C[x], 0, -1)]  # first document, everything before is a gap
        if x >= 1:
            opts.append((D2[x] + sc.log_shared[x], 1, x))
            opts.append((D[x - 1] + sc.log_not_shared[x], 2, x - 1))
        if x >= 2 and M[x - 2] > NEG:
            opts.append((M[x - 2] + C[x] + sc.log_not_shared[x], 3, Marg[x - 2]))
        best = max(opts, key=lambda o: o[0])
        B[x], btype[x], bprev[x] = start[x] + best[0], best[1], best[2]
        single = B[x] + lp[1] * prior_on[x] + sc.end[x]
        D[x], d_single[x] = (single, True) if single > D2[x] else (D2[x], False)
        cand = D[x] - C[x + 1]
        if x == 0 or cand > M[x - 1]:
            M[x], Marg[x] = cand, x
        else:
            M[x], Marg[x] = M[x - 1], Marg[x - 1]

    e = int(Marg[n - 1])
    total = float(M[n - 1] + C[n]) - BIG * len(forced)
    segs = []
    table = "D"
    while e >= 0:
        a = e if table == "D" and d_single[e] else int(bp2[e])
        t = ("first", "shared", "adjacent", "gap")[btype[a]]
        segs.append((a, e, t))
        if t == "first":
            break
        e, table = (a, "D2") if t == "shared" else (int(bprev[a]), "D")
    segs.reverse()
    return segs, total


# ── assembly ──────────────────────────────────────────────────────────────────


@dataclass
class Segment:
    start: int  # first scan position
    end: int  # last scan position
    kind: str  # toc | subdoc | unindexed | non-document
    boundary: str  # first | shared | adjacent | gap (how it follows the previous document)
    start_logit: float
    end_logit: float
    toc_rows: list[int] = field(default_factory=list)  # rows in inv.toc starting here
    parent_row: int | None = None  # ToC row this segment is a subdocument of
    align_score: float | None = None
    court: dict | None = None  # court case from the EMDCCR dataset (court_records.py)
    derived: dict | None = None  # ToC entry derived from another version of the text (versions.py)


@dataclass
class Result:
    inventory_number: str
    segments: list[Segment]
    placements: list[Placement]
    score: float

    def documents(self) -> list[Segment]:
        return [sg for sg in self.segments if sg.kind != "non-document"]


def _nesting(inv, placements: list[Placement]) -> dict[int, int]:
    """Placed ToC entries whose page range lies within an earlier entry's range."""
    parent: dict[int, int] = {}
    stack: list[int] = []
    for pl in placements:
        e = inv.toc.loc[pl.entry]
        while stack:
            top = inv.toc.loc[stack[-1]]
            if (
                not pd.isna(e.folio_start)
                and not pd.isna(top.end_eff)
                and top.folio_sequence == e.folio_sequence
                and top.folio_start <= e.folio_start
                and e.end_eff <= top.end_eff
                and (e.folio_start, e.end_eff) != (top.folio_start, top.end_eff)
            ):
                parent[pl.entry] = stack[-1]
                break
            stack.pop()
        stack.append(pl.entry)
    return parent


def _entry_spans(inv: InventoryData, placements: list[Placement]) -> list[tuple[int, int]]:
    """Scan span of each placed entry: to the next placement, capped at the
    span its page range implies (with some slack)."""
    f = inv.features
    numbers = NumberIndex(f["_number"].to_numpy(), f["_run"].to_numpy(), inv.scans["layout"])
    starts = sorted({pl.scan for pl in placements}) + [inv.n]
    spans = []
    for pl in placements:
        nxt = next(x for x in starts if x > pl.scan)
        e = inv.toc.loc[pl.entry]
        end = nxt - 1
        if not pd.isna(e.folio_start) and not pd.isna(e.end_eff):
            implied = (e.end_eff - e.folio_start + 1) * numbers.rate(numbers.run[pl.scan])
            end = min(end, pl.scan + int(math.ceil(implied * 1.2)) + 2)
        spans.append((pl.scan, end))
    return spans


def scan_scores(inv: InventoryData, model: SegmentationModel, use_toc: bool = True) -> ScanScores:
    if inv.features is None:
        compute_features(inv)
    f = inv.features
    if not use_toc:
        f = f.copy()
        for c in ToCNumberPredictor.TOC_FEATURES + ToCTextPredictor.TOC_FEATURES:
            if c in f:
                f[c] = 0
    p_shared = 1 / (1 + np.exp(-model.shared.logit(f)))
    p_shared = np.clip(p_shared, 1e-4, 1 - 1e-4)
    return ScanScores(
        start=model.start.logit(f) + model.start_offset,
        end=model.end.logit(f),
        log_shared=np.log(p_shared),
        log_not_shared=np.log(1 - p_shared),
        nondoc=model.nondoc.logit(f),
    )


def segment_inventory(inv: InventoryData, model: SegmentationModel, use_toc: bool = True) -> Result:
    if inv.features is None:
        compute_features(inv)
    sc = scan_scores(inv, model, use_toc)
    placements = align_toc(inv, sc.start, model.align) if use_toc else []
    forced = {pl.scan for pl in placements}
    use_prior = None
    if placements and not model.length_prior_in_toc:
        # inside a placed ToC entry its length is explained by the ToC; the
        # generic prior (fitted on mostly short documents) would split it up
        use_prior = np.ones(inv.n, dtype=bool)
        for a, b in _entry_spans(inv, placements):
            use_prior[a : b + 1] = False
    raw, dp_score = segment_scans(sc, model.length_prior, model.max_length, forced, use_prior)

    by_scan: dict[int, list[Placement]] = {}
    placed_at = {pl.entry: pl.scan for pl in placements}
    for pl in placements:
        by_scan.setdefault(pl.scan, []).append(pl)
    parent_of = _nesting(inv, placements)
    f = inv.features
    numbers = NumberIndex(f["_number"].to_numpy(), f["_run"].to_numpy(), inv.scans["layout"])

    segments: list[Segment] = []
    owner: int | None = None  # ToC row of the latest placed entry
    prev_end = -1
    for a, e, bt in raw:
        if a > prev_end + 1:  # non-document scans before this document
            segments.append(Segment(prev_end + 1, a - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[a - 1])))
        prev_end = max(prev_end, e)
        here = by_scan.get(a, [])
        if here:
            owner = here[-1].entry
            segments.append(
                Segment(a, e, "toc", bt, float(sc.start[a]), float(sc.end[e]), [pl.entry for pl in here], parent_of.get(here[0].entry), sum(pl.score for pl in here))
            )
            continue
        kind, parent = "unindexed", None
        num = numbers.number_at(a)
        row = owner
        while row is not None:  # innermost enclosing ToC entry whose range covers this start
            end_eff = inv.toc.at[row, "end_eff"]
            same_run = numbers.run[placed_at[row]] == numbers.run[a]
            if same_run and (pd.isna(end_eff) or pd.isna(num) or num <= end_eff + 0.5):
                kind, parent = "subdoc", row
                break
            row = parent_of.get(row)
        segments.append(Segment(a, e, kind, bt, float(sc.start[a]), float(sc.end[e]), parent_row=parent))
    if prev_end < inv.n - 1:
        segments.append(Segment(prev_end + 1, inv.n - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[-1])))

    score = dp_score + sum(pl.score for pl in placements)
    return Result(inv.inventory_number, segments, placements, score)
