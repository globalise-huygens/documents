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
4. Every document is labelled: 'toc' (starts a ToC entry), 'gm' (a General
   Missive identified by hand that has no ToC entry here), 'subdoc' (inside
   the page range of the ToC entry it follows, e.g. an enclosure or appendix;
   when that range or the page number is unknown, only if no non-document
   scans separate them; never beyond 1.5 times the scans the entry's page
   range implies, plus 5), or 'unindexed' (outside any entry's range, or no ToC
   at all); runs of
   non-document scans are reported as 'non-document'.

Hard rules, whatever the model scores say: no document starts on a blank
scan (forced starts move to the next scan with text), and every scan with
text belongs to a document, so non-document runs hold blank scans only; the
exception are scans whose text mostly recurs on nearby scans (a small sheet
over a larger page, RecurringTextPredictor), which may lie outside documents.

General Missives identified by hand (segment_inventory(gm=...)) override the
alignment: their start is a forced start (their ToC entry, when there is one,
is placed there), no other document starts inside their span, and their title
and dates win over the ToC entry's (result_rows / db_import). Within the span
the model still looks for subdocuments (enclosures, appendices): a second pass
segments the span on its own, and what starts inside it becomes a 'subdoc' of
the missive.
"""

import datetime
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .header_dates import HeaderDate
from .inventory import InventoryData
from .model import AlignParams, SegmentationModel
from .predictors import RECUR_SKIP, REGISTER_TITLE_RE, ToCNumberPredictor, ToCTextPredictor, compute_features, register_heading

BIG = 1e6


def next_text_scan(blank: np.ndarray) -> np.ndarray:
    """Per position: the first position at or after it that is not blank (itself
    when only blank scans follow)."""
    n = len(blank)
    out = np.arange(n)
    nxt = None
    for i in range(n - 1, -1, -1):
        if not blank[i]:
            nxt = i
        if nxt is not None:
            out[i] = nxt
    return out


# ── page/folio number index ───────────────────────────────────────────────────


EXTRAPOLATE_MAX = 10  # also extrapolate from a page number this close to the one sought
SECTION_START, SECTION_REACH = 3, 100  # the first folios of a section: extrapolated back from a run's first numbers up to this high


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
        """
        Interpolated scan positions of `number` per run that covers it: between
        two numbers that follow each other in scan order and bracket it (the
        closest such pair), and extrapolated at the run's rate from the nearest
        number below and above it when these are close (or nothing brackets it). Numbers
        of another section that ended up in the run (a restart that was not
        recognised) then cannot pull the position away, as they would when the
        run's numbers are sorted by value; where both positions differ, the
        alignment chooses.
        """
        out = []
        for r, (idx, val) in self.points.items():
            lo, hi = val.min(), val.max()
            # the start of a section whose first numbers were not read (fo. 1 when fo. 51 is the first read: 3598)
            section_start = number <= SECTION_START and lo <= SECTION_REACH
            if not (lo - margin <= number <= hi + margin or section_start):
                continue
            rate = self.rate(r)
            run_scans = np.flatnonzero(self.run == r)
            found = []
            pairs = [k for k in range(len(idx) - 1) if val[k] <= number <= val[k + 1]]
            if pairs:
                k = min(pairs, key=lambda k: val[k + 1] - val[k])
                v1, v2, i1, i2 = val[k], val[k + 1], idx[k], idx[k + 1]
                found.append(i1 + (number - v1) * (i2 - i1) / (v2 - v1) if v2 > v1 else i1)
            for side in (val <= number, val >= number):  # from the nearest number below, and above
                if side.any():
                    k = np.flatnonzero(side)[np.argmin(np.abs(val[side] - number))]
                    if not pairs or abs(val[k] - number) <= EXTRAPOLATE_MAX:
                        found.append(idx[k] + (number - val[k]) * rate)
            for pos in found:
                pos = float(min(max(pos, run_scans[0]), run_scans[-1]))
                if all(abs(pos - q) > 1 for rr, q in out if rr == r):
                    out.append((r, pos))
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
    how: str  # 'exact' | 'interpolated' (page numbers) | 'text' (description matches the text) | 'register' (unfoliated register just before its folio) | 'evidence' | 'forced' (placed between neighbours)
    note: str = ""  # e.g. a register placed out of ToC order


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


FIXED_BONUS = 1e4  # alignment score of a hand-identified start: the other entries are aligned around it


def _restrict(cands: list[tuple[int, float, str]], text_at: np.ndarray, blocked: np.ndarray) -> list[tuple[int, float, str]]:
    """Candidates moved off blank scans to the next scan with text, without
    those inside a hand-identified General Missive."""
    out: dict[int, tuple[float, str]] = {}
    for c, score, how in cands:
        c = int(text_at[c])
        if not blocked[c] and (c not in out or out[c][0] < score):
            out[c] = (score, how)
    return sorted(((c, sc, how) for c, (sc, how) in out.items()), key=lambda t: -t[1])


def align_toc(inv: InventoryData, s: np.ndarray, p: AlignParams, fixed: dict[int, int] | None = None, blocked: np.ndarray | None = None) -> list[Placement]:
    """
    Place the ToC entries. `fixed` maps rows in inv.toc to the scan they start
    on (General Missives identified by hand): they are placed there, the others
    around them, never inside `blocked` (the interiors of those missives) and
    never on a blank scan.
    """
    fixed = fixed or {}
    blank = inv.scans["is_blank"].to_numpy(dtype=bool)
    text_at = next_text_scan(blank)
    blocked = np.zeros(inv.n, dtype=bool) if blocked is None else blocked
    toc = inv.toc[inv.toc["toc_order"].notna()].reset_index()  # 'index' = row in inv.toc
    if toc.empty:
        return [Placement(row, scan, 0.0, "gm") for row, scan in sorted(fixed.items(), key=lambda t: t[1])]
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
    cands = []
    for k, e in enumerate(toc.itertuples()):
        row = int(e.index)
        if row in fixed:
            c = fixed[row]
            cands.append([(c, FIXED_BONUS + p.w_start * s[c] + dterms[k][c], "gm")])
        else:
            cands.append(_restrict(_candidates(e, numbers, number, layout, s, dterms[k], sims[k], p, n), text_at, blocked))

    # An unfoliated register gets the folio of the next numbered leaf in the ToC:
    # where that leaf is not a register page itself but one lies just before it,
    # the register is placed on that page.
    texts = inv.texts if inv.texts is not None else pd.Series("", index=range(n))
    heading = np.array([register_heading(t) for t in texts])
    for k, e in enumerate(toc.itertuples()):
        if int(e.index) in fixed or not REGISTER_TITLE_RE.match(str(e.title or "")):
            continue
        moved_back = []
        for c, score, how in cands[k]:
            j = next((j for j in range(c - 1, max(-1, c - 1 - REGISTER_LOOKBACK), -1) if heading[j] and np.isnan(number[j]) and not blocked[j]), None)
            if how == "exact" and not heading[c] and j is not None:
                moved_back.append((j, p.exact_number + p.w_start * s[j] + dterms[k][j], "register"))
            else:
                moved_back.append((c, score, how))
        cands[k] = sorted({c: (c, sc, h) for c, sc, h in sorted(moved_back, key=lambda t: t[1])}.values(), key=lambda t: -t[1])

    # A register page found by what it lists is strong evidence for its register,
    # as strong as a page number matching the ToC (the move below handles
    # registers on the other side of their papers).
    for rp in inv.register_pages or []:
        if rp.register >= len(toc) or int(toc.at[rp.register, "index"]) in fixed:
            continue
        c = int(text_at[rp.scan])
        if not blocked[c]:
            ck = [cd for cd in cands[rp.register] if cd[0] != c]
            cands[rp.register] = sorted(ck + [(c, p.exact_number + p.w_start * s[c] + dterms[rp.register][c], "register")], key=lambda t: -t[1])

    def run(order, cands_k):
        objective, placed = _align_in_order(toc.iloc[order].reset_index(drop=True), [cands_k[k] for k in order], fs_arr[order], numbers, p)
        return objective, placed

    order = np.arange(len(toc))
    objective, placements = run(order, cands)
    # A register of papers may be bound on the other side of the papers it lists
    # than the ToC has it. Where its page was found by what it lists
    # (predictors.find_register_pages), the register goes on that page, on the
    # other side of its papers, unless that disrupts the other entries. Its own
    # description is no help in deciding: registers are worded alike, and the
    # current placement is often on another register's page that matches the
    # wording as well; so it scores the same in both orders.
    moved = {}
    for rp in (inv.register_pages or []) if p.move_registers else []:
        if rp.register >= len(toc) or int(toc.at[rp.register, "index"]) in fixed:
            continue
        c = int(text_at[rp.scan])
        row = int(toc.at[rp.register, "index"])
        if blocked[c]:
            continue
        own = next((pl.score for pl in placements if pl.entry == row), 0.0)  # the register's term as it is now
        alt_cands = list(cands)
        alt_cands[rp.register] = [(c, own, "text")]
        rest = [k for k in order if k != rp.register]
        at = rest.index(min(rp.papers)) if rp.side == "after" else rest.index(max(rp.papers)) + 1
        alt = np.array(rest[:at] + [rp.register] + rest[at:])
        alt_objective, alt_placements = run(alt, alt_cands)
        if alt_objective >= objective - REGISTER_MOVE_TOLERANCE and any(pl.entry == row for pl in alt_placements):
            order, cands, objective, placements = alt, alt_cands, alt_objective, alt_placements
            moved[int(toc.at[rp.register, "index"])] = rp.side
    toc_o = toc.iloc[order].reset_index(drop=True)
    placements += _place_unnumbered(inv, toc_o, placements, s, [dterms[k] for k in order], p, blank | blocked)
    for pl in placements:
        if pl.entry in moved:
            pl.note = "register moved " + ("before" if moved[pl.entry] == "after" else "after") + " the papers its page lists"
    
    # hand-identified starts win, also where the ToC order contradicts them (the DP then dropped them)
    placements = [pl for pl in placements if pl.entry not in fixed]
    placements += [Placement(row, scan, 0.0, "gm") for row, scan in fixed.items()]
    for pl in placements:
        if pl.how != "gm":
            pl.scan = int(text_at[pl.scan])  # no room between the neighbours but a blank scan
    placements.sort(key=lambda pl: (pl.scan, pl.how != "gm", inv.toc.at[pl.entry, "toc_order"]))
    return placements


SUBDOC_SPAN_FACTOR, SUBDOC_SPAN_SLACK = 1.5, 5  # a subdocument starts within 1.5x its parent's implied span + 5 scans
REGISTER_LOOKBACK = 6  # scans before the numbered leaf where an unfoliated register may begin
REGISTER_MOVE_TOLERANCE = 2.0  # a register is moved to its page unless the other entries then fit this much worse


def _align_in_order(toc: pd.DataFrame, cands: list, fs_arr: np.ndarray, numbers: NumberIndex, p: AlignParams) -> tuple[float, list[Placement]]:
    """The monotone DP over the entries in this order: the placements of the
    entries it places, and its objective (to compare orders)."""
    placeable = [k for k in range(len(toc)) if cands[k]]
    seq = toc["folio_sequence"].to_numpy(dtype=float) if "folio_sequence" in toc else np.full(len(toc), np.nan)
    end_eff = toc["end_eff"].to_numpy(dtype=float) if "end_eff" in toc else np.full(len(toc), np.nan)

    def next_folio(kp, k):
        """Start folio of entry k counted in kp's foliation, and whether k starts a new one: after
        "Makassar 1, fo. 1-90" a new sequence's fo. 1 lies 90 folios on, not on the same leaf (3598)."""
        if not (np.isnan(seq[kp]) or np.isnan(seq[k])) and seq[k] != seq[kp]:
            return (end_eff[kp] + 1 if not np.isnan(end_eff[kp]) else np.nan), True
        return fs_arr[k], False

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
                    f1, restart = next_folio(kp, k)
                    if ppos == pos:
                        v += share_cost(fs_arr[kp], f1, p)
                    v += _length_term(fs_arr[kp], f1, ppos, pos, numbers, p, across_runs=restart)
                    if v > scores[ci]:
                        scores[ci], bp[ci] = v, (tp, cpi)
        best.append(scores)
        back.append(bp)

    placements: list[Placement] = []
    if not placeable:
        return 0.0, placements
    total = [best[t].max() + p.skip * (len(placeable) - 1 - t) if len(best[t]) else -np.inf for t in range(len(placeable))]
    t = int(np.argmax(total))
    objective = float(total[t])
    ci = int(np.argmax(best[t]))
    while True:
        k = placeable[t]
        pos, local, how = cands[k][ci]
        placements.append(Placement(int(toc.at[k, "index"]), int(pos), float(local - (FIXED_BONUS if how == "gm" else 0)), how))
        if back[t][ci] is None:
            break
        t, ci = back[t][ci]
    placements.reverse()
    return objective, placements


def share_cost(f0: float, f1: float, p: AlignParams) -> float:
    """Score of two consecutive ToC entries starting on the same scan: cheap for
    entries with the same start folio (a letter and its enclosures), more
    expensive the further apart the ToC puts them."""
    if np.isnan(f0) or np.isnan(f1):
        return p.same_start
    return p.same_start - p.same_start_folio * math.log1p(max(0.0, abs(f1 - f0) - 1))


def _length_term(fs0: float, fs1: float, ppos: int, pos: int, numbers: NumberIndex, p: AlignParams, across_runs: bool = False) -> float:
    """How well the scan distance fits the folio distance (at the first entry's run rate); only within
    one numbering run, unless the second entry starts a new foliation sequence (then fs1 is the folio
    after the first entry's end)."""
    if np.isnan(fs0) or np.isnan(fs1) or fs1 < fs0 or (numbers.run[ppos] != numbers.run[pos] and not across_runs):
        return 0.0
    expected = (fs1 - fs0) * numbers.rate(numbers.run[ppos])
    return -p.length_weight * abs(math.log((pos - ppos + 1) / (expected + 1)))


def _place_unnumbered(inv, toc, placed: list[Placement], s, dterms, p: AlignParams, avoid: np.ndarray) -> list[Placement]:
    """
    Place every entry the number-based alignment left out, between its placed
    neighbours in index order, on the scans with the best start evidence
    (start log-odds + header-date agreement). Positions are non-decreasing;
    sharing a scan with a neighbour is allowed at share_cost (for when there
    is no room). Where both neighbours have a start folio, an entry with one is
    drawn towards the position its folio implies between them
    (index_interp_weight), which spreads entries over a stretch without
    usable page numbers. Scans in `avoid` (blank, or inside a General Missive
    identified by hand) are taken only when there is nothing else. A
    placement is 'evidence' when its evidence clears unnumbered_threshold,
    'forced' otherwise.
    """
    placed_rows = {pl.entry for pl in placed}
    row_to_k = {int(r): k for k, r in enumerate(toc["index"])}
    fs = toc["folio_start"].to_numpy(dtype=float)
    anchors = [(-1, -1)] + sorted((row_to_k[pl.entry], pl.scan) for pl in placed) + [(len(toc), inv.n)]
    out = []
    for (ka, ca), (kb, cb) in zip(anchors, anchors[1:]):
        todo = [k for k in range(ka + 1, kb) if int(toc.at[k, "index"]) not in placed_rows]
        if not todo:
            continue
        lo, hi = max(ca, 0), min(cb, inv.n - 1)  # inclusive; the ends share a scan with an anchor
        J = hi - lo + 1
        pos = np.arange(lo, hi + 1)
        fa = fs[ka] if ka >= 0 else np.nan
        fb = fs[kb] if kb < len(toc) else np.nan
        interp = not np.isnan(fa) and not np.isnan(fb) and fb > fa and cb > ca >= 0
        evidence, gain = [], []
        for k in todo:
            ev = p.w_start * s[lo : hi + 1] + dterms[k][lo : hi + 1]
            ev = ev + np.where(pos == ca, share_cost(fa, fs[k], p), 0.0) + np.where(pos == cb, share_cost(fs[k], fb, p), 0.0)
            g = ev - FIXED_BONUS * avoid[lo : hi + 1]
            if interp and not np.isnan(fs[k]) and p.index_interp_weight:
                expected = ca + (fs[k] - fa) / (fb - fa) * (cb - ca)
                scale = max(1.0, 2 * (cb - ca) / (fb - fa))  # scans per two folios
                g = g - p.index_interp_weight * np.abs(pos - expected) / scale
            evidence.append(ev)
            gain.append(g)
        gain = np.array(gain)
        U = len(todo)
        best = np.full((U, J), -np.inf)
        from_same = np.zeros((U, J), dtype=bool)  # True: previous entry on the same scan
        best[0] = gain[0]
        for u in range(1, U):
            prefix = np.maximum.accumulate(best[u - 1])  # best previous position <= j
            earlier = np.concatenate([[-np.inf], prefix[:-1]])  # previous position < j
            same = best[u - 1] + share_cost(fs[todo[u - 1]], fs[todo[u]], p)
            from_same[u] = same > earlier
            best[u] = gain[u] + np.maximum(earlier, same)
        j = int(np.argmax(best[U - 1]))
        for u in range(U - 1, -1, -1):
            g = float(evidence[u][j])
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


def segment_scans(
    sc: ScanScores,
    length_prior,
    max_length: int,
    forced=frozenset(),
    use_prior: np.ndarray | None = None,
    blank: np.ndarray | None = None,
    no_start: np.ndarray | None = None,
    skippable: np.ndarray | None = None,
):
    """
    Semi-Markov DP over scans. A segmentation is a sequence of documents
    [a, e]; between two documents the boundary is 'shared' (the next starts on
    the previous one's last scan), 'adjacent' (it starts on the next scan) or
    'gap' (non-document scans in between). Score = sum over documents of
    start[a] + end[e] + length prior, plus log P(shared)/log P(not shared) per
    boundary, plus nondoc[i] for every scan in a gap (including before the
    first and after the last document). A one-scan document on a shared scan
    gets no end[e]: that end is already the previous document's (counted twice,
    a certain end would add a one-scan document on every closing page).

    With `blank` (per scan) two hard rules hold: no document starts on a blank
    scan, and only blank scans (and `skippable` ones) lie outside the documents. No document starts on
    a `no_start` scan either (inside a General Missive identified by hand).
    Forced starts override both; callers move them off blank scans first. When
    no document beats leaving everything out (all blank), the result is empty.

    Returns ([(start, end, boundary_type)], score); boundary_type of the first
    document is 'first'.
    """
    n = len(sc.start)
    L = min(max(max_length, _longest_run(no_start) + 2), n)  # a hand-identified missive can be longer than max_length
    lp = np.concatenate([[0.0], length_prior(np.arange(1, L + 1))])
    prior_on = np.ones(n) if use_prior is None else use_prior.astype(float)
    start = sc.start.astype(float).copy()
    nondoc = sc.nondoc.astype(float).copy()
    if blank is not None:
        start[blank] -= BIG
        nondoc[~(blank | skippable) if skippable is not None else ~blank] -= BIG
    if no_start is not None:
        start[no_start] -= BIG
    for f_ in forced:
        start[f_] = sc.start[f_] + BIG
    C = np.concatenate([[0.0], np.cumsum(nondoc)])  # C[x+1] = sum nondoc[0..x]; gap(a..b) = C[b+1]-C[a]

    NEG = -np.inf
    B = np.full(n, NEG)  # best score up to and including the start of a document at a
    btype = np.zeros(n, dtype=np.int8)  # 0 first, 1 shared, 2 adjacent, 3 gap
    bprev = np.full(n, -1)  # end of the previous document (gap case)
    D = np.full(n, NEG)  # best score with a document ending at e
    D2 = np.full(n, NEG)  # ... of length >= 2
    d_single = np.zeros(n, dtype=bool)
    stype = np.zeros(n, dtype=np.int8)  # btype/bprev of a one-scan document at x (may differ from B's)
    sprev = np.full(n, -1)
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
        # a one-scan document: its end is scored unless it shares the scan with the previous document's end
        single, stype[x], sprev[x] = max(
            (start[x] + o[0] + lp[1] * prior_on[x] + (0.0 if o[1] == 1 else sc.end[x]), o[1], o[2]) for o in opts
        )
        D[x], d_single[x] = (single, True) if single > D2[x] else (D2[x], False)
        cand = D[x] - C[x + 1]
        if x == 0 or cand > M[x - 1]:
            M[x], Marg[x] = cand, x
        else:
            M[x], Marg[x] = M[x - 1], Marg[x - 1]

    e = int(Marg[n - 1])
    total = float(M[n - 1] + C[n]) - BIG * len(forced)
    if not forced and C[n] > total:
        return [], float(C[n])  # nothing but non-document scans
    segs = []
    table = "D"
    while e >= 0:
        one = table == "D" and d_single[e]
        a = e if one else int(bp2[e])
        kind, prev = (stype[a], sprev[a]) if one else (btype[a], bprev[a])
        t = ("first", "shared", "adjacent", "gap")[kind]
        segs.append((a, e, t))
        if t == "first":
            break
        e, table = (a, "D2") if t == "shared" else (int(prev), "D")
    segs.reverse()
    return segs, total


def _longest_run(mask: np.ndarray | None) -> int:
    if mask is None or not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def skippable_scans(inv: InventoryData) -> np.ndarray:
    """Scans with text that may lie outside the documents: their text mostly recurs on nearby scans."""
    f = inv.features
    if f is None or "_recurring_share" not in f:
        return np.zeros(inv.n, dtype=bool)
    return f["_recurring_share"].to_numpy() >= RECUR_SKIP


def scan_rule_violations(inv: InventoryData, result: "Result") -> list[str]:
    """Breaches of the hard rules: documents starting on a blank scan, scans
    with text outside every document."""
    blank = inv.scans["is_blank"].to_numpy(dtype=bool)
    fn = inv.scans["filename"]
    covered = np.zeros(inv.n, dtype=bool)
    out = []
    for sg in result.documents():
        covered[sg.start : sg.end + 1] = True
        if blank[sg.start]:
            out.append(f"{sg.kind} document starts on blank scan {fn.iat[sg.start]}")
    out += [f"scan with text {fn.iat[i]} is in no document" for i in np.flatnonzero(~covered & ~blank & ~skippable_scans(inv))]
    return out


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
    gm: list[dict] = field(default_factory=list)  # General Missives identified by hand starting here (gm_overrides)
    parent_gm: str | None = None  # General Missive (gm_id) this segment is a subdocument of


@dataclass
class Result:
    inventory_number: str
    segments: list[Segment]
    placements: list[Placement]
    score: float

    def documents(self) -> list[Segment]:
        return [sg for sg in self.segments if sg.kind != "non-document"]


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


def gm_overrides(inv: InventoryData, gm: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[int, int], np.ndarray]:
    """
    The General Missives identified by hand (inventory.load_gm_overrides), with
    their start moved off blank scans ('start'; those without any text are
    dropped), the ToC row each one belongs to ('row', by its index id), the
    fixed start per ToC row for align_toc, and the scans no other document may
    start on (inside a missive).
    """
    blocked = np.zeros(inv.n, dtype=bool)
    if gm is None or gm.empty:
        return pd.DataFrame(columns=["gm_id", "start", "end", "row"]), {}, blocked
    blank = inv.scans["is_blank"].to_numpy(dtype=bool)
    text_at = next_text_scan(blank)
    gm = gm.copy()
    gm["start"] = [int(text_at[a]) for a in gm["start"]]
    gm = gm[(gm["start"] <= gm["end"]) & ~blank[gm["start"]]].sort_values(["start", "end"], ascending=[True, False])
    row_of = {int(c): r for r, c in inv.toc["csv_id"].items() if not pd.isna(c)}
    gm["row"] = [row_of.get(int(c)) if not pd.isna(c) else None for c in gm["csv_id"]]
    fixed: dict[int, int] = {}
    for r in gm.itertuples():
        if r.row is not None and not pd.isna(r.row):
            fixed.setdefault(int(r.row), int(r.start))
        blocked[r.start + 1 : r.end + 1] = True
    blocked[list(gm["start"])] = False  # a missive starting inside another one (a secret letter with the general one)
    return gm.reset_index(drop=True), fixed, blocked


COUNT_UNITS = {
    "twee": 2, "twe": 2, "drie": 3, "drij": 3, "vier": 4, "vijf": 5, "vijff": 5, "vyf": 5, "vyff": 5, "ses": 6, "zes": 6,
    "seven": 7, "zeven": 7, "acht": 8, "agt": 8, "negen": 9, "tien": 10, "thien": 10, "elf": 11, "elff": 11,
    "twaalf": 12, "twaalff": 12, "twaelf": 12, "dertien": 13, "veertien": 14, "vijftien": 15, "sestien": 16, "zestien": 16,
    "seventien": 17, "zeventien": 17, "achtien": 18, "agtien": 18, "negentien": 19,
}
COUNT_TENS = {"twintig": 20, "twingtig": 20, "dertig": 30, "veertig": 40, "vijftig": 50}  # also spelled -tigh
_UNIT = "|".join(sorted(COUNT_UNITS, key=len, reverse=True))
_TENS = "|".join(COUNT_TENS)
_NUMBER = rf"(?:({_UNIT})(?:\s+en\s+({_TENS})h?)?|({_TENS})h?|(\d{{1,2}}))"
_DOCS = (  # plural document words
    r"(?:stu[ck]x|stuks|stu[ck]ken|brieven|briefjes|brieffjes|missiven|missives|copi[ej]?n|copijen|extracten|translaten|"
    r"translaeten|rapporten|raporten|resoluti[eë]n|verklaringen|lijsten|memori[eë]n|memories|papieren|requesten|acten|"
    r"attestati[eë]n|rendementen|berigten|registers|notiti[eë]n|rollen|reekeningen|rekeningen|sententi[eë]n|olas|"
    r"contracten|instructi[eë]n|processen|protesten|factura'?s|facturen|bijlagen|bylagen)(?!\w)"
)
# "Vier brieven", "Ses en twintig stux", "Dertig ...": a written number at the start counts as it is
COUNT_WORDS_RE = re.compile(rf"^\W{{0,2}}(?:({_UNIT})(?:\s+en\s+({_TENS})h?)?|({_TENS})h?)\s+\w", re.I)
# digits at the start only before a plural document word ("31 Resolutien", "12 stux"): "-20 Missive" and
# "40 Een missive" are entry numbers
COUNT_DIGITS_RE = re.compile(rf"^[^\w-]{{0,2}}(\d{{1,2}})\s+(?:\S+\s+){{0,2}}?{_DOCS}", re.I)
# after copy words ("Copie van drie missiven", "Translaet van twee Moorsche brieven", "Dito van twee diverse
# brieven"), also only before a plural document word: "Idem van veertien personen" is not about documents
COUNT_AFTER_COPY_RE = re.compile(
    r"^\W{0,2}(?:(?:een|copia|copie|copij|copije|copien|translaat|translaet|translaten|translaeten|extract|extracten|dito|idem|"
    r"authenticque|authentique|"
    rf"originele|origineele|origineel|secrete|secreete)\s+)+(?:van\s+)?{_NUMBER}\s+(?:\S+\s+){{0,2}}?{_DOCS}",
    re.I,
)
# enclosures announced after the main document ("... annex twee translaten", "met vijf stuks bijlagen", "nevens twee
# requesten"): that many parts more; "uijt twee brieven" or "den ontfang van twee brieven" mention documents only
COUNT_ENCLOSED_RE = re.compile(
    rf"\b(?:annex|nevens|benevens|met(?:\s+daar\s+agter)?|item|mitsgaders|alsmede)\s+{_NUMBER}\s+(?:\S+\s+){{0,2}}?{_DOCS}", re.I
)
COUNT_BONUS_MAX = 8.0  # the most a title's count may add to the start log-odds inside its entry


def announced_parts(title) -> int | None:
    """The number of documents a ToC title announces ("Vier brieven van Ternaten ...", "Ses en twintig stux
    rendementen ...", "31 Resolutien ...", "Copie van drie missiven ..."), plus the enclosures it announces
    ("Missive ... annex twee translaten" = 3), if more than one."""
    t = str(title or "")
    if (m := COUNT_WORDS_RE.match(t)):
        n = _number(*m.groups(), None)
    elif (m := COUNT_DIGITS_RE.match(t)):
        n = int(m.group(1))
    elif (m := COUNT_AFTER_COPY_RE.match(t)):
        n = _number(*m.groups())
    else:
        n = 1  # one main document, unless enclosures follow
    n += sum(_number(*e.groups()) for e in COUNT_ENCLOSED_RE.finditer(t))
    return n if 2 <= n <= 60 else None


def _number(unit, tens, tens_only, digits) -> int:
    if unit or tens_only:
        return COUNT_TENS[tens_only.lower()] if tens_only else COUNT_UNITS[unit.lower()] + (COUNT_TENS[tens.lower()] if tens else 0)
    return int(digits)


def _count_bonus(inv: InventoryData, sc: ScanScores, model: SegmentationModel, placements: list[Placement], blank: np.ndarray,
                 max_bonus: float | None = None) -> ScanScores:
    """
    Start scores raised inside the entries whose title announces n documents
    and where the model alone finds fewer: the entry's span is segmented on its
    own, and the smallest bonus (up to max_bonus) that gives n parts is added to
    the start log-odds of its scans after the first. Strong evidence against a
    split still wins; a title that announces fewer parts than the model finds
    changes nothing.
    """
    max_bonus = COUNT_BONUS_MAX if max_bonus is None else max_bonus
    if max_bonus <= 0 or not placements:
        return sc
    start = sc.start.copy()
    for pl, (a, b) in zip(placements, _entry_spans(inv, placements)):
        n = announced_parts(inv.toc.at[pl.entry, "title"])
        if n is None or b - a + 1 < n:
            continue
        bl = blank[a : b + 1]
        prior = None if model.length_prior_in_toc else np.zeros(b - a + 1, dtype=bool)

        def parts(bonus):
            st = sc.start[a : b + 1].copy()
            st[1:] += bonus
            part = ScanScores(st, sc.end[a : b + 1], sc.log_shared[a : b + 1], sc.log_not_shared[a : b + 1], sc.nondoc[a : b + 1])
            return len(segment_scans(part, model.length_prior, model.max_length, {0}, prior, bl)[0])

        if parts(0.0) >= n:
            continue
        lo, hi = 0.0, max_bonus
        if parts(hi) >= n:
            for _ in range(6):  # the smallest bonus that gives n parts
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if parts(mid) >= n else (mid, hi)
        start[a + 1 : b + 1] += hi
    return ScanScores(start, sc.end, sc.log_shared, sc.log_not_shared, sc.nondoc)


def _gm_subdocs(sc: ScanScores, model: SegmentationModel, blank: np.ndarray, g: dict, b: int, forced: set[int], row: int | None) -> list[Segment]:
    """Subdocuments inside a hand-identified missive starting at g['start'] (up
    to scan b): the model segments the span on its own; every document starting
    after the missive's first scan is part of it (other forced starts excepted)."""
    a = g["start"]
    if b <= a:
        return []
    part = ScanScores(**{k: getattr(sc, k)[a : b + 1] for k in ("start", "end", "log_shared", "log_not_shared", "nondoc")})
    raw, _ = segment_scans(part, model.length_prior, model.max_length, {0}, blank=blank[a : b + 1])
    out = []
    for s0, e0, _ in raw:
        s0, e0 = s0 + a, e0 + a
        if s0 > a and s0 not in forced:
            out.append(Segment(s0, e0, "subdoc", "subdocument", float(sc.start[s0]), float(sc.end[e0]), parent_row=row, parent_gm=g["gm_id"]))
    return out


def _gm_info(r) -> dict:
    def val(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT else str(x)

    return {"gm_id": r.gm_id, "title": val(r.title), "date_begin": val(r.date_begin), "date_end": val(r.date_end),
            "csv_id": None if pd.isna(r.csv_id) else int(r.csv_id), "start": int(r.start), "end": int(r.end)}


def segment_inventory(inv: InventoryData, model: SegmentationModel, use_toc: bool = True, gm: pd.DataFrame | None = None) -> Result:
    """Segment one inventory; `gm`: the General Missives identified by hand
    (inventory.load_gm_overrides), which override the model and the ToC alignment."""
    if inv.features is None:
        compute_features(inv)
    sc = scan_scores(inv, model, use_toc)
    blank = inv.scans["is_blank"].to_numpy(dtype=bool)
    skippable = skippable_scans(inv)
    gm, fixed, blocked = gm_overrides(inv, gm)
    if not use_toc:
        fixed = {}
    placements = align_toc(inv, sc.start, model.align, fixed, blocked) if use_toc else []
    forced = {pl.scan for pl in placements} | set(gm["start"])
    sc_dp = _count_bonus(inv, sc, model, placements, blank)  # reported scores stay the model's own
    use_prior = None
    if (placements or len(gm)) and not model.length_prior_in_toc:
        # inside a placed ToC entry its length is explained by the ToC; the
        # generic prior (fitted on mostly short documents) would split it up
        use_prior = np.ones(inv.n, dtype=bool)
        for a, b in _entry_spans(inv, placements) + list(zip(gm["start"], gm["end"])):
            use_prior[a : b + 1] = False
    raw, dp_score = segment_scans(sc_dp, model.length_prior, model.max_length, forced, use_prior, blank, blocked, skippable)
    gm_at: dict[int, list[dict]] = {}
    for r in gm.itertuples():
        gm_at.setdefault(int(r.start), []).append(_gm_info(r))

    by_scan: dict[int, list[Placement]] = {}
    placed_at = {pl.entry: pl.scan for pl in placements}
    span_end = {pl.entry: e for pl, (_, e) in zip(placements, _entry_spans(inv, placements))}
    for pl in placements:
        by_scan.setdefault(pl.scan, []).append(pl)
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
                # a ToC entry is never part of another one: an index range inside an earlier one is a typo in the
                # index (fo 2287-2888) or, mostly, a misread foliation sequence (step 18)
                Segment(a, e, "toc", bt, float(sc.start[a]), float(sc.end[e]), [pl.entry for pl in here], None,
                        sum(pl.score for pl in here), gm=gm_at.get(a, []))
            )
        elif a in gm_at:  # a missive without a ToC entry in this inventory
            segments.append(Segment(a, e, "gm", bt, float(sc.start[a]), float(sc.end[e]), gm=gm_at[a]))
        if a in gm_at:
            g = gm_at[a][0]
            segments += _gm_subdocs(sc, model, blank, g, min(e, g["end"]), forced, here[0].entry if here else None)
        if here or a in gm_at:
            continue
        kind, parent = "unindexed", None
        num = numbers.number_at(a)
        row = owner
        if row is not None:  # the latest ToC entry, when its range covers this start
            end_eff = inv.toc.at[row, "end_eff"]
            same_run = numbers.run[placed_at[row]] == numbers.run[a]
            unknown = pd.isna(end_eff) or pd.isna(num)
            in_range = not unknown and num <= end_eff + 0.5
            # not far beyond the scans the entry's page range implies: where the page numbers are
            # unread or misread, the interpolated number says little (3588: a two-folio entry
            # followed by 160 scans of "subdocuments")
            in_span = a <= placed_at[row] + SUBDOC_SPAN_FACTOR * (span_end.get(row, inv.n) - placed_at[row]) + SUBDOC_SPAN_SLACK
            # after non-document scans the entry has ended, unless its range is known to cover this start
            if same_run and in_span and (in_range if not unknown else bt != "gap"):
                kind, parent = "subdoc", row
        segments.append(Segment(a, e, kind, bt, float(sc.start[a]), float(sc.end[e]), parent_row=parent))
    if prev_end < inv.n - 1:
        segments.append(Segment(prev_end + 1, inv.n - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[-1])))

    score = dp_score + sum(pl.score for pl in placements)
    return Result(inv.inventory_number, segments, placements, score)
