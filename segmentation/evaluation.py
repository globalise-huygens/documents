"""
Ground-truth labels, model fitting and evaluation.

Two kinds of ground truth:
  validated  – ~22 inventories segmented by hand, read from the
               "<inv> - Document Segmentation.csv" files: documents,
               subdocuments, same-scan boundaries and non-document page types.
               Complete for the whole inventory.
  GM         – hand-checked General Missives: reliable starts, and no
               boundaries inside a missive. A missive's ToC entry may also
               cover appendices after it, so ends are not evaluated there.
               Where the annotation starts on the missive's title page, the
               start is moved to the text (title pages are outside documents,
               as in the ToC and the validated inventories).

The model is fitted on the validated inventories only (adding the General
Missives made every metric worse, including on the missives themselves).
Evaluation is cross-validated by inventory.
"""

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ground_truth import csv_path, load_truth
from .inventory import (
    InventoryData,
    connect,
    gm_inventories,
    load_baseline,
    load_general_missives,
    load_inventory,
    load_validated,
    validated_inventories,
)
from .model import AlignParams, LengthPrior, Logistic, SegmentationModel, design
from .predictors import compute_features, feature_columns
from .segmenter import Result, segment_inventory

# context columns per model: scan i plus its neighbours
SHIFTS = {"start": (0, -1), "end": (0, 1), "shared": (0, -1), "nondoc": (-1, 0, 1)}


@dataclass
class Item:
    """One inventory with features and ground truth, ready for fit/eval."""

    inv: InventoryData
    source: str  # 'validated' | 'gm'
    truth: pd.DataFrame  # document spans (start/end positions, part_of_id, csv_id)
    baseline: pd.DataFrame
    page_type: pd.Series | None = None  # validated: non-document page type per scan ('' = in a document)


def prepare(conn, inventory_number: str, source: str) -> Item:
    inv = load_inventory(conn, inventory_number)
    compute_features(inv)
    page_type = None
    if source == "validated" and (path := csv_path(inventory_number)):
        truth, page_type = load_truth(inv, path)
    elif source == "validated":
        truth = load_validated(conn, inv)
    else:
        truth = strip_title_pages(load_general_missives(conn, inv), inv)
    return Item(inv, source, truth, load_baseline(conn, inv), page_type)


TITLE_MAX_CHARS = 120  # ~70% of the validated title pages are shorter; <3% of document scans
TITLE_WITH_VERSO_MAX_CHARS = 1000  # a title page followed by a blank verso can hold more text


def strip_title_pages(truth: pd.DataFrame, inv: InventoryData) -> pd.DataFrame:
    """
    The General Missives annotation sometimes starts a missive on its title
    page; the ToC (and the validated ground truth) keep title pages outside
    the document. Move such starts to the first text scan: a short scan
    followed by a blank verso and text, or a very short scan followed by text.
    Adds a boolean column 'title_page'.
    """
    if truth.empty:
        return truth.assign(title_page=False)
    length = inv.scans["text_len"].to_numpy()
    truth = truth.copy()
    truth["title_page"] = False
    for k, r in truth.iterrows():
        a = r.start
        if r.end - a >= 2 and length[a] < TITLE_WITH_VERSO_MAX_CHARS and length[a + 1] == 0 and length[a + 2] >= TITLE_MAX_CHARS:
            truth.at[k, "start"], truth.at[k, "title_page"] = a + 2, True
        elif r.end - a >= 1 and length[a] < TITLE_MAX_CHARS and length[a + 1] >= TITLE_MAX_CHARS:
            truth.at[k, "start"], truth.at[k, "title_page"] = a + 1, True
    return truth


def load_items(sources=("validated", "gm"), cache: str | None = None, gm_limit: int | None = None) -> list[Item]:
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            items = pickle.load(f)
        return [it for it in items if it.source in sources]
    conn = connect()
    items = []
    if "validated" in sources:
        items += [prepare(conn, n, "validated") for n in validated_inventories(conn)]
    if "gm" in sources:
        validated = {it.inv.inventory_number for it in items}
        gm = [n for n in gm_inventories(conn) if n not in validated][:gm_limit]
        items += [prepare(conn, n, "gm") for n in gm]
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(items, f)
    return items


# ── flat view of the ground truth ─────────────────────────────────────────────


def flat_segments(truth: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    The ground truth as the segmenter sees it: a sequence of documents over
    the scans (a subdocument starts a new segment), each with its end and the
    type of the boundary before it. For segment j with start s_j and next
    start s_{j+1}, the end is the last end of a document starting in
    [s_j, s_{j+1}): capped at s_{j+1} - 1 when that document runs on past the
    next start (a subdocument inside its parent), equal to s_{j+1} for a
    same-scan boundary, and earlier when non-document scans follow.
    """
    starts = sorted(set(truth["start"]))
    rows = []
    for j, s in enumerate(starts):
        nxt = starts[j + 1] if j + 1 < len(starts) else n
        ends = truth.loc[(truth["start"] >= s) & (truth["start"] < nxt), "end"]
        e = int(ends.max())
        if e > nxt:
            e = nxt - 1
        rows.append({"start": s, "end": min(e, n - 1)})
    flat = pd.DataFrame(rows)
    prev_end = flat["end"].shift(1)
    flat["boundary"] = np.where(
        prev_end.isna(), "first", np.where(prev_end == flat["start"], "shared", np.where(prev_end == flat["start"] - 1, "adjacent", "gap"))
    )
    return flat


def labels(item: Item) -> pd.DataFrame:
    """Per-scan training targets for the four models (validated inventories)."""
    n = item.inv.n
    lab = pd.DataFrame({"start": 0.0, "end": 0.0, "shared": np.nan, "nondoc": 1.0}, index=range(n))
    flat = flat_segments(item.truth, n)
    for r in flat.itertuples():
        lab.iat[r.start, 0] = 1
        lab.iat[r.end, 1] = 1
        lab.iloc[r.start : r.end + 1, 3] = 0
        if r.boundary != "first":
            lab.iat[r.start, 2] = float(r.boundary == "shared")
    return lab


# ── fitting ───────────────────────────────────────────────────────────────────


def fit_model(train: list[Item], l2: float = 10.0, align: AlignParams | None = None) -> SegmentationModel:
    """Fit the start / end / shared / nondoc models and the length prior on validated inventories."""
    train = [it for it in train if it.source == "validated" and not it.truth.empty]
    cols = feature_columns(train[0].inv.features)
    X = {k: [] for k in SHIFTS}
    y = {k: [] for k in SHIFTS}
    lengths = []
    for it in train:
        lab = labels(it)
        for k, shifts in SHIFTS.items():
            d = design(it.inv.features, cols, shifts)
            m = lab[k].notna().to_numpy()
            X[k].append(d[m])
            y[k].append(lab[k].to_numpy()[m])
        flat = flat_segments(it.truth, it.inv.n)
        lengths.append((flat["end"] - flat["start"] + 1).to_numpy())
    fitted = {k: Logistic.fit(pd.concat(X[k], ignore_index=True).fillna(0), np.concatenate(y[k]), l2) for k in SHIFTS}
    lengths = np.concatenate(lengths)
    return SegmentationModel(
        start=fitted["start"],
        end=fitted["end"],
        shared=fitted["shared"],
        nondoc=fitted["nondoc"],
        length_prior=LengthPrior.fit(lengths, 1 / lengths.mean()),
        align=align or AlignParams(),
    )


# ── metrics ───────────────────────────────────────────────────────────────────


def _match(pred: set[int], true: set[int], tol: int) -> tuple[int, int, int]:
    """(true positives, #pred, #true) with one-to-one matching within tol."""
    if tol == 0:
        return len(pred & true), len(pred), len(true)
    unmatched = set(true)
    tp = 0
    for p in sorted(pred):
        hit = min((t for t in unmatched if abs(t - p) <= tol), key=lambda t: abs(t - p), default=None)
        if hit is not None:
            unmatched.discard(hit)
            tp += 1
    return tp, len(pred), len(true)


@dataclass
class Counts:
    c: dict = field(default_factory=dict)

    def add(self, key, tp, npred, ntrue):
        a = self.c.setdefault(key, [0, 0, 0])
        a[0] += tp
        a[1] += npred
        a[2] += ntrue

    def prf(self, key):
        tp, npred, ntrue = self.c.get(key, [0, 0, 0])
        p = tp / npred if npred else 0.0
        r = tp / ntrue if ntrue else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(item: Item, result: Result | None, counts: Counts, label: str, baseline: bool = False):
    """Add this inventory's counts under `label`."""
    t = item.truth
    if t.empty or (baseline and item.source == "validated"):
        return  # step 15 deleted the baseline's links on validated pages
    if baseline:
        pred_all = pred_top = set(item.baseline["start"])
        docs = []
    else:
        docs = result.documents()
        pred_all = {sg.start for sg in docs}
        pred_top = {sg.start for sg in docs if sg.kind in ("toc", "unindexed")}

    if item.source == "validated":
        flat = flat_segments(t, item.inv.n)
        for tol in (0, 1):
            counts.add(f"{label}|all starts|±{tol}", *_match(pred_all, set(flat["start"]), tol))
            counts.add(f"{label}|top-level starts|±{tol}", *_match(pred_top, set(t.loc[t["part_of_id"].isna(), "start"]), tol))
            counts.add(f"{label}|all ends|±{tol}", *_match({sg.end for sg in docs}, set(flat["end"]), tol))
        # end and boundary type of documents whose start is exactly right
        by_start = {sg.start: sg for sg in docs}
        for r in flat.itertuples():
            sg = by_start.get(r.start)
            if sg is None:
                continue
            counts.add(f"{label}|end correct, given the start|±0", int(sg.end == r.end), 1, 1)
            counts.add(f"{label}|end correct, given the start|±1", int(abs(sg.end - r.end) <= 1), 1, 1)
            if r.boundary != "first":
                counts.add(f"{label}|same-scan boundaries", int(sg.boundary == "shared" and r.boundary == "shared"), int(sg.boundary == "shared"), int(r.boundary == "shared"))
                counts.add(f"{label}|boundary type correct", int(sg.boundary == r.boundary), 1, 1)
        # non-document scans
        true_nd = np.ones(item.inv.n, dtype=bool)
        for r in flat.itertuples():
            true_nd[r.start : r.end + 1] = False
        pred_nd = np.zeros(item.inv.n, dtype=bool)
        for sg in result.segments:
            if sg.kind == "non-document":
                pred_nd[sg.start : sg.end + 1] = True
        counts.add(f"{label}|non-document scans", int((true_nd & pred_nd).sum()), int(pred_nd.sum()), int(true_nd.sum()))
    else:
        gm_starts = set(t["start"])
        for tol in (0, 1):
            tp = sum(1 for g in gm_starts if any(abs(p - g) <= tol for p in pred_all))
            counts.add(f"{label}|GM start recall|±{tol}", tp, len(gm_starts), len(gm_starts))
        interior = [(r.start + 2, r.end - 1) for r in t.itertuples()]  # ±1 around the start is tolerated
        n_int = sum(max(0, b - a + 1) for a, b in interior)
        false = sum(1 for p in pred_all for a, b in interior if a <= p <= b)
        counts.add(f"{label}|GM false starts inside missives", false, n_int, n_int)

    if result is None or baseline:
        return
    # ToC placement for truth documents that carry an index id
    placed = {int(item.inv.toc.at[pl.entry, "csv_id"]): pl.scan for pl in result.placements if not pd.isna(item.inv.toc.at[pl.entry, "csv_id"])}
    in_toc = set(item.inv.toc["csv_id"].dropna().astype(int))
    for r in t[t["csv_id"].notna() & t["part_of_id"].isna()].itertuples():
        cid = int(r.csv_id)
        if cid not in in_toc:
            continue
        pos = placed.get(cid)
        counts.add(f"{label}|ToC entry placed ({item.source})", int(pos is not None), 1, 1)
        for tol in (0, 1):
            counts.add(f"{label}|ToC entry start correct ({item.source})|±{tol}", int(pos is not None and abs(pos - r.start) <= tol), 1, 1)


def cross_validate(items: list[Item], l2: float = 10.0, align: AlignParams | None = None) -> Counts:
    """Leave-one-inventory-out over the validated inventories; the General
    Missives are scored with a model fitted on all validated inventories."""
    counts = Counts()
    valid = [it for it in items if it.source == "validated"]
    for it in valid:
        _eval_all(it, fit_model([x for x in valid if x is not it], l2, align), counts)
    gm = [it for it in items if it.source == "gm"]
    if gm:
        model = fit_model(valid, l2, align)
        for it in gm:
            _eval_all(it, model, counts)
    return counts


def _eval_all(it: Item, model: SegmentationModel, counts: Counts):
    evaluate(it, segment_inventory(it.inv, model, use_toc=True), counts, "model")
    if it.inv.toc["toc_order"].notna().any():
        evaluate(it, segment_inventory(it.inv, model, use_toc=False), counts, "no ToC")
    evaluate(it, None, counts, "baseline", baseline=True)


def format_counts(counts: Counts) -> str:
    rows = []
    for key in sorted(counts.c, key=lambda k: (k.split("|")[1], k.split("|")[2] if k.count("|") > 1 else "", k.split("|")[0])):
        label, metric, *tol = key.split("|")
        name = metric + (" " + tol[0] if tol else "")
        tp, npred, ntrue = counts.c[key]
        p, r, f1 = counts.prf(key)
        if metric.startswith("GM false"):
            rows.append(f"{name:<40} {label:<9} {tp / max(npred, 1) * 100:6.2f} per 100 interior scans ({tp}/{npred})")
        elif npred == ntrue and metric.startswith(("GM start", "ToC", "end correct", "boundary type")):
            rows.append(f"{name:<40} {label:<9} {r:6.1%}  ({tp}/{ntrue})")
        else:
            rows.append(f"{name:<40} {label:<9} P {p:6.1%}  R {r:6.1%}  F1 {f1:6.1%}  (tp {tp}, pred {npred}, true {ntrue})")
    return "\n".join(rows)
