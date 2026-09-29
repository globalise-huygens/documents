"""
Human-readable evidence for the segmentation: what the predictors read on
each scan and how the models weighed it.

scan_evidence()     one row per scan: header text and the date parsed from
                    it, header/date verdicts against the previous headers and
                    across a possible start, page numbers (as read and
                    cleaned), blank/text length, signature, and the four model
                    probabilities with their strongest contributions.
document_evidence() per predicted document: how it starts and ends, and for
                    ToC documents how the entry was placed.
"""

import numpy as np
import pandas as pd

from .inventory import InventoryData
from .model import Logistic, SegmentationModel, shifted
from .predictors import CLOSING_RE, GENRE_RE, SALUTATION_RE, scan_texts, signature_kind
from .segmenter import Placement, Result, Segment, scan_scores

FEATURE_LABELS = {
    "blank": "blank page",
    "after_blank": "previous page blank",
    "blank_run_before": "blank pages before",
    "first_content": "first page with text",
    "sig_collation_here": "collation mark on this page",
    "prev_sig_collation": "collation mark on previous page",
    "sig_signed_here": "signature on this page",
    "prev_sig_signed": "signature on previous page",
    "sig_quire_here": "quire mark on this page",
    "prev_sig_quire": "quire mark on previous page",
    "has_number": "page number present",
    "number_restart": "page numbering restarts here",
    "number_restart_ahead": "page numbering restarts just after",
    "header_present": "running header present",
    "header_text_change": "header wording differs from previous header",
    "header_after_gap": "header after pages without header",
    "header_text_change_across": "next header differs from last header",
    "has_header_date": "date in header",
    "header_date_changed": "header date differs from previous",
    "header_date_uncertain": "header date differs by one misread character",
    "header_date_same": "header date same as previous",
    "header_date_changed_across": "next header date differs from last",
    "header_date_same_across": "next header date same as last",
    "language_changed": "language changes",
    "non_dutch": "not Dutch",
    "marginalia": "marginalia",
    "prev_marginalia": "marginalia on previous page",
    "log_text_len": "amount of text",
    "rel_text_len": "text length relative to inventory",
    "short_text": "very little text",
    "text_len_change_prev": "text length differs from previous page",
    "near_inventory_start": "near start of inventory",
    "near_inventory_end": "near end of inventory",
    "toc_start_number": "ToC entry starts on this page number",
    "toc_end_number": "ToC entry ends on this page number",
    "toc_end_start_number": "ToC: one entry ends and next starts on this number",
    "open_genre": "text opens with a genre word (Copia, Extract, Register, ...)",
    "open_salutation": "salutation near the top (Hoog Edele ...)",
    "close_formula": "closing formula at the end (Accordeert, onderstond, ...)",
    "close_place_date": "place and date at the end",
    "close_then_open": "closing formula followed by a new opening on the page",
    "leaders_per_1000": "dotted leaders (table of contents)",
    "digit_share": "share of digits",
    "no_text": "no text",
    "toc_title_match": "text matches a ToC description",
    "toc_title_match_title_page": "title page before it matches a ToC description",
    "toc_entries_matched": "text matches several ToC descriptions (ToC page)",
    "catch_word": "catch-word at the bottom (text continues)",
    "text_top": "how low on the page the text starts",
    "text_bottom": "how far down the page the text reaches",
    "max_line_gap": "largest gap between lines",
    "closing_mid_page": "closing formula halfway down the page, text below",
    "opening_below_top": "opening (genre word or salutation) below the top",
    "heading_below_top": "centred heading below the top",
    "n_paragraph_lines": "number of text lines",
    "has_layout": "page layout available",
}
MODELS = {"start": "document starts here", "end": "document ends here", "shared": "starts on previous document's last page", "nondoc": "not part of a document"}


def label(feature: str) -> str:
    base, _, k = feature.partition("@")
    text = FEATURE_LABELS.get(base, base.replace("_", " "))
    if k == "-1":
        return f"{text} (previous scan)"
    if k == "+1":
        return f"{text} (next scan)"
    return text


def contributions(lg: Logistic, features: pd.DataFrame, top: int = 4) -> list[list]:
    """
    Per scan: the `top` features that move the log-odds most away from a
    typical scan, weight × (value − training mean), as [label, contribution].
    """
    names = list(lg.weights)
    if not names:
        return [[] for _ in range(len(features))]
    cols = np.column_stack([shifted(features, n) if shifted(features, n) is not None else np.zeros(len(features)) for n in names])
    means = np.array([lg.means.get(n, 0.0) for n in names])
    cols = np.where(np.isnan(cols), means, cols)  # missing = neutral
    contrib = (cols - means) * np.array([lg.weights[n] for n in names])
    out = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:top]
        out.append([[label(names[i]), round(float(row[i]), 2)] for i in idx if abs(row[i]) >= 0.05])
    return out


def _formulas(f: pd.DataFrame, i: int) -> str:
    names = {
        "open_genre": "opens with genre word",
        "open_salutation": "salutation",
        "close_formula": "closing formula",
        "close_place_date": "place and date at end",
        "close_then_open": "closing then new opening on the page",
    }
    return ", ".join(v for k, v in names.items() if k in f and f.at[i, k])


def _best_matches(inv: InventoryData) -> list[str]:
    """Per scan: the ToC entry whose description best matches its text (if any)."""
    sim = inv.toc_text_sim
    toc = inv.toc[inv.toc["toc_order"].notna()]
    if sim is None or not len(toc):
        return [""] * inv.n
    k = sim.argmax(axis=0)
    best = sim.max(axis=0)
    titles = toc["title"].astype(str).to_numpy()
    return [f"{best[i]:.0%}: {titles[k[i]][:80]}" if best[i] >= 0.4 else "" for i in range(inv.n)]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _verdict(f: pd.DataFrame, i: int, prefix: str) -> str:
    if f.at[i, f"{prefix}changed"]:
        return "changed"
    if f.at[i, f"{prefix}uncertain"] if f"{prefix}uncertain" in f else False:
        return "uncertain (one character)"
    if f.at[i, f"{prefix}same"]:
        return "same"
    return ""


def scan_evidence(inv: InventoryData, model: SegmentationModel) -> pd.DataFrame:
    sc = scan_scores(inv, model)
    f = inv.features
    s = inv.scans
    dates = f["_header_date"]
    ev = pd.DataFrame(
        {
            "position": np.arange(inv.n),
            "filename": s["filename"],
            "blank": s["is_blank"],
            "text_len": s["text_len"],
            "numbers_read": s["folios"].map(lambda l: ", ".join(map(str, l))),
            "number": f["_number"],
            "numbering_run": f["_run"] + 1,
            "number_restart": f["number_restart"].astype(bool),
            "header": s["header"],
            "header_date": dates.map(lambda d: d.date.isoformat() if d else ""),
            "header_date_precision": dates.map(lambda d: d.precision if d else ""),
            "header_change_vs_previous": f["header_text_change"].round(2),
            "header_change_across": f["header_text_change_across"].round(2),
            "header_date_vs_previous": [_verdict(f, i, "header_date_") for i in range(inv.n)],
            "header_date_across": np.where(f["header_date_changed_across"] > 0, "changed", np.where(f["header_date_same_across"] > 0, "same", "")),
            "signature": s["signatures"],
            "signature_kind": s["signatures"].map(signature_kind).replace("none", ""),
            "languages": s["languages"],
            "text_start": scan_texts(inv).map(lambda t: t.strip()[:160]),
            "text_end": scan_texts(inv).map(lambda t: t.strip()[-160:] if len(t.strip()) > 160 else ""),
            "formulas": [_formulas(f, i) for i in range(inv.n)],
            "best_toc_match": _best_matches(inv),
            "p_start": _sigmoid(sc.start).round(3),
            "p_end": _sigmoid(sc.end).round(3),
            "p_same_scan": np.exp(sc.log_shared).round(3),
            "p_nondoc": _sigmoid(sc.nondoc).round(3),
        }
    )
    for name in MODELS:
        ev[f"why_{name}"] = contributions(getattr(model, name), f)
    return ev


def document_evidence(inv: InventoryData, result: Result, segment: Segment, ev: pd.DataFrame) -> dict:
    """What the model saw at a document's first and last scan, and how its ToC entry was placed."""
    a, e = segment.start, segment.end
    out = {
        "kind": segment.kind,
        "boundary": segment.boundary,
        "start": {k: _plain(ev.at[a, k]) for k in ("filename", "p_start", "p_same_scan", "header", "header_date", "header_date_across", "number", "text_start", "formulas", "best_toc_match", "why_start")},
        "end": {k: _plain(ev.at[e, k]) for k in ("filename", "p_end", "signature", "signature_kind", "text_end", "formulas", "why_end")},
    }
    placements = [pl for pl in result.placements if pl.entry in segment.toc_rows]
    if placements:
        pl = placements[0]
        entry = inv.toc.loc[pl.entry]
        out["toc"] = {
            "placed_by": pl.how,
            "placement_score": round(pl.score, 2),
            "toc_start_number": _plain(entry.folio_start),
            "toc_end_number": _plain(entry.end_eff),
            "number_on_start_scan": _plain(ev.at[a, "number"]),
            "numbers_around_start": [_plain(ev.at[i, "number"]) for i in range(max(0, a - 1), min(len(ev), a + 2))],
            "toc_date": "" if entry.date_begin is None or pd.isna(entry.date_begin) else f"{entry.date_begin} – {entry.date_end}",
            "header_date_near_start": next((d for d in ev["header_date"].iloc[a : a + 3] if d), ""),
        }
    return out


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
