"""Review of machine translations, per document (/review/translations).

Translations come from the globalise-translation-experiments repo: one parquet
per inventory and run, a row per translated scan (filename, inventory,
translation, model, generated_at), in
TRANSLATIONS_DIR/<run>/<inventory>/<inventory>.parquet (TRANSLATIONS_DIR: the output/
folder under GLOBALISE_TRANSLATIONS_DIR, read from that repo's .env). Which scans were skipped,
and why, is in the plan next to it: ../runs/<run>/plans/<inventory>.json. The
Dutch is the page text in data/texts (the source the translations were made
from).

Documents are the top-level documents of the best identification method in
the inventory (the same order as the translation pipeline uses); scans that
no document covers are grouped as "scans outside documents", so every
translated scan can be reached.

Reviews go to the translation_review table (models.TranslationReview),
append-only: the latest row per (run, inventory, item_key) is the current one.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache

_EXPERIMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "globalise-translation-experiments")


def _translations_root() -> str:
    """The translations folder of globalise-translation-experiments (runs/
    and output/): GLOBALISE_TRANSLATIONS_DIR from the environment or from
    that repo's .env (only that line is read), else its translations/."""
    root = os.environ.get("GLOBALISE_TRANSLATIONS_DIR")
    env = os.path.join(_EXPERIMENTS, ".env")
    if not root and os.path.exists(env):
        with open(env, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key.strip() == "GLOBALISE_TRANSLATIONS_DIR" and value.strip():
                    root = value.strip().strip("'\"")
    return os.path.expanduser(root) if root else os.path.join(_EXPERIMENTS, "translations")


# TRANSLATIONS_DIR (the output folder itself) still overrides.
TRANSLATIONS_DIR = os.environ.get("TRANSLATIONS_DIR") or os.path.join(_translations_root(), "output")
DOCUMENT_METHODS = [
    "Manual Validation (document)",
    "General Missives Ground Truth",
    "Segmentation model",
    "TANAP Digitized Index",
]
VERDICTS = ("good", "minor", "major", "unusable", "unsure")
ISSUES = {
    "omission": "Text left out",
    "addition": "Text added / invented",
    "mistranslation": "Mistranslation",
    "too_free": "Too free (structure lost)",
    "too_literal": "Too literal (hard to read)",
    "names": "Person or ship name changed",
    "places": "Place name modernized or wrong",
    "terms": "VOC term or unit wrong",
    "brackets": "Notes or brackets added",
    "boundaries": "English on the wrong scan",
    "untranslated": "Dutch left untranslated",
}


def runs() -> list[dict]:
    """Translation runs and their inventories, the full run first. An
    inventory is a folder in the run's output (one listing per run, cached
    until the folder changes: a full run has ~7,000 on a slow external
    drive)."""
    out = []
    if not os.path.isdir(TRANSLATIONS_DIR):
        return out
    for entry in sorted(os.scandir(TRANSLATIONS_DIR), key=lambda e: (e.name != "full", e.name)):
        if entry.is_dir():
            invs = _inventories(entry.path, entry.stat().st_mtime)
            if invs:
                out.append({"run": entry.name, "inventories": invs})
    return out


@lru_cache(maxsize=64)
def _inventories(run_dir: str, mtime: float) -> list[str]:
    def key(name: str):
        digits = "".join(c for c in name if c.isdigit())
        return (int(digits) if digits else 0, name)

    with os.scandir(run_dir) as entries:
        return sorted((e.name for e in entries if e.is_dir()), key=key)


def source_texts(run: str, inventory_number: str) -> dict[str, str] | None:
    """The Dutch as it was sent to the model (written by prepare: margin notes
    in <margin> lines, hyphenations joined), when the run has it."""
    path = os.path.join(os.path.dirname(TRANSLATIONS_DIR), "runs", run, "sources", f"{inventory_number}.parquet")
    if not os.path.exists(path):
        return None
    return _source_texts(path, os.path.getmtime(path))


@lru_cache(maxsize=8)
def _source_texts(path: str, mtime: float) -> dict[str, str]:
    import pandas as pd

    t = pd.read_parquet(path, columns=["filename", "text"])
    return dict(zip(t["filename"], t["text"].fillna("")))


def translation_path(run: str, inventory_number: str) -> str:
    return os.path.join(TRANSLATIONS_DIR, run, inventory_number, f"{inventory_number}.parquet")


def plan_path(run: str, inventory_number: str) -> str:
    return os.path.join(os.path.dirname(TRANSLATIONS_DIR), "runs", run, "plans", f"{inventory_number}.json")


@lru_cache(maxsize=8)
def _translations(path: str, mtime: float, plan: str, plan_mtime: float) -> dict[str, dict]:
    """filename -> status (translated, skipped, missing), skip_reason, text_en, model."""
    import pandas as pd

    t = pd.read_parquet(path, columns=["filename", "translation", "model"])
    out = {
        f: {"filename": f, "status": "translated", "skip_reason": None, "text_en": en, "model": m}
        for f, en, m in zip(t["filename"], t["translation"], t["model"])
    }
    if plan_mtime:
        with open(plan, encoding="utf-8") as handle:
            for s in json.load(handle)["scans"]:
                if s["filename"] not in out:
                    skipped = s["status"] == "skip"
                    out[s["filename"]] = {
                        "filename": s["filename"], "status": "skipped" if skipped else "missing",
                        "skip_reason": s.get("reason") if skipped else None, "text_en": None, "model": None,
                    }
    return out


def translations(run: str, inventory_number: str) -> dict[str, dict]:
    path = translation_path(run, inventory_number)
    if not os.path.exists(path):
        raise KeyError(f"{run}/{inventory_number}")
    plan = plan_path(run, inventory_number)
    return _translations(path, os.path.getmtime(path), plan, os.path.getmtime(plan) if os.path.exists(plan) else 0.0)


def document_items(conn, inv_id: str, scans: list[dict], translated: dict[str, dict]) -> tuple[list[dict], str | None]:
    """Top-level documents of the best method, in scan order, each covering
    its first to last scan; runs of scans outside any document as 'gap' items."""
    position = {s["f"]: k for k, s in enumerate(scans)}
    rows = conn.execute(
        """SELECT d.id, m.name, d.title, s.filename
           FROM document d
           JOIN document_identification_method m ON m.id = d.method_id
           JOIN page2document pd ON pd.document_id = d.id
           JOIN page p ON p.id = pd.page_id
           JOIN scan s ON s.id = p.scan_id
           WHERE d.inventory_id = ? AND d.part_of_id IS NULL""",
        (inv_id,),
    ).fetchall()
    docs: dict[str, dict] = {}
    for doc_id, method, title, filename in rows:
        if filename not in position:
            continue
        d = docs.setdefault(doc_id, {"method": method, "title": title or "", "lo": position[filename], "hi": position[filename]})
        d["lo"] = min(d["lo"], position[filename])
        d["hi"] = max(d["hi"], position[filename])
    method = next((m for m in DOCUMENT_METHODS if any(d["method"] == m for d in docs.values())), None)
    chosen = sorted(((k, d) for k, d in docs.items() if d["method"] == method), key=lambda kd: (kd[1]["lo"], kd[1]["hi"]))

    def counts(lo, hi):
        status = [(translated.get(scans[k]["f"]) or {}).get("status") for k in range(lo, hi + 1)]
        return {"n": hi - lo + 1, "translated": status.count("translated"), "missing": status.count("missing")}

    items, covered = [], set()
    for doc_id, d in chosen:
        items.append({"key": f"doc:{doc_id}", "kind": "doc", "title": d["title"] or "(untitled)", "lo": d["lo"], "hi": d["hi"], **counts(d["lo"], d["hi"])})
        covered.update(range(d["lo"], d["hi"] + 1))
    # scans outside any document, in runs
    k = 0
    while k < len(scans):
        if k in covered:
            k += 1
            continue
        lo = k
        while k + 1 < len(scans) and k + 1 not in covered:
            k += 1
        c = counts(lo, k)
        if c["translated"]:
            items.append({"key": f"gap:{scans[lo]['f']}", "kind": "gap", "title": "Scans outside documents", "lo": lo, "hi": k, **c})
        k += 1
    items.sort(key=lambda it: (it["lo"], it["hi"]))
    for it in items:
        it["start"], it["end"] = scans[it["lo"]]["f"], scans[it["hi"]]["f"]
    return items, method


def current_reviews(conn, run: str | None = None, inventory_number: str | None = None) -> dict[tuple[str, str, str], dict]:
    """Latest review row per (run, inventory, item_key); cleared reviews are left out."""
    where, params = [], []
    if run:
        where.append("run = ?"); params.append(run)
    if inventory_number:
        where.append("inventory = ?"); params.append(inventory_number)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    try:
        cur = conn.execute(
            f"SELECT * FROM translation_review WHERE id IN "
            f"(SELECT max(id) FROM translation_review {clause} GROUP BY run, inventory, item_key)",
            params,
        )
    except Exception:  # table not created yet
        return {}
    cols = [c[0] for c in cur.description]
    out = {}
    for row in cur.fetchall():
        rv = dict(zip(cols, row))
        if rv["verdict"] or rv["note"] or rv["issues"] or rv["flagged_scans"]:
            out[(rv["run"], rv["inventory"], rv["item_key"])] = rv
    return out


def load(conn, run: str, inventory_number: str) -> dict:
    """Everything the review page needs, except the texts (fetched per document)."""
    from segmentation.review import load_scans

    inv_id, scans = load_scans(conn, inventory_number)
    translated = translations(run, inventory_number)
    items, method = document_items(conn, inv_id, scans, translated)
    reviews = {k: v for (_, _, k), v in current_reviews(conn, run, inventory_number).items()}
    for it in items:
        it["review"] = reviews.get(it["key"])
    return {"run": run, "inventory": inventory_number, "scans": scans, "items": items, "method": method}


def document_rows(scans: list[dict], lo: int, hi: int, texts: dict[str, str], translated: dict[str, dict]) -> list[dict]:
    rows = []
    for s in scans[lo : hi + 1]:
        t = translated.get(s["f"]) or {}
        rows.append({
            **s,
            "nl": texts.get(s["f"], ""),
            "en": t.get("text_en"),
            "status": t.get("status") or "not in run",
            "reason": t.get("skip_reason"),
        })
    return rows


def build_row(data: dict, payload: dict, now: str | None = None) -> dict:
    items = {it["key"]: it for it in data["items"]}
    it = items[payload["key"]]
    clear = bool(payload.get("clear"))
    verdict = None if clear else (payload.get("verdict") or None)
    if verdict and verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict}")
    issues = [] if clear else [i for i in payload.get("issues") or [] if i in ISSUES]
    in_doc = {s["f"] for s in data["scans"][it["lo"] : it["hi"] + 1]}
    flagged = [] if clear else [f for f in payload.get("flagged") or [] if f in in_doc]
    return {
        "run": data["run"], "inventory": data["inventory"], "item_key": it["key"], "kind": it["kind"],
        "title": it["title"], "first_scan": it["start"], "last_scan": it["end"],
        "verdict": verdict, "issues": ",".join(issues) or None, "flagged_scans": ",".join(flagged) or None,
        "note": None if clear else (payload.get("note") or "").strip() or None,
        "reviewer": (payload.get("reviewer") or "").strip() or None,
        "created_at": now or datetime.now().isoformat(timespec="seconds"),
    }


def export_rows(conn):
    import pandas as pd

    rows = list(current_reviews(conn).values())
    return pd.DataFrame(rows).drop(columns=["id"], errors="ignore")
