"""
Store segmentation output (the CSV written by `python -m segmentation run`)
as documents of the identification method "Segmentation model".

  - One Document per ToC entry (entries starting on the same scan each get
    their own), per subdocument and per unindexed document; non-document runs
    are skipped.
  - ToC documents copy title, dates, page range and settlement from their ToC
    entry and are linked to the entry's index ids: OBP_INDEX, TANAP and
    DIGITIZED TYPOSCRIPTS (the same ExternalID rows) and NT (the entry's id in
    OBP NT_gecorrigeerd.xlsx; ExternalID rows with context "NT").
  - ToC entries derived from other versions of the text (entry_id;
    versions.py) take title and dates from the CSV; their source and the
    other versions are in their evidence (no index ids: those belong to the
    other volume's ToC).
  - Court cases (kind 'case', from the EMDCCR dataset; court_records.py) take
    title, dates and place from the CSV and are linked to their case id
    (ExternalID context "EMDCCR"); the ranges of individual accused are
    subdocuments of their case.
  - Subdocuments and nested ToC entries get part_of_id = their parent's
    document; a parent's pages include those of all its descendants.
  - Page2Document rows for every page, in scan order (verso before recto on
    double scans), source SEGMENTATION_MODEL, confidence CANDIDATE.
  - The model's evidence per document goes into document_evidence (JSON).

Every committed inventory is appended to <csv>.imported; `import --resume`
skips those, so an interrupted import continues where it stopped.

Re-importing an inventory first deletes that method's documents there. So
importing a CSV whose court-record inventories were segmented without the
court cases (e.g. a segments.csv written before court_records.py existed)
replaces the court cases with model-only documents: the import warns about
this; re-import the court segments (court_segments.csv) afterwards, or re-run
`segmentation run` for those inventories.
"""

import datetime
import json
import os
import logging
import time
import uuid
from collections import defaultdict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import Base, LinkConfidence

logger = logging.getLogger("segmentation")

METHOD_NAME = "Segmentation model"
METHOD_DESCRIPTION = (
    "Documents, subdocuments and unindexed documents found by the segmentation "
    "model (segmentation/ in this repository): per-scan start/end/non-document "
    "scores from all predictors, aligned with the ToC where there is one."
)
SOURCE = "SEGMENTATION_MODEL"
NT_CONTEXT = "NT"
COURT_CONTEXT = "EMDCCR"
DERIVED_TOC = "derived_toc.csv"  # entries derived from other versions (versions.py)
DOC_COLUMNS = (
    "folio_start", "folio_end", "date_earliest_begin", "date_latest_begin",
    "date_earliest_end", "date_latest_end", "location_id",
)


def _method_id(session: Session) -> str:
    row = session.execute(text("SELECT id FROM document_identification_method WHERE name = :n"), {"n": METHOD_NAME}).first()
    if row:
        return row[0]
    mid = str(uuid.uuid4())
    session.execute(
        text("INSERT INTO document_identification_method (id, name, description, date) VALUES (:id, :n, :d, :dt)"),
        {"id": mid, "n": METHOD_NAME, "d": METHOD_DESCRIPTION, "dt": datetime.date.today()},
    )
    return mid


def _in_chunks(session: Session, sql: str, ids: list[str]):
    for k in range(0, len(ids), 500):
        chunk = {f"p{j}": v for j, v in enumerate(ids[k : k + 500])}
        session.execute(text(sql.format(ph=",".join(f":p{j}" for j in range(len(chunk))))), chunk)


def _delete_previous(session: Session, method_id: str, inventory_id: str) -> int:
    ids = [r[0] for r in session.execute(text("SELECT id FROM document WHERE method_id = :m AND inventory_id = :i"), {"m": method_id, "i": inventory_id})]
    for table in ("page2document", "document2external_id", "document2documenttype", "document_evidence"):
        _in_chunks(session, f"DELETE FROM {table} WHERE document_id IN ({{ph}})", ids)
    _in_chunks(session, "UPDATE document SET part_of_id = NULL WHERE id IN ({ph})", ids)
    _in_chunks(session, "DELETE FROM document WHERE id IN ({ph})", ids)
    return len(ids)


def _nt_external_ids(session: Session, nt_indexes: set[int], dry_run: bool) -> dict[int, str]:
    """ExternalID id per NT index, created where missing."""
    ids = _external_ids(session, NT_CONTEXT, {str(i) for i in nt_indexes}, dry_run)
    return {int(k): v for k, v in ids.items()}


def _external_ids(session: Session, context: str, identifiers: set[str], dry_run: bool) -> dict[str, str]:
    """ExternalID id per identifier in `context`, created where missing."""
    have = {}
    for r in session.execute(text("SELECT identifier, id FROM external_id WHERE context = :c"), {"c": context}):
        have[str(r[0])] = r[1]
    new = [{"id": str(uuid.uuid4()), "identifier": i, "context": context} for i in identifiers if i not in have]
    if new and not dry_run:
        session.execute(text('INSERT INTO external_id (id, "URL", identifier, context) VALUES (:id, NULL, :identifier, :context)'), new)
    have.update({r["identifier"]: r["id"] for r in new})
    return have


def _settlements(session: Session) -> dict[str, str]:
    return {r[0]: r[1] for r in session.execute(text("SELECT label, settlement_id FROM settlement_label"))}


def _toc_entries(conn, inv_id: str) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """ToC entries of the inventory by csv id, and the ExternalID rows of each."""
    toc = pd.read_sql(
        text(
            "SELECT CAST(e.identifier AS INTEGER) AS csv_id, d.id AS toc_doc_id, d.title, d.nt_index, "
            + ", ".join(f"d.{c}" for c in DOC_COLUMNS)
            + " FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' "
            "JOIN document2external_id de ON de.document_id = d.id JOIN external_id e ON e.id = de.external_id AND e.context = 'OBP_INDEX' "
            "WHERE d.inventory_id = :i"
        ),
        conn,
        params={"i": inv_id},
    ).drop_duplicates("csv_id").set_index("csv_id")
    ext = pd.read_sql(
        text(
            "SELECT de.document_id, de.external_id FROM document2external_id de "
            "JOIN document d ON d.id = de.document_id JOIN document_identification_method m ON m.id = d.method_id "
            "AND m.name = 'TANAP Digitized Index' WHERE d.inventory_id = :i"
        ),
        conn,
        params={"i": inv_id},
    )
    by_doc = ext.groupby("document_id")["external_id"].apply(list).to_dict()
    return toc, {cid: by_doc.get(r.toc_doc_id, []) for cid, r in toc.iterrows()}


def import_segments(csv_path: str, database_url: str, dry_run: bool = False, resume: bool = False):
    """Import the segments; every committed inventory is recorded in <csv>.imported,
    and with resume=True the inventories listed there are skipped."""
    seg = pd.read_csv(csv_path, dtype={"inventory": str, "toc_csv_ids": str}, low_memory=False)
    seg = seg[seg["kind"] != "non-document"]
    extra = ("court_case", "parent_court_case", "title", "date_begin", "date_end", "place", "entry_id", "parent_entry_id")
    for c in extra:
        if c not in seg:
            seg[c] = None  # CSVs written before court records / derived ToCs were added
    seg = seg.astype({c: object for c in extra})
    warnings = _stale_warnings(seg, csv_path)
    for w in warnings:
        logger.warning(w)
    progress_path = csv_path + ".imported"
    done = set()
    if resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            done = {line.strip() for line in f if line.strip()}
        logger.info("Resuming: %d inventories already imported (%s)", len(done), progress_path)
    elif not dry_run and os.path.exists(progress_path):
        os.remove(progress_path)  # a new import of this CSV
    seg = seg[~seg["inventory"].isin(done)]
    t0, n_inv = time.time(), seg["inventory"].nunique()
    engine = create_engine(database_url)
    if not dry_run:
        Base.metadata.create_all(engine)  # document_evidence
    with Session(engine) as session:
        method_id = _method_id(session)
        settlements = _settlements(session)
        court_ext = _external_ids(session, COURT_CONTEXT, {c for c in seg["court_case"].dropna()}, dry_run)
        totals = defaultdict(int)
        for inv_number, rows in seg.groupby("inventory", sort=False):
            inv_id = session.execute(text("SELECT id FROM inventory WHERE inventory_number = :n"), {"n": inv_number}).scalar()
            if inv_id is None:
                logger.warning("Inventory %s not found; skipped", inv_number)
                continue
            conn = session.connection()
            scans = pd.read_sql(text("SELECT id AS scan_id, filename FROM scan WHERE inventory_id = :i ORDER BY scan_order, filename"), conn, params={"i": inv_id})
            pos = {f: i for i, f in enumerate(scans["filename"])}
            pages = pd.read_sql(text("SELECT id AS page_id, scan_id, recto_verso FROM page WHERE inventory_id = :i"), conn, params={"i": inv_id})
            pages["side"] = pages["recto_verso"].map({"Verso": 0, "Recto": 1}).fillna(0)
            pages_by_scan = pages.sort_values(["scan_id", "side"]).groupby("scan_id")["page_id"].apply(list).to_dict()
            toc, toc_ext = _toc_entries(conn, inv_id)
            nt_ext = _nt_external_ids(session, {int(v) for v in toc["nt_index"].dropna()}, dry_run)
            if not dry_run:
                totals["replaced"] += _delete_previous(session, method_id, inv_id)

            docs: list[dict] = []
            scans_of: dict[str, set[int]] = {}
            parent_csv: dict[str, int] = {}
            doc_by_csv: dict[int, str] = {}
            evidence: dict[str, str] = {}
            case_doc: dict[str, str] = {}  # court case id -> its (first) document
            case_docs: list[tuple[str, str]] = []  # (court case id, document) of every court case document
            parent_case: dict[str, str] = {}  # document -> court case id of its parent
            entry_doc: dict[str, str] = {}  # derived ToC entry id -> its document
            parent_entry: dict[str, str] = {}  # document -> derived entry id of its parent
            for r in rows.itertuples():
                span = set(range(pos[r.start_scan], pos[r.end_scan] + 1))
                csv_ids = [int(x) for x in str(r.toc_csv_ids).split(";") if x and x != "nan"]
                for k, cid in enumerate(csv_ids or [None]):
                    doc_id = str(uuid.uuid4())
                    entry = toc.loc[cid] if cid is not None and cid in toc.index else None
                    doc = {"id": doc_id, "inventory_id": inv_id, "method_id": method_id, "part_of_id": None, "date_text": None,
                           "title": entry["title"] if entry is not None else None}
                    for c in DOC_COLUMNS:
                        v = entry[c] if entry is not None else None
                        doc[c] = None if v is None or pd.isna(v) else (int(v) if c.startswith("folio") else v)
                    docs.append(doc)
                    scans_of[doc_id] = span
                    if cid is not None:
                        doc_by_csv[cid] = doc_id
                    if k > 0:
                        parent_csv[doc_id] = csv_ids[0]  # another entry starting on the same scan
                    elif not pd.isna(r.parent_csv_id):
                        parent_csv[doc_id] = int(r.parent_csv_id)
                    if isinstance(r.evidence, str):
                        evidence[doc_id] = r.evidence
                    if isinstance(r.entry_id, str):  # ToC entry derived from another version (versions.py)
                        doc["title"] = r.title if isinstance(r.title, str) else None
                        doc["date_earliest_begin"] = r.date_begin if isinstance(r.date_begin, str) else None
                        doc["date_latest_end"] = r.date_end if isinstance(r.date_end, str) else None
                        entry_doc[r.entry_id] = doc_id
                    if isinstance(r.parent_entry_id, str):
                        parent_entry[doc_id] = r.parent_entry_id
                    if isinstance(r.court_case, str):
                        doc["title"] = r.title if isinstance(r.title, str) else None
                        doc["date_earliest_begin"] = r.date_begin if isinstance(r.date_begin, str) else None
                        doc["date_latest_end"] = r.date_end if isinstance(r.date_end, str) else None
                        doc["location_id"] = settlements.get(r.place) if isinstance(r.place, str) else None
                        if r.kind == "case":
                            case_doc.setdefault(r.court_case, doc_id)
                            case_docs.append((r.court_case, doc_id))
                        if isinstance(r.parent_court_case, str):
                            parent_case[doc_id] = r.parent_court_case
            for d in docs:
                if d["id"] in parent_csv:
                    d["part_of_id"] = doc_by_csv.get(parent_csv[d["id"]])
                elif d["id"] in parent_case:
                    d["part_of_id"] = case_doc.get(parent_case[d["id"]])
                elif d["id"] in parent_entry:
                    d["part_of_id"] = entry_doc.get(parent_entry[d["id"]])
            # a parent spans all its descendants
            children = defaultdict(list)
            for d in docs:
                if d["part_of_id"]:
                    children[d["part_of_id"]].append(d["id"])

            def all_scans(doc_id, seen=()):
                out = set(scans_of[doc_id])
                for c in children[doc_id]:
                    if c not in seen:
                        out |= all_scans(c, seen + (doc_id,))
                return out

            links, ext_links, ev_rows = [], [], []
            for d in docs:
                idx = 0
                for p in sorted(all_scans(d["id"])):
                    for page_id in pages_by_scan.get(scans.at[p, "scan_id"], []):
                        links.append({"id": str(uuid.uuid4()), "page_id": page_id, "document_id": d["id"], "index": idx,
                                      "source": SOURCE, "confidence": LinkConfidence.CANDIDATE.value})
                        idx += 1
            for cid, doc_id in doc_by_csv.items():
                for ext_id in toc_ext.get(cid, []):
                    ext_links.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": ext_id})
                nt = toc.at[cid, "nt_index"] if cid in toc.index else None
                if nt is not None and not pd.isna(nt):
                    ext_links.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": nt_ext[int(nt)]})
            for case_id, doc_id in case_docs:
                ext_links.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": court_ext[case_id]})
            ev_rows = [{"document_id": k, "evidence": v} for k, v in evidence.items()]

            totals["documents"] += len(docs)
            totals["ToC documents"] += len(doc_by_csv)
            totals["court case documents"] += len(case_docs)
            totals["derived ToC documents"] += len(entry_doc)
            totals["subdocuments"] += sum(1 for d in docs if d["part_of_id"])
            totals["page links"] += len(links)
            totals["index id links"] += len(ext_links)
            totals["inventories"] += 1
            logger.info("%d/%d %s: %d documents (%d ToC, %d part of another), %d page links, %d index id links (%.0fs)",
                        totals["inventories"], n_inv, inv_number, len(docs), len(doc_by_csv), sum(1 for d in docs if d["part_of_id"]),
                        len(links), len(ext_links), time.time() - t0)
            if dry_run:
                continue
            cols = list(docs[0].keys())
            # parents first (part_of_id is a foreign key)
            docs.sort(key=lambda d: d["part_of_id"] is not None)
            session.execute(text(f"INSERT INTO document ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"), docs)
            if ext_links:
                session.execute(text("INSERT INTO document2external_id (id, document_id, external_id) VALUES (:id, :document_id, :external_id)"), ext_links)
            if links:
                session.execute(text('INSERT INTO page2document (id, page_id, document_id, "index", source, confidence) '
                                     "VALUES (:id, :page_id, :document_id, :index, :source, :confidence)"), links)
            if ev_rows:
                session.execute(text("INSERT INTO document_evidence (document_id, evidence) VALUES (:document_id, :evidence)"), ev_rows)
            session.commit()  # per inventory: a large import can be interrupted and resumed
            with open(progress_path, "a") as f:
                f.write(inv_number + "\n")
        if dry_run:
            session.rollback()
            logger.info("Dry run — nothing written. Would create %s", dict(totals))
        else:
            session.commit()
            logger.info("Imported %s", dict(totals))
    for w in warnings:
        logger.warning(w)


def _stale_warnings(seg: pd.DataFrame, csv_path: str) -> list[str]:
    """Warnings for inventories whose court cases or derived ToC entries this CSV would replace."""
    out = []
    without_cases = _court_inventories_without_cases(seg)
    if without_cases:
        out.append(_court_warning(without_cases, csv_path))
    if os.path.exists(DERIVED_TOC) and os.path.abspath(csv_path) != os.path.abspath(DERIVED_TOC):
        derived = set(pd.read_csv(DERIVED_TOC, usecols=["inventory"], dtype=str)["inventory"])
        has = set(seg.loc[seg["entry_id"].notna() | seg["court_case"].notna(), "inventory"])  # court cases take precedence
        missing = sorted((set(seg["inventory"]) & derived) - has, key=lambda n: (len(n), n))
        if missing:
            shown = ", ".join(missing[:10]) + (f" … ({len(missing)} in all)" if len(missing) > 10 else "")
            out.append(f"{csv_path} has no derived ToC entries for {len(missing)} inventories that have them in {DERIVED_TOC} ({shown}); "
                       "importing it replaces their derived ToC documents with model-only documents. Re-import the derived "
                       "segments afterwards (uv run python -m segmentation import derived_segments.csv) or re-run "
                       "`segmentation run` for these inventories.")
    return out


def _court_inventories_without_cases(seg: pd.DataFrame) -> list[str]:
    """Inventories in the CSV that have court cases (data/EMDCCR*.xlsx) but
    were segmented without them."""
    from .court_records import load_court_cases

    _, ranges = load_court_cases()
    has_cases = set(seg.loc[seg["court_case"].notna(), "inventory"])
    return sorted((set(seg["inventory"]) & set(ranges)) - has_cases, key=lambda n: (len(n), n))


def _court_warning(inventories: list[str], csv_path: str) -> str:
    shown = ", ".join(inventories[:10]) + (f" … ({len(inventories)} in all)" if len(inventories) > 10 else "")
    return (f"{csv_path} has no court cases for {len(inventories)} inventories with court records ({shown}); "
            "importing it replaces their court cases with model-only documents. Re-import the court segments "
            "afterwards (uv run python -m segmentation import court_segments.csv) or re-run "
            "`segmentation run` for these inventories.")
