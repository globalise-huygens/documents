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
  - General Missives identified by hand (gm_ids; the documents of "General
    Missives Ground Truth") take title, date text and dates from that
    document, and its index ids when it has no ToC entry in the inventory;
    further missives starting on the same scan (a secret letter with the
    general one) span their own scans and are part of the first, unless they
    have an index entry of their own (ToC documents are never parts). Subdocuments
    found inside a missive (parent_gm_id) are part of its document.
  - Court cases (kind 'case', from the EMDCCR dataset; court_records.py) take
    title, dates and place from the CSV and are linked to their case id
    (ExternalID context "EMDCCR"); the ranges of individual accused are
    subdocuments of their case.
  - Subdocuments get part_of_id = their parent's document; a parent's pages
    include those of all its descendants. ToC documents are never part of
    another document.
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
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from models import Base, LinkConfidence
from segmentation.doc_titles import Places

logger = logging.getLogger("segmentation")

METHOD_NAME = "Segmentation model"
METHOD_DESCRIPTION = (
    "Documents, subdocuments and unindexed documents found by the segmentation "
    "model (segmentation/ in this repository): per-scan start/end/non-document "
    "scores from all predictors, aligned with the ToC where there is one."
)
SOURCE = "SEGMENTATION_MODEL"
NT_CONTEXT = "NT"
# document.title_source: who made the title
TITLE_VOC = "VOC"  # the VOC itself: the index, or a General Missive identified by hand
TITLE_OTHER_VERSION = "VOC (other version)"  # the index entry of another version of the text
TITLE_COURT = "Resilient Diversity"  # the court case dataset (EMDCCR)
TITLE_INFERRED = "inferred"  # by this project, from the text
SQLITE_CACHE_MB = 2000  # page cache of the import's connection
INFERRED_TITLES = "inferred_titles.csv"  # infer-titles (doc_titles.py)
BRACKETED = {TITLE_INFERRED, TITLE_COURT}  # titles not given by the VOC itself are stored as [title]
COURT_TYPE = "processtuk"  # GLOBALISE type (Dutch label) of a court case
LETTER_TYPE = "brief"  # ... of a General Missive without an index entry
COURT_CONTEXT = "EMDCCR"
DERIVED_TOC = "derived_toc.csv"  # entries derived from other versions (versions.py)
DOC_COLUMNS = (
    "folio_start", "folio_end", "date_earliest_begin", "date_latest_begin",
    "date_earliest_end", "date_latest_end", "location_id", "place_glob_id",
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
    for table in ("page2document", "document2external_id", "document2documenttype", "document_evidence", "document_settlement"):
        _in_chunks(session, f"DELETE FROM {table} WHERE document_id IN ({{ph}})", ids)
    _in_chunks(session, "UPDATE document SET part_of_id = NULL WHERE id IN ({ph})", ids)
    _in_chunks(session, "DELETE FROM document WHERE id IN ({ph})", ids)
    return len(ids)


def _nt_external_ids(session: Session, nt_indexes: set[int], dry_run: bool) -> dict[int, str]:
    """ExternalID id per NT index, created where missing."""
    ids = _external_ids(session, NT_CONTEXT, {str(i) for i in nt_indexes}, dry_run)
    return {int(k): v for k, v in ids.items()}


_known_ids: dict[str, dict[str, str]] = {}  # context -> identifier -> ExternalID id, read once per import


def _external_ids(session: Session, context: str, identifiers: set[str], dry_run: bool) -> dict[str, str]:
    """ExternalID id per identifier in `context`, created where missing."""
    if context not in _known_ids:
        _known_ids[context] = {
            str(r[0]): r[1] for r in session.execute(text("SELECT identifier, id FROM external_id WHERE context = :c"), {"c": context})
        }
    have = _known_ids[context]
    new = [{"id": str(uuid.uuid4()), "identifier": i, "context": context} for i in identifiers if i not in have]
    if new and not dry_run:
        session.execute(text('INSERT INTO external_id (id, "URL", identifier, context) VALUES (:id, NULL, :identifier, :context)'), new)
    have.update({r["identifier"]: r["id"] for r in new})
    return have


GM_METHOD = "General Missives Ground Truth"
GM_COLUMNS = ("title", "date_text", "date_earliest_begin", "date_latest_begin", "date_earliest_end", "date_latest_end")


def _general_missives(conn, inv_id: str) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """The hand-identified General Missives of the inventory by document id
    (GM_COLUMNS, csv_id, first and last scan), and their ExternalID rows."""
    gm = pd.read_sql(
        text(
            "SELECT d.id, " + ", ".join(f"d.{c}" for c in GM_COLUMNS) + ", "
            "(SELECT CAST(e.identifier AS INTEGER) FROM document2external_id de JOIN external_id e ON e.id = de.external_id "
            " AND e.context = 'OBP_INDEX' WHERE de.document_id = d.id) AS csv_id, "
            "(SELECT min(s.filename) FROM page2document p JOIN page pg ON pg.id = p.page_id JOIN scan s ON s.id = pg.scan_id WHERE p.document_id = d.id) AS first_scan, "
            "(SELECT max(s.filename) FROM page2document p JOIN page pg ON pg.id = p.page_id JOIN scan s ON s.id = pg.scan_id WHERE p.document_id = d.id) AS last_scan "
            "FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = :m WHERE d.inventory_id = :i"
        ),
        conn,
        params={"m": GM_METHOD, "i": inv_id},
    )
    ext = pd.read_sql(
        text(
            "SELECT de.document_id, de.external_id FROM document2external_id de JOIN document d ON d.id = de.document_id "
            "JOIN document_identification_method m ON m.id = d.method_id AND m.name = :m WHERE d.inventory_id = :i"
        ),
        conn,
        params={"m": GM_METHOD, "i": inv_id},
    )
    info = {r["id"]: {k: (None if pd.isna(v) else v) for k, v in r.items()} for r in gm.to_dict("records")}
    return info, ext.groupby("document_id")["external_id"].apply(list).to_dict()


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


def _toc_types(conn, inv_id: str) -> dict[str, list[str]]:
    """Document types (GLOBALISE and OBP/TANAP concepts) of the inventory's index entries."""
    rows = conn.execute(
        text(
            "SELECT x.document_id, x.document_type_id FROM document2documenttype x JOIN document d ON d.id = x.document_id "
            "JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' WHERE d.inventory_id = :i"
        ),
        {"i": inv_id},
    ).fetchall()
    out: dict[str, list[str]] = {}
    for doc, t in rows:
        out.setdefault(doc, []).append(t)
    return out


def _globalise_types(session: Session) -> dict[str, str]:
    """GLOBALISE document type concepts by Dutch label."""
    return {r[0]: r[1] for r in session.execute(text("SELECT pref_label_nl, id FROM document_type WHERE scheme = 'GLOBALISE'"))}


def _settlement_globs(session: Session) -> dict[str, str]:
    """GLOB id -> settlement id. Settlements are only looked up: other places get just a place_glob_id."""
    return {r[0]: r[1] for r in session.execute(text("SELECT glob_id, id FROM settlement WHERE glob_id IS NOT NULL"))}


def _toc_settlements(conn, inv_id: str) -> dict[str, list[str]]:
    """All settlements of the inventory's index entries (step 21 merges entries listed once per settlement)."""
    rows = conn.execute(
        text(
            "SELECT ds.document_id, ds.settlement_id FROM document_settlement ds JOIN document d ON d.id = ds.document_id "
            "JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' WHERE d.inventory_id = :i"
        ),
        {"i": inv_id},
    ).fetchall()
    out: dict[str, list[str]] = {}
    for doc, settlement in rows:
        out.setdefault(doc, []).append(settlement)
    return out


def _inferred_titles(path: str) -> dict[tuple[str, str], dict]:
    """Inferred titles (doc_titles.py) by (inventory, segment), if the CSV exists."""
    if not path or not os.path.exists(path):
        return {}
    t = pd.read_csv(path, dtype={"inventory": str, "segment": str}, keep_default_na=False, na_values=[""])
    return {(r.inventory, r.segment): r._asdict() for r in t.itertuples(index=False)}


def _date_range(iso: str | None, precision: str | None) -> tuple[str | None, str | None]:
    """First and last day a date of this precision covers."""
    if not isinstance(iso, str):
        return None, None
    d = pd.Timestamp(iso)
    if precision == "year":
        return f"{d.year}-01-01", f"{d.year}-12-31"
    if precision == "month":
        return d.strftime("%Y-%m-01"), (d + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    return iso, iso


def import_segments(csv_path: str, database_url: str, dry_run: bool = False, resume: bool = False, titles_path: str = INFERRED_TITLES):
    """Import the segments; every committed inventory is recorded in <csv>.imported,
    and with resume=True the inventories listed there are skipped. Documents without
    a title from the index, a court case or a General Missive get their inferred
    title (titles_path, from infer-titles) with title_source 'inferred'."""
    inferred = _inferred_titles(titles_path)
    if inferred:
        logger.info("Inferred titles for %d segments (%s)", len(inferred), titles_path)
    seg = pd.read_csv(csv_path, dtype={"inventory": str, "toc_csv_ids": str}, low_memory=False)
    seg = seg[seg["kind"] != "non-document"]
    extra = ("court_case", "parent_court_case", "title", "date_begin", "date_end", "place", "entry_id", "parent_entry_id", "gm_ids", "parent_gm_id")
    for c in extra:
        if c not in seg:
            seg[c] = None  # CSVs written before court records / derived ToCs / General Missives were added
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
    _known_ids.clear()
    if engine.dialect.name == "sqlite":
        # the database is large (12 GB): a big page cache keeps the indexes the deletes and inserts touch in memory
        # (about twice as fast as the 2 MB default); synchronous=NORMAL syncs less often, still safe in rollback mode
        @event.listens_for(engine, "connect")
        def _pragmas(con, _):
            con.execute(f"PRAGMA cache_size = -{SQLITE_CACHE_MB * 1000}")
            con.execute("PRAGMA synchronous = NORMAL")
            con.execute("PRAGMA temp_store = MEMORY")

    if not dry_run:
        Base.metadata.create_all(engine)  # document_evidence
    with Session(engine) as session:
        method_id = _method_id(session)
        settlements = _settlements(session)
        settlement_ids = _settlement_globs(session)  # GLOB id -> settlement id
        settlement_glob = {v: k for k, v in settlement_ids.items() if k in Places().ids}  # settlement id -> GLOB id of the places dataset
        type_ids = _globalise_types(session)  # Dutch GLOBALISE type -> document_type id
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
            toc_settlements = _toc_settlements(conn, inv_id)
            toc_types = _toc_types(conn, inv_id)
            nt_ext = _nt_external_ids(session, {int(v) for v in toc["nt_index"].dropna()}, dry_run)
            gm_info, gm_ext = _general_missives(conn, inv_id)
            stale = {g for v in rows["gm_ids"].dropna() for g in str(v).split(";") if g and g not in gm_info}
            if stale:
                logger.warning("%s: %d General Missives of %s are no longer in the database (8_import_GM.py was run again?); "
                               "re-run `segmentation run` for this inventory to use the current ones", inv_number, len(stale), csv_path)
            if not dry_run:
                totals["replaced"] += _delete_previous(session, method_id, inv_id)

            docs: list[dict] = []
            doc_type: dict[str, str] = {}  # document -> Dutch GLOBALISE type (inferred, court case, missive)
            scans_of: dict[str, set[int]] = {}
            parent_csv: dict[str, int] = {}
            parent_doc: dict[str, str] = {}  # document -> the first document starting on the same scan
            doc_by_csv: dict[int, str] = {}
            gm_links: list[tuple[str, str]] = []  # (document, General Missive) whose index ids it takes over
            gm_doc: dict[str, str] = {}  # General Missive -> its document
            parent_gm: dict[str, str] = {}  # document -> General Missive of its parent
            evidence: dict[str, str] = {}
            case_doc: dict[str, str] = {}  # court case id -> its (first) document
            case_docs: list[tuple[str, str]] = []  # (court case id, document) of every court case document
            parent_case: dict[str, str] = {}  # document -> court case id of its parent
            entry_doc: dict[str, str] = {}  # derived ToC entry id -> its document
            parent_entry: dict[str, str] = {}  # document -> derived entry id of its parent
            for r in rows.itertuples():
                span = set(range(pos[r.start_scan], pos[r.end_scan] + 1))
                csv_ids = [int(x) for x in str(r.toc_csv_ids).split(";") if x and x != "nan"]
                # (ToC entry, General Missive) per document: each missive with its own entry, then the other entries
                units, rest = [], list(csv_ids)
                for g in (x for x in str(r.gm_ids).split(";") if x and x != "nan" and x in gm_info):
                    c = gm_info[g]["csv_id"]
                    c = int(c) if c is not None and int(c) in rest else None
                    if c is not None:
                        rest.remove(c)
                    units.append((c, g))
                units += [(c, None) for c in rest]
                first_doc = None
                for k, (cid, g) in enumerate(units or [(None, None)]):
                    doc_id = str(uuid.uuid4())
                    entry = toc.loc[cid] if cid is not None and cid in toc.index else None
                    doc = {"id": doc_id, "inventory_id": inv_id, "method_id": method_id, "part_of_id": None, "date_text": None,
                           "title": entry["title"] if entry is not None else None,
                           "title_source": TITLE_VOC if entry is not None and isinstance(entry["title"], str) else None}
                    for c in DOC_COLUMNS:
                        v = entry[c] if entry is not None else None
                        doc[c] = None if v is None or pd.isna(v) else (int(v) if c.startswith("folio") else v)
                    docs.append(doc)
                    scans_of[doc_id] = span
                    if g is not None:  # identified by hand: its title and dates win
                        gm_doc[g] = doc_id
                        doc.update({c: gm_info[g][c] for c in GM_COLUMNS})
                        doc["title_source"] = TITLE_VOC if doc.get("title") else doc["title_source"]
                        if k > 0 and gm_info[g]["last_scan"] in pos:
                            scans_of[doc_id] = set(range(pos[r.start_scan], min(pos[gm_info[g]["last_scan"]], pos[r.end_scan]) + 1))
                        if cid is None:
                            gm_links.append((doc_id, g))
                            doc_type[doc_id] = LETTER_TYPE  # a missive without an index entry of its own
                    if cid is not None:
                        doc_by_csv[cid] = doc_id
                    if k > 0 and cid is None:
                        parent_doc[doc_id] = first_doc  # a further missive starting on the same scan; ToC entries are never parts
                    elif isinstance(r.parent_gm_id, str):
                        parent_gm[doc_id] = r.parent_gm_id
                    elif not pd.isna(r.parent_csv_id):
                        parent_csv[doc_id] = int(r.parent_csv_id)
                    first_doc = first_doc or doc_id
                    if isinstance(r.evidence, str):
                        evidence[doc_id] = r.evidence
                    if isinstance(r.entry_id, str):  # ToC entry derived from another version (versions.py)
                        doc["title"] = r.title if isinstance(r.title, str) else None
                        doc["title_source"] = TITLE_OTHER_VERSION if doc["title"] else None
                        doc["date_earliest_begin"] = r.date_begin if isinstance(r.date_begin, str) else None
                        doc["date_latest_end"] = r.date_end if isinstance(r.date_end, str) else None
                        entry_doc[r.entry_id] = doc_id
                    if isinstance(r.parent_entry_id, str):
                        parent_entry[doc_id] = r.parent_entry_id
                    inf = inferred.get((inv_number, str(r.segment)))
                    if inf is not None and doc["title"] is None and not isinstance(r.court_case, str) and isinstance(inf.get("title"), str):
                        doc["title"], doc["title_source"] = inf["title"], TITLE_INFERRED
                        if doc.get("date_earliest_begin") is None:
                            doc["date_earliest_begin"], doc["date_latest_end"] = _date_range(inf.get("date"), inf.get("date_precision"))
                        if isinstance(inf.get("type_nl"), str):
                            doc_type[doc_id] = inf["type_nl"]
                        if isinstance(inf.get("place_glob_id"), str) and doc.get("location_id") is None:
                            doc["place_glob_id"] = inf["place_glob_id"]
                            doc["location_id"] = settlement_ids.get(inf["place_glob_id"])
                    if isinstance(r.court_case, str):
                        doc_type[doc_id] = COURT_TYPE
                        doc["title"] = r.title if isinstance(r.title, str) else None
                        doc["title_source"] = TITLE_COURT if doc["title"] else None
                        doc["date_earliest_begin"] = r.date_begin if isinstance(r.date_begin, str) else None
                        doc["date_latest_end"] = r.date_end if isinstance(r.date_end, str) else None
                        doc["location_id"] = settlements.get(r.place) if isinstance(r.place, str) else None
                        if r.kind == "case":
                            case_doc.setdefault(r.court_case, doc_id)
                            case_docs.append((r.court_case, doc_id))
                        if isinstance(r.parent_court_case, str):
                            parent_case[doc_id] = r.parent_court_case
            for d in docs:
                if d["id"] in parent_doc:
                    d["part_of_id"] = parent_doc[d["id"]]
                elif d["id"] in parent_gm:
                    d["part_of_id"] = gm_doc.get(parent_gm[d["id"]])
                elif d["id"] in parent_csv:
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
            for doc_id, g in gm_links:
                for ext_id in gm_ext.get(g, []):
                    ext_links.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": ext_id})
            for case_id, doc_id in case_docs:
                ext_links.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": court_ext[case_id]})
            seen = set()  # a merged index entry carries its NT numbers both as nt_index and as ids
            ext_links = [x for x in ext_links if (x["document_id"], x["external_id"]) not in seen and not seen.add((x["document_id"], x["external_id"]))]
            for d in docs:  # court cases (and ToC entries imported before step 22): the GLOB id of their settlement
                if d.get("location_id") and not d.get("place_glob_id"):
                    d["place_glob_id"] = settlement_glob.get(d["location_id"])
            settlement_rows = {(d["id"], d["location_id"]) for d in docs if d.get("location_id")}
            type_rows = {(doc_id, type_ids[t]) for doc_id, t in doc_type.items() if t in type_ids}
            for cid, doc_id in doc_by_csv.items():
                if cid in toc.index:
                    settlement_rows |= {(doc_id, x) for x in toc_settlements.get(toc.at[cid, "toc_doc_id"], [])}
                    type_rows |= {(doc_id, x) for x in toc_types.get(toc.at[cid, "toc_doc_id"], [])}  # GLOBALISE and OBP (TANAP) types
            for d in docs:  # titles not given by the VOC itself are marked as ours (or another project's)
                if d.get("title") and d.get("title_source") in BRACKETED and not d["title"].startswith("["):
                    d["title"] = f"[{d['title']}]"
            ev_rows = [{"document_id": k, "evidence": v} for k, v in evidence.items()]

            totals["documents"] += len(docs)
            totals["ToC documents"] += len(doc_by_csv)
            totals["court case documents"] += len(case_docs)
            totals["derived ToC documents"] += len(entry_doc)
            totals["General Missives"] += sum(1 for r in rows.itertuples() for x in str(r.gm_ids).split(";") if x in gm_info)
            totals["subdocuments"] += sum(1 for d in docs if d["part_of_id"])
            totals["page links"] += len(links)
            totals["index id links"] += len(ext_links)
            totals["document type links"] += len(type_rows)
            totals["settlement links"] += len(settlement_rows)
            totals["inferred titles"] += sum(d.get("title_source") == TITLE_INFERRED for d in docs)
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
            if settlement_rows:
                session.execute(text("INSERT OR IGNORE INTO document_settlement (document_id, settlement_id) VALUES (:d, :s)"),
                                [{"d": d, "s": x} for d, x in settlement_rows])
            if type_rows:
                session.execute(text("INSERT INTO document2documenttype (id, document_id, document_type_id) VALUES (:id, :d, :t)"),
                                [{"id": str(uuid.uuid4()), "d": d, "t": t} for d, t in type_rows])
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
    """Warnings for inventories whose court cases this CSV would replace. (Derived ToC
    entries are no longer used by default: the model segments those inventories.)"""
    out = []
    without_cases = _court_inventories_without_cases(seg)
    if without_cases:
        out.append(_court_warning(without_cases, csv_path))
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
