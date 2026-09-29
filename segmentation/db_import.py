"""
Store segmentation output (the CSV written by `python -m segmentation run`)
as documents of the identification method "Segmentation model".

Per segment (non-document runs are skipped):
  - a Document in the segment's inventory; ToC segments copy title, dates,
    page range and settlement from their ToC entry and link to the same
    OBP_INDEX external id; subdocuments (and nested ToC entries) get
    part_of_id = the document of their parent ToC entry
  - Page2Document rows for every page of the segment's scans, in scan order
    (verso before recto on double scans), source SEGMENTATION_MODEL,
    confidence CANDIDATE

Re-importing an inventory first deletes that method's documents there.
"""

import datetime
import logging
import uuid

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import LinkConfidence

logger = logging.getLogger("segmentation")

METHOD_NAME = "Segmentation model"
METHOD_DESCRIPTION = (
    "Documents, subdocuments and unindexed documents found by the segmentation "
    "model (segmentation/ in this repository): per-scan start/end/non-document "
    "scores from all predictors, aligned with the ToC where there is one."
)
SOURCE = "SEGMENTATION_MODEL"


def _method_id(session: Session) -> str:
    row = session.execute(
        text("SELECT id FROM document_identification_method WHERE name = :n"), {"n": METHOD_NAME}
    ).first()
    if row:
        return row[0]
    mid = str(uuid.uuid4())
    session.execute(
        text("INSERT INTO document_identification_method (id, name, description, date) VALUES (:id, :n, :d, :dt)"),
        {"id": mid, "n": METHOD_NAME, "d": METHOD_DESCRIPTION, "dt": datetime.date.today()},
    )
    return mid


def _delete_previous(session: Session, method_id: str, inventory_id: str) -> int:
    ids = [r[0] for r in session.execute(
        text("SELECT id FROM document WHERE method_id = :m AND inventory_id = :i"), {"m": method_id, "i": inventory_id}
    )]
    for k in range(0, len(ids), 500):
        chunk = {f"p{j}": v for j, v in enumerate(ids[k : k + 500])}
        ph = ",".join(f":p{j}" for j in range(len(chunk)))
        for table in ("page2document", "document2external_id", "document2documenttype"):
            session.execute(text(f"DELETE FROM {table} WHERE document_id IN ({ph})"), chunk)
        session.execute(text(f"UPDATE document SET part_of_id = NULL WHERE id IN ({ph})"), chunk)
        session.execute(text(f"DELETE FROM document WHERE id IN ({ph})"), chunk)
    return len(ids)


def import_segments(csv_path: str, database_url: str, dry_run: bool = False):
    seg = pd.read_csv(csv_path, dtype={"inventory": str, "toc_csv_ids": str})
    seg = seg[seg["kind"] != "non-document"]
    engine = create_engine(database_url)
    with Session(engine) as session:
        method_id = _method_id(session)
        totals = {"documents": 0, "page links": 0, "replaced": 0}
        for inv_number, rows in seg.groupby("inventory", sort=False):
            inv_id = session.execute(text("SELECT id FROM inventory WHERE inventory_number = :n"), {"n": inv_number}).scalar()
            if inv_id is None:
                logger.warning("Inventory %s not found; skipped", inv_number)
                continue
            conn = session.connection()
            scans = pd.read_sql(
                text("SELECT id AS scan_id, filename FROM scan WHERE inventory_id = :i ORDER BY scan_order, filename"),
                conn, params={"i": inv_id},
            )
            pos = {f: i for i, f in enumerate(scans["filename"])}
            pages = pd.read_sql(
                text("SELECT id AS page_id, scan_id, recto_verso FROM page WHERE inventory_id = :i"), conn, params={"i": inv_id}
            )
            pages["side"] = pages["recto_verso"].map({"Verso": 0, "Recto": 1}).fillna(0)
            pages = pages.sort_values(["scan_id", "side"])
            pages_by_scan = pages.groupby("scan_id")["page_id"].apply(list).to_dict()
            toc = pd.read_sql(
                text("SELECT CAST(e.identifier AS INTEGER) AS csv_id, e.id AS ext_id, d.title, d.folio_start, d.folio_end, "
                "       d.date_earliest_begin, d.date_latest_begin, d.date_earliest_end, d.date_latest_end, d.location_id "
                "FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' "
                "JOIN document2external_id de ON de.document_id = d.id JOIN external_id e ON e.id = de.external_id AND e.context = 'OBP_INDEX' "
                "WHERE d.inventory_id = :i"),
                conn, params={"i": inv_id},
            ).drop_duplicates("csv_id").set_index("csv_id")

            if not dry_run:
                totals["replaced"] += _delete_previous(session, method_id, inv_id)
            doc_rows, link_rows, ext_rows, parent_of = [], [], [], {}
            doc_by_csv: dict[int, str] = {}
            for r in rows.itertuples():
                doc_id = str(uuid.uuid4())
                csv_ids = [int(x) for x in str(r.toc_csv_ids).split(";") if x and x != "nan"]
                entry = toc.loc[csv_ids[0]] if csv_ids and csv_ids[0] in toc.index else None
                doc = {
                    "id": doc_id, "inventory_id": inv_id, "method_id": method_id, "part_of_id": None,
                    "title": entry["title"] if entry is not None else None,
                    "folio_start": None, "folio_end": None, "date_earliest_begin": None, "date_latest_begin": None,
                    "date_earliest_end": None, "date_latest_end": None, "location_id": None, "date_text": None,
                }
                if entry is not None:
                    for c in ("folio_start", "folio_end", "date_earliest_begin", "date_latest_begin", "date_earliest_end", "date_latest_end", "location_id"):
                        doc[c] = None if pd.isna(entry[c]) else (int(entry[c]) if c.startswith("folio") else entry[c])
                doc_rows.append(doc)
                for c in csv_ids:
                    doc_by_csv[c] = doc_id
                    if c in toc.index:
                        ext_rows.append({"id": str(uuid.uuid4()), "document_id": doc_id, "external_id": toc.at[c, "ext_id"]})
                if not pd.isna(r.parent_csv_id):
                    parent_of[doc_id] = int(r.parent_csv_id)
                idx = 0
                for p in range(pos[r.start_scan], pos[r.end_scan] + 1):
                    for page_id in pages_by_scan.get(scans.at[p, "scan_id"], []):
                        link_rows.append({"id": str(uuid.uuid4()), "page_id": page_id, "document_id": doc_id, "index": idx,
                                          "source": SOURCE, "confidence": LinkConfidence.CANDIDATE.value})
                        idx += 1
            for d in doc_rows:
                if d["id"] in parent_of:
                    d["part_of_id"] = doc_by_csv.get(parent_of[d["id"]])
            totals["documents"] += len(doc_rows)
            totals["page links"] += len(link_rows)
            logger.info("%s: %d documents, %d page links", inv_number, len(doc_rows), len(link_rows))
            if dry_run:
                continue
            cols = list(doc_rows[0].keys()) if doc_rows else []
            if doc_rows:
                session.execute(text(f"INSERT INTO document ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"), doc_rows)
            if ext_rows:
                session.execute(text("INSERT INTO document2external_id (id, document_id, external_id) VALUES (:id, :document_id, :external_id)"), ext_rows)
            if link_rows:
                session.execute(text('INSERT INTO page2document (id, page_id, document_id, "index", source, confidence) '
                                     "VALUES (:id, :page_id, :document_id, :index, :source, :confidence)"), link_rows)
        if dry_run:
            session.rollback()
            logger.info("Dry run — nothing written. Would create %s", totals)
        else:
            session.commit()
            logger.info("Imported %s", totals)
