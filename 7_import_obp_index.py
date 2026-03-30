#!/usr/bin/env python3
"""
Import GLOBALISE Digitized Indexes of the Dutch East India Company OBP (1602–1799)
from the CSV spreadsheet into the SQLAlchemy database.

This is import script #7 in the sequence. Scripts 1–6 must have been run first so
that Inventory, DocumentType, and Settlement records already exist in the database.

Column mapping
──────────────
Mapped:
  DESCRIPTION                          → Document.title
  INVENTORY NUMBER                     → FK to existing Inventory (by inventory_number)
  YEAR (EARLIEST)                      → Document.date_earliest_begin (Jan 1) /
                                          Document.date_latest_begin  (Dec 31)
  YEAR (LATEST)                        → Document.date_earliest_end  (Jan 1) /
                                          Document.date_latest_end    (Dec 31)
  DOCUMENT TYPE URI (TANAP)            → Document2DocumentType rows (UUIDs extracted
  DOCUMENT TYPE URI (GLOBALISE)          from PoolParty URIs, split on ";")
  ID                                   → ExternalID(context="OBP_INDEX")
  ID (TANAP)                           → ExternalID(context="TANAP")           [nullable]
  ID (DIGITIZED TYPOSCRIPTS)           → ExternalID(context="DIGITIZED TYPOSCRIPTS") [nullable]
  SETTLEMENT                           → Document.location_id via SettlementLabel
                                          lookup (script 6 must have run first);
                                          left NULL when label is not in the table.
  FOLIONUMBER (START OF DOCUMENT)      → Document.folio_start
  FOLIONUMBER (END OF DOCUMENT)        → Document.folio_end

Not mapped (no corresponding schema field):
  SECTION
  DOCUMENT TYPE (TANAP)                (legacy label column — superseded by URI columns)
  FOLIONUMBERS (AS THEY APPEAR IN TYPOSCRIPT)
  YEARS (ALL)
  LOCATION (TANAP)
  GEOGRAPHICAL COVERAGE OF INV. NUMBER

Settlement matching
───────────────────
The SETTLEMENT column value is matched case-insensitively against the label column
of the settlement_label table.  If a match is found the corresponding settlement.id
is stored in document.settlement_id.  Unmatched values are logged as warnings and
the field is left NULL — no document row is skipped because of a missing settlement.
"""

import os
import sys
import uuid
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import (
    Base,
    Document,
    Document2DocumentType,
    Document2ExternalID,
    DocumentIdentificationMethod,
    ExternalID,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METHOD_NAME = "TANAP Digitized Index"
BATCH_SIZE = 5_000

CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "data",
    "globalise_digitized_indexes.csv",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def year_to_start(year) -> Optional[date]:
    """Convert an integer year to Jan 1 of that year."""
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return None
    try:
        return date(int(year), 1, 1)
    except (ValueError, TypeError):
        return None


def year_to_end(year) -> Optional[date]:
    """Convert an integer year to Dec 31 of that year."""
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return None
    try:
        return date(int(year), 12, 31)
    except (ValueError, TypeError):
        return None


def int_or_none(value) -> Optional[str]:
    """
    Return the value as a clean integer string, or None.
    Handles the float representation pandas uses for nullable integer columns
    (e.g. 2.0 → "2").
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return None


def int_field(value) -> Optional[int]:
    """Return a Python int or None; safe against NaN and non-numeric values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


PLACEHOLDER_VALUES = {"#name", "-"}


def is_placeholder(value) -> bool:
    """
    Return True if a cell value is an Excel error or a dash standing in for
    'no value' — specifically '#NAME?' (Excel formula error) and '-'.
    These rows should be skipped entirely during import.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower().rstrip("?") in PLACEHOLDER_VALUES


def parse_type_uris(raw) -> list[str]:
    """
    Split a semicolon-separated list of PoolParty URIs and return the UUID
    extracted from the last path segment of each.

    Leading/trailing whitespace is stripped from every entry.
    Segments that are not valid UUIDs are logged and skipped.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    results = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        segment = part.rstrip("/").rsplit("/", 1)[-1]
        try:
            results.append(str(uuid.UUID(segment)))
        except ValueError:
            logger.warning(f"  Could not extract UUID from document-type URI: {part!r}")
    return results


# ── database helpers ──────────────────────────────────────────────────────────

def get_or_create_method(session: Session) -> str:
    """Return the ID of the TANAP identification method, creating it if needed."""
    existing = (
        session.query(DocumentIdentificationMethod)
        .filter(DocumentIdentificationMethod.name == METHOD_NAME)
        .first()
    )
    if existing:
        logger.info(f"Using existing identification method: {existing.id}")
        return existing.id

    method = DocumentIdentificationMethod(
        id=str(uuid.uuid4()),
        name=METHOD_NAME,
        description=(
            "Documents identified from the GLOBALISE Digitized Indexes of the "
            "Dutch East India Company OBP (1602–1799) spreadsheet. Each row "
            "represents a distinct archival document as catalogued in the TANAP "
            "typoscript indexes."
        ),
        date=datetime.now().date(),
        url="https://datasets.iisg.amsterdam/dataset.xhtml?persistentId=hdl:10622/APNBFT",
    )
    session.add(method)
    session.commit()
    logger.info(f"Created identification method: {method.id}")
    return method.id


def check_already_imported(session: Session, method_id: str) -> int:
    """Return the number of documents already imported with this method."""
    result = session.execute(
        text("SELECT COUNT(*) FROM document WHERE method_id = :mid"),
        {"mid": method_id},
    ).scalar()
    return result or 0


def preload_inventories(session: Session, inventory_numbers: set[str]) -> dict[str, str]:
    """Return {inventory_number: inventory.id} for all requested numbers."""
    result: dict[str, str] = {}
    inv_list = list(inventory_numbers)
    chunk = 900
    for i in range(0, len(inv_list), chunk):
        subset = inv_list[i : i + chunk]
        placeholders = ",".join([f":p{j}" for j in range(len(subset))])
        params = {f"p{j}": v for j, v in enumerate(subset)}
        rows = session.execute(
            text(
                f"SELECT inventory_number, id FROM inventory "
                f"WHERE inventory_number IN ({placeholders})"
            ),
            params,
        ).all()
        for inv_num, inv_id in rows:
            result[inv_num] = inv_id
    return result


def preload_document_type_ids(session: Session) -> set[str]:
    """Return the set of all document_type UUIDs present in the database."""
    rows = session.execute(text("SELECT id FROM document_type")).all()
    return {row[0] for row in rows}


def preload_settlement_labels(session: Session) -> dict[str, str]:
    """
    Return a case-insensitive label → settlement_id mapping.

    Built from the settlement_label table populated by script 6.
    When a label appears multiple times (shouldn't happen in practice)
    the last encountered settlement_id wins — the import log will show a warning.
    """
    rows = session.execute(
        text("SELECT sl.label, sl.settlement_id FROM settlement_label sl")
    ).all()

    mapping: dict[str, str] = {}
    for label, settlement_id in rows:
        key = label.strip().lower()
        if key in mapping and mapping[key] != settlement_id:
            logger.warning(
                f"Duplicate settlement label '{label}' maps to multiple settlements; "
                f"using settlement_id={settlement_id}."
            )
        mapping[key] = settlement_id

    logger.info(f"Preloaded {len(mapping)} settlement labels.")
    return mapping


# ── core import ───────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV file not found: {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df = df.where(pd.notnull(df), None)
    logger.info(f"Loaded {len(df)} rows from CSV ({os.path.basename(CSV_PATH)})")
    return df


def bulk_insert(session: Session, table, rows: list[dict], label: str) -> int:
    """Insert rows in batches; returns total inserted."""
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        if batch:
            session.execute(table.insert(), batch)
            inserted += len(batch)
    session.commit()
    logger.info(f"Inserted {inserted:,} {label}")
    return inserted


def main():
    df = load_csv()
    session = Session(engine)

    try:
        method_id = get_or_create_method(session)

        # Guard against double-importing
        already = check_already_imported(session, method_id)
        if already > 0:
            logger.warning(
                f"{already:,} documents with method '{METHOD_NAME}' already exist. "
                "Aborting to avoid duplicates. Drop those rows first if you want to re-import."
            )
            return

        # Preload inventories keyed by string inventory number
        inv_numbers = {str(int(v)) for v in df["INVENTORY NUMBER"].dropna().unique()}
        logger.info(f"Preloading {len(inv_numbers):,} inventories...")
        inventories = preload_inventories(session, inv_numbers)

        # Preload all known document-type UUIDs for FK validation
        known_type_ids = preload_document_type_ids(session)
        logger.info(f"Preloaded {len(known_type_ids):,} document types.")

        # Preload settlement labels (script 6 must have run first)
        settlement_labels = preload_settlement_labels(session)
        if not settlement_labels:
            logger.warning(
                "No settlement labels found in the database. "
                "Run script 6 (6_import_settlements.py) first to populate them. "
                "Continuing import — all document.settlement_id fields will be NULL."
            )

        missing_inventories: set[str] = set()
        unknown_type_ids: set[str] = set()
        unmatched_settlements: set[str] = set()
        matched_settlement_count = 0

        doc_rows: list[dict] = []
        doc_type_rows: list[dict] = []
        ext_id_rows: list[dict] = []
        doc_ext_id_rows: list[dict] = []

        for _, row in df.iterrows():
            # Skip rows where the document type is a placeholder value
            # ('#NAME?' is an Excel formula error; '-' means no value)
            tanap_raw = row.get("DOCUMENT TYPE URI (TANAP)")
            globalise_raw = row.get("DOCUMENT TYPE URI (GLOBALISE)")
            if is_placeholder(tanap_raw) or is_placeholder(globalise_raw):
                continue

            inv_number = str(int(row["INVENTORY NUMBER"]))
            inv_id = inventories.get(inv_number)
            if not inv_id:
                missing_inventories.add(inv_number)
                continue

            # ── Settlement lookup ──────────────────────────────────────────────
            settlement_id: Optional[str] = None
            raw_settlement = row.get("SETTLEMENT")
            if raw_settlement is not None and not (
                isinstance(raw_settlement, float) and pd.isna(raw_settlement)
            ):
                settlement_label = str(raw_settlement).strip()
                settlement_id = settlement_labels.get(settlement_label.lower())
                if settlement_id:
                    matched_settlement_count += 1
                else:
                    unmatched_settlements.add(settlement_label)

            doc_id = str(uuid.uuid4())

            doc_rows.append(
                {
                    "id": doc_id,
                    "inventory_id": inv_id,
                    "title": row.get("DESCRIPTION"),
                    "date_earliest_begin": year_to_start(row.get("YEAR (EARLIEST)")),
                    "date_latest_begin": year_to_end(row.get("YEAR (EARLIEST)")),
                    "date_earliest_end": year_to_start(row.get("YEAR (LATEST)")),
                    "date_latest_end": year_to_end(row.get("YEAR (LATEST)")),
                    "date_text": None,
                    "part_of_id": None,
                    "location_id": settlement_id,     # NULL when not found in settlement_label
                    "folio_start": int_field(row.get("FOLIONUMBER (START OF DOCUMENT)")),
                    "folio_end": int_field(row.get("FOLIONUMBER (END OF DOCUMENT)")),
                    "method_id": method_id,
                }
            )

            # Document types — collect UUIDs from both URI columns, deduplicate per doc
            type_uuids_for_doc: set[str] = set()
            for col in ("DOCUMENT TYPE URI (TANAP)", "DOCUMENT TYPE URI (GLOBALISE)"):
                for type_uuid in parse_type_uris(row.get(col)):
                    if type_uuid not in known_type_ids:
                        unknown_type_ids.add(type_uuid)
                        continue
                    type_uuids_for_doc.add(type_uuid)

            for type_uuid in type_uuids_for_doc:
                doc_type_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "document_id": doc_id,
                        "document_type_id": type_uuid,
                    }
                )

            # External IDs — one per non-null identifier column
            for context, raw_value in (
                ("OBP_INDEX", row.get("ID")),
                ("TANAP", row.get("ID (TANAP)")),
                ("DIGITIZED TYPOSCRIPTS", row.get("ID (DIGITIZED TYPOSCRIPTS)")),
            ):
                identifier = int_or_none(raw_value)
                if identifier is None:
                    continue
                ext_id = str(uuid.uuid4())
                ext_id_rows.append(
                    {
                        "id": ext_id,
                        "identifier": identifier,
                        "context": context,
                        "URL": None,
                    }
                )
                doc_ext_id_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "document_id": doc_id,
                        "external_id": ext_id,
                    }
                )

        # ── Warnings ──────────────────────────────────────────────────────────
        if missing_inventories:
            logger.warning(
                f"Skipped {len(missing_inventories)} row(s) — "
                f"inventory numbers not found in DB: {sorted(missing_inventories)}"
            )
        if unknown_type_ids:
            logger.warning(
                f"Skipped {len(unknown_type_ids)} document-type UUID(s) not found in "
                f"document_type table (run script 5 first?): {sorted(unknown_type_ids)}"
            )
        if unmatched_settlements:
            logger.warning(
                f"{len(unmatched_settlements)} SETTLEMENT value(s) from the CSV had no "
                f"matching label in the settlement_label table — document.settlement_id "
                f"left NULL for those rows. Unmatched values: "
                f"{sorted(unmatched_settlements)}"
            )

        logger.info(
            f"Prepared {len(doc_rows):,} documents "
            f"({matched_settlement_count:,} with settlement, "
            f"{len(unmatched_settlements):,} unmatched), "
            f"{len(doc_type_rows):,} document-type links, "
            f"{len(ext_id_rows):,} external IDs"
        )

        bulk_insert(session, Document.__table__, doc_rows, "documents")
        bulk_insert(session, Document2DocumentType.__table__, doc_type_rows, "document-type links")
        bulk_insert(session, ExternalID.__table__, ext_id_rows, "external IDs")
        bulk_insert(
            session,
            Document2ExternalID.__table__,
            doc_ext_id_rows,
            "document ↔ external ID links",
        )

        logger.info("OBP index import completed successfully.")

    except Exception as e:
        logger.exception(f"Error during import: {e}")
        session.rollback()
        raise
    finally:
        session.close()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("GLOBALISE OBP Index Import  (script 7 of 7)")
    print("=" * 60)
    print(f"Source : {os.path.basename(CSV_PATH)}")
    print(f"DB     : {DATABASE_URL}")
    print(
        "\nThis script requires scripts 1–6 to have been run first "
        "(inventories, pages, hierarchy, baseline documents, document types, settlements)."
    )
    print("=" * 60)

    response = input("\nProceed with import? (yes/no): ")
    if response.lower() != "yes":
        print("Import cancelled.")
        sys.exit(0)

    main()
