"""
Import script for overview_general_missives.csv into the GLOBALISE database.

Each CSV row maps to one Document, with:
  - ExternalID          via Document2ExternalID  (context="OBP_INDEX")
  - Inventory           looked up by inventory_number
  - Document.title      from "Beschrijving in TANAP"
  - Document.date_text  from "Datum"
  - date_earliest_begin / date_latest_begin from "Datum (numeriek)"
    Single date  → both fields get the same value
    Range X/Y    → earliest_begin=X, latest_begin=Y
  - Page2Document links to Pages that belong to the first- and last-scan
    Scans are looked up by filename ("Bestandsnaam van eerste/laatste scan")



"""

import argparse
import logging
import sys
import uuid
import os
from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Adjust this import path if models.py lives elsewhere.
# ---------------------------------------------------------------------------
from models import (
    Base,
    Document,
    Document2ExternalID,
    DocumentIdentificationMethod,
    ExternalID,
    Inventory,
    Page,
    Page2Document,
    Scan,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METHOD_NAME = "General Missives Ground Truth"
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "data",
    "overview_general_missives.csv",
)

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(value: str) -> Optional[date]:
    """Parse an ISO date string (YYYY-MM-DD) into a date object."""
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def split_date_range(
    raw: Optional[str],
) -> Tuple[Optional[date], Optional[date]]:
    """
    Parse "Datum (numeriek)" into (earliest_begin, latest_begin).

    Single date  "1618-10-05"           → (1618-10-05, 1618-10-05)
    Range        "1619-10-07/1619-10-15" → (1619-10-07, 1619-10-15)
    Missing                              → (None, None)
    """
    if not raw or pd.isna(raw):
        return None, None
    raw = str(raw).strip()
    if "/" in raw:
        parts = raw.split("/", 1)
        return parse_date(parts[0]), parse_date(parts[1])
    d = parse_date(raw)
    return d, d


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_or_create_method(session: Session) -> str:
    """Return the ID of the TANAP identification method, creating it if needed."""
    existing = (
        session.query(DocumentIdentificationMethod)
        .filter(DocumentIdentificationMethod.name == METHOD_NAME)
        .first()
    )
    if existing:
        logger.info(f"Using existing identification method: {existing.id}")
        return existing

    method = DocumentIdentificationMethod(
        id=str(uuid.uuid4()),
        name=METHOD_NAME,
        description=(
            "Document boundaries for General Missives identified by Kay Pepping."
        ),
        date=datetime.now().date(),
        url="https://doi.org/10.34894/SRDMFU",
    )
    session.add(method)
    session.commit()
    logger.info(f"Created identification method: {method.id}")
    return method

def lookup_inventory(session: Session, inv_number: str) -> Optional[Inventory]:
    stmt = select(Inventory).where(Inventory.inventory_number == str(inv_number))
    return session.scalars(stmt).first()


def lookup_scan_by_filename(session: Session, filename: str) -> Optional[Scan]:
    """
    Look up a Scan by filename.  The CSV sometimes omits the leading zero on
    the scan number (e.g. "_617" vs "_0617"), so we try the raw value first
    and fall back to a suffix-normalised search.
    """
    stmt = select(Scan).where(Scan.filename == filename)
    scan = session.scalars(stmt).first()
    if scan:
        return scan

    # Normalise: strip trailing underscored number and repad to 4 digits
    # e.g. "NL-HaNA_1.04.02_1068_617" → "NL-HaNA_1.04.02_1068_0617"
    parts = filename.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        normalised = f"{parts[0]}_{int(parts[1]):04d}"
        stmt2 = select(Scan).where(Scan.filename == normalised)
        scan = session.scalars(stmt2).first()
        if scan:
            return scan

    return None


def get_pages_for_scan(session: Session, scan: Scan) -> list[Page]:
    """Return all Pages that belong to this scan."""
    stmt = select(Page).where(Page.scan_id == scan.id)
    return list(session.scalars(stmt).all())


def create_or_get_external_id(
    session: Session, identifier: str, context: str
) -> ExternalID:
    """Find or create an ExternalID with the given identifier and context."""
    stmt = select(ExternalID).where(
        ExternalID.identifier == identifier,
        ExternalID.context == context,
    )
    ext = session.scalars(stmt).first()
    if not ext:
        ext = ExternalID(
            id=str(uuid.uuid4()),
            identifier=identifier,
            context=context,
        )
        session.add(ext)
    return ext


# ---------------------------------------------------------------------------
# Row importer
# ---------------------------------------------------------------------------

def import_row(
    session: Session,
    row: pd.Series,
    method: DocumentIdentificationMethod,
    *,
    dry_run: bool,
    stats: dict,
) -> None:
    row_id = int(row["ID"])

    # ---- Inventory -------------------------------------------------------
    inv_number = row["Inv.nr. Nationaal Archief (1.04.02)"]
    if pd.isna(inv_number):
        logger.warning("Row %d: Has no Inventory — skipping.", row_id)
        stats["skipped_no_inventory"] += 1
        return

    inv_number = str(int(inv_number)).strip()
    inventory = lookup_inventory(session, inv_number)
    if inventory is None:
        logger.warning("Row %d: Inventory %r not found — skipping.", row_id, inv_number)
        stats["skipped_no_inventory"] += 1
        return

    # ---- Dates -----------------------------------------------------------
    earliest_begin, latest_begin = split_date_range(row.get("Datum (numeriek)"))
    earliest_end, latest_end = split_date_range(row.get("Datum (numeriek)"))
    date_text = row.get("Datum") if pd.notna(row.get("Datum")) else None

    # ---- Document --------------------------------------------------------
    title = row.get("Beschrijving in TANAP")
    title = str(title).strip() if pd.notna(title) else None

    document = Document(
        id=str(uuid.uuid4()),
        inventory_id=inventory.id,
        title=title,
        date_text=date_text,
        date_earliest_begin=earliest_begin,
        date_latest_begin=latest_begin,
        date_earliest_end=earliest_end,
        date_latest_end=latest_end,
        method_id=method.id,
    )
    session.add(document)

    # ---- ExternalID (TANAP) ----------------------------------------------
    tanap_id_raw = row.get("ID in TANAP database")
    if pd.notna(tanap_id_raw):
        tanap_id = str(int(tanap_id_raw))
        ext_id = create_or_get_external_id(session, tanap_id, "OBP_INDEX")
        link = Document2ExternalID(
            id=str(uuid.uuid4()),
            document_id=document.id,
            external_id=ext_id.id,
        )
        session.add(link)

    # ---- Scan / Page links -----------------------------------------------
    first_scan_name = row.get("Bestandsnaam van eerste scan")
    begin_scan_raw = row.get("Beginscan")
    end_scan_raw = row.get("Eindscan")
 
    if pd.isna(first_scan_name) or pd.isna(begin_scan_raw) or pd.isna(end_scan_raw):
        logger.warning("Row %d: Missing scan range columns — skipping scan links.", row_id)
        stats["imported"] += 1
        return
 
    first_scan_name = str(first_scan_name).strip()
    begin_scan = int(begin_scan_raw)
    end_scan = int(end_scan_raw)
 
    # Derive the filename prefix from the first scan name.
    # Pattern: "<prefix>_<zero-padded-number>"
    # e.g. "NL-HaNA_1.04.02_1068_0611" -> prefix "NL-HaNA_1.04.02_1068", start 611
    name_parts = first_scan_name.rsplit("_", 1)
    if len(name_parts) != 2 or not name_parts[1].isdigit():
        logger.warning(
            "Row %d: Cannot parse scan filename prefix from %r — skipping scan links.",
            row_id,
            first_scan_name,
        )
        stats["imported"] += 1
        return
 
    scan_prefix = name_parts[0]
    next_index = 0
 
    for scan_number in range(begin_scan, end_scan + 1):
        filename = f"{scan_prefix}_{scan_number:04d}"
        scan = lookup_scan_by_filename(session, filename)
        if scan is None:
            logger.warning(
                "Row %d: Scan %r not found in database — skipping.",
                row_id,
                filename,
            )
            stats["missing_scans"] += 1
            continue
 
        pages = get_pages_for_scan(session, scan)
        if not pages:
            logger.warning(
                "Row %d: Scan %r has no pages — skipping.",
                row_id,
                filename,
            )
            stats["scans_without_pages"] += 1
            continue
 
        for page in pages:
            p2d = Page2Document(
                id=str(uuid.uuid4()),
                page_id=page.id,
                document_id=document.id,
                index=next_index,
            )
            session.add(p2d)
            next_index += 1
 
    stats["imported"] += 1
    logger.debug("Row %d: Document %s imported.", row_id, document.id)

    # # ---- Scan / Page links -----------------------------------------------
    # first_scan_name = row.get("Bestandsnaam van eerste scan")
    # last_scan_name = row.get("Bestandsnaam van laatste scan")

    # linked_page_ids: set[str] = set()
    # next_index = 0

    # def link_scan(filename: str) -> None:
    #     nonlocal next_index
    #     if not filename or pd.isna(filename):
    #         return
    #     filename = str(filename).strip()
    #     scan = lookup_scan_by_filename(session, filename)
    #     if scan is None:
    #         logger.warning(
    #             "Row %d: Scan %r not found in database — skipping scan link.",
    #             row_id,
    #             filename,
    #         )
    #         stats["missing_scans"] += 1
    #         return

    #     pages = get_pages_for_scan(session, scan)
    #     if not pages:
    #         logger.warning(
    #             "Row %d: Scan %r has no pages — skipping scan link.",
    #             row_id,
    #             filename,
    #         )
    #         stats["scans_without_pages"] += 1
    #         return

    #     for page in pages:
    #         if page.id in linked_page_ids:
    #             continue  # first == last scan: don't double-link
    #         linked_page_ids.add(page.id)
    #         p2d = Page2Document(
    #             id=str(uuid.uuid4()),
    #             page_id=page.id,
    #             document_id=document.id,
    #             index=next_index,
    #         )
    #         session.add(p2d)
    #         next_index += 1

    # link_scan(first_scan_name)
    # if pd.notna(last_scan_name) and last_scan_name != first_scan_name:
    #     link_scan(last_scan_name)

    # stats["imported"] += 1
    # logger.debug("Row %d: Document %s imported.", row_id, document.id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Import general missives CSV into GLOBALISE database.")
    # parser.add_argument("--csv", required=True, help="Path to overview_general_missives.csv")
    # parser.add_argument("--db", required=True, help="SQLAlchemy database URL, e.g. sqlite:///globalise.db")
    # parser.add_argument("--method-id", default=None, help="UUID of an existing DocumentIdentificationMethod")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without writing to the database")
    parser.add_argument("--batch-size", type=int, default=100, help="Flush session every N rows (default 100)")
    args = parser.parse_args()


    # ---- Load CSV --------------------------------------------------------
    logger.info("Loading CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH, dtype=str)  # read everything as str; we cast as needed
    logger.info("Rows to import: %d", len(df))

    # ---- Database --------------------------------------------------------
    engine = create_engine(DATABASE_URL, echo=False)
    # Uncomment the next line if you need to create tables from scratch:
    # Base.metadata.create_all(engine)

    stats = {
        "imported": 0,
        "skipped_no_inventory": 0,
        "missing_scans": 0,
        "scans_without_pages": 0,
    }

    with Session(engine) as session:
        method = get_or_create_method(session)

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            import_row(session, row, method, dry_run=args.dry_run, stats=stats)

            if not args.dry_run and i % args.batch_size == 0:
                session.flush()
                logger.info("  … flushed after %d rows", i)

        if args.dry_run:
            logger.info("Dry run — rolling back.")
            session.rollback()
        else:
            session.commit()
            logger.info("Committed.")

    logger.info(
        "Done. imported=%d  skipped_no_inventory=%d  missing_scans=%d  scans_without_pages=%d",
        stats["imported"],
        stats["skipped_no_inventory"],
        stats["missing_scans"],
        stats["scans_without_pages"],
    )


if __name__ == "__main__":
    main()
