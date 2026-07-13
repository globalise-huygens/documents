"""
Import script for a folder of "Document Segmentation" CSVs into the GLOBALISE
database (one CSV per inventory, e.g. "8697_-_Document_Segmentation.csv").

Each CSV has one row per scan-level boundary event, with columns:

  - "Scan File_Name"              e.g. NL-HaNA_1.04.02_8697_0001
  - "TANAP Boundaries"            "START" / "END" / "SAME AS <filename>" / blank
  - "TANAP ID"                    numeric TANAP document id (filled for every
                                   scan that belongs to a TANAP document)
  - "Subdocument boundaries"      "/"-separated sequence of START/END tokens
                                   (subdocuments have no id of their own — they
                                   are purely defined by boundary events)
  - "Type of non-document page"  Cover / Empty / Section title page / Table of
                                   contents / ... — marks a scan that is not
                                   part of any document

This script creates two kinds of Document rows per file:

  1. "TANAP documents" — one per distinct TANAP ID, linked to an ExternalID
     (context=TANAP_ID_CONTEXT) so re-imports can find them again.
  2. "Subdocuments" — one per START/END span in "Subdocument boundaries".
     These have no natural identifier, so a synthetic one is generated
     (f"{inventory_number}-SUBDOC-{n:04d}") and stored the same way. Each
     subdocument is linked to its enclosing TANAP document via
     Document.part_of_id, when one is open at the time the subdocument starts.

Both are linked to their scans' Pages via Page2Document, with
confidence=LinkConfidence.DEFINITIVE and source="SEGMENTATION".

A scan filename can legitimately appear on more than one row (e.g. one
document ends and another begins on the very same physical page — think of
a letter that ends halfway down a page and a new one starting right below
it). Rather than trying to split a scan's pages across those rows, every row
that references a scan links ALL of that scan's pages to the relevant
document(s) — Page2Document is a many-to-many junction, so a page can
legitimately belong to more than one document.

"SAME AS <filename>" rows mean "this scan is a duplicate/reproduction of
<filename> and belongs to whichever TANAP document(s) that scan belongs to".
Because the referenced filename can appear earlier OR later in the file, these
rows are resolved in a second pass after the whole file has been read once.

Run against a folder full of these CSVs with:

    python import_segmentation.py --input-dir data/segmentation

See --help for the rest of the options.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import uuid
from collections import defaultdict
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Adjust this import path if models.py lives elsewhere.
# ---------------------------------------------------------------------------
from models import (
    Document,
    Document2ExternalID,
    DocumentIdentificationMethod,
    ExternalID,
    Inventory,
    LinkConfidence,
    Page,
    Page2Document,
    Scan,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

# Identification methods created/looked up for the two kinds of Documents
# this script produces.
TANAP_METHOD_NAME = "Manual Validation (document)"
TANAP_METHOD_DESC = (
    "TANAP document boundaries identified in the per-inventory document "
    "segmentation ground truth."
)
SUBDOC_METHOD_NAME = "Manual Validation (subdocument)"
SUBDOC_METHOD_DESC = (
    "Subdocument boundaries identified in the per-inventory document "
    "segmentation ground truth. Subdocuments have no external identifier of "
    "their own; a synthetic one is generated from the inventory number and "
    "a running counter."
)

# ExternalID.context values used for the two kinds of documents.
# NOTE: "OBP_INDEX" is the same context 8_import_GM.py uses for its "ID in
# TANAP database" column — these are the same identifier namespace, so using
# the same context here means a TANAP document already created by the GM
# import (or a previous run of this script) gets reused instead of
# duplicated, regardless of which script created it first.
TANAP_ID_CONTEXT = "OBP_INDEX"
SUBDOC_ID_CONTEXT = "SEGMENTATION_SUBDOC"

P2D_SOURCE = "SEGMENTATION"

# Confidence tiers, strongest to weakest (see LinkConfidence docstring in
# models.py: VALIDATED > DEFINITIVE > FOLIO_RANGE > INTERPOLATED > CANDIDATE).
# This script writes DEFINITIVE-confidence links. Segmentation imports can
# happen at any point in the pipeline — before or after scripts like
# 10_match_folios.py or 12_interpolate_documents.py have run — so rather than
# having every other script defensively avoid stepping on manually validated
# data, THIS import is the one responsible for cleaning up anything weaker
# that conflicts with what it now knows to be ground truth. See
# supersede_weaker_links() below.
_CONFIDENCE_RANK: dict[str, int] = {c.value: i for i, c in enumerate(LinkConfidence)}
_DEFINITIVE_RANK = _CONFIDENCE_RANK[LinkConfidence.DEFINITIVE.value]

# Matches "SAME AS <filename>"
SAME_AS_RE = re.compile(r"^SAME AS\s+(.+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = {
    "Scan File_Name",
    "TANAP Boundaries",
    "TANAP ID",
    "Subdocument boundaries",
    "Type of non-document page",
}


def read_segmentation_csv(path: str) -> pd.DataFrame:
    """
    Read one segmentation CSV, tolerant of the BOM Excel likes to add and of
    the delimiter being either ';' or ',' (some files export as one, some as
    the other). Whichever delimiter actually produces the expected columns
    wins; if neither does, raise with both attempts' columns for debugging.
    """
    attempts = {}
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
        except Exception as exc:  # malformed for this delimiter — try the other
            attempts[sep] = f"<failed to parse: {exc}>"
            continue
        df.columns = [c.strip() for c in df.columns]
        attempts[sep] = list(df.columns)
        if REQUIRED_COLUMNS <= set(df.columns):
            return df

    raise ValueError(
        f"{path}: could not find expected columns with ';' or ',' as the "
        f"delimiter. Columns found — ';': {attempts.get(';')} | ',': {attempts.get(',')}"
    )


def inventory_number_from_scan_filename(filename: str) -> Optional[str]:
    """
    "NL-HaNA_1.04.02_8697_0001" -> "8697"

    Scan filenames are "<archive>_<series>_<inventory>_<scan-number>".
    """
    parts = filename.strip().split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        return None
    return parts[-2]


def detect_inventory_number(path: str, df: pd.DataFrame) -> Optional[str]:
    """
    Prefer deriving the inventory number from the scan filenames themselves
    (most reliable); fall back to the leading digits of the CSV filename;
    warn if the two disagree.
    """
    from_filename = None
    m = re.match(r"^(\d+)", os.path.basename(path))
    if m:
        from_filename = m.group(1)

    from_data = None
    for name in df["Scan File_Name"].dropna():
        from_data = inventory_number_from_scan_filename(name)
        if from_data:
            break

    if from_data and from_filename and from_data != from_filename:
        logger.warning(
            "%s: inventory number from data (%s) != from filename (%s); using %s",
            path,
            from_data,
            from_filename,
            from_data,
        )
    return from_data or from_filename


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_or_create_method(
    session: Session, name: str, description: str
) -> DocumentIdentificationMethod:
    existing = (
        session.query(DocumentIdentificationMethod)
        .filter(DocumentIdentificationMethod.name == name)
        .first()
    )
    if existing:
        return existing
    method = DocumentIdentificationMethod(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
    )
    session.add(method)
    session.flush()
    logger.info("Created identification method %r: %s", name, method.id)
    return method


def lookup_inventory(session: Session, inv_number: str) -> Optional[Inventory]:
    stmt = select(Inventory).where(Inventory.inventory_number == str(inv_number))
    return session.scalars(stmt).first()


def lookup_scan_by_filename(session: Session, filename: str) -> Optional[Scan]:
    """
    Look up a Scan by filename, falling back to a zero-padding-normalised
    search (the CSVs are consistently zero-padded, but this keeps the script
    robust to files that aren't).
    """
    stmt = select(Scan).where(Scan.filename == filename)
    scan = session.scalars(stmt).first()
    if scan:
        return scan

    parts = filename.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        normalised = f"{parts[0]}_{int(parts[1]):04d}"
        if normalised != filename:
            stmt2 = select(Scan).where(Scan.filename == normalised)
            scan = session.scalars(stmt2).first()
    return scan


def get_pages_for_scan(session: Session, scan: Scan) -> list[Page]:
    """All Pages for a scan, in a stable order (Recto before Verso)."""
    stmt = select(Page).where(Page.scan_id == scan.id)
    pages = list(session.scalars(stmt).all())
    pages.sort(key=lambda p: (0 if p.recto_verso and p.recto_verso.value == "Recto" else 1, p.id))
    return pages


def create_or_get_external_id(
    session: Session, identifier: str, context: str
) -> ExternalID:
    stmt = select(ExternalID).where(
        ExternalID.identifier == identifier,
        ExternalID.context == context,
    )
    ext = session.scalars(stmt).first()
    if not ext:
        ext = ExternalID(id=str(uuid.uuid4()), identifier=identifier, context=context)
        session.add(ext)
        session.flush()
    return ext


def already_imported(session: Session, inventory: Inventory) -> bool:
    """
    Has this script already linked pages for this inventory? Checked by the
    presence of any Page2Document row with source=P2D_SOURCE, rather than by
    DocumentIdentificationMethod, because a TANAP document reused from a
    prior GM import keeps GM's method_id, not this script's.
    """
    return (
        session.query(Page2Document)
        .join(Document, Document.id == Page2Document.document_id)
        .filter(
            Document.inventory_id == inventory.id,
            Page2Document.source == P2D_SOURCE,
        )
        .first()
        is not None
    )


def _confidence_rank(value) -> int:
    value = getattr(value, "value", value)  # unwrap LinkConfidence -> str if needed
    return _CONFIDENCE_RANK.get(value, len(_CONFIDENCE_RANK))


def supersede_weaker_links(
    session: Session,
    stats: dict,
    *,
    page_id: Optional[str] = None,
    document_id: Optional[str] = None,
) -> None:
    """
    Remove existing Page2Document rows at DEFINITIVE confidence or weaker
    (DEFINITIVE, FOLIO_RANGE, INTERPOLATED, CANDIDATE) for the given page
    and/or document. VALIDATED rows — an explicit human sign-off, stronger
    than what a dataset import establishes — are never touched.

    This is how manually validated / segmentation-derived material overrides
    earlier, less precise linking: since a segmentation import can happen at
    any point in the pipeline (not necessarily before scripts like
    10_match_folios.py or 12_interpolate_documents.py), it's this import's
    job to clean up whatever weaker data is already there when it runs,
    rather than have every other script defensively work around it.

    Called at two scopes:
      - document_id: the first time this run touches a document, clearing
        out anything previously attached to that exact document (e.g. a
        coarser prior import, or an earlier run of this script).
      - page_id: every page this run links, clearing out weaker guesses that
        pointed that page at some *other* document (e.g. a folio-range or
        interpolated match that turned out to be wrong).
    """
    if page_id is None and document_id is None:
        return
    query = session.query(Page2Document)
    if page_id is not None:
        query = query.filter(Page2Document.page_id == page_id)
    if document_id is not None:
        query = query.filter(Page2Document.document_id == document_id)

    to_delete = [row for row in query.all() if _confidence_rank(row.confidence) >= _DEFINITIVE_RANK]
    for row in to_delete:
        session.delete(row)
    if to_delete:
        stats["weaker_links_superseded"] += len(to_delete)
        logger.debug(
            "Superseded %d weaker link(s) (page_id=%s, document_id=%s).",
            len(to_delete),
            page_id,
            document_id,
        )


# ---------------------------------------------------------------------------
# Per-document page linking
# ---------------------------------------------------------------------------


class PageLinker:
    """Keeps a running Page2Document index per document so pages appended
    across several rows (and across the two passes) keep increasing indices.
    Also makes sure every page it links has any weaker, conflicting links to
    other documents cleared out first (see supersede_weaker_links).
    """

    def __init__(self, session: Session, stats: dict):
        self.session = session
        self.stats = stats
        self._next_index: dict[str, int] = defaultdict(int)
        self._cleared_pages: set[str] = set()

    def link(self, document: Document, scan: Scan) -> None:
        pages = get_pages_for_scan(self.session, scan)
        if not pages:
            logger.warning("Scan %r has no pages — skipping link.", scan.filename)
            self.stats["scans_without_pages"] += 1
            return
        for page in pages:
            if page.id not in self._cleared_pages:
                self._cleared_pages.add(page.id)
                supersede_weaker_links(self.session, self.stats, page_id=page.id)

            idx = self._next_index[document.id]
            self.session.add(
                Page2Document(
                    id=str(uuid.uuid4()),
                    page_id=page.id,
                    document_id=document.id,
                    index=idx,
                    source=P2D_SOURCE,
                    confidence=LinkConfidence.DEFINITIVE,
                )
            )
            self._next_index[document.id] = idx + 1
            self.stats["pages_linked"] += 1


# ---------------------------------------------------------------------------
# Row-by-row import
# ---------------------------------------------------------------------------


def import_file(
    session: Session,
    path: str,
    tanap_method: DocumentIdentificationMethod,
    subdoc_method: DocumentIdentificationMethod,
    *,
    force: bool,
    stats: dict,
) -> None:
    df = read_segmentation_csv(path)
    inv_number = detect_inventory_number(path, df)
    if not inv_number:
        logger.warning("%s: could not determine inventory number — skipping file.", path)
        stats["skipped_files"] += 1
        return

    inventory = lookup_inventory(session, inv_number)
    if inventory is None:
        logger.warning("%s: Inventory %r not found — skipping file.", path, inv_number)
        stats["skipped_files"] += 1
        return

    if not force and already_imported(session, inventory):
        logger.warning(
            "%s: inventory %s already has segmentation-derived page links — "
            "skipping (use --force to re-import).",
            path,
            inv_number,
        )
        stats["skipped_files"] += 1
        return

    logger.info("Importing %s (inventory %s, %d rows)", path, inv_number, len(df))
    linker = PageLinker(session, stats)
    cleared_document_ids: set[str] = set()

    open_tanap_docs: dict[str, Document] = {}
    current_tanap_doc: Optional[Document] = None

    current_subdoc: Optional[Document] = None
    subdoc_counter = 0

    # filename -> list of TANAP Documents directly linked to it (built as we
    # go, used to resolve forward- or backward-referencing "SAME AS" rows).
    scan_to_tanap_docs: dict[str, list[Document]] = defaultdict(list)
    # (scan_filename, referenced_filename, is_non_document_page)
    same_as_pending: list[tuple[str, str, bool]] = []

    for row_num, row in df.iterrows():
        scan_filename = row.get("Scan File_Name")
        if pd.isna(scan_filename):
            continue
        scan_filename = scan_filename.strip()

        boundary = row.get("TANAP Boundaries")
        boundary = boundary.strip() if pd.notna(boundary) else None
        tanap_id = row.get("TANAP ID")
        tanap_id = tanap_id.strip() if pd.notna(tanap_id) else None
        subdoc_raw = row.get("Subdocument boundaries")
        non_doc_type = row.get("Type of non-document page")
        is_non_document_page = pd.notna(non_doc_type)

        docs_to_link: list[Document] = []

        # ---- TANAP documents ----------------------------------------
        same_as_match = SAME_AS_RE.match(boundary) if boundary else None
        if same_as_match:
            same_as_pending.append(
                (scan_filename, same_as_match.group(1).strip(), is_non_document_page)
            )
        elif tanap_id:
            doc = open_tanap_docs.get(tanap_id)
            if doc is None:
                ext = create_or_get_external_id(session, tanap_id, TANAP_ID_CONTEXT)
                existing_link = (
                    session.query(Document2ExternalID)
                    .filter(Document2ExternalID.external_id == ext.id)
                    .first()
                )
                if existing_link is not None:
                    doc = existing_link.document
                    logger.debug(
                        "Row %d: reusing existing Document for TANAP ID %s", row_num, tanap_id
                    )
                else:
                    doc = Document(
                        id=str(uuid.uuid4()),
                        inventory_id=inventory.id,
                        method_id=tanap_method.id,
                    )
                    session.add(doc)
                    session.flush()
                    session.add(
                        Document2ExternalID(
                            id=str(uuid.uuid4()),
                            document_id=doc.id,
                            external_id=ext.id,
                        )
                    )
                    stats["tanap_documents_created"] += 1
                if doc.id not in cleared_document_ids:
                    cleared_document_ids.add(doc.id)
                    supersede_weaker_links(session, stats, document_id=doc.id)
                open_tanap_docs[tanap_id] = doc

            docs_to_link.append(doc)
            scan_to_tanap_docs[scan_filename].append(doc)
            current_tanap_doc = doc
            tanap_doc_closing_this_row = boundary == "END"
        else:
            tanap_doc_closing_this_row = False

        # ---- Subdocuments ---------------------------------------------
        # (uses current_tanap_doc as the parent for any newly-started
        # subdocument — even if this row's TANAP document also closes here,
        # since a subdocument starting on this scan is still physically part
        # of it. The close is applied below, after subdocuments are handled.)
        tokens = (
            [t.strip() for t in subdoc_raw.split("/") if t.strip()]
            if pd.notna(subdoc_raw)
            else []
        )
        if tokens:
            for token in tokens:
                if token.upper() == "START":
                    subdoc_counter += 1
                    identifier = f"{inv_number}-SUBDOC-{subdoc_counter:04d}"
                    ext = create_or_get_external_id(session, identifier, SUBDOC_ID_CONTEXT)
                    existing_subdoc_link = (
                        session.query(Document2ExternalID)
                        .filter(Document2ExternalID.external_id == ext.id)
                        .first()
                    )
                    if existing_subdoc_link is not None:
                        current_subdoc = existing_subdoc_link.document
                        current_subdoc.part_of_id = (
                            current_tanap_doc.id if current_tanap_doc else None
                        )
                    else:
                        current_subdoc = Document(
                            id=str(uuid.uuid4()),
                            inventory_id=inventory.id,
                            method_id=subdoc_method.id,
                            part_of_id=current_tanap_doc.id if current_tanap_doc else None,
                        )
                        session.add(current_subdoc)
                        session.flush()
                        session.add(
                            Document2ExternalID(
                                id=str(uuid.uuid4()),
                                document_id=current_subdoc.id,
                                external_id=ext.id,
                            )
                        )
                        stats["subdocuments_created"] += 1
                    if current_subdoc.id not in cleared_document_ids:
                        cleared_document_ids.add(current_subdoc.id)
                        supersede_weaker_links(session, stats, document_id=current_subdoc.id)
                    docs_to_link.append(current_subdoc)
                elif token.upper() == "END":
                    if current_subdoc is not None:
                        docs_to_link.append(current_subdoc)
                        current_subdoc = None
                    else:
                        logger.warning(
                            "Row %d (%s): 'END' subdocument token with no open "
                            "subdocument — ignoring.",
                            row_num,
                            scan_filename,
                        )
                else:
                    logger.warning(
                        "Row %d (%s): unrecognised subdocument token %r — ignoring.",
                        row_num,
                        scan_filename,
                        token,
                    )
        elif current_subdoc is not None:
            docs_to_link.append(current_subdoc)

        if tanap_doc_closing_this_row:
            open_tanap_docs.pop(tanap_id, None)
            current_tanap_doc = None

        # ---- Non-document pages ----------------------------------------
        if is_non_document_page:
            stats["non_document_pages"] += 1
            stats["non_document_page_types"][non_doc_type.strip()] += 1
            continue  # never link a flagged non-document page to anything

        if not docs_to_link:
            continue

        scan = lookup_scan_by_filename(session, scan_filename)
        if scan is None:
            logger.warning("Row %d: Scan %r not found — skipping.", row_num, scan_filename)
            stats["missing_scans"] += 1
            continue

        for doc in dict.fromkeys(docs_to_link):  # de-dupe, keep order
            linker.link(doc, scan)

    # ---- Second pass: resolve "SAME AS <filename>" rows -------------------
    for scan_filename, ref_filename, is_non_document_page in same_as_pending:
        if is_non_document_page:
            stats["non_document_pages"] += 1
            continue
        target_docs = scan_to_tanap_docs.get(ref_filename)
        if not target_docs:
            logger.warning(
                "%s: 'SAME AS %s' on scan %r could not be resolved — "
                "referenced scan has no TANAP document.",
                path,
                ref_filename,
                scan_filename,
            )
            stats["same_as_unresolved"] += 1
            continue

        scan = lookup_scan_by_filename(session, scan_filename)
        if scan is None:
            logger.warning(
                "Row referencing 'SAME AS %s': scan %r not found — skipping.",
                ref_filename,
                scan_filename,
            )
            stats["missing_scans"] += 1
            continue

        for doc in dict.fromkeys(target_docs):
            linker.link(doc, scan)
        stats["same_as_resolved"] += 1

    stats["files_imported"] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_stats() -> dict:
    return {
        "files_imported": 0,
        "skipped_files": 0,
        "tanap_documents_created": 0,
        "subdocuments_created": 0,
        "pages_linked": 0,
        "weaker_links_superseded": 0,
        "missing_scans": 0,
        "scans_without_pages": 0,
        "non_document_pages": 0,
        "non_document_page_types": defaultdict(int),
        "same_as_resolved": 0,
        "same_as_unresolved": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a folder of Document Segmentation CSVs into the GLOBALISE database."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing the segmentation CSVs (e.g. '8697_-_Document_Segmentation.csv').",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern (relative to --input-dir) selecting which files to import. Default: *.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --input-dir recursively for files matching --pattern.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without committing to the database.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import a file even if the inventory already has documents "
        "from the TANAP-boundaries identification method.",
    )
    args = parser.parse_args()

    if args.recursive:
        paths = sorted(glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True))
    else:
        paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    if not paths:
        logger.warning("No files matching %r found in %s", args.pattern, args.input_dir)
        return

    logger.info("Found %d file(s) to import.", len(paths))

    engine = create_engine(DATABASE_URL, echo=False)
    # Uncomment if you need to create tables from scratch:
    # Base.metadata.create_all(engine)

    stats = make_stats()

    with Session(engine) as session:
        tanap_method = get_or_create_method(session, TANAP_METHOD_NAME, TANAP_METHOD_DESC)
        subdoc_method = get_or_create_method(session, SUBDOC_METHOD_NAME, SUBDOC_METHOD_DESC)

        for path in paths:
            try:
                import_file(
                    session,
                    path,
                    tanap_method,
                    subdoc_method,
                    force=args.force,
                    stats=stats,
                )
                session.flush()
            except Exception:
                logger.exception("Failed to import %s — rolling back this file's changes.", path)
                session.rollback()
                stats["skipped_files"] += 1

        if args.dry_run:
            logger.info("Dry run — rolling back.")
            session.rollback()
        else:
            session.commit()
            logger.info("Committed.")

    logger.info("Done.")
    logger.info(
        "files_imported=%d skipped_files=%d tanap_documents_created=%d "
        "subdocuments_created=%d pages_linked=%d weaker_links_superseded=%d",
        stats["files_imported"],
        stats["skipped_files"],
        stats["tanap_documents_created"],
        stats["subdocuments_created"],
        stats["pages_linked"],
        stats["weaker_links_superseded"],
    )
    logger.info(
        "missing_scans=%d scans_without_pages=%d same_as_resolved=%d "
        "same_as_unresolved=%d non_document_pages=%d",
        stats["missing_scans"],
        stats["scans_without_pages"],
        stats["same_as_resolved"],
        stats["same_as_unresolved"],
        stats["non_document_pages"],
    )
    if stats["non_document_page_types"]:
        logger.info("non_document_page_types=%s", dict(stats["non_document_page_types"]))


if __name__ == "__main__":
    main()
