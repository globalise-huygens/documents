"""
interpolate_scan_documents.py
─────────────────────────────
Fill gaps in scan→document links using neighbour interpolation.

For every inventory, scans are visited in scan_order.  A scan that has no
linked document is a candidate for interpolation: if the nearest linked scan
*before* it and the nearest linked scan *after* it both resolve to exactly
one document, and that document is the same on both sides, and the gap between
them is no wider than `max_gap`, every unlinked scan in that gap is linked to
that document with source="INTERPOLATED" / confidence=INTERPOLATED.

Usage (as a module):
    from interpolate_scan_documents import interpolate_inventory, interpolate_all

Usage (CLI):
    python interpolate_scan_documents.py \
        --db sqlite:///archive.db \
        --max-gap 3 \
        [--inventory 1.04.02]   # omit to process every inventory
        [--dry-run]             # print what would be inserted, don't commit
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import (
    Inventory,
    Scan,
    Page,
    Page2Document,
    LinkConfidence,
)

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

# ---------------------------------------------------------------------------
# Core query: documents linked to a single scan (via page2document)
# ---------------------------------------------------------------------------

_DOCS_FOR_SCAN_SQL = text(
    """
    SELECT DISTINCT d.id
    FROM scan s
    JOIN page p            ON p.scan_id      = s.id
    JOIN page2document p2d ON p2d.page_id    = p.id
    JOIN document d        ON d.id           = p2d.document_id
    WHERE s.id = :scan_id
      AND p2d.source IN ('FOLIO_RANGE')   -- only trust explicit folio matches
    """
)


def _document_ids_for_scan(session: Session, scan_id: str) -> set[str]:
    """Return the set of document ids linked to *scan_id* via page2document."""
    rows = session.execute(_DOCS_FOR_SCAN_SQL, {"scan_id": scan_id}).fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Per-inventory interpolation
# ---------------------------------------------------------------------------

def interpolate_inventory(
    session: Session,
    inventory: Inventory,
    max_gap: int = 5,
    dry_run: bool = False,
    source: str = "INTERPOLATED",
) -> int:
    """
    Interpolate missing scan→document links for one inventory.

    Parameters
    ----------
    session     : active SQLAlchemy session
    inventory   : Inventory ORM object (must already be in the session)
    max_gap     : maximum number of consecutive unlinked scans that may be
                  filled in a single interpolation step
    dry_run     : if True, log what would be inserted but do not flush/commit
    source      : value written to Page2Document.source for new rows

    Returns
    -------
    Number of Page2Document rows inserted (or that *would* be inserted in
    dry-run mode).
    """
    # ------------------------------------------------------------------
    # 1. Collect ordered scans for this inventory
    # ------------------------------------------------------------------
    scans: list[Scan] = (
        session.query(Scan)
        .filter(Scan.inventory_id == inventory.id)
        .order_by(Scan.scan_order.nullslast(), Scan.filename)
        .all()
    )

    if not scans:
        log.debug("Inventory %s has no scans – skipping.", inventory.inventory_number)
        return 0

    # ------------------------------------------------------------------
    # 2. Build a list of (scan, frozenset[doc_id]) for every scan
    # ------------------------------------------------------------------
    scan_docs: list[tuple[Scan, frozenset[str]]] = []
    for scan in scans:
        doc_ids = _document_ids_for_scan(session, scan.id)
        scan_docs.append((scan, frozenset(doc_ids)))

    # ------------------------------------------------------------------
    # 3. Walk the list and collect gap runs
    # ------------------------------------------------------------------
    # A "gap run" is a maximal contiguous sub-sequence of scans whose
    # doc-set is empty.  We record:
    #   - the indices of the gap
    #   - the doc-set of the preceding linked scan (left neighbour)
    #   - the doc-set of the succeeding linked scan (right neighbour)

    inserted_total = 0
    n = len(scan_docs)
    i = 0

    while i < n:
        _, docs = scan_docs[i]
        if docs:           # this scan already has a document – move on
            i += 1
            continue

        # Found the start of a gap – find its end
        gap_start = i
        while i < n and not scan_docs[i][1]:
            i += 1
        gap_end = i - 1    # inclusive

        gap_length = gap_end - gap_start + 1

        # Left neighbour: the last linked scan before the gap
        left_docs: Optional[frozenset[str]] = None
        if gap_start > 0:
            left_docs = scan_docs[gap_start - 1][1]

        # Right neighbour: the first linked scan after the gap
        right_docs: Optional[frozenset[str]] = None
        if gap_end + 1 < n:
            right_docs = scan_docs[gap_end + 1][1]

        # ------------------------------------------------------------------
        # 4. Decide whether the gap can be filled
        # ------------------------------------------------------------------
        # Conditions:
        #   (a) gap is not wider than max_gap
        #   (b) both neighbours exist
        #   (c) each neighbour links to exactly one document
        #   (d) both neighbours agree on the same document
        # ------------------------------------------------------------------
        if (
            gap_length <= max_gap
            and left_docs is not None
            and right_docs is not None
            and len(left_docs) == 1
            and len(right_docs) == 1
            and left_docs == right_docs
        ):
            target_doc_id = next(iter(left_docs))

            for idx in range(gap_start, gap_end + 1):
                gap_scan, _ = scan_docs[idx]
                inserted = _link_scan_to_document(
                    session,
                    gap_scan,
                    target_doc_id,
                    source=source,
                    dry_run=dry_run,
                )
                inserted_total += inserted

                if inserted:
                    log.info(
                        "[%s] scan %s (order=%s) → document %s (%s)",
                        inventory.inventory_number,
                        gap_scan.filename,
                        gap_scan.scan_order,
                        target_doc_id,
                        "DRY RUN" if dry_run else "INSERTED",
                    )
        else:
            if gap_length > max_gap:
                reason = f"gap too wide ({gap_length} > {max_gap})"
            elif left_docs is None or right_docs is None:
                reason = "missing neighbour (start or end of inventory)"
            elif len(left_docs) != 1 or len(right_docs) != 1:
                reason = (
                    f"ambiguous neighbours "
                    f"(left={len(left_docs or set())} docs, "
                    f"right={len(right_docs or set())} docs)"
                )
            else:
                reason = "neighbours disagree on document"

            log.debug(
                "[%s] gap scans %d–%d skipped: %s",
                inventory.inventory_number,
                gap_start,
                gap_end,
                reason,
            )

    return inserted_total


def _link_scan_to_document(
    session: Session,
    scan: Scan,
    document_id: str,
    source: str,
    dry_run: bool,
) -> int:
    """
    Create Page2Document rows for every page of *scan* that is not yet linked
    to *document_id*.

    Returns the number of rows inserted (or that would be inserted).
    """
    pages: list[Page] = (
        session.query(Page).filter(Page.scan_id == scan.id).all()
    )

    if not pages:
        log.warning("Scan %s has no pages – cannot link to document.", scan.filename)
        return 0

    count = 0
    for page in pages:
        # Skip if this page is already linked to the same document
        already_linked = any(
            p2d.document_id == document_id for p2d in page.documents
        )
        if already_linked:
            continue

        # Determine next index for this page
        existing_indices = [p2d.index for p2d in page.documents]
        next_index = (max(existing_indices) + 1) if existing_indices else 0

        if not dry_run:
            p2d = Page2Document(
                id=str(uuid.uuid4()),
                page_id=page.id,
                document_id=document_id,
                index=next_index,
                source=source,
                confidence=LinkConfidence.INTERPOLATED,
            )
            session.add(p2d)

        count += 1

    return count


# ---------------------------------------------------------------------------
# Process all inventories
# ---------------------------------------------------------------------------

def interpolate_all(
    session: Session,
    max_gap: int = 3,
    dry_run: bool = False,
    source: str = "INTERPOLATED",
) -> dict[str, int]:
    """
    Run interpolation over every inventory.

    Returns a dict mapping inventory_number → number of rows inserted.
    """
    inventories: list[Inventory] = session.query(Inventory).all()
    results: dict[str, int] = {}

    for inv in inventories:
        n = interpolate_inventory(
            session, inv, max_gap=max_gap, dry_run=dry_run, source=source
        )
        if n:
            results[inv.inventory_number] = n

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpolate missing scan→document links within inventories."
    )
    p.add_argument(
        "--db",
        default=DATABASE_URL,
        help="SQLAlchemy database URL (default: env DATABASE_URL or sqlite:///globalise_documents.db)",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="Maximum number of consecutive unlinked scans to fill (default: 3)",
    )
    p.add_argument(
        "--inventory",
        default=None,
        help="Process only this inventory number (default: all inventories)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be inserted without writing to the database",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    engine = create_engine(args.db)

    with Session(engine) as session:
        if args.inventory:
            inv = (
                session.query(Inventory)
                .filter(Inventory.inventory_number == args.inventory)
                .one_or_none()
            )
            if inv is None:
                log.error("Inventory %r not found in database.", args.inventory)
                return

            n = interpolate_inventory(
                session, inv,
                max_gap=args.max_gap,
                dry_run=args.dry_run,
            )
            log.info(
                "Inventory %s: %d row(s) %s.",
                args.inventory,
                n,
                "would be inserted (dry run)" if args.dry_run else "inserted",
            )
        else:
            results = interpolate_all(
                session,
                max_gap=args.max_gap,
                dry_run=args.dry_run,
            )
            total = sum(results.values())
            log.info(
                "Done. %d row(s) %s across %d inventory/ies.",
                total,
                "would be inserted (dry run)" if args.dry_run else "inserted",
                len(results),
            )
            for inv_num, count in sorted(results.items()):
                log.info("  %-12s  %d", inv_num, count)

        if not args.dry_run:
            session.commit()
            log.info("Changes committed.")


if __name__ == "__main__":
    main()
