#!/usr/bin/env python3
"""
Fill missing boundary pages of TANAP documents using baseline segmentation.
Step 12 in the import sequence.

TANAP documents carry a fixed, ground-truth folio_start/folio_end range.
Script 10 links a page to a document when the page's OWN folio label parses
and falls inside that range. Some pages near the start or end of a document
never get linked this way — their folio label is missing, smudged, or just
didn't parse — even though they are clearly part of the document.

Example this script targets:
    TANAP document: folio_start=10, folio_end=15  (6 folios, fixed)
    Currently linked (script 10): folios 11, 12, 13, 14 only
    Missing folios: 10 and 15
    Surrounding documents: unlinked

    The baseline segmentation (script 4, blank pages & signatures) draws a
    document boundary that is a bit WIDER than the linked block — it
    includes a scan just before folio 11 and one just after folio 14.
    Since the document is missing exactly folios 10 and 15, and there are
    exactly that many unlinked scans immediately adjacent to the linked
    block, those scans are almost certainly folios 10 and 15.

How it works, per TANAP document D:
    1. covered = the folio numbers already linked to D (from D's own
       linked pages' index values, D.folio_start/D.folio_end unchanged).
    2. missing = every folio number in [folio_start, folio_end] that isn't
       covered yet.
    3. Find the baseline document that contains most of D's linked pages,
       and locate D's contiguous block inside it.
    4. Walk outward from that block, both backward and forward, collecting
       unlinked scans, but STOP as soon as a scan is already linked to a
       DIFFERENT document (never cross into another document's territory).
    5. Pair the closest unlinked scans with the closest missing folio
       numbers (nearest scan <-> nearest missing folio), one-to-one. If a
       side has more unlinked scans than missing folios (the baseline
       boundary over-reached — "boundaries are decent, but can be wrong"),
       only the innermost ones are filled and the excess is left alone. If
       there are fewer scans than missing folios, only what's physically
       there gets filled.

This script NEVER changes folio_start/folio_end (ground truth) and NEVER
reassigns a page that's already linked to a different TANAP document.

Run with --dry-run first to preview what would be linked.
"""

import os
import uuid
import logging
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from models import Base, LinkConfidence

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
BASELINE_METHOD_NAME = "Baseline: Empty Pages & Signatures"
SOURCE = "BASELINE_BOUNDARY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def fill_baseline_boundaries(database_url: str, dry_run: bool = False) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    stats = {
        "inventories_processed": 0,
        "documents_extended": 0,
        "pages_filled": 0,
        "folios_still_missing": 0,
    }

    with session_factory() as session:
        try:
            baseline_method_row = session.execute(
                text("SELECT id FROM document_identification_method WHERE name = :name"),
                {"name": BASELINE_METHOD_NAME},
            ).first()
            if not baseline_method_row:
                logger.warning("Baseline identification method not found. Run script 4 first.")
                return stats
            baseline_method_id = baseline_method_row[0]

            inv_rows = session.execute(
                text("SELECT DISTINCT inventory_id FROM document WHERE folio_start IS NOT NULL")
            ).all()
            inventory_ids = [r[0] for r in inv_rows]
            logger.info(f"Processing {len(inventory_ids)} inventories...")

            for inv_id in inventory_ids:
                inv_stats = _process_inventory(session, inv_id, baseline_method_id, dry_run)
                if inv_stats["pages_filled"]:
                    stats["inventories_processed"] += 1
                stats["documents_extended"] += inv_stats["documents_extended"]
                stats["pages_filled"] += inv_stats["pages_filled"]
                stats["folios_still_missing"] += inv_stats["folios_still_missing"]

            if not dry_run:
                session.commit()
            else:
                session.rollback()

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise

    return stats


def _process_inventory(
    session: Session, inv_id: str, baseline_method_id: str, dry_run: bool
) -> Dict[str, int]:
    stats = {"documents_extended": 0, "pages_filled": 0, "folios_still_missing": 0}

    # --- Baseline segmentation: ordered pages per baseline document -------
    baseline_rows = session.execute(
        text(
            "SELECT p2d.document_id, p2d.page_id, p2d.\"index\" "
            "FROM page2document p2d "
            "JOIN document d ON d.id = p2d.document_id "
            "WHERE d.inventory_id = :inv_id AND d.method_id = :method_id "
            "ORDER BY p2d.document_id, p2d.\"index\""
        ),
        {"inv_id": inv_id, "method_id": baseline_method_id},
    ).all()

    baseline_doc_pages: Dict[str, List[str]] = defaultdict(list)
    page_to_baseline_doc: Dict[str, str] = {}
    for b_doc_id, page_id, _idx in baseline_rows:
        baseline_doc_pages[b_doc_id].append(page_id)
        page_to_baseline_doc[page_id] = b_doc_id

    if not baseline_doc_pages:
        return stats  # no baseline segmentation for this inventory yet

    # --- Every existing TANAP link in this inventory (any source) ---------
    tanap_link_rows = session.execute(
        text(
            "SELECT p2d.document_id, p2d.page_id, p2d.\"index\" "
            "FROM page2document p2d "
            "JOIN document d ON d.id = p2d.document_id "
            "WHERE d.inventory_id = :inv_id AND d.folio_start IS NOT NULL"
        ),
        {"inv_id": inv_id},
    ).all()

    links_by_doc: Dict[str, List[Tuple[str, Optional[int]]]] = defaultdict(list)
    linked_pages: Set[str] = set()  # any page already linked to ANY TANAP doc
    for doc_id, page_id, idx in tanap_link_rows:
        links_by_doc[doc_id].append((page_id, idx))
        linked_pages.add(page_id)

    # --- TANAP documents (ground truth ranges) -----------------------------
    doc_rows = session.execute(
        text(
            "SELECT id, folio_start, folio_end FROM document "
            "WHERE inventory_id = :inv_id AND folio_start IS NOT NULL"
        ),
        {"inv_id": inv_id},
    ).all()

    new_rows: List[Dict] = []

    for doc_id, folio_start, folio_end_raw in doc_rows:
        # folio_end is nullable (single-folio documents). Mirror script 10's
        # convention: treat a missing/invalid end as a one-folio range.
        folio_end = (
            folio_end_raw
            if (folio_end_raw is not None and folio_end_raw >= folio_start)
            else folio_start
        )

        doc_links = links_by_doc.get(doc_id, [])
        if not doc_links:
            continue  # nothing to anchor to; script 10/11 found no match at all

        doc_linked_pages = {pid for pid, _idx in doc_links}
        covered = {
            idx for _pid, idx in doc_links
            if idx is not None and folio_start <= idx <= folio_end
        }
        if not covered:
            continue  # no reliable folio numbers to anchor an extension on

        missing = sorted(set(range(folio_start, folio_end + 1)) - covered)
        if not missing:
            continue  # document already fully covered

        min_covered, max_covered = min(covered), max(covered)
        before_missing = sorted((f for f in missing if f < min_covered), reverse=True)
        after_missing = sorted(f for f in missing if f > max_covered)

        # Anchor baseline document: whichever baseline doc holds most of
        # this TANAP document's currently linked pages.
        candidates = Counter(
            page_to_baseline_doc[pid] for pid in doc_linked_pages if pid in page_to_baseline_doc
        )
        if not candidates:
            continue
        anchor_id, _count = candidates.most_common(1)[0]
        baseline_pages = baseline_doc_pages[anchor_id]

        positions = [i for i, pid in enumerate(baseline_pages) if pid in doc_linked_pages]
        if not positions:
            continue
        min_pos, max_pos = min(positions), max(positions)

        # Walk backward from the block, collecting unlinked scans, closest first.
        left_gap: List[str] = []
        i = min_pos - 1
        while i >= 0:
            pid = baseline_pages[i]
            if pid in linked_pages:
                break  # belongs to a different document, or somehow already linked
            left_gap.append(pid)
            i -= 1

        # Walk forward from the block, collecting unlinked scans, closest first.
        right_gap: List[str] = []
        i = max_pos + 1
        while i < len(baseline_pages):
            pid = baseline_pages[i]
            if pid in linked_pages:
                break
            right_gap.append(pid)
            i += 1

        filled_this_doc = 0

        for page_id, folio_num in zip(left_gap, before_missing):
            new_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "page_id": page_id,
                    "document_id": doc_id,
                    "index": folio_num,
                    "source": SOURCE,
                    "confidence": LinkConfidence.INTERPOLATED.value,
                }
            )
            linked_pages.add(page_id)  # don't let another document also claim it
            filled_this_doc += 1

        for page_id, folio_num in zip(right_gap, after_missing):
            new_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "page_id": page_id,
                    "document_id": doc_id,
                    "index": folio_num,
                    "source": SOURCE,
                    "confidence": LinkConfidence.INTERPOLATED.value,
                }
            )
            linked_pages.add(page_id)
            filled_this_doc += 1

        remaining_missing = len(missing) - filled_this_doc
        stats["folios_still_missing"] += max(remaining_missing, 0)

        if filled_this_doc:
            stats["documents_extended"] += 1
            stats["pages_filled"] += filled_this_doc
            logger.info(
                f"  Document {doc_id} (folio {folio_start}-{folio_end}): "
                f"filled {filled_this_doc}/{len(missing)} missing folio(s)"
                + (" [dry-run]" if dry_run else "")
            )

    if new_rows and not dry_run:
        session.execute(
            text(
                "INSERT INTO page2document (id, page_id, document_id, \"index\", source, confidence) "
                "VALUES (:id, :page_id, :document_id, :index, :source, :confidence)"
            ),
            new_rows,
        )

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing boundary folios using baseline segmentation (step 12)."
    )
    parser.add_argument("--database", default=DATABASE_URL)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing to the database",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GLOBALISE — Baseline Boundary Filling  (step 12)")
    print("=" * 60)

    results = fill_baseline_boundaries(args.database, dry_run=args.dry_run)

    print("\n=== Summary ===")
    print(f"  Inventories with changes    : {results['inventories_processed']:,}")
    print(f"  Documents extended          : {results['documents_extended']:,}")
    print(f"  Pages filled                : {results['pages_filled']:,}")
    print(f"  Missing folios still absent : {results['folios_still_missing']:,}")
    if args.dry_run:
        print("  (dry run — nothing was written)")
    print("✓ Done.")


if __name__ == "__main__":
    main()
