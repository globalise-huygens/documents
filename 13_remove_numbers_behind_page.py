#!/usr/bin/env python3
"""
Remove FOLIO_RANGE links that were likely created from a misplaced,
repeated folio number.

Step 12 in the import sequence (cleanup pass that runs after scripts 10
and 11).

Background
----------
Script 10 links a page to every document whose folio_start-folio_end
range contains one of the page's folio numbers. That's usually safe, but
sometimes a folio number doesn't belong where it appears -- for example a
correction slip, a cross-reference, or a stamp that was misread off a
neighbouring leaf. This can happen in two recognisable patterns:

  1. A batch of consecutive scans all end up carrying the exact same
     folio number -- e.g. because the number was misread from the leaf
     behind each of them. A short run like that is normal (a recto/verso
     pair sharing one folio number), but a long one is not: real folio
     numbering advances roughly one-per-leaf, so more than a handful of
     scans in a row reading the identical number is a strong signal of a
     systemic misread rather than genuine numbering.

  2. A single scan carries an extra, spurious folio number alongside its
     real one (e.g. "103, 200"), so script 10 links it to two documents
     instead of one.

This script applies one rule to clean this up:

  Long repeated runs: if a folio number appears identically on more than
  --max-repeat consecutive scans, every FOLIO_RANGE link that came from
  that number is deleted, whether or not the page in question was
  otherwise ambiguous. Affected pages simply lose that link; if that
  leaves them with no document at all, script 11's neighbour
  interpolation is the intended next step for them, not this script.

A page that ends up linked to more than one document for some other
reason (not a long repeated run) is left untouched by this script.

Only FOLIO_RANGE links (source='FOLIO_RANGE') are touched. VALIDATED and
DEFINITIVE links are ground truth and are never removed. INTERPOLATED and
CANDIDATE links from other scripts are left for those scripts to manage.
"""

import os
import logging
import argparse
from typing import Optional, List, Dict, Tuple, Any
import re

from sqlalchemy import create_engine, text, delete
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from models import Base, Page2Document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
SOURCE = "FOLIO_RANGE"          # the only link source this script ever touches
BATCH_SIZE = 5_000

# How many consecutive scans may legitimately share the exact same folio
# number (a recto/verso pair, or a small amount of slack) before the run
# is treated as a misread and its links are deleted outright.
DEFAULT_MAX_REPEAT = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_folio_numbers(raw: Optional[str]) -> List[int]:
    """Parse strictly valid folio numbers like '695, 696' into [695, 696].

    Rejects any part that is not a clean integer. Kept identical to the
    version in 10_match_folios.py so the two scripts agree on what counts
    as a folio number.
    """
    if not raw:
        return []

    results = []
    for part in raw.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            results.append(int(part))
        else:
            continue

    return results


def find_runs(
    occurrence_index: Dict[int, List[int]], max_repeat: int
) -> List[Tuple[int, List[int]]]:
    """Find folio values that appear on more than max_repeat scans in a row.

    occurrence_index maps a folio value to every scan-order position (an
    index into `seq`) where it appears. "In a row" means those positions
    are physically consecutive scans, not just nearby.

    Returns a list of (value, run_positions) for every run longer than
    max_repeat.
    """
    runs: List[Tuple[int, List[int]]] = []

    for value, positions in occurrence_index.items():
        positions_sorted = sorted(set(positions))
        n = len(positions_sorted)
        run_start = 0
        for i in range(1, n + 1):
            contiguous = i < n and positions_sorted[i] == positions_sorted[i - 1] + 1
            if contiguous:
                continue
            run_len = i - run_start
            if run_len > max_repeat:
                runs.append((value, positions_sorted[run_start:i]))
            run_start = i

    return runs


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def remove_misplaced_folio_links(
    database_url: str,
    max_repeat: int = DEFAULT_MAX_REPEAT,
    dry_run: bool = False,
) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "inventories_processed": 0,
        "suspect_runs_detected": 0,
        "links_removed_run_rule": 0,
    }

    with Session(engine) as session:
        try:
            # ---------------------------------------------------------------- #
            # 1. Find inventories that have FOLIO_RANGE links at all            #
            # ---------------------------------------------------------------- #
            inv_rows = session.execute(
                text(
                    "SELECT DISTINCT document.inventory_id "
                    "FROM page2document "
                    "JOIN document ON document.id = page2document.document_id "
                    "WHERE page2document.source = :source"
                ),
                {"source": SOURCE},
            ).all()

            if not inv_rows:
                logger.warning("No FOLIO_RANGE links found. Run script 10 first.")
                return stats

            inventory_ids = [r[0] for r in inv_rows]
            logger.info(f"Processing {len(inventory_ids)} inventories...")

            for inv_id in inventory_ids:
                # ------------------------------------------------------------ #
                # 2. Build the scan-order sequence of folio numbers            #
                # ------------------------------------------------------------ #
                # Grouped by scan, not by page: a scan with both a recto and a
                # verso page is one physical scan and must only count once
                # towards a run's length, even though it produced two Page
                # rows (usually reading the same folio number on both sides).
                page_rows = session.execute(
                    text(
                        "SELECT page.id, page.page_or_folio_number, scan.id "
                        "FROM page "
                        "JOIN scan ON scan.id = page.scan_id "
                        "WHERE page.inventory_id = :inv_id "
                        "AND scan.scan_order IS NOT NULL "
                        "ORDER BY scan.scan_order, page.recto_verso"
                    ),
                    {"inv_id": inv_id},
                ).all()

                seq: List[Dict[str, Any]] = []
                scan_pos: Dict[str, int] = {}
                occurrence_index: Dict[int, List[int]] = {}

                for p_id, p_str, scan_id in page_rows:
                    folios = parse_folio_numbers(p_str)

                    if scan_id not in scan_pos:
                        scan_pos[scan_id] = len(seq)
                        seq.append({"page_ids": [], "folios": set()})

                    idx = scan_pos[scan_id]
                    seq[idx]["page_ids"].append(p_id)
                    seq[idx]["folios"].update(folios)

                for idx, entry in enumerate(seq):
                    for f in entry["folios"]:
                        occurrence_index.setdefault(f, []).append(idx)

                if not seq:
                    continue

                # ------------------------------------------------------------ #
                # 3. Load document folio ranges for this inventory             #
                # ------------------------------------------------------------ #
                doc_rows = session.execute(
                    text(
                        "SELECT id, folio_start, folio_end FROM document "
                        "WHERE inventory_id = :inv_id AND folio_start IS NOT NULL"
                    ),
                    {"inv_id": inv_id},
                ).all()

                doc_ranges: Dict[str, Tuple[int, int]] = {}
                for doc_id, f_start, f_end in doc_rows:
                    actual_end = (
                        f_end if (f_end is not None and f_end >= f_start) else f_start
                    )
                    doc_ranges[doc_id] = (f_start, actual_end)

                # ------------------------------------------------------------ #
                # 4. Load current FOLIO_RANGE links for this inventory         #
                # ------------------------------------------------------------ #
                link_rows = session.execute(
                    text(
                        "SELECT page2document.page_id, page2document.document_id, "
                        "       page2document.id "
                        "FROM page2document "
                        "JOIN document ON document.id = page2document.document_id "
                        "WHERE document.inventory_id = :inv_id "
                        "AND page2document.source = :source"
                    ),
                    {"inv_id": inv_id, "source": SOURCE},
                ).all()

                links_by_page: Dict[str, List[Tuple[str, str]]] = {}
                for page_id, document_id, link_id in link_rows:
                    links_by_page.setdefault(page_id, []).append((link_id, document_id))

                if not links_by_page:
                    continue

                links_to_remove: List[str] = []

                # ------------------------------------------------------------ #
                # 5. Rule 1 -- delete links from long repeated runs           #
                # ------------------------------------------------------------ #
                for value, run_positions in find_runs(occurrence_index, max_repeat):
                    run_deletions_before = len(links_to_remove)
                    affected_page_ids: List[str] = []

                    for idx in run_positions:
                        for page_id in seq[idx]["page_ids"]:
                            page_links = links_by_page.get(page_id, [])
                            remaining = []
                            deleted_here = False
                            for link_id, doc_id in page_links:
                                rng = doc_ranges.get(doc_id)
                                if rng and rng[0] <= value <= rng[1]:
                                    links_to_remove.append(link_id)
                                    stats["links_removed_run_rule"] += 1
                                    deleted_here = True
                                else:
                                    remaining.append((link_id, doc_id))
                            links_by_page[page_id] = remaining
                            if deleted_here:
                                affected_page_ids.append(page_id)

                    if len(links_to_remove) > run_deletions_before:
                        stats["suspect_runs_detected"] += 1
                        logger.info(
                            f"  [run] folio {value} repeats on {len(run_positions)} "
                            f"consecutive scans -- deleted "
                            f"{len(links_to_remove) - run_deletions_before} "
                            f"associated link(s) on pages: "
                            f"{', '.join(affected_page_ids)}"
                        )

                # ------------------------------------------------------------ #
                # 6. Delete, commit per inventory                              #
                # ------------------------------------------------------------ #
                if links_to_remove:
                    stats["inventories_processed"] += 1

                    if dry_run:
                        logger.info(
                            f"  Inventory {inv_id}: would remove "
                            f"{len(links_to_remove)} link(s) [dry run]"
                        )
                    else:
                        for i in range(0, len(links_to_remove), BATCH_SIZE):
                            batch = links_to_remove[i : i + BATCH_SIZE]
                            session.execute(
                                delete(Page2Document).where(Page2Document.id.in_(batch))
                            )
                        session.commit()
                        logger.info(
                            f"  Inventory {inv_id}: removed "
                            f"{len(links_to_remove)} link(s)"
                        )

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove FOLIO_RANGE links caused by a repeated/misplaced folio "
            "number (step 12)."
        )
    )
    parser.add_argument("--database", default=DATABASE_URL)
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=DEFAULT_MAX_REPEAT,
        help=(
            "How many consecutive scans may legitimately share the exact "
            "same folio number before the run is deleted outright "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting anything",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GLOBALISE — Remove Misplaced Folio Links  (step 12)")
    print("=" * 60)
    if args.dry_run:
        print("(dry run — no changes will be made)")

    results = remove_misplaced_folio_links(
        args.database, max_repeat=args.max_repeat, dry_run=args.dry_run
    )

    print("\n=== Summary ===")
    print(f"  Inventories affected      : {results['inventories_processed']}")
    print(f"  Long repeated runs found  : {results['suspect_runs_detected']}")
    print(f"  Links removed             : {results['links_removed_run_rule']:,}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
