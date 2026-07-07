#!/usr/bin/env python3
"""
Remove FOLIO_RANGE links that were likely created from a misplaced,
out-of-sequence folio number.

Step 12 in the import sequence (cleanup pass that runs after scripts 10
and 11).

Background
----------
Script 10 links a page to every document whose folio_start-folio_end
range contains one of the page's folio numbers. That's usually safe, but
sometimes a folio number is repeated on a scan where it doesn't belong --
for example a correction slip, a cross-reference, or a stamp that was
copied from another page. When that happens, the *same* folio number
shows up on more than one scan in the inventory. Only one of those scans
is genuinely at that point in the document; the others are out of step
with their own neighbours in scan order.

Script 10 doesn't know the difference, so it links the misplaced scan to
whichever document owns that folio number -- in addition to the document
the scan actually belongs to (the one consistent with its neighbours).
The scan ends up attached to two documents.

This script looks at every page that currently has more than one
FOLIO_RANGE link, and uses the page's neighbours in scan order to work
out which of its folio numbers is the one that actually belongs there.
It then removes the FOLIO_RANGE link(s) that came from the other,
out-of-sequence folio number -- but only once it has confirmed that the
number really is a repeat, i.e. it has a legitimate home somewhere else
in the same inventory. A number that is merely unusual but not repeated
elsewhere is left alone and flagged for manual review instead of being
guessed at.

A neighbour is only trustworthy if it isn't itself part of the problem.
Two things disqualify a page from being used as a trustworthy neighbour:

  1. It is itself ambiguous (linked to more than one document) -- its own
     reading hasn't been resolved yet.
  2. It belongs to a run of more than --max-repeat consecutive pages that
     all carry the exact same folio number. A short run like that is
     normal (a recto/verso pair sharing one folio number), but a long run
     usually means a batch of scans were mislabelled with the same wrong
     number -- for example because the number was misread from a
     neighbouring leaf. Trusting that run would make the misreading look
     like a legitimate, well-supported sequence instead of the anomaly it
     is. When the local neighbourhood is this contaminated, the script
     looks further afield for a trustworthy bound and, if none can be
     found, leaves the page for manual review rather than guessing.

Only FOLIO_RANGE links (source='FOLIO_RANGE') are touched. VALIDATED and
DEFINITIVE links are ground truth and are never removed. INTERPOLATED and
CANDIDATE links from other scripts are left for those scripts to manage.

Note on scope: this script only fixes pages that ended up linked to *two*
documents. A run of consecutive pages that were all mislabelled with the
same wrong folio number, but each only produced a *single* (wrong) link,
won't show up as ambiguous and isn't touched here -- that's a data-entry
problem for the source folio numbers themselves, not a duplicate-link
problem. The script logs detected runs so they're at least visible.
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

# How far a folio number may drift from its neighbours' values and still
# count as "in sequence". 0 means strictly non-decreasing. Raise this a
# little if real inventories have small amounts of noise (e.g. a slip of
# paper numbered a page ahead of where it's bound).
DEFAULT_TOLERANCE = 0

# How many consecutive scans may legitimately share the exact same folio
# number (recto/verso pairs are the normal case, hence 2). A run longer
# than this is treated as a probable labelling error rather than a
# trustworthy stretch of sequence.
DEFAULT_MAX_REPEAT = 2

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


def is_in_sequence(
    folio_num: int,
    prev_val: Optional[int],
    next_val: Optional[int],
    tolerance: int = 0,
) -> bool:
    """Does folio_num fit between its neighbours in scan order?

    Folio numbers are expected to be non-decreasing as scan order
    increases. A value that dips below its predecessor or jumps past its
    successor is breaking that trend and is treated as out of sequence.
    """
    if prev_val is not None and folio_num < prev_val - tolerance:
        return False
    if next_val is not None and folio_num > next_val + tolerance:
        return False
    return True


def local_bounds(
    scaffold_value: Dict[int, int], excluded_idx: set, seq_len: int, idx: int
) -> Tuple[Optional[int], Optional[int]]:
    """The nearest trustworthy folio values immediately before/after idx.

    Any index in excluded_idx is skipped when looking for neighbours --
    that includes pages that are themselves ambiguous (their own reading
    hasn't been resolved yet) and pages that belong to a suspiciously long
    run of identical values (see find_suspect_runs). Skipping past both
    means a neighbour is only trusted once it looks like a genuine,
    individually-supported reading.
    """
    prev_val = None
    i = idx - 1
    while i >= 0:
        if i not in excluded_idx and i in scaffold_value:
            prev_val = scaffold_value[i]
            break
        i -= 1

    next_val = None
    j = idx + 1
    while j < seq_len:
        if j not in excluded_idx and j in scaffold_value:
            next_val = scaffold_value[j]
            break
        j += 1

    return prev_val, next_val


def find_suspect_runs(
    scaffold_value: Dict[int, int], ordered_idx: List[int], max_repeat: int
) -> Tuple[set, int]:
    """Indices that belong to a run of more than max_repeat identical values.

    ordered_idx is the list of scaffold indices in scan-order (already
    excludes ambiguous pages). Runs are measured over this filtered list,
    so an ambiguous page sitting between two identical values doesn't
    break up an otherwise-contiguous run of the same wrong number.

    Returns (suspect_indices, number_of_runs_found).
    """
    suspect: set = set()
    run_count = 0
    run_start = 0
    n = len(ordered_idx)

    for i in range(1, n + 1):
        same_as_prev = (
            i < n and scaffold_value[ordered_idx[i]] == scaffold_value[ordered_idx[run_start]]
        )
        if same_as_prev:
            continue

        run_len = i - run_start
        if run_len > max_repeat:
            run_value = scaffold_value[ordered_idx[run_start]]
            run_indices = ordered_idx[run_start:i]
            suspect.update(run_indices)
            run_count += 1
            logger.info(
                f"  [suspect run] folio {run_value} repeats on "
                f"{run_len} consecutive scans -- excluding from trend baseline"
            )
        run_start = i

    return suspect, run_count


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def remove_misplaced_folio_links(
    database_url: str,
    tolerance: int = DEFAULT_TOLERANCE,
    max_repeat: int = DEFAULT_MAX_REPEAT,
    dry_run: bool = False,
) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "inventories_processed": 0,
        "pages_examined": 0,
        "pages_resolved": 0,
        "links_removed": 0,
        "pages_needing_manual_review": 0,
        "suspect_runs_detected": 0,
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
                page_rows = session.execute(
                    text(
                        "SELECT page.id, page.page_or_folio_number, scan.scan_order "
                        "FROM page "
                        "JOIN scan ON scan.id = page.scan_id "
                        "WHERE page.inventory_id = :inv_id "
                        "AND scan.scan_order IS NOT NULL "
                        "ORDER BY scan.scan_order, page.recto_verso"
                    ),
                    {"inv_id": inv_id},
                ).all()

                seq: List[Dict[str, Any]] = []
                page_index: Dict[str, int] = {}
                occurrence_index: Dict[int, List[int]] = {}

                for p_id, p_str, _scan_order in page_rows:
                    folios = parse_folio_numbers(p_str)
                    if not folios:
                        continue
                    idx = len(seq)
                    seq.append({"page_id": p_id, "folios": folios})
                    page_index[p_id] = idx
                    for f in folios:
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
                # 4. Find pages with more than one FOLIO_RANGE link            #
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

                ambiguous_pages = {
                    p: links for p, links in links_by_page.items() if len(links) > 1
                }

                if not ambiguous_pages:
                    continue

                ambiguous_idx = {
                    page_index[p] for p in ambiguous_pages if p in page_index
                }

                # ------------------------------------------------------------ #
                # 4b. Resolve a single trustworthy value per non-ambiguous     #
                #     page, then flag any suspiciously long run of identical  #
                #     values so it isn't trusted as a neighbour either.       #
                # ------------------------------------------------------------ #
                scaffold_value: Dict[int, int] = {}
                for idx, entry in enumerate(seq):
                    if idx in ambiguous_idx:
                        continue
                    page_links = links_by_page.get(entry["page_id"], [])
                    candidates = entry["folios"]

                    if len(page_links) == 1:
                        rng = doc_ranges.get(page_links[0][1])
                        matched = (
                            [f for f in candidates if rng[0] <= f <= rng[1]]
                            if rng else []
                        )
                        if matched:
                            scaffold_value[idx] = min(matched)
                            continue

                    if len(candidates) == 1:
                        scaffold_value[idx] = candidates[0]
                    # Otherwise: multiple raw readings and no resolved link
                    # to disambiguate them -- no trustworthy value here.

                ordered_idx = sorted(scaffold_value.keys())
                suspect_idx, run_count = find_suspect_runs(
                    scaffold_value, ordered_idx, max_repeat
                )
                stats["suspect_runs_detected"] += run_count

                excluded_idx = ambiguous_idx | suspect_idx

                # ------------------------------------------------------------ #
                # 5. Resolve each ambiguous page                               #
                # ------------------------------------------------------------ #
                link_ids_to_remove: List[str] = []

                for page_id, links in ambiguous_pages.items():
                    stats["pages_examined"] += 1

                    if page_id not in page_index:
                        # Page has folio-based links but no parseable folio
                        # number / scan order any more -- leave for manual
                        # review rather than guessing.
                        logger.info(
                            f"  [review] page {page_id}: has {len(links)} FOLIO_RANGE "
                            f"links but no ordered folio number to check against"
                        )
                        stats["pages_needing_manual_review"] += 1
                        continue

                    idx = page_index[page_id]
                    prev_val, next_val = local_bounds(
                        scaffold_value, excluded_idx, len(seq), idx
                    )
                    candidates = seq[idx]["folios"]

                    classified = []
                    for link_id, doc_id in links:
                        rng = doc_ranges.get(doc_id)
                        if rng is None:
                            continue
                        f_start, f_end = rng
                        matched = [f for f in candidates if f_start <= f <= f_end]
                        if not matched:
                            continue
                        representative = min(matched)
                        classified.append({
                            "link_id": link_id,
                            "doc_id": doc_id,
                            "folio": representative,
                            "in_seq": is_in_sequence(
                                representative, prev_val, next_val, tolerance
                            ),
                        })

                    in_seq_links = [c for c in classified if c["in_seq"]]
                    out_seq_links = [c for c in classified if not c["in_seq"]]

                    if len(in_seq_links) != 1 or not out_seq_links:
                        # Either no single, clear "home" document, or nothing
                        # looks out of place -- don't guess.
                        logger.info(
                            f"  [review] page {page_id}: {len(in_seq_links)} in-sequence "
                            f"link(s), {len(out_seq_links)} out-of-sequence link(s) "
                            f"-- ambiguous, skipping"
                        )
                        stats["pages_needing_manual_review"] += 1
                        continue

                    to_remove = []
                    for c in out_seq_links:
                        f = c["folio"]
                        has_home_elsewhere = False
                        for other_idx in occurrence_index.get(f, []):
                            if other_idx == idx:
                                continue
                            o_prev, o_next = local_bounds(
                                scaffold_value, excluded_idx, len(seq), other_idx
                            )
                            if is_in_sequence(f, o_prev, o_next, tolerance):
                                has_home_elsewhere = True
                                break

                        if has_home_elsewhere:
                            to_remove.append(c)
                        else:
                            logger.info(
                                f"  [review] page {page_id}: folio {f} is out of "
                                f"sequence (neighbours: {prev_val}-{next_val}) but has "
                                f"no confirmed home elsewhere -- skipping, needs a look"
                            )
                            stats["pages_needing_manual_review"] += 1

                    if not to_remove:
                        continue

                    for c in to_remove:
                        logger.info(
                            f"  page {page_id}: removing link to document "
                            f"{c['doc_id']} (folio {c['folio']}, neighbours "
                            f"{prev_val}-{next_val}); keeping document "
                            f"{in_seq_links[0]['doc_id']} (folio "
                            f"{in_seq_links[0]['folio']})"
                        )
                        link_ids_to_remove.append(c["link_id"])

                    stats["pages_resolved"] += 1

                # ------------------------------------------------------------ #
                # 6. Delete the resolved bad links, commit per inventory       #
                # ------------------------------------------------------------ #
                if link_ids_to_remove:
                    stats["links_removed"] += len(link_ids_to_remove)
                    stats["inventories_processed"] += 1

                    if dry_run:
                        logger.info(
                            f"  Inventory {inv_id}: would remove "
                            f"{len(link_ids_to_remove)} link(s) [dry run]"
                        )
                    else:
                        for i in range(0, len(link_ids_to_remove), BATCH_SIZE):
                            batch = link_ids_to_remove[i : i + BATCH_SIZE]
                            session.execute(
                                delete(Page2Document).where(Page2Document.id.in_(batch))
                            )
                        session.commit()
                        logger.info(
                            f"  Inventory {inv_id}: removed "
                            f"{len(link_ids_to_remove)} link(s)"
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
            "Remove FOLIO_RANGE links likely caused by an out-of-sequence, "
            "repeated folio number (step 12)."
        )
    )
    parser.add_argument("--database", default=DATABASE_URL)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=(
            "How many folios a value may drift from its neighbours and still "
            "count as in-sequence (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=DEFAULT_MAX_REPEAT,
        help=(
            "How many consecutive scans may legitimately share the exact same "
            "folio number before the run is treated as a probable labelling "
            "error and excluded from the trend baseline (default: %(default)s)"
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
        args.database,
        tolerance=args.tolerance,
        max_repeat=args.max_repeat,
        dry_run=args.dry_run,
    )

    print("\n=== Summary ===")
    print(f"  Inventories affected      : {results['inventories_processed']}")
    print(f"  Suspect runs detected     : {results['suspect_runs_detected']}")
    print(f"  Ambiguous pages examined  : {results['pages_examined']}")
    print(f"  Pages resolved            : {results['pages_resolved']}")
    print(f"  Links removed             : {results['links_removed']:,}")
    print(f"  Needing manual review     : {results['pages_needing_manual_review']}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
