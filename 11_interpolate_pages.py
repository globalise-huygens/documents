#!/usr/bin/env python3
"""
Interpolate page–document links for unlinked pages using scan-order neighbours.
Step 11 in the import sequence — run after step 10.

New features:
- Safe reruns: script can be executed multiple times to fill new gaps.
- Configurable propagation depth via CLI argument.

Algorithm (per inventory):
  1. Load all pages in scan order.
  2. Identify unlinked pages.
  3. For each unlinked page, find nearest linked neighbours.
  4. Interpolate ONLY if:
       - both neighbours exist
       - both resolve to exactly one document
       - both are the same document
  5. Propagation depth controls whether interpolated pages can be reused.

Confidence assigned: INTERPOLATED
"""

import os
import uuid
import logging
import argparse
from typing import Dict, List, Optional, Set, Tuple, Any

from sqlalchemy import create_engine, text, insert
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from models import Base, Page2Document, LinkConfidence

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
SOURCE = "INTERPOLATED"
BATCH_SIZE = 5_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_neighbour_doc(
    page_index: int,
    ordered_page_ids: List[str],
    page_to_docs: Dict[str, Set[str]],
    direction: int,
) -> Optional[str]:
    i = page_index + direction
    while 0 <= i < len(ordered_page_ids):
        docs = page_to_docs.get(ordered_page_ids[i], set())
        if docs:
            return next(iter(docs)) if len(docs) == 1 else None
        i += direction
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def interpolate(database_url: str, propagation_depth: int) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "inventories_processed": 0,
        "pages_interpolated": 0,
        "pages_skipped_boundary": 0,
        "pages_skipped_no_neighbour": 0,
    }

    with Session(engine) as session:
        try:
            inv_rows = session.execute(
                text(
                    "SELECT DISTINCT d.inventory_id "
                    "FROM page2document p2d "
                    "JOIN document d ON d.id = p2d.document_id"
                )
            ).all()

            inventory_ids = [r[0] for r in inv_rows]

            total_inventories = len(inventory_ids)

            for idx_inv, inv_id in enumerate(inventory_ids, start=1):
                logger.info(f"Processing inventory {idx_inv}/{total_inventories} (ID={inv_id})...")
                page_rows = session.execute(
                    text(
                        "SELECT p.id, p.recto_verso "
                        "FROM page p "
                        "LEFT JOIN scan s ON s.id = p.scan_id "
                        "WHERE p.inventory_id = :inv_id "
                        "ORDER BY COALESCE(s.filename, ''), "
                        "         CASE p.recto_verso "
                        "             WHEN 'Recto' THEN 0 "
                        "             WHEN 'Verso' THEN 1 "
                        "             ELSE 2 END"
                    ),
                    {"inv_id": inv_id},
                ).all()

                ordered_page_ids: List[str] = [r[0] for r in page_rows]

                link_rows = session.execute(
                    text(
                        "SELECT p2d.page_id, p2d.document_id \
                        FROM page2document p2d \
                        JOIN document d ON d.id = p2d.document_id \
                        WHERE d.inventory_id = :inv_id \
                        AND p2d.confidence IN ('DEFINITIVE', 'FOLIO_RANGE')"
                    ),
                    {"inv_id": inv_id},
                ).all()

                page_to_docs: Dict[str, Set[str]] = {}
                for page_id, doc_id in link_rows:
                    page_to_docs.setdefault(page_id, set()).add(doc_id)

                base_page_to_docs = {pid: docs.copy() for pid, docs in page_to_docs.items()}

                for depth_iter in range(propagation_depth):
                    logger.info(f"  Pass {depth_iter + 1}/{propagation_depth}...")
                    new_links_this_round = False

                    unlinked_indices = [
                        i for i, pid in enumerate(ordered_page_ids)
                        if pid not in page_to_docs
                    ]

                    total_unlinked = len(unlinked_indices)

                    for idx_i, idx in enumerate(unlinked_indices, start=1):
                        if idx_i % 500 == 0:
                            logger.info(f"    Processed {idx_i}/{total_unlinked} unlinked pages...")
                        page_id = ordered_page_ids[idx]

                        source_map = page_to_docs if propagation_depth > 1 else base_page_to_docs

                        before_doc = _find_neighbour_doc(idx, ordered_page_ids, source_map, -1)
                        after_doc = _find_neighbour_doc(idx, ordered_page_ids, source_map, +1)

                        if before_doc is None or after_doc is None:
                            stats["pages_skipped_no_neighbour"] += 1
                            continue

                        if before_doc != after_doc:
                            stats["pages_skipped_boundary"] += 1
                            continue

                        doc_id = before_doc

                        page_to_docs.setdefault(page_id, set()).add(doc_id)

                        session.execute(insert(Page2Document), {
                            "id": str(uuid.uuid4()),
                            "page_id": page_id,
                            "document_id": doc_id,
                            "index": 0,
                            "source": SOURCE,
                            "confidence": LinkConfidence.INTERPOLATED.value,
                        })

                        stats["pages_interpolated"] += 1
                        new_links_this_round = True

                    session.commit()

                    if not new_links_this_round:
                        break

                stats["inventories_processed"] += 1

        except SQLAlchemyError as e:
            session.rollback()
            raise

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=DATABASE_URL)
    parser.add_argument(
        "--propagation-depth",
        type=int,
        default=1,
        help="How many interpolation passes to run (default: 1, safe mode)",
    )
    args = parser.parse_args()

    results = interpolate(args.database, args.propagation_depth)

    print("\n=== Summary ===")
    print(f"  Inventories processed : {results['inventories_processed']}")
    print(f"  Pages interpolated    : {results['pages_interpolated']:,}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
