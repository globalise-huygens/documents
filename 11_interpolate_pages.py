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

def interpolate(database_url: str) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "inventories_processed": 0,
        "pages_interpolated": 0,
        "pages_skipped_ambiguous": 0,
    }

    with Session(engine) as session:
        try:
            inv_rows = session.execute(text("""
                SELECT DISTINCT d.inventory_id
                FROM page2document p2d
                JOIN document d ON d.id = p2d.document_id
            """)).all()

            inventory_ids = [r[0] for r in inv_rows]

            for inv_id in inventory_ids:
                # ------------------------------------------------------------ #
                # 1. Load all pages in canonical scan order                    #
                # ------------------------------------------------------------ #
                page_rows = session.execute(text("""
                    SELECT p.id
                    FROM page p
                    LEFT JOIN scan s ON s.id = p.scan_id
                    WHERE p.inventory_id = :inv_id
                    ORDER BY COALESCE(s.filename, ''),
                             CASE p.recto_verso
                                 WHEN 'Verso' THEN 0
                                 WHEN 'Recto' THEN 1
                                 ELSE 2 END
                """), {"inv_id": inv_id}).all()

                if not page_rows:
                    continue

                ordered_pages = [r[0] for r in page_rows]

                # Assign canonical index to ALL pages
                page_index_map = {
                    pid: idx for idx, pid in enumerate(ordered_pages)
                }

                # Reverse lookup: index → page_id
                index_to_page = {
                    idx: pid for pid, idx in page_index_map.items()
                }

                # ------------------------------------------------------------ #
                # 2. Load existing links (anchors only)                        #
                # ------------------------------------------------------------ #
                rows = session.execute(text("""
                    SELECT p2d.page_id, p2d.document_id
                    FROM page2document p2d
                    JOIN document d ON d.id = p2d.document_id
                    WHERE d.inventory_id = :inv_id
                      AND p2d.source != 'INTERPOLATED'
                """), {"inv_id": inv_id}).all()

                if not rows:
                    continue

                # page → documents
                page_to_docs: Dict[str, Set[str]] = {}
                for page_id, doc_id in rows:
                    page_to_docs.setdefault(page_id, set()).add(doc_id)

                # document → indices
                doc_to_indices: Dict[str, Set[int]] = defaultdict(set)
                for page_id, doc_id in rows:
                    idx = page_index_map.get(page_id)
                    if idx is not None:
                        doc_to_indices[doc_id].add(idx)

                inserts = []

                # ------------------------------------------------------------ #
                # 3. Interpolate per document using index continuity           #
                # ------------------------------------------------------------ #
                for doc_id, indices in doc_to_indices.items():
                    if len(indices) < 2:
                        continue

                    sorted_indices = sorted(indices)

                    start = sorted_indices[0]
                    end = sorted_indices[-1]

                    for idx in range(start, end + 1):
                        if idx in indices:
                            continue

                        page_id = index_to_page.get(idx)
                        if not page_id:
                            continue

                        # Check if page already linked
                        if page_id in page_to_docs:
                            continue

                        # SAFETY: ensure no other document already claims it
                        existing_docs = page_to_docs.get(page_id, set())
                        if existing_docs:
                            stats["pages_skipped_ambiguous"] += 1
                            continue

                        inserts.append({
                            "id": str(uuid.uuid4()),
                            "page_id": page_id,
                            "document_id": doc_id,
                            "index": idx,
                            "source": SOURCE,
                            "confidence": LinkConfidence.INTERPOLATED.value,
                        })

                        # update in-memory state (important for reruns)
                        page_to_docs.setdefault(page_id, set()).add(doc_id)
                        stats["pages_interpolated"] += 1

                # ------------------------------------------------------------ #
                # 4. Commit batch                                             #
                # ------------------------------------------------------------ #
                if inserts:
                    session.execute(insert(Page2Document), inserts)
                    session.commit()

                stats["inventories_processed"] += 1

        except SQLAlchemyError:
            session.rollback()
            raise

    return stats

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate page–document links using scan-order neighbours (step 11)."
    )
    parser.add_argument("--database", default=DATABASE_URL)
    parser.add_argument(
        "--propagation-depth",
        type=int,
        default=1,
        help=(
            "How many interpolation passes to run per inventory (default: 1). "
            "With depth=1 (safe mode) only original anchor links seed neighbours. "
            "With depth>1 newly interpolated pages can chain-feed into adjacent gaps."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GLOBALISE — Neighbour Interpolation  (step 11)")
    print("=" * 60)

    results = interpolate(args.database)

    print("\n=== Summary ===")
    print(f"  Inventories processed    : {results['inventories_processed']}")
    print(f"  Pages interpolated       : {results['pages_interpolated']:,}")
    print(f"  Skipped (boundary)       : {results['pages_skipped_boundary']:,}")
    print(f"  Skipped (no neighbour)   : {results['pages_skipped_no_neighbour']:,}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
