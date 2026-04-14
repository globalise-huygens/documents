#!/usr/bin/env python3
"""
Match pages to OBP documents by folio range.
Step 10 in the import sequence.

Assigns confidence tier FOLIO_RANGE to every page whose folio number falls
inside a document's folio_start–folio_end range.

Pages that are unmatched here (no folio or folio outside all ranges) are
picked up by script 11, which uses neighbour interpolation.
"""

import os
import uuid
import logging
import argparse
from typing import Optional, List, Dict, Set, Tuple, Any
import re

from sqlalchemy import create_engine, text, insert
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from models import Base, Page2Document, LinkConfidence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
SOURCE = "FOLIO_RANGE"
BATCH_SIZE = 5_000

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
    
    Rejects any part that is not a clean integer.
    """
    if not raw:
        return []

    results = []
    for part in raw.split(","):
        part = part.strip()

        # Only accept pure digits (no signs, no letters, no noise)
        if re.fullmatch(r"\d+", part):
            results.append(int(part))
        else:
            # Optional: log rejected values for debugging
            # logger.debug(f"Rejected folio value: '{part}'")
            continue

    return results


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def match_folios(database_url: str) -> Dict[str, int]:
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "inventories_processed": 0,
        "rows_inserted": 0,
        "already_linked_skipped": 0,
    }

    with Session(engine) as session:
        try:
            # ---------------------------------------------------------------- #
            # 1. Identify inventories that have documents with folio data       #
            # ---------------------------------------------------------------- #
            inv_rows = session.execute(
                text("SELECT DISTINCT inventory_id FROM document WHERE folio_start IS NOT NULL")
            ).all()

            if not inv_rows:
                logger.warning("No documents with folio_start found. Run script 7 first.")
                return stats

            inventory_ids = [r[0] for r in inv_rows]
            logger.info(f"Processing {len(inventory_ids)} inventories...")

            for inv_id in inventory_ids:
                # ------------------------------------------------------------ #
                # 2. Load pages for this inventory                              #
                # ------------------------------------------------------------ #
                page_rows = session.execute(
                    text(
                        "SELECT id, page_or_folio_number FROM page "
                        "WHERE inventory_id = :inv_id AND page_or_folio_number IS NOT NULL"
                    ),
                    {"inv_id": inv_id},
                ).all()

                if not page_rows:
                    continue

                # Index: folio_number → [page_id, ...]
                folio_to_pages: Dict[int, List[str]] = {}
                for p_id, p_str in page_rows:
                    for num in parse_folio_numbers(p_str):
                        folio_to_pages.setdefault(num, []).append(p_id)

                # ------------------------------------------------------------ #
                # 3. Load documents with folio ranges                          #
                # ------------------------------------------------------------ #
                doc_rows = session.execute(
                    text(
                        "SELECT id, folio_start, folio_end FROM document "
                        "WHERE inventory_id = :inv_id AND folio_start IS NOT NULL"
                    ),
                    {"inv_id": inv_id},
                ).all()

                # ------------------------------------------------------------ #
                # 4. Load existing links (all methods) to prevent duplicates   #
                # ------------------------------------------------------------ #
                existing_links: Set[Tuple[str, str]] = set(
                    session.execute(
                        text(
                            "SELECT page_id, document_id FROM page2document "
                            "JOIN document ON document.id = page2document.document_id "
                            "WHERE document.inventory_id = :inv_id"
                        ),
                        {"inv_id": inv_id},
                    ).all()
                )

                # ------------------------------------------------------------ #
                # 5. Build new rows                                             #
                # ------------------------------------------------------------ #
                inv_new_rows: List[Dict[str, Any]] = []

                for doc_id, f_start, f_end in doc_rows:
                    actual_end = (
                        f_end if (f_end is not None and f_end >= f_start) else f_start
                    )

                    for folio_num in range(f_start, actual_end + 1):
                        for page_id in folio_to_pages.get(folio_num, []):
                            if (page_id, doc_id) in existing_links:
                                stats["already_linked_skipped"] += 1
                                continue

                            inv_new_rows.append({
                                "id": str(uuid.uuid4()),
                                "page_id": page_id,
                                "document_id": doc_id,
                                "index": folio_num,
                                "source": SOURCE,
                                "confidence": LinkConfidence.FOLIO_RANGE.value,
                            })
                            # Track to prevent in-batch duplicates when a page
                            # has multiple folio numbers that fall in the same range.
                            existing_links.add((page_id, doc_id))

                # ------------------------------------------------------------ #
                # 6. Bulk insert in batches, commit per inventory              #
                # ------------------------------------------------------------ #
                if inv_new_rows:
                    for i in range(0, len(inv_new_rows), BATCH_SIZE):
                        batch = inv_new_rows[i : i + BATCH_SIZE]
                        session.execute(insert(Page2Document), batch)

                    session.commit()
                    stats["rows_inserted"] += len(inv_new_rows)
                    stats["inventories_processed"] += 1
                    logger.info(
                        f"  Inventory {inv_id}: linked {len(inv_new_rows)} pages "
                        f"(confidence={LinkConfidence.FOLIO_RANGE.value})"
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
        description="Match pages to documents by folio range (step 10)."
    )
    parser.add_argument("--database", default=DATABASE_URL)
    args = parser.parse_args()

    print("=" * 60)
    print("GLOBALISE — Folio Range Matching  (step 10)")
    print("=" * 60)

    results = match_folios(args.database)

    print("\n=== Summary ===")
    print(f"  Inventories processed : {results['inventories_processed']}")
    print(f"  New links created     : {results['rows_inserted']:,}")
    print(f"  Duplicates skipped    : {results['already_linked_skipped']:,}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
