"""
Put TANAP index documents in the inventory that OBP NT_gecorrigeerd.xlsx
assigns them to.

The OBP CSV (script 7) stores inventory numbers as integers, so documents in
suffixed inventories (9014A, 1430A, 1457C, …) end up under the base number.
Two things go wrong because of that:
  - the base number does not exist in the database (e.g. 9014), so script 7
    silently skipped the document;
  - the base number does exist (e.g. 1430 next to 1430A), so the document was
    imported into the wrong inventory.

NT_gecorrigeerd has the corrected inventory numbers. For every NT document
matched to a CSV id (see toc_sources.match_nt_to_csv) whose NT inventory
differs from the CSV's and exists in the database, this script:
  - creates the document (same rows as script 7 would) if it is missing, or
  - moves the existing document to the NT inventory and deletes its
    page2document links to pages of other inventories (only links weaker than
    DEFINITIVE; those were computed against the wrong inventory's pages).

NT inventories that do not exist in the database are left alone (the CSV's
inventory is kept). The CSV id stays the document's identifier.

Page links for created/moved documents are not recomputed here.

Run after script 7; safe to rerun.
"""

import argparse
import importlib
import logging
import os
from collections import Counter

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import toc_sources

obp = importlib.import_module("7_import_obp_index")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")


def main(db_url: str, dry_run: bool = False):
    csv = toc_sources.load_obp_csv()
    nt = toc_sources.match_nt_to_csv(toc_sources.load_nt(), csv)

    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        inventories = dict(
            session.execute(text("SELECT inventory_number, id FROM inventory")).all()
        )
        method_id = obp.get_or_create_method(session)

        pairs = nt.dropna(subset=["csv_id"]).astype({"csv_id": int}).merge(
            csv, on="csv_id"
        )
        pairs = pairs[pairs["nt_inventory"] != pairs["inventory_number"]]
        in_db = pairs["nt_inventory"].isin(inventories)
        logger.info(
            "%d documents have a different inventory in NT; %d of those NT "
            "inventories exist in the database. Left alone (NT inventory not in DB): %s",
            len(pairs),
            in_db.sum(),
            dict(Counter(pairs.loc[~in_db, "nt_inventory"])),
        )
        pairs = pairs[in_db]

        existing = {
            int(identifier): (doc_id, inv_number)
            for identifier, doc_id, inv_number in session.execute(
                text(
                    "SELECT e.identifier, d.id, i.inventory_number "
                    "FROM external_id e "
                    "JOIN document2external_id de ON de.external_id = e.id "
                    "JOIN document d ON d.id = de.document_id "
                    "JOIN inventory i ON i.id = d.inventory_id "
                    "WHERE e.context = 'OBP_INDEX' AND d.method_id = :mid"
                ),
                {"mid": method_id},
            ).all()
        }

        known_type_ids = obp.preload_document_type_ids(session)
        settlement_labels = obp.preload_settlement_labels(session)
        acc = obp.new_accumulator()
        moves: list[dict] = []
        skipped_placeholder = 0
        created = Counter()

        for _, row in pairs.iterrows():
            target_id = inventories[row["nt_inventory"]]
            hit = existing.get(row["csv_id"])
            if hit is None:
                if obp.is_placeholder(row.get("DOCUMENT TYPE URI (TANAP)")) or obp.is_placeholder(
                    row.get("DOCUMENT TYPE URI (GLOBALISE)")
                ):
                    skipped_placeholder += 1  # script 7 skips these too
                    continue
                obp.build_document_rows(
                    row, target_id, method_id, known_type_ids, settlement_labels, acc
                )
                created[row["nt_inventory"]] += 1
            elif hit[1] != row["nt_inventory"]:
                moves.append(
                    {"doc_id": hit[0], "inv_id": target_id, "from": hit[1], "to": row["nt_inventory"]}
                )

        logger.info(
            "To create: %d documents %s (skipped %d with placeholder types)",
            len(acc["doc_rows"]),
            dict(created),
            skipped_placeholder,
        )
        logger.info(
            "To move: %d documents %s",
            len(moves),
            dict(Counter(f"{m['from']}→{m['to']}" for m in moves)),
        )

        if dry_run:
            logger.info("Dry run — nothing written")
            return

        obp.bulk_insert(session, obp.Document.__table__, acc["doc_rows"], "documents")
        obp.bulk_insert(
            session, obp.Document2DocumentType.__table__, acc["doc_type_rows"], "document-type links"
        )
        obp.bulk_insert(session, obp.ExternalID.__table__, acc["ext_id_rows"], "external IDs")
        obp.bulk_insert(
            session,
            obp.Document2ExternalID.__table__,
            acc["doc_ext_id_rows"],
            "document ↔ external ID links",
        )

        removed = 0
        for m in moves:
            session.execute(
                text("UPDATE document SET inventory_id = :inv_id WHERE id = :doc_id"), m
            )
            removed += session.execute(
                text(
                    "DELETE FROM page2document WHERE id IN ("
                    "  SELECT p2d.id FROM page2document p2d "
                    "  JOIN page p ON p.id = p2d.page_id "
                    "  WHERE p2d.document_id = :doc_id "
                    "  AND p2d.confidence NOT IN ('VALIDATED', 'DEFINITIVE') "
                    "  AND p.inventory_id IS NOT :inv_id)"
                ),
                m,
            ).rowcount
        session.commit()
        logger.info(
            "Moved %d documents; removed %d page links to pages of their old inventory",
            len(moves),
            removed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DATABASE_URL, help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()
    main(args.db, dry_run=args.dry_run)
