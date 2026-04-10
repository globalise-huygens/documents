"""
Import annotation page availability flags from annotationpages.csv.

Sets has_transcriptions, has_entities, and has_events on Scan rows
based on the CSV produced by transform_annotationspages.py.

CSV columns: filename, transcriptions, entities, events
Values are 0/1 integers.
"""

import argparse
import logging
import os
import time

import pandas as pd
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from models import Base, Scan

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "annotationpages.csv")


def main(csv_path: str, db_url: str, dry_run: bool = False):
    logger.info("Reading %s …", csv_path)
    df = pd.read_csv(csv_path, dtype={"filename": str, "transcriptions": int, "entities": int, "events": int})
    logger.info("Loaded %d rows from CSV", len(df))

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Build a lookup of scan filename -> scan id
        scans = session.query(Scan.id, Scan.filename).all()
        filename_to_id = {s.filename: s.id for s in scans}
        logger.info("Found %d scans in database", len(filename_to_id))

        # Reset all flags to False first
        session.execute(
            update(Scan).values(
                has_transcriptions=False,
                has_entities=False,
                has_events=False,
            )
        )

        updated = 0
        skipped = 0
        t0 = time.time()

        # Batch updates
        batch = []
        for row in df.itertuples(index=False):
            scan_id = filename_to_id.get(row.filename)
            if scan_id is None:
                skipped += 1
                continue

            batch.append({
                "id": scan_id,
                "has_transcriptions": bool(row.transcriptions),
                "has_entities": bool(row.entities),
                "has_events": bool(row.events),
            })
            updated += 1

        if batch:
            session.execute(update(Scan), batch)

        elapsed = time.time() - t0
        logger.info(
            "Updated %d scans, skipped %d filenames not found (%.1fs)",
            updated, skipped, elapsed,
        )

        if dry_run:
            logger.info("Dry run — rolling back")
            session.rollback()
        else:
            session.commit()
            logger.info("Committed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV_PATH, help="Path to annotationpages.csv")
    parser.add_argument("--db", default=DATABASE_URL, help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Do not commit changes")
    args = parser.parse_args()
    main(args.csv, args.db, dry_run=args.dry_run)
