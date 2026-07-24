"""
Import per-scan language predictions from data/language_data_per_scan.parquet.

The parquet file has two columns:
  - scan   → Scan.filename (e.g. NL-HaNA_1.04.02_9524I_0146)
  - langs  → comma-separated ISO 639-3 codes (e.g. "fra,nld"), or the
             literal string "unknown" when no language was detected

This script stores the raw `langs` value on Scan.languages. Rows whose scan
filename has no matching Scan in the database are skipped. A handful of scan
filenames appear more than once in the parquet file; the first occurrence is
kept.
"""

import argparse
import logging
import os
import time

import pandas as pd
from sqlalchemy import create_engine, inspect, text, update
from sqlalchemy.orm import Session

from models import Base, Scan

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
PARQUET_PATH = os.path.join(SCRIPT_DIR, "data", "language_data_per_scan.parquet")


def ensure_languages_column(engine):
    """Add the `languages` column to the `scan` table if it is missing.

    Base.metadata.create_all() only creates tables that don't exist yet, so
    an already-populated `scan` table needs an explicit ALTER TABLE to gain
    the new column.
    """
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("scan")}
    if "languages" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE scan ADD COLUMN languages TEXT"))
        logger.info("Added 'languages' column to scan table")


def main(parquet_path: str, db_url: str, dry_run: bool = False):
    logger.info("Reading %s …", parquet_path)
    df = pd.read_parquet(parquet_path, columns=["scan", "langs"])
    before = len(df)
    df = df.drop_duplicates(subset="scan", keep="first")
    logger.info(
        "Loaded %d rows from parquet (%d duplicate scan rows dropped)",
        len(df),
        before - len(df),
    )

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    ensure_languages_column(engine)

    with Session(engine) as session:
        scans = session.query(Scan.id, Scan.filename).all()
        filename_to_id = {s.filename: s.id for s in scans}
        logger.info("Found %d scans in database", len(filename_to_id))

        updated = 0
        skipped = 0
        t0 = time.time()

        batch = []
        for row in df.itertuples(index=False):
            scan_id = filename_to_id.get(row.scan)
            if scan_id is None:
                skipped += 1
                continue

            batch.append({"id": scan_id, "languages": row.langs})
            updated += 1

        if batch:
            session.execute(update(Scan), batch)

        elapsed = time.time() - t0
        logger.info(
            "Updated %d scans, skipped %d filenames not found (%.1fs)",
            updated,
            skipped,
            elapsed,
        )

        if dry_run:
            logger.info("Dry run — rolling back")
            session.rollback()
        else:
            session.commit()
            logger.info("Committed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", default=PARQUET_PATH, help="Path to language_data_per_scan.parquet"
    )
    parser.add_argument("--db", default=DATABASE_URL, help="Database URL")
    parser.add_argument("--dry-run", action="store_true", help="Do not commit changes")
    args = parser.parse_args()
    main(args.parquet, args.db, dry_run=args.dry_run)
