#!/usr/bin/env python3
"""
Replace the titles taken from the OBP CSV's DESCRIPTION (which contain line
breaks) with those of the "- no linebreaks.xlsx" version of the same file.

Applies to every document linked to an OBP_INDEX id — the "TANAP Digitized
Index" documents of script 7 and the "Segmentation model" documents that
copied their title from them — but only where the title is the CSV's
DESCRIPTION for that id up to whitespace, '¬' and hyphens (so also earlier
versions of the xlsx), so titles set otherwise are left alone. Safe to rerun.
"""

import argparse
import logging
import os
import re

import pandas as pd
from sqlalchemy import create_engine, text

from toc_sources import load_obp_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(
    SCRIPT_DIR,
    "data",
    "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799) - no linebreaks.xlsx",
)


def text_key(s: pd.Series) -> pd.Series:
    """The text without whitespace, line-break markers and hyphens."""
    return s.astype(str).map(lambda t: re.sub(r"[\s¬-]", "", t))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = parser.parse_args()

    logger.info("Reading %s …", XLSX_PATH)
    new = pd.read_excel(XLSX_PATH, usecols=["ID", "DESCRIPTION"]).set_index("ID")["DESCRIPTION"]
    old = load_obp_csv().set_index("ID")["DESCRIPTION"]

    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        docs = pd.read_sql(
            text(
                "SELECT d.id, d.title, m.name AS method, CAST(e.identifier AS INTEGER) AS csv_id FROM document d "
                "JOIN document_identification_method m ON m.id = d.method_id "
                "JOIN document2external_id de ON de.document_id = d.id "
                "JOIN external_id e ON e.id = de.external_id AND e.context = 'OBP_INDEX'"
            ),
            conn,
        )
        docs["old"] = docs["csv_id"].map(old)
        docs["new"] = docs["csv_id"].map(new)
        same = docs["title"].notna() & docs["old"].notna() & (text_key(docs["title"]) == text_key(docs["old"]))
        todo = docs[same & docs["new"].notna() & (docs["title"] != docs["new"])]
        kept = docs[~same & (docs["title"] != docs["new"])]

        logger.info("%d documents linked to an OBP_INDEX id; %d titles to replace:", len(docs), len(todo))
        for method, n in todo["method"].value_counts().items():
            logger.info("  %-30s %d", method, n)
        if len(kept):
            logger.info("%d titles differ from the CSV and are left alone", len(kept))

        if args.dry_run:
            logger.info("Dry run: nothing written.")
            return
        conn.execute(
            text("UPDATE document SET title = :title WHERE id = :id"),
            [{"id": r.id, "title": r.new} for r in todo.itertuples()],
        )
        logger.info("Updated %d titles.", len(todo))


if __name__ == "__main__":
    main()
