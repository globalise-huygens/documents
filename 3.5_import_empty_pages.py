#!/usr/bin/env python3
"""
Set Page.is_blank using normalized transcription lengths from parquet.

This script reads a parquet file with at least:
- filename
- normalized_text

For each scan filename, it computes character length of normalized_text and sets
is_blank for all linked pages:
- True  when char_length < threshold
- False when char_length >= threshold

Default threshold is 20 characters.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")


def resolve_parquet_path(user_path: str | None) -> Path:
    """Resolve parquet path from CLI or common defaults."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Parquet file not found: {p}")

    candidates = [
        Path("data/normalized_texts.parquet"),
        Path("normalized_texts.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find normalized text parquet. Tried: data/normalized_texts.parquet, normalized_texts.parquet"
    )


def _chunked(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _prepare_temp_table(session: Session) -> None:
    session.execute(text("DROP TABLE IF EXISTS tmp_blank_flags"))
    session.execute(text("""
            CREATE TEMPORARY TABLE tmp_blank_flags (
                filename TEXT PRIMARY KEY,
                is_blank BOOLEAN NOT NULL
            )
            """))


def _insert_flags_batch(session: Session, rows: list[dict], chunk_size: int) -> int:
    inserted = 0
    insert_sql = text(
        "INSERT INTO tmp_blank_flags (filename, is_blank) VALUES (:filename, :is_blank)"
    )
    for batch in _chunked(rows, chunk_size):
        session.execute(insert_sql, batch)
        inserted += len(batch)
    return inserted


def _load_flags_with_duckdb(
    session: Session,
    parquet_path: Path,
    threshold: int,
    chunk_size: int,
) -> int:
    import duckdb

    con = duckdb.connect()
    try:
        cols = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet_path)]
            ).fetchall()
        }
        required = {"filename", "normalized_text"}
        missing = required - cols
        if missing:
            raise ValueError(
                f"Parquet is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(cols)}"
            )

        query = """
            SELECT
                trim(filename) AS filename,
                CAST(MAX(length(COALESCE(normalized_text, ''))) < ? AS BOOLEAN) AS is_blank
            FROM read_parquet(?)
            WHERE filename IS NOT NULL
              AND trim(filename) <> ''
            GROUP BY trim(filename)
        """

        reader = con.execute(query, [threshold, str(parquet_path)]).to_arrow_reader(
            batch_size=chunk_size
        )

        inserted = 0
        for batch in reader:
            data = batch.to_pydict()
            filenames = data.get("filename", [])
            flags = data.get("is_blank", [])
            rows = [
                {"filename": fn, "is_blank": bool(flag)}
                for fn, flag in zip(filenames, flags)
                if fn
            ]
            if rows:
                inserted += _insert_flags_batch(session, rows, chunk_size)

        return inserted
    finally:
        con.close()


def _load_flags_with_pandas(
    session: Session,
    parquet_path: Path,
    threshold: int,
    chunk_size: int,
) -> int:
    # Fallback path when DuckDB is unavailable.
    df = pd.read_parquet(parquet_path, columns=["filename", "normalized_text"])
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df[df["filename"] != ""]
    df["char_length"] = df["normalized_text"].fillna("").astype(str).str.len()

    # Collapse duplicates while preserving a conservative blank/non-blank decision.
    agg = df.groupby("filename", sort=False, as_index=False)["char_length"].max()
    agg["is_blank"] = agg["char_length"] < threshold

    rows = [
        {"filename": filename, "is_blank": bool(is_blank)}
        for filename, is_blank in agg[["filename", "is_blank"]].itertuples(
            index=False, name=None
        )
    ]
    return _insert_flags_batch(session, rows, chunk_size)


def load_flags_into_temp_table(
    session: Session,
    parquet_path: Path,
    threshold: int,
    chunk_size: int,
) -> tuple[int, str]:
    _prepare_temp_table(session)

    try:
        inserted = _load_flags_with_duckdb(session, parquet_path, threshold, chunk_size)
        return inserted, "duckdb"
    except Exception as exc:
        print(f"DuckDB fast path unavailable, falling back to pandas: {exc}")
        inserted = _load_flags_with_pandas(session, parquet_path, threshold, chunk_size)
        return inserted, "pandas"


def apply_blank_flags(session: Session) -> tuple[int, int, int]:
    """Apply blank flags to pages via a single joined SQL update."""
    matched_scans = int(session.execute(text("""
                SELECT COUNT(*)
                FROM scan s
                INNER JOIN tmp_blank_flags t ON t.filename = s.filename
                """)).scalar_one())

    unmatched_filenames = int(session.execute(text("""
                SELECT COUNT(*)
                FROM tmp_blank_flags t
                LEFT JOIN scan s ON s.filename = t.filename
                WHERE s.id IS NULL
                """)).scalar_one())

    result = session.execute(text("""
            UPDATE page
            SET is_blank = (
                SELECT t.is_blank
                FROM scan s
                INNER JOIN tmp_blank_flags t ON t.filename = s.filename
                WHERE s.id = page.scan_id
            )
            WHERE EXISTS (
                SELECT 1
                FROM scan s
                INNER JOIN tmp_blank_flags t ON t.filename = s.filename
                WHERE s.id = page.scan_id
            )
            """))

    updated_pages = int(getattr(result, "rowcount", 0) or 0)
    session.commit()
    return matched_scans, unmatched_filenames, updated_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set page.is_blank from normalized text parquet"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to parquet file (default: auto-detect)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Character threshold for blank pages (default: 20)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=DATABASE_URL,
        help="Database URL (default: env DATABASE_URL or sqlite:///globalise_documents.db)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Batch size for temp-table inserts and DuckDB streaming (default: 100000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.threshold < 0:
        print("Threshold must be >= 0")
        return 1

    try:
        parquet_path = resolve_parquet_path(args.parquet)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    print(f"Using parquet: {parquet_path}")
    print(f"Using threshold: {args.threshold} characters")

    engine = create_engine(args.database_url, echo=False)
    session = Session(engine)
    try:
        prepared_flags, backend = load_flags_into_temp_table(
            session,
            parquet_path,
            args.threshold,
            args.chunk_size,
        )
        print(f"Prepared {prepared_flags} filename flags using {backend}")

        matched_scans, unmatched_filenames, updated_pages = apply_blank_flags(session)
    except Exception as exc:
        session.rollback()
        print(f"Failed to import blank flags: {exc}")
        return 1
    finally:
        session.close()

    print(f"Matched scans: {matched_scans}")
    print(f"Unmatched filenames: {unmatched_filenames}")
    print(f"Updated pages: {updated_pages}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
