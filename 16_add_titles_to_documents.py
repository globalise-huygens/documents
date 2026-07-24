#!/usr/bin/env python3
"""
Set titles for baseline documents from the first header found on their pages.

Rule:
- For each document created by the baseline method, use the first non-empty
  Page.header in page sequence order (Page2Document.index) as Document.title.

By default, only empty titles are filled. Use --overwrite to replace existing
 titles as well.
"""

import argparse
import ast
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
BASELINE_METHOD_NAME = "Baseline: Empty Pages & Signatures"


def normalize_header_value(raw_header: str | None) -> str | None:
    """Normalize stored page header value to a plain title string.

    Header values may be stored as Python-list strings, e.g.
    "['foo', 'bar']". In that case we concatenate non-empty items.
    """
    if raw_header is None:
        return None

    value = str(raw_header).strip()
    if not value:
        return None

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = value

    if isinstance(parsed, (list, tuple)):
        parts = [str(item).strip() for item in parsed if str(item).strip()]
        return " ".join(parts) if parts else None

    if isinstance(parsed, str):
        parsed = parsed.strip()
        return parsed or None

    converted = str(parsed).strip()
    return converted or None


def add_titles_to_baseline_documents(
    database_url: str,
    method_name: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    engine = create_engine(database_url, echo=False)

    stats = {
        "baseline_documents": 0,
        "documents_with_header_candidate": 0,
        "eligible_documents": 0,
        "updated_documents": 0,
    }

    with Session(engine) as session:
        baseline_documents = session.execute(
            text("""
                SELECT COUNT(*)
                FROM document d
                JOIN document_identification_method m ON m.id = d.method_id
                WHERE m.name = :method_name
                """),
            {"method_name": method_name},
        ).scalar_one()
        stats["baseline_documents"] = int(baseline_documents)

        if stats["baseline_documents"] == 0:
            if not dry_run:
                session.commit()
            return stats

        session.execute(text("DROP TABLE IF EXISTS tmp_first_headers"))
        session.execute(
            text("""
                CREATE TEMPORARY TABLE tmp_first_headers AS
                WITH ranked AS (
                    SELECT
                        d.id AS document_id,
                        trim(p.header) AS header,
                        row_number() OVER (
                            PARTITION BY d.id
                            ORDER BY p2d."index", s.filename, p.id
                        ) AS rn
                    FROM document d
                    JOIN document_identification_method m ON m.id = d.method_id
                    JOIN page2document p2d ON p2d.document_id = d.id
                    JOIN page p ON p.id = p2d.page_id
                    LEFT JOIN scan s ON s.id = p.scan_id
                    WHERE m.name = :method_name
                      AND p.header IS NOT NULL
                      AND trim(p.header) <> ''
                )
                SELECT document_id, header
                FROM ranked
                WHERE rn = 1
                """),
            {"method_name": method_name},
        )
        session.execute(
            text(
                "CREATE INDEX idx_tmp_first_headers_document_id ON tmp_first_headers(document_id)"
            )
        )

        header_rows = session.execute(
            text("SELECT document_id, header FROM tmp_first_headers")
        ).all()

        parsed_rows = []
        for document_id, raw_header in header_rows:
            normalized_title = normalize_header_value(raw_header)
            if normalized_title:
                parsed_rows.append(
                    {"document_id": document_id, "title": normalized_title}
                )

        session.execute(text("DROP TABLE IF EXISTS tmp_first_titles"))
        session.execute(text("""
                CREATE TEMPORARY TABLE tmp_first_titles (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                )
                """))
        if parsed_rows:
            for i in range(0, len(parsed_rows), 50_000):
                session.execute(
                    text("""
                        INSERT INTO tmp_first_titles (document_id, title)
                        VALUES (:document_id, :title)
                        """),
                    parsed_rows[i : i + 50_000],
                )

        docs_with_header_candidate = session.execute(
            text("SELECT COUNT(*) FROM tmp_first_titles")
        ).scalar_one()
        stats["documents_with_header_candidate"] = int(docs_with_header_candidate)

        eligibility_condition = (
            "" if overwrite else "AND (d.title IS NULL OR trim(d.title) = '')"
        )

        eligible_documents = session.execute(
            text(f"""
                SELECT COUNT(*)
                FROM document d
                JOIN tmp_first_titles fh ON fh.document_id = d.id
                WHERE 1 = 1
                {eligibility_condition}
                """),
        ).scalar_one()
        stats["eligible_documents"] = int(eligible_documents)

        if dry_run or stats["eligible_documents"] == 0:
            session.rollback()
            return stats

        update_result = session.execute(
            text(f"""
                UPDATE document AS d
                SET title = (
                    SELECT fh.title
                    FROM tmp_first_titles fh
                    WHERE fh.document_id = d.id
                )
                WHERE d.id IN (SELECT document_id FROM tmp_first_titles)
                {eligibility_condition}
                """),
        )

        stats["updated_documents"] = int(getattr(update_result, "rowcount", 0) or 0)
        session.commit()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set baseline document titles from first sequential page header"
    )
    parser.add_argument(
        "--database",
        default=DATABASE_URL,
        help="Database URL (default: env DATABASE_URL or sqlite:///globalise_documents.db)",
    )
    parser.add_argument(
        "--method-name",
        default=BASELINE_METHOD_NAME,
        help=f"DocumentIdentificationMethod.name to target (default: {BASELINE_METHOD_NAME})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty document titles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts without writing updates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("GLOBALISE — Add Baseline Document Titles (step 16)")
    print("=" * 60)
    print(f"Method name: {args.method_name}")
    print(f"Overwrite existing titles: {args.overwrite}")
    print(f"Dry run: {args.dry_run}")

    stats = add_titles_to_baseline_documents(
        database_url=args.database,
        method_name=args.method_name,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print("\n=== Summary ===")
    print(f"  Baseline documents             : {stats['baseline_documents']:,}")
    print(
        "  Documents with header candidate : "
        f"{stats['documents_with_header_candidate']:,}"
    )
    print(f"  Eligible documents             : {stats['eligible_documents']:,}")
    print(f"  Updated documents              : {stats['updated_documents']:,}")

    if args.dry_run:
        print("\nDry run complete (no changes written).")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
