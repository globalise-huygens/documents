#!/usr/bin/env python3
"""
Migrate page2document.confidence from FLOAT to the LinkConfidence enum (VARCHAR).
Step 9.5 in the import sequence — run once, between step 9 and step 10.

Strategy (works for both SQLite and PostgreSQL):
  1. Audit: report all distinct float values currently in the column.
  2. Add a new column  confidence_new  (TEXT / VARCHAR).
  3. Backfill using the mapping table; unmapped values → CANDIDATE (logged).
  4. Drop the old column, rename the new one.
  5. For PostgreSQL: additionally add a CHECK constraint matching the enum values.

SQLite does not support ALTER COLUMN TYPE, so the add/backfill/swap approach is
used for both backends to keep the script uniform.
"""

import os
import sys
import logging
import argparse

from sqlalchemy import create_engine, text, inspect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")

# Known float → enum-string mappings.
# Add extra entries here if any other scripts wrote non-standard values.
FLOAT_TO_ENUM: dict[str, str] = {
    "1.0":  "DEFINITIVE",
    "0.8":  "FOLIO_RANGE",
}
FALLBACK_CONFIDENCE = "CANDIDATE"

VALID_ENUM_VALUES = {"VALIDATED", "DEFINITIVE", "FOLIO_RANGE", "INTERPOLATED", "CANDIDATE"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_postgres(engine) -> bool:
    return engine.dialect.name == "postgresql"


def _is_sqlite(engine) -> bool:
    return engine.dialect.name == "sqlite"


def _audit(conn) -> dict[str, int]:
    """Return {str(confidence_value): count} for all distinct values."""
    rows = conn.execute(
        text("SELECT CAST(confidence AS TEXT), COUNT(*) FROM page2document GROUP BY 1 ORDER BY 2 DESC")
    ).all()
    return {r[0]: r[1] for r in rows}


def _column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(
        text(f"SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = :col"),
        {"col": column},
    )
    return result.scalar() > 0


def _column_exists_pg(conn, table: str, column: str) -> bool:
    result = conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = :tbl AND column_name = :col"
        ),
        {"tbl": table, "col": column},
    )
    return result.scalar() > 0


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------

def migrate(engine, dry_run: bool = False) -> dict[str, int]:
    is_pg = _is_postgres(engine)
    stats = {"rows_backfilled": 0, "rows_fallback": 0, "already_migrated": 0}

    with engine.begin() as conn:
        # ------------------------------------------------------------------ #
        # 1. Audit current state                                               #
        # ------------------------------------------------------------------ #
        current_values = _audit(conn)
        logger.info("Current distinct confidence values in page2document:")
        for val, cnt in current_values.items():
            label = FLOAT_TO_ENUM.get(val) or ("already enum" if val in VALID_ENUM_VALUES else "→ CANDIDATE (fallback)")
            logger.info(f"  {val!r:>12}  ×{cnt:>10,}  {label}")

        # Check if column is already migrated (all values are valid enum strings)
        all_migrated = all(v in VALID_ENUM_VALUES for v in current_values)
        if all_migrated:
            logger.info("Column appears to already contain enum string values. Nothing to do.")
            stats["already_migrated"] = sum(current_values.values())
            return stats

        if dry_run:
            logger.info("[DRY RUN] No changes written.")
            return stats

        # ------------------------------------------------------------------ #
        # 2. Add confidence_new column                                         #
        # ------------------------------------------------------------------ #
        logger.info("Adding temporary column 'confidence_new'...")
        if is_pg:
            col_exists = _column_exists_pg(conn, "page2document", "confidence_new")
        else:
            col_exists = _column_exists(conn, "page2document", "confidence_new")

        if not col_exists:
            conn.execute(text("ALTER TABLE page2document ADD COLUMN confidence_new TEXT"))

        # ------------------------------------------------------------------ #
        # 3. Backfill known values                                             #
        # ------------------------------------------------------------------ #
        for float_val, enum_val in FLOAT_TO_ENUM.items():
            # CAST to REAL for robust float comparison across backends
            result = conn.execute(
                text(
                    "UPDATE page2document "
                    "SET confidence_new = :enum_val "
                    "WHERE confidence_new IS NULL "
                    "  AND CAST(confidence AS REAL) = CAST(:float_val AS REAL)"
                ),
                {"enum_val": enum_val, "float_val": float(float_val)},
            )
            n = result.rowcount
            stats["rows_backfilled"] += n
            if n:
                logger.info(f"  {float_val!r} → '{enum_val}': {n:,} rows")

        # ------------------------------------------------------------------ #
        # 4. Fallback: anything still NULL                                     #
        # ------------------------------------------------------------------ #
        result = conn.execute(
            text(
                "UPDATE page2document "
                "SET confidence_new = :fallback "
                "WHERE confidence_new IS NULL"
            ),
            {"fallback": FALLBACK_CONFIDENCE},
        )
        n = result.rowcount
        stats["rows_fallback"] = n
        if n:
            logger.warning(
                f"  {n:,} rows had unrecognised confidence values → '{FALLBACK_CONFIDENCE}'. "
                "Review these manually and update to the correct tier."
            )

        # ------------------------------------------------------------------ #
        # 5. Swap columns                                                      #
        # ------------------------------------------------------------------ #
        logger.info("Swapping columns...")

        if is_pg:
            # PostgreSQL supports ALTER COLUMN … TYPE directly.
            conn.execute(text("ALTER TABLE page2document DROP COLUMN confidence"))
            conn.execute(text("ALTER TABLE page2document RENAME COLUMN confidence_new TO confidence"))
            conn.execute(text("ALTER TABLE page2document ALTER COLUMN confidence SET NOT NULL"))
            conn.execute(text("ALTER TABLE page2document ALTER COLUMN confidence SET DEFAULT 'DEFINITIVE'"))
            # Add CHECK constraint
            valid_list = ", ".join(f"'{v}'" for v in sorted(VALID_ENUM_VALUES))
            conn.execute(
                text(
                    f"ALTER TABLE page2document "
                    f"ADD CONSTRAINT ck_page2document_confidence "
                    f"CHECK (confidence IN ({valid_list}))"
                )
            )
        else:
            # SQLite: no DROP/RENAME support before 3.35 — use table-rebuild via
            # simple column drop + rename (supported since SQLite 3.35 / 2021-03-12).
            # If your SQLite is older, see the commented block below.
            conn.execute(text("ALTER TABLE page2document DROP COLUMN confidence"))
            conn.execute(
                text("ALTER TABLE page2document RENAME COLUMN confidence_new TO confidence")
            )

            # --- Fallback for SQLite < 3.35 (uncomment if needed) ---
            # conn.execute(text("""
            #     CREATE TABLE page2document_new AS
            #     SELECT id, page_id, document_id, "index", source, confidence_new AS confidence
            #     FROM page2document
            # """))
            # conn.execute(text("DROP TABLE page2document"))
            # conn.execute(text("ALTER TABLE page2document_new RENAME TO page2document"))
            # ---------------------------------------------------------

        total = stats["rows_backfilled"] + stats["rows_fallback"]
        logger.info(f"Migration complete. {total:,} rows updated.")

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate page2document.confidence from FLOAT to LinkConfidence enum."
    )
    parser.add_argument("--database", default=DATABASE_URL, help="SQLAlchemy database URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit only — print what would change without writing anything",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GLOBALISE — Confidence column migration  (step 9.5)")
    print("=" * 60)

    if args.dry_run:
        print("*** DRY RUN — no changes will be written ***\n")

    engine = create_engine(args.database, echo=False)

    if _is_sqlite(engine):
        logger.info("Backend: SQLite")
    elif _is_postgres(engine):
        logger.info("Backend: PostgreSQL")
    else:
        logger.warning(f"Untested backend: {engine.dialect.name}. Proceeding anyway.")

    results = migrate(engine, dry_run=args.dry_run)

    print("\n=== Summary ===")
    if results["already_migrated"]:
        print(f"  Already migrated      : {results['already_migrated']:,} rows (no-op)")
    else:
        print(f"  Rows backfilled       : {results['rows_backfilled']:,}")
        print(f"  Fallback (CANDIDATE)  : {results['rows_fallback']:,}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
