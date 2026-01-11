#!/usr/bin/env python3
"""
Import page metadata into the SQLAlchemy database.
Reads data/page_metadata.csv and updates Scan.scan_type and Page records.
"""
import os
import sys
import logging
import uuid
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import Base, Page, PageType, RectoVerso

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
engine = create_engine(DATABASE_URL, echo=False)

# Ensure tables exist
Base.metadata.create_all(engine)


def map_scan_type(value: str):
    if not value:
        return None
    v = str(value).strip().lower()
    if v == "single":
        return PageType.SINGLE
    if v == "double":
        return PageType.DOUBLE
    return PageType.OTHER


def map_scan_type_str(value):
    """Return the DB enum NAME string (SINGLE/DOUBLE/OTHER) or None to match SQLAlchemy Enum storage."""
    st = map_scan_type(value)
    return st.name if st else None


def read_pages_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    csv_path_1 = os.path.join(data_dir, "page_metadata.csv")
    csv_path_2 = os.path.join(
        data_dir, "page_metadata_new_inventories.csv"
    )  # <-- new inventories
    if not os.path.exists(csv_path_1):
        logger.error(f"Missing CSV file: {csv_path_1}")
        sys.exit(1)
    if not os.path.exists(csv_path_2):
        logger.error(f"Missing CSV file: {csv_path_2}")
        sys.exit(1)
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.where(pd.notnull(df), None)
    return df


def main():
    df = read_pages_csv()
    session = Session(engine)

    try:
        # Prepare target sets from CSV
        rows = df.to_dict(orient="records")
        filenames = {
            str(r.get("doc_id") or "").strip() for r in rows if r.get("doc_id")
        }
        inv_numbers = {
            str(r.get("inventory") or "").strip() for r in rows if r.get("inventory")
        }

        # Preload scans (id and current scan_type) only for relevant filenames
        logger.info(f"Preloading {len(filenames)} scans by filename...")
        scans = {}
        fn_list = list(filenames)
        chunk = 900
        for i in range(0, len(fn_list), chunk):
            subset = fn_list[i : i + chunk]
            placeholders = ",".join([f":p{j}" for j in range(len(subset))])
            params = {f"p{j}": subset[j] for j in range(len(subset))}
            sql = text(
                f"SELECT filename, id, scan_type FROM scan WHERE filename IN ({placeholders})"
            )
            for filename, scan_id, scan_type in session.execute(sql, params).all():
                scans[filename] = {"id": scan_id, "scan_type": scan_type}

        # Preload inventories (id by inventory_number) for relevant numbers
        logger.info(f"Preloading {len(inv_numbers)} inventories by inventory_number...")
        inventories = {}
        inv_list = list(inv_numbers)
        for i in range(0, len(inv_list), chunk):
            subset = inv_list[i : i + chunk]
            placeholders = ",".join([f":p{j}" for j in range(len(subset))])
            params = {f"p{j}": subset[j] for j in range(len(subset))}
            sql = text(
                f"SELECT inventory_number, id FROM inventory WHERE inventory_number IN ({placeholders})"
            )
            for inv_num, inv_id in session.execute(sql, params).all():
                inventories[inv_num] = inv_id

        # Bulk update scan.scan_type where needed
        logger.info("Preparing scan_type updates...")
        updates = []
        for r in rows:
            filename = str(r.get("doc_id") or "").strip()
            if not filename:
                continue
            scan = scans.get(filename)
            if not scan:
                logger.warning(
                    f"Scan not found for filename '{filename}', skipping row"
                )
                continue
            new_type = map_scan_type_str(r.get("scan_type"))
            if new_type and scan.get("scan_type") != new_type:
                updates.append({"scan_type": new_type, "id": scan["id"]})

        # Execute updates in batches
        scan_type_updates = 0
        if updates:
            for i in range(0, len(updates), 20000):
                batch = updates[i : i + 20000]
                session.execute(
                    text("UPDATE scan SET scan_type=:scan_type WHERE id=:id"), batch
                )
                scan_type_updates += len(batch)
            session.commit()

        # Build page inserts (assume fresh DB for pages: no updates, only inserts)
        logger.info("Preparing page inserts (fresh DB assumed)...")
        page_rows = []
        for r in rows:
            filename = str(r.get("doc_id") or "").strip()
            if not filename:
                continue
            scan = scans.get(filename)
            if not scan:
                continue
            scan_id = scan["id"]
            inv_number = str(r.get("inventory") or "").strip()
            inv_id = inventories.get(inv_number) if inv_number else None
            headers = r.get("headers")
            signature_marks = r.get("signature_marks")
            page_numbers = r.get("page_numbers")
            has_marginalia = r.get("has_marginalia")
            is_blank = r.get("is_blank")
            new_type = map_scan_type_str(r.get("scan_type"))

            if new_type == PageType.DOUBLE.name:
                # Verso
                page_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "page_or_folio_number": page_numbers,
                        "recto_verso": RectoVerso.VERSO.name,
                        "header": headers,
                        "inventory_id": inv_id,
                        "scan_id": scan_id,
                        "detected_languages": None,
                        "rotation": 0,
                        "signatures": signature_marks,
                        "has_marginalia": has_marginalia,
                        "has_table": None,
                        "has_illustration": None,
                        "has_print": None,
                        "is_blank": is_blank,
                    }
                )
                # Recto
                page_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "page_or_folio_number": page_numbers,
                        "recto_verso": RectoVerso.RECTO.name,
                        "header": headers,
                        "inventory_id": inv_id,
                        "scan_id": scan_id,
                        "detected_languages": None,
                        "rotation": 0,
                        "signatures": signature_marks,
                        "has_marginalia": has_marginalia,
                        "has_table": None,
                        "has_illustration": None,
                        "has_print": None,
                        "is_blank": is_blank,
                    }
                )
            else:
                page_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "page_or_folio_number": page_numbers,
                        "recto_verso": None,
                        "header": headers,
                        "inventory_id": inv_id,
                        "scan_id": scan_id,
                        "detected_languages": None,
                        "rotation": 0,
                        "signatures": signature_marks,
                        "has_marginalia": has_marginalia,
                        "has_table": None,
                        "has_illustration": None,
                        "has_print": None,
                        "is_blank": is_blank,
                    }
                )

        # Insert pages in batches using Core executemany
        create_count = 0
        for i in range(0, len(page_rows), 20000):
            batch = page_rows[i : i + 20000]
            if batch:
                session.execute(Page.__table__.insert(), batch)
                create_count += len(batch)
        session.commit()

        logger.info("Page import completed")
        logger.info(f"Scan type updates: {scan_type_updates}")
        logger.info(f"Pages created: {create_count}")
        logger.info("Pages updated: 0 (fresh insert)")

    except Exception as e:
        logger.exception(f"Error during page import: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
