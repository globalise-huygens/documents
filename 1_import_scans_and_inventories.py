"""
Import data from CSV and JSON files into SQLAlchemy database.
Converted from Django import.py script.
"""

import os
import sys
import json
import uuid
import pandas as pd
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from models import Base, Inventory, InventoryTitle, Scan

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
engine = create_engine(DATABASE_URL, echo=False)

# Create tables if they don't exist
Base.metadata.create_all(engine)


def process_dates(dates):
    """
    Process date strings into start and end dates.
    Handles various date formats from the inventory data.
    """
    parsed_dates = []

    for date in dates:
        if "/" in date:
            start, end = date.split("/")

            # Normalize start date
            if len(start) == 4:
                start = f"{start}-01-01"
                start_as_date = datetime.strptime(start, "%Y-%m-%d").date()
            elif len(start) == 7:
                start = f"{start}-01"
                start_as_date = datetime.strptime(start, "%Y-%m-%d").date()
            elif len(start) == 8:
                # Handle YYYYMMDD format
                start_as_date = datetime.strptime(start, "%Y%m%d").date()
            else:
                start_as_date = datetime.strptime(start, "%Y-%m-%d").date()

            # Normalize end date
            if len(end) == 4:
                end = f"{end}-12-31"
                end_as_date = datetime.strptime(end, "%Y-%m-%d").date()
            elif len(end) == 7:
                # Add last day of month using datetime and pandas
                end_year, end_month = end.split("-")
                end_dt = datetime(
                    int(end_year), int(end_month), 1
                ) + pd.offsets.MonthEnd(1)
                end_as_date = end_dt.date()
            elif len(end) == 8:
                # Handle YYYYMMDD format
                end_as_date = datetime.strptime(end, "%Y%m%d").date()
            else:
                end_as_date = datetime.strptime(end, "%Y-%m-%d").date()

            parsed_dates.append(start_as_date)
            parsed_dates.append(end_as_date)

        else:
            if len(date) == 4:
                start_as_date = datetime.strptime(f"{date}-01-01", "%Y-%m-%d").date()
                end_as_date = datetime.strptime(f"{date}-12-31", "%Y-%m-%d").date()

                parsed_dates.append(start_as_date)
                parsed_dates.append(end_as_date)
            elif len(date) == 8:
                # Handle YYYYMMDD format (e.g., '17140214')
                start_as_date = datetime.strptime(date, "%Y%m%d").date()
                parsed_dates.append(start_as_date)
            else:
                start_as_date = datetime.strptime(date, "%Y-%m-%d").date()
                parsed_dates.append(start_as_date)

    return min(parsed_dates), max(parsed_dates)


def main():
    """Main import function."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Load CSV data
    logger.info("Loading CSV data...")
    csv_path1 = os.path.join(data_dir, "documents_for_django.csv")
    csv_path2 = os.path.join(data_dir, "documents_for_django_2025.csv")
    df_scans1 = pd.read_csv(csv_path1)
    df_scans2 = pd.read_csv(csv_path2)
    df_scans = pd.concat([df_scans1, df_scans2], ignore_index=True)
    logger.info(f"Loaded {len(df_scans)} scan records from CSV")

    # Load JSON files
    logger.info("Loading JSON files...")
    with open(os.path.join(data_dir, "inventory2dates.json")) as f:
        inventory2dates = json.load(f)

    with open(os.path.join(data_dir, "inventory2handle.json")) as f:
        inventory2handle = json.load(f)

    with open(os.path.join(data_dir, "inventory2titles.json")) as f:
        inventory2titles = json.load(f)

    # Keep track of processed inventories to avoid duplicates (maps inventory_number -> generated id)
    processed_inventories = {}

    # Statistics for reporting
    stats = {
        "inventories_created": 0,
        "titles_created": 0,
        "scans_created": 0,
        "errors": 0,
    }

    logger.info("Starting import process...")

    # First pass: prepare all the data
    inventory_data = {}
    scan_data = []

    # Collect all data first before making database changes
    logger.info("Collecting data from CSV and JSON files...")
    for i, row in df_scans.iterrows():
        # Extract inventory data
        inventory_number = str(row["inventory_number"])
        na_identifier_inventory = (
            row["na_identifier_inventory"]
            if not pd.isna(row["na_identifier_inventory"])
            else None
        )

        # Prepare scan data
        file_name_scan = row["file_name_scan"]
        na_identifier_scan = (
            row["na_identifier_scan"]
            if not pd.isna(row["na_identifier_scan"])
            else None
        )

        # Convert na_identifier_scan to canonical UUID string if present
        na_scan_uuid = None
        if na_identifier_scan:
            try:
                na_scan_uuid = str(uuid.UUID(str(na_identifier_scan)))
            except (ValueError, AttributeError, TypeError):
                logger.warning(
                    f"Invalid UUID for scan {file_name_scan}: {na_identifier_scan}"
                )
                na_scan_uuid = None

        # IIIF info URL (now optional; store None if empty)
        iiif_info_url = (
            row["iiif_info_url"]
            if (
                "iiif_info_url" in row
                and not pd.isna(row["iiif_info_url"])
                and str(row["iiif_info_url"]).strip()
            )
            else None
        )
        width = int(row["width"]) if not pd.isna(row["width"]) else 0
        height = int(row["height"]) if not pd.isna(row["height"]) else 0

        # Only collect inventory data once
        if inventory_number not in inventory_data:
            # Get inventory data from JSON files
            handle = None
            date_start = None
            date_end = None
            titles = []

            if inventory_number in inventory2handle:
                handle = inventory2handle[inventory_number]
            else:
                logger.warning(f"No handle found for inventory {inventory_number}")

            if inventory_number in inventory2dates:
                dates = inventory2dates[inventory_number]
                try:
                    date_start, date_end = process_dates(dates)
                except Exception as e:
                    logger.warning(
                        f"Error processing dates for inventory {inventory_number}: {e}"
                    )
                    date_start = None
                    date_end = None
            else:
                logger.warning(f"No dates found for inventory {inventory_number}")

            if inventory_number in inventory2titles:
                titles = inventory2titles[inventory_number]
            else:
                logger.warning(f"No titles found for inventory {inventory_number}")

            # Convert na_identifier string to canonical UUID string if present
            na_id_uuid = None
            if na_identifier_inventory:
                try:
                    na_id_uuid = str(uuid.UUID(str(na_identifier_inventory)))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(
                        f"Invalid UUID for inventory {inventory_number}: {na_identifier_inventory}"
                    )
                    na_id_uuid = None

            # Store inventory data
            inventory_data[inventory_number] = {
                "na_identifier": na_id_uuid,
                "handle": handle,
                "date_start": date_start,
                "date_end": date_end,
                "titles": titles,
            }

        # Store scan data (even if IIIF info URL is missing)
        scan_data.append(
            {
                "inventory_number": inventory_number,
                "file_name": file_name_scan,
                "na_identifier": na_scan_uuid,
                "iiif_info_url": iiif_info_url,
                "width": width,
                "height": height,
            }
        )

    # Start database session
    logger.info("Starting database operations...")
    session = Session(engine)

    try:
        # Normalize any legacy invalid UUIDs written as numeric types (no-op on fresh DB)
        with engine.begin() as conn:
            conn.execute(text("PRAGMA synchronous = OFF"))
            conn.execute(text("PRAGMA journal_mode = OFF"))
            conn.execute(text("PRAGMA temp_store = MEMORY"))

        # Create inventories and titles using fast bulk inserts
        logger.info(f"Creating {len(inventory_data)} inventories...")
        inventory_rows = []
        title_rows = []
        for inventory_number, inv_data in inventory_data.items():
            inv_id = str(uuid.uuid4())
            processed_inventories[inventory_number] = inv_id
            inventory_rows.append(
                {
                    "id": inv_id,
                    "inventory_number": inventory_number,
                    "na_identifier": inv_data["na_identifier"],
                    "handle": inv_data["handle"],
                    "date_start": inv_data["date_start"],
                    "date_end": inv_data["date_end"],
                }
            )
            stats["inventories_created"] += 1

            for title in inv_data["titles"]:
                title_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "inventory_id": inv_id,
                    }
                )
                stats["titles_created"] += 1

        with engine.begin() as conn:
            if inventory_rows:
                conn.execute(Inventory.__table__.insert(), inventory_rows)
            if title_rows:
                conn.execute(InventoryTitle.__table__.insert(), title_rows)

        # Create scans via fast executemany (assume fresh DB; unique by filename)
        logger.info(
            f"Preparing {len(scan_data)} scans for bulk insert (unique by filename)..."
        )
        seen = set()
        batch = []
        batch_size = 20000
        inserted = 0
        for scan_item in scan_data:
            inventory_number = scan_item["inventory_number"]
            inv_id = processed_inventories.get(inventory_number)
            if not inv_id:
                logger.warning(
                    f"Inventory {inventory_number} not found for scan {scan_item['file_name']}"
                )
                stats["errors"] += 1
                continue
            filename = scan_item["file_name"]
            if filename in seen:
                continue
            seen.add(filename)
            batch.append(
                {
                    "id": str(uuid.uuid4()),
                    "filename": filename,
                    "na_identifier": scan_item["na_identifier"],
                    "iiif_image_info": scan_item["iiif_info_url"],
                    "inventory_id": inv_id,
                    "height": int(scan_item["height"] or 0),
                    "width": int(scan_item["width"] or 0),
                    "scan_type": None,
                }
            )
            if len(batch) >= batch_size:
                with engine.begin() as conn:
                    conn.execute(Scan.__table__.insert(), batch)
                inserted += len(batch)
                batch = []
        if batch:
            with engine.begin() as conn:
                conn.execute(Scan.__table__.insert(), batch)
            inserted += len(batch)
        stats["scans_created"] += inserted

        logger.info("Import process completed successfully!")

    except Exception as e:
        logger.error(f"Error during import: {e}")
        session.rollback()
        stats["errors"] += 1
        raise

    finally:
        session.close()

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("IMPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Inventories created: {stats['inventories_created']}")
    logger.info(f"Inventory titles created: {stats['titles_created']}")
    logger.info(f"Scans created: {stats['scans_created']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Check if data files exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    required_files = [
        "documents_for_django.csv",
        "documents_for_django_2025.csv",  # <-- new inventories
        "inventory2dates.json",
        "inventory2handle.json",
        "inventory2titles.json",
    ]

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    if missing_files:
        logger.error("Missing required data files:")
        for filename in missing_files:
            logger.error(f"  - {filename}")
        logger.error(f"Please ensure all data files are in: {data_dir}")
        sys.exit(1)

    # Confirm before importing
    print("=" * 60)
    print("GLOBALISE Document Archive - Real Data Import")
    print("=" * 60)
    print("\nThis will import data from:")
    print(
        "  - CSV: documents_for_django.csv + documents_for_django_2025.csv (~5M scans)"
    )
    print("  - JSON: inventory metadata files")
    print(f"\nDatabase: {DATABASE_URL}")
    print("\n⚠️  WARNING: This will add data to your existing database.")
    print("   Make sure to backup first if needed!")
    print("=" * 60)

    response = input("\nProceed with import? (yes/no): ")
    if response.lower() != "yes":
        print("Import cancelled.")
        sys.exit(0)

    main()
