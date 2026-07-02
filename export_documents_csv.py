"""
Export documents to CSV with identifier, inventory number, scan filenames,
title, date, settlement, method, and start/end scan types.

Export documents to a CSV file.

options:
  -h, --help            show this help message and exit
  --filename, -f FILENAME
                        Output filename (default: data/s3/document/documents.csv)
  --gzip                Gzip-compress the output file (default)
  --no-gzip             Write plain CSV without gzip compression

"""

import os
import csv
import argparse
import gzip
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base, Document

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


def get_settlement_label(document):
    """Get the settlement label for a document, or empty string if none."""
    if document.location:
        if document.location.labels:
            return document.location.labels[0].label
        return document.location.glob_id
    return ""


def get_inventory_number(document):
    """Get the inventory number for a document."""
    if document.inventory:
        return document.inventory.inventory_number
    return ""


def get_start_scan_filename(document):
    """Get the filename of the first scan for a document."""
    if not document.pages:
        return ""

    # Sort Page2Document entries by index to get the first one
    sorted_pages = sorted(document.pages, key=lambda p: p.index)
    first_page_link = sorted_pages[0]

    if first_page_link.page and first_page_link.page.scan:
        return first_page_link.page.scan.filename
    return ""


def get_end_scan_filename(document):
    """Get the filename of the last scan for a document."""
    if not document.pages:
        return ""

    # Sort Page2Document entries by index to get the last one
    sorted_pages = sorted(document.pages, key=lambda p: p.index)
    last_page_link = sorted_pages[-1]

    if last_page_link.page and last_page_link.page.scan:
        return last_page_link.page.scan.filename
    return ""


def get_start_end_scan_types(document):
    """Get the scan types of the first and last scans for a document."""
    if not document.pages:
        return "", ""

    sorted_pages = sorted(document.pages, key=lambda p: p.index)
    first_page_link = sorted_pages[0]
    last_page_link = sorted_pages[-1]

    start_scan_type = ""
    end_scan_type = ""

    if first_page_link.page and first_page_link.page.scan:
        scan_type = getattr(first_page_link.page.scan, "scan_type", None)
        if scan_type:
            start_scan_type = (
                scan_type.value if hasattr(scan_type, "value") else str(scan_type)
            )

    if last_page_link.page and last_page_link.page.scan:
        scan_type = getattr(last_page_link.page.scan, "scan_type", None)
        if scan_type:
            end_scan_type = (
                scan_type.value if hasattr(scan_type, "value") else str(scan_type)
            )

    return start_scan_type, end_scan_type


def get_date_start(document):
    """Get the start date in ISO 8601 format."""
    if document.date_earliest_begin:
        return document.date_earliest_begin.isoformat()
    return ""


def get_date_end(document):
    """Get the end date in ISO 8601 format."""
    if document.date_latest_end:
        return document.date_latest_end.isoformat()
    return ""


def get_identification_method(document):
    """Get the identification method name."""
    if document.method:
        return document.method.name
    return ""


def export_documents_csv(
    output_file="data/s3/document/documents.csv", gzip_output=True
):
    """Export all documents to CSV.

    Args:
        output_file: Output filename for the CSV export.
        gzip_output: When True, gzip-compress the output file.
    """
    with Session(engine) as session:
        # Query all documents
        documents = session.query(Document).all()

        logger.info(f"Found {len(documents)} documents to export")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        open_fn = gzip.open if gzip_output else open
        with open_fn(output_file, "wt", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "identifier",
                    "inventory_number",
                    "start_scan_filename",
                    "end_scan_filename",
                    "start_scan_type",
                    "end_scan_type",
                    "title",
                    "date_start",
                    "date_end",
                    "settlement",
                    "method",
                ]
            )

            # Write data rows
            for document in documents:
                start_scan_type, end_scan_type = get_start_end_scan_types(document)
                writer.writerow(
                    [
                        document.id,
                        get_inventory_number(document),
                        get_start_scan_filename(document),
                        get_end_scan_filename(document),
                        start_scan_type,
                        end_scan_type,
                        document.title or "",
                        get_date_start(document),
                        get_date_end(document),
                        get_settlement_label(document),
                        get_identification_method(document),
                    ]
                )

        if gzip_output:
            logger.info(
                f"Exported {len(documents)} documents to gzip file {output_file}"
            )
            logger.info(
                "Upload with: aws s3 sync data/s3/ s3://globalise-data/objects/document/ --acl=public-read --content-encoding gzip"
            )

        else:
            logger.info(f"Exported {len(documents)} documents to {output_file}")
            logger.info(
                "Upload with: aws s3 sync data/s3/ s3://globalise-data/objects/document/ --acl=public-read"
            )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export documents to a CSV file.")
    parser.add_argument(
        "--filename",
        "-f",
        default="data/s3/document/documents.csv",
        help="Output filename (default: data/s3/document/documents.csv)",
    )
    parser.add_argument(
        "--gzip",
        dest="gzip",
        action="store_true",
        help="Gzip-compress the output file (default)",
    )
    parser.add_argument(
        "--no-gzip",
        dest="gzip",
        action="store_false",
        help="Write plain CSV without gzip compression",
    )
    parser.set_defaults(gzip=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_documents_csv(output_file=args.filename, gzip_output=args.gzip)
