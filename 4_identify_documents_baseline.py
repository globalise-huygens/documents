"""
Document identification baseline method using page features.

This script identifies document boundaries based on:
1. Empty pages (is_blank=True): indicate document boundaries
2. Pages with signatures: indicate the end of a document

A new document starts:
- At the beginning of an inventory, after skipping initial empty pages/covers
- After a sequence of empty pages
- After a page with a signature

This creates a baseline document identification method and assigns pages to documents.
"""

import uuid
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import (
    Inventory,
    Page,
    Scan,
    Document,
    DocumentIdentificationMethod,
    Page2Document,
)
import os
from typing import Optional

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")


def create_identification_method(session: Session) -> str:
    """
    Create or retrieve the baseline document identification method.

    Args:
        session: SQLAlchemy session

    Returns:
        Method ID (UUID string)
    """
    # Check if method already exists
    existing_method = (
        session.query(DocumentIdentificationMethod)
        .filter(
            DocumentIdentificationMethod.name == "Baseline: Empty Pages & Signatures"
        )
        .first()
    )

    if existing_method:
        print(f"Using existing method: {existing_method.id}")
        return existing_method.id

    # Create new method
    method = DocumentIdentificationMethod(
        id=str(uuid.uuid4()),
        name="Baseline: Empty Pages & Signatures",
        description=(
            "Identifies documents in early modern archival inventories:\n"
            "1. Skips empty pages (is_blank=True) at inventory start\n"
            "2. First document starts with first non-blank page\n"
            "3. Empty page sequences (is_blank=True) indicate document boundaries\n"
            "4. Pages with signatures indicate document end; new document starts after"
        ),
        date=datetime.now().date(),
    )
    session.add(method)
    session.commit()
    print(f"✓ Created new identification method: {method.id}")
    return method.id


def identify_documents_for_inventory(
    session: Session,
    inventory: Inventory,
    method_id: str,
) -> tuple[int, int]:
    """
    Identify documents for a single inventory using baseline rules.

    For early modern archival documents:
    1. Skip empty pages (is_blank=True) at the beginning of inventory
    2. Start first document with first non-blank page
    3. Empty page sequences indicate document boundaries
    4. Pages with signatures indicate document end

    Args:
        session: SQLAlchemy session
        inventory: Inventory to process
        method_id: Document identification method ID

    Returns:
        Tuple of (documents_created, pages_processed)
    """
    # Get all pages for this inventory, ordered by scan filename and page id
    pages = (
        session.query(Page)
        .filter(Page.inventory_id == inventory.id)
        .join(Scan)
        .order_by(Scan.filename, Page.id)
        .all()
    )

    if not pages:
        return 0, 0

    # Find the first non-blank page
    first_content_idx = None
    for i, page in enumerate(pages):
        if not page.is_blank:
            first_content_idx = i
            break

    if first_content_idx is None:
        # No non-blank pages in this inventory
        return 0, 0

    documents_created = 0
    pages_processed = 0
    current_document = None
    empty_page_sequence = False
    previous_page_had_signature = False
    page_index = 0

    for i in range(first_content_idx, len(pages)):
        page = pages[i]

        # Determine if we should start a new document
        start_new_document = False

        if i == first_content_idx:
            # Start first document with first non-blank page
            start_new_document = True
        elif previous_page_had_signature:
            # Previous page had signature - start new document on next non-blank page
            if not page.is_blank:
                start_new_document = True
                previous_page_had_signature = False
            empty_page_sequence = False
        elif page.is_blank:
            # Mark empty page sequence
            empty_page_sequence = True
        elif empty_page_sequence and not page.is_blank:
            # Non-blank page after empty sequence = new document boundary
            start_new_document = True
            empty_page_sequence = False
        else:
            empty_page_sequence = False

        # Create new document and add page (only for non-blank pages)
        if start_new_document and not page.is_blank:
            current_document = Document(
                id=str(uuid.uuid4()),
                inventory_id=inventory.id,
                method_id=method_id,
            )
            session.add(current_document)
            session.flush()  # Get the ID
            documents_created += 1
            page_index = 0

            page2doc = Page2Document(
                id=str(uuid.uuid4()),
                page_id=page.id,
                document_id=current_document.id,
                index=page_index,
            )
            session.add(page2doc)
            page_index += 1
            pages_processed += 1

            # If page has signature, mark for document boundary on next page
            if page.signatures:
                previous_page_had_signature = True
        elif current_document and not page.is_blank:
            # Add non-blank page to existing document
            page2doc = Page2Document(
                id=str(uuid.uuid4()),
                page_id=page.id,
                document_id=current_document.id,
                index=page_index,
            )
            session.add(page2doc)
            page_index += 1
            pages_processed += 1

            # If page has signature, mark for document boundary on next page
            if page.signatures:
                previous_page_had_signature = True

    session.commit()
    return documents_created, pages_processed


def identify_documents_baseline(
    inventory_id: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Identify documents for one or all inventories using baseline rules.

    Optimized for early modern archival documents:
    1. Skip empty pages (is_blank=True) at the beginning of inventory
    2. First document starts with first non-blank page
    3. Empty page sequences (is_blank=True) indicate document boundaries
    4. Pages with signatures indicate document end; new document starts after

    Args:
        inventory_id: Specific inventory to process, or None for all
        verbose: Print detailed progress information

    Returns:
        Dictionary with statistics
    """
    engine = create_engine(DATABASE_URL, echo=False)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    try:
        # Create identification method
        method_id = create_identification_method(session)

        # Get inventories to process
        query = session.query(Inventory)
        if inventory_id:
            query = query.filter(Inventory.id == inventory_id)

        inventories = query.all()

        if not inventories:
            print("✗ No inventories found to process")
            return {
                "inventories_processed": 0,
                "documents_created": 0,
                "pages_processed": 0,
            }

        if verbose:
            print(f"\n{'='*60}")
            print("Document Identification - Baseline Method")
            print("Using is_blank metric only")
            print(f"{'='*60}")

        total_documents = 0
        total_pages = 0

        for inventory in inventories:
            if verbose:
                print(f"\nProcessing: {inventory.inventory_number}")

            docs_created, pages_processed = identify_documents_for_inventory(
                session, inventory, method_id
            )

            total_documents += docs_created
            total_pages += pages_processed

            if verbose:
                print(f"  ✓ Documents created: {docs_created}")
                print(f"  ✓ Pages processed: {pages_processed}")

        if verbose:
            print(f"\n{'='*60}")
            print("Summary:")
            print(f"  - Inventories processed: {len(inventories)}")
            print(f"  - Total documents created: {total_documents}")
            print(f"  - Total pages processed: {total_pages}")
            print(f"  - Method ID: {method_id}")
            print(f"{'='*60}\n")

        return {
            "inventories_processed": len(inventories),
            "documents_created": total_documents,
            "pages_processed": total_pages,
            "method_id": method_id,
        }

    finally:
        session.close()


if __name__ == "__main__":

    # Optional: process specific inventory
    # inventory_id = "your-inventory-id-here"
    identify_documents_baseline()
