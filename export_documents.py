"""
Export individual CuratedHolding and HumanMadeObject JSON-LD records.

These are the objects currently returned directly from models by `inventory_to_jsonld`
and `document_physical_to_jsonld` and embedded inside `rdfs:seeAlso` links.
This script serializes these individual entities to disk.

Output paths:
  data/s3/objects/document/<uuid>.json
  data/s3/objects/inventory/<inventory_number>.json
"""

import gzip
import json
import os
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, selectinload

from models import Inventory, Document, Series
from export import document_physical_to_jsonld, inventory_to_jsonld, series_to_jsonld

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
OUTPUT_DIR = os.environ.get("DOCUMENTS_OUTPUT_DIR", "data/s3/objects")


def natural_inv_sort_key(inv):
    """Sort inventory numbers naturally: numeric prefix then alphabetic suffix."""
    s = inv.inventory_number or ""
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    num = int(s[:i]) if i > 0 else 0
    suffix = s[i:].upper()
    return (num, suffix)


def export_documents():
    engine = create_engine(DATABASE_URL, echo=False)
    session = Session(engine)

    doc_dir = os.path.join(OUTPUT_DIR, "document")
    inv_dir = os.path.join(OUTPUT_DIR, "inventory")
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(inv_dir, exist_ok=True)

    print("Loading inventories...")
    inventories = session.query(Inventory).all()
    inventories.sort(key=natural_inv_sort_key)
    print(f"Loaded {len(inventories)} inventories.")

    t0 = time.time()

    # 1. Export Inventories
    for i, inventory in enumerate(inventories, 1):
        inv_data = inventory_to_jsonld(inventory)

        # Override the seeAlso embedding and subject_of if necessary (this logic
        # has been moved to export.py, but here we just directly serialize the JSON-LD).
        out_path = os.path.join(inv_dir, f"{inventory.inventory_number}.json")
        json_bytes = json.dumps(inv_data, ensure_ascii=False, indent=2).encode("utf-8")

        with gzip.open(out_path, "wb") as f:
            f.write(json_bytes)

        if i % 100 == 0:
            print(f"  Exported {i} inventory objects...")

    elapsed_inv = time.time() - t0
    print(f"Exported {len(inventories)} inventories in {elapsed_inv:.1f}s.")

    # 1b. Export the Series (Sets) into the same file system, resolving subsets and curated holdings.
    print("Loading Series (Sets)...")
    series_all = (
        session.query(Series)
        .options(selectinload(Series.sub_series), selectinload(Series.inventories))
        .all()
    )
    print(f"Loaded {len(series_all)} series records.")

    for s in series_all:
        s_data = series_to_jsonld(s)

        # Inject the members (which are sub-series and inventories)
        members = []

        # 1. Sub-sets
        for sub_s in sorted(s.sub_series, key=lambda x: x.title or ""):
            members.append(
                {
                    "id": f"https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/series:{sub_s.id}",
                    "type": "Set",
                    "_label": sub_s.title,
                }
            )

        # 2. CuratedHoldings
        sorted_invs = sorted(s.inventories, key=natural_inv_sort_key)
        for inv in sorted_invs:
            members.append(
                {
                    "id": f"https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/inventory:{inv.inventory_number}",
                    "type": "CuratedHolding",
                    "_label": f"Inventory {inv.inventory_number}",
                }
            )

        if members:
            # We want to declare the type of the members just generally
            # (they may be mixed Set and CuratedHolding)
            # s_data["members_are_type"] = ... (Not strictly required for mixed)
            s_data["member"] = members

        out_path = os.path.join(inv_dir, f"series_{s.id}.json")
        json_bytes = json.dumps(s_data, ensure_ascii=False, indent=2).encode("utf-8")
        with gzip.open(out_path, "wb") as f:
            f.write(json_bytes)

    # 1c. Export the top-level global Set of all top-level series
    print("Generating top level Set metadata...")
    top_level_series = [s for s in series_all if s.part_of_id is None]
    top_level_series.sort(key=lambda x: x.title or "")

    # Recursively build the full embedded structure
    def build_nested_member(s):
        s_data = series_to_jsonld(s)

        # Remove 'member_of' if it exists since we're nesting top-down
        if "member_of" in s_data:
            del s_data["member_of"]

        members = []
        for sub_s in sorted(s.sub_series, key=lambda x: x.title or ""):
            members.append(build_nested_member(sub_s))

        for inv in sorted(s.inventories, key=natural_inv_sort_key):
            members.append(
                {
                    "id": f"https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/inventory:{inv.inventory_number}",
                    "type": "CuratedHolding",
                    "_label": f"Inventory {inv.inventory_number}",
                }
            )

        if members:
            s_data["member"] = members

        return s_data

    set_data = {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": "https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/inventory:set",
        "type": "Set",
        "_label": "Set of the Globalise corpus",
        "identified_by": [
            {
                "type": "Name",
                "classified_as": [
                    {
                        "id": "http://vocab.getty.edu/aat/300404670",
                        "type": "Type",
                        "_label": "Primary Name",
                    }
                ],
                "content": "Set of all VOC inventories that are part of the Globalise corpus",
            }
        ],
        "member": [build_nested_member(s) for s in top_level_series],
        "subject_of": [
            {
                "type": "LinguisticObject",
                "digitally_carried_by": [
                    {
                        "type": "DigitalObject",
                        "access_point": [
                            {
                                "id": "https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/inventory:collection",
                                "type": "DigitalObject",  # Also Manifest?
                                "_label": "IIIF Collection of all inventories in the Globalise corpus",
                            }
                        ],
                        "conforms_to": [
                            {
                                "id": "http://iiif.io/api/presentation/",
                                "type": "InformationObject",
                            }
                        ],
                        "format": "application/ld+json;profile='http://iiif.io/api/presentation/3/context.json'",
                    }
                ],
            },
        ],
    }
    set_out_path = os.path.join(inv_dir, "set.json")
    with gzip.open(set_out_path, "wb") as f:
        f.write(json.dumps(set_data, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Exported global Set object to {set_out_path}")

    # How to save 500K files?
    # 2. Export Documents

    print("Loading documents...")
    # documents = session.query(Document).all()

    # For now, only inventory 1053 and 3598
    documents = (
        session.query(Document)
        .join(Document.inventory)
        .filter(Inventory.inventory_number.in_(["1053", "3598"]))
        .all()
    )

    total_docs = len(documents)
    print(f"Loaded {total_docs} documents.")

    t1 = time.time()
    for i, doc in enumerate(documents, 1):
        doc_data = document_physical_to_jsonld(doc)

        out_path = os.path.join(doc_dir, f"{doc.id}.json")
        json_bytes = json.dumps(doc_data, ensure_ascii=False, indent=2).encode("utf-8")

        with gzip.open(out_path, "wb") as f:
            f.write(json_bytes)

        if i % 1000 == 0:
            print(f"  Exported {i}/{total_docs} document objects...")

    elapsed_doc = time.time() - t1
    print(f"Exported {total_docs} documents in {elapsed_doc:.1f}s.")

    print("\nDone.")
    print("Output available in data/s3/objects/")
    session.close()


if __name__ == "__main__":
    export_documents()
# ensure the formatting is right
