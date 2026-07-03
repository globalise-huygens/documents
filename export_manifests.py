"""
Export IIIF Manifests for all inventories in the database.

Generates gzipped JSON files ready for upload to the object store:
    aws s3 sync objects/ s3://globalise-data/objects --acl=public-read --content-encoding gzip
"""

import gzip
import json
import os
import sys
import time

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, selectinload

from models import Base, Inventory, Document, Scan, Page, Series
from export import inventory_to_manifest_jsonld

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
OUTPUT_DIR = os.environ.get("MANIFEST_OUTPUT_DIR", "data/s3/objects/inventory")
BASE_URI = "https://data.globalise.huygens.knaw.nl/hdl:20.500.14722"


def export_all_manifests():
    engine = create_engine(DATABASE_URL, echo=False)
    session = Session(engine)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = session.query(func.count(Inventory.id)).scalar()
    print(f"Exporting manifests for {total} inventories to {OUTPUT_DIR}/")

    inventories = session.query(Inventory).all()

    t0 = time.time()
    for i, inventory in enumerate(inventories, 1):
        inv_num = inventory.inventory_number
        manifest_uri = f"{BASE_URI}/inventory:{inv_num}.manifest"
        manifest = inventory_to_manifest_jsonld(inventory, manifest_uri)

        out_path = os.path.join(OUTPUT_DIR, f"{inv_num}.manifest.json")
        json_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

        with gzip.open(out_path, "wb") as f:
            f.write(json_bytes)

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone. {total} manifests written to {OUTPUT_DIR}/ in {elapsed:.1f}s")
    print("\nUpload with:")
    print(
        "  aws s3 sync objects/inventory/ s3://globalise-data/objects/inventory "
        "--acl=public-read --content-encoding gzip"
    )

    session.close()


if __name__ == "__main__":
    export_all_manifests()
