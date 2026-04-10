"""
Export a IIIF Collection that references all inventory manifests.

Generates a gzipped JSON file ready for upload to the object store alongside the manifests:
    aws s3 sync objects/ s3://globalise-data/objects --acl=public-read --content-encoding gzip
"""

import gzip
import json
import os
import time

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, selectinload

from models import Inventory, InventoryTitle, Scan

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
OUTPUT_DIR = os.environ.get("MANIFEST_OUTPUT_DIR", "objects")
BASE_URI = "https://data.globalise.huygens.knaw.nl/hdl:20.500.14722"


def natural_inv_sort_key(inv):
    """Sort inventory numbers naturally: numeric prefix then alphabetic suffix."""
    s = inv.inventory_number or ""
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    num = int(s[:i]) if i > 0 else 0
    suffix = s[i:].upper()
    return (num, suffix)


def export_collection():
    engine = create_engine(DATABASE_URL, echo=False)
    session = Session(engine)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    inventories = (
        session.query(Inventory)
        .options(
            selectinload(Inventory.titles),
            selectinload(Inventory.scans),
        )
        .all()
    )
    inventories.sort(key=natural_inv_sort_key)

    print(f"Building IIIF Collection for {len(inventories)} inventories...")

    items = []
    for inventory in inventories:
        inv_num = inventory.inventory_number

        # Build label from titles or fall back to inventory number
        if inventory.titles:
            title_text = "; ".join(t.title for t in inventory.titles if t.title)
            label_text = f"{inv_num} - {title_text}"
        else:
            label_text = f"{inv_num}"

        manifest_ref = {
            "id": f"{BASE_URI}/inventory:{inv_num}.manifest",
            "type": "Manifest",
            "label": {"en": [label_text]},
        }

        # Add navDate if date information is available
        if inventory.date_start:
            manifest_ref["navDate"] = f"{inventory.date_start}T00:00:00+00:00"
        elif inventory.date_end:
            manifest_ref["navDate"] = f"{inventory.date_end}T00:00:00+00:00"

        # Add thumbnail from first scan
        if inventory.scans:
            first_scan = sorted(inventory.scans, key=lambda s: s.filename or "")[0]
            thumb_url = first_scan.get_image_url(size="982,")
            service_id = None
            if getattr(first_scan, "iiif_image_info", False):
                service_id = first_scan.iiif_image_info.replace("/info.json", "")
            if thumb_url and service_id:
                manifest_ref["thumbnail"] = [
                    {
                        "id": thumb_url,
                        "type": "Image",
                        "height": first_scan.height,
                        "width": first_scan.width,
                        "service": [
                            {
                                "@id": service_id,
                                "@type": "ImageService3",
                                "profile": "level2",
                                "format": "image/jpeg",
                            }
                        ],
                        "format": "image/jpeg",
                    }
                ]

        items.append(manifest_ref)

    collection = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": f"{BASE_URI}/inventory:collection",
        "type": "Collection",
        "label": {
            "en": ["GLOBALISE — Dutch East India Company archives (1.04.02)"],
        },
        "summary": {
            "en": [
                "A collection of digitised inventories from the archives of the "
                "Dutch East India Company (VOC), part of the National Archives of "
                "the Netherlands (NL-HaNA), access number 1.04.02."
            ],
        },
        "requiredStatement": {
            "label": {"en": ["Attribution"]},
            "value": {
                "en": [
                    "<span>GLOBALISE Project. "
                    '<a href="https://creativecommons.org/publicdomain/zero/1.0/">'
                    '<img src="https://licensebuttons.net/l/zero/1.0/88x31.png" '
                    'alt="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"/> '
                    "</a> </span>"
                ]
            },
        },
        "rights": "http://creativecommons.org/publicdomain/zero/1.0/",
        "homepage": [
            {
                "id": "https://globalise.huygens.knaw.nl",
                "type": "Text",
                "label": {"en": ["GLOBALISE Project"]},
                "format": "text/html",
            }
        ],
        "provider": [
            {
                "id": "https://globalise.huygens.knaw.nl",
                "type": "Agent",
                "label": {"en": ["GLOBALISE Project"]},
                "homepage": [
                    {
                        "id": "https://globalise.huygens.knaw.nl",
                        "type": "Text",
                        "label": {"en": ["GLOBALISE Project"]},
                        "format": "text/html",
                    }
                ],
                "logo": [
                    {
                        "id": "https://objectstore.surf.nl/87435b768620494e8e911c83d1997f24:globalise-data/static/img/globalise.png",
                        "type": "Image",
                        "height": 182,
                        "width": 1200,
                        "format": "image/png",
                    }
                ],
            }
        ],
        "items": items,
    }

    out_path = os.path.join(OUTPUT_DIR, "inventory", "collection.json")
    json_bytes = json.dumps(collection, ensure_ascii=False, indent=2).encode("utf-8")

    with gzip.open(out_path, "wb") as f:
        f.write(json_bytes)

    print(f"Done. Collection with {len(items)} manifests written to {out_path}")
    print("\nUpload with:")
    print(
        "  aws s3 sync objects/inventory/ s3://globalise-data/objects/inventory "
        "--acl=public-read --content-encoding gzip"
    )

    session.close()


if __name__ == "__main__":
    export_collection()
