"""
Export helpers: JSON/JSON-LD serializers for GLOBALISE entities.
Moved out of app.py to keep routes lean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import RectoVerso


def slugify(value: str) -> str:
    """Simple slugify: lowercase, replace spaces/underscores with hyphens, strip non-alnum-hyphen."""
    if not value:
        return "unknown"
    import re

    v = value.lower().strip()
    v = re.sub(r"[\s_]+", "-", v)
    v = re.sub(r"[^a-z0-9\-]", "", v)
    return v or "unknown"


# Helper to serialize a Scan to JSON-LD (preliminary, can be extended)
# Some fields are placeholders or require further mapping


def scan_to_jsonld(scan) -> Dict[str, Any]:
    return {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": f"urn:uuid:{scan.id}",
        "type": "DigitalObject",
        "_label": f"Scan {scan.filename}",
        "classified_as": [
            {
                "id": "http://vocab.getty.edu/aat/300417380",
                "type": "Type",
                "_label": "Digitized image",
            }
        ],
        "identified_by": [
            {
                "type": "Identifier",
                "classified_as": None,  # TODO: map if needed
                "content": scan.na_identifier or "",
            },
            {
                "type": "Identifier",
                "classified_as": None,  # TODO: map if needed
                "content": scan.id,
            },
        ],
        "dimension": [
            {
                "type": "Dimension",
                "classified_as": {
                    "id": "http://vocab.getty.edu/aat/300055644",
                    "type": "Type",
                    "_label": "Height",
                },
                "value": scan.height,
                "unit": {
                    "id": "http://vocab.getty.edu/aat/300266190",
                    "type": "MeasurementUnit",
                    "_label": "pixels",
                },
            },
            {
                "type": "Dimension",
                "classified_as": {
                    "id": "http://vocab.getty.edu/aat/300055647",
                    "type": "Type",
                    "_label": "Width",
                },
                "value": scan.width,
                "unit": {
                    "id": "http://vocab.getty.edu/aat/300266190",
                    "type": "MeasurementUnit",
                    "_label": "pixels",
                },
            },
        ],
        "access_point": {
            "id": scan.get_image_url(size="max") or "",
            "type": "DigitalObject",
        },
        "format": "image/jpeg",
        # 'digitally_carries' and 'digitally_shows' are placeholders for now
        "digitally_carries": None,
        "digitally_shows": None,
    }


# Helper to serialize a Page to JSON-LD (preliminary, can be extended)
# Some fields are placeholders or require further mapping


def page_to_jsonld(page) -> Dict[str, Any]:
    # Compose the id using a placeholder pattern; adjust as needed for your real IDs
    page_id_url = f"https://data.globalise.huygens.nl/hdl:20.500.14722/document/doc1-physical/{page.id}"
    # Determine recto/verso classification id and label
    if page.recto_verso == RectoVerso.RECTO:
        classification_id = "http://vocab.getty.edu/aat/300078817"  # Recto
        recto_verso_label = "Recto"
    elif page.recto_verso == RectoVerso.VERSO:
        classification_id = "http://vocab.getty.edu/aat/300010292"  # Verso
        recto_verso_label = "Verso"
    else:
        classification_id = (
            "http://vocab.getty.edu/aat/300241583"  # Generic part type fallback???
        )
        recto_verso_label = "Page"

    # Shallow reference to the associated Scan (if present)
    scan_ref = None
    if page.scan is not None:
        scan_ref = {
            "id": f"urn:uuid:{page.scan.id}",
            "type": "DigitalObject",
            "_label": f"Scan {page.scan.filename}",
        }

    return {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": page_id_url,
        "type": "PhysicalHumanMadeThing",
        "_label": f"Page {page.page_or_folio_number or page.id[:8]}",
        "classified_as": [
            {
                "id": classification_id,
                "type": "Type",
                "_label": recto_verso_label,
                "classified_as": [
                    {
                        "id": "http://vocab.getty.edu/aat/300241583",
                        "type": "Type",
                        "_label": "Part Type",
                    }
                ],
            }
        ],
        "carries": {
            "id": None,  # TODO: supply LinguisticObject id when available
            "type": "LinguisticObject",
            "_label": "Textual content of the page",
            "digitally_carried_by": {
                "id": "",
                "type": "DigitalObject",
                "_label": f"PageXML of {page.scan.filename}",
            },
        },
        "shows": {
            "id": None,  # TODO: supply VisualItem id when available
            "type": "VisualItem",
            "_label": "Visual depiction of the page.",
            "digitally_shown_by": scan_ref,
        },
    }


# Physical Document JSON-LD (material manifestation of a conceptual document)


def document_physical_to_jsonld(document) -> Dict[str, Any]:
    base_id = f"https://data.globalise.huygens.nl/hdl:20.500.14722/document/{document.id}-physical"
    # Classification from first document type if available
    if document.document_types:
        doc_type_str = document.document_types[0].document_type
        doc_type_slug = slugify(doc_type_str)
        classified = {
            "id": f"https://data.globalise.huygens.nl/hdl:20.500.14722/concept/documenttype/{doc_type_slug}",
            "type": "Type",
            "_label": f"{doc_type_str} (document type)",
        }
    else:
        classified = {
            "id": "https://data.globalise.huygens.nl/hdl:20.500.14722/concept/documenttype/unknown",
            "type": "Type",
            "_label": "Unknown document type",
        }

    # Title object
    title_obj = None
    if getattr(document, "title", None):
        title_obj = {
            "type": "Title",
            "content": document.title,
            "_label": "Title of the document",
        }

    # Timespan
    timespan = None
    if (getattr(document, "date_earliest_begin", None) is not None) or (
        getattr(document, "date_latest_end", None) is not None
    ):
        timespan = {
            "type": "Timespan",
            "begin_of_the_begin": (
                str(document.date_earliest_begin)
                if getattr(document, "date_earliest_begin", None) is not None
                else None
            ),
            "end_of_the_end": (
                str(document.date_latest_end)
                if getattr(document, "date_latest_end", None) is not None
                else None
            ),
        }
        if getattr(document, "date_text", None):
            timespan["referred_to_by"] = {
                "id": "",
                "type": "LinguisticObject",
                "content": document.date_text,
                "_label": "Original date expression",
            }  # type: ignore[assignment]

    # Parts: physical pages (recto/verso) from Page2Document
    parts: List[Dict[str, Any]] = []
    if document.pages:
        # Order by index
        sorted_page_links = sorted(document.pages, key=lambda p: p.index)
        for link in sorted_page_links:
            pg = link.page
            label: Optional[str]
            if pg.page_or_folio_number and pg.recto_verso:
                suffix = "r" if pg.recto_verso == RectoVerso.RECTO else "v"
                label = f"Fol. {pg.page_or_folio_number}{suffix}"
            elif pg.page_or_folio_number:
                label = f"Page {pg.page_or_folio_number}"
            else:
                label = f"Physical Page {pg.id[:8]}"
            parts.append(
                {
                    "id": f"https://data.globalise.huygens.nl/hdl:20.500.14722/document/{document.id}-physical/page/{pg.id}",
                    "type": "PhysicalHumanMadeThing",
                    "_label": label,
                }
            )

    subject_of = {
        "id": f"https://globalise.huygens.knaw.nl/document/{document.id}",
        "type": "DigitalObject",
        "_label": "Digital representation of this document",
    }

    return {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": base_id,
        "type": "PhysicalHumanMadeThing",
        "_label": f"Document {document.id}",
        "classified_as": [classified],
        "title": title_obj,
        "produced_by": {
            "type": "Production",
            "classified_as": None,  # placeholder
            "took_place_at": None,  # no place data
            "timespan": timespan,
            "carried_out_by": None,  # unknown actor
        },
        "part": parts,
        "carries": {
            "id": None,
            "type": "LinguisticObject",
            "_label": "Textual content of the document",
            "digitally_carried_by": None,
        },
        "subject_of": subject_of,
    }


# Series (Set) JSON-LD


def series_to_jsonld(series) -> Dict[str, Any]:
    series_id = f"https://data.globalise.huygens.nl/hdl:20.500.14722/series/{series.id}"

    # Determine classification based on hierarchy level
    # If it has a parent, it's likely a sub-grouping; otherwise a top-level grouping
    if getattr(series, "part_of_id", None) is not None:
        classified_as = {
            "id": "http://vocab.getty.edu/aat/300404023",
            "type": "Type",
            "_label": "Archival SubGrouping",
        }
    else:
        classified_as = {
            "id": "http://vocab.getty.edu/aat/300404022",
            "type": "Type",
            "_label": "Archival Grouping",
        }

    # Identified by Name
    identified_by = [
        {
            "type": "Name",
            "classified_as": [
                {
                    "id": "http://vocab.getty.edu/aat/300404670",
                    "type": "Type",
                    "_label": "Primary Name",
                }
            ],
            "content": series.title,
        }
    ]

    # Member of (parent Series if exists)
    member_of = None
    if getattr(series, "part_of", None) is not None:
        parent = series.part_of
        member_of = [
            {
                "id": f"https://data.globalise.huygens.nl/hdl:20.500.14722/series/{parent.id}",
                "type": "Set",
                "_label": parent.title,
            }
        ]

    result: Dict[str, Any] = {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": series_id,
        "type": "Set",
        "_label": series.title,
        "classified_as": [classified_as],
        "identified_by": identified_by,
    }

    if member_of:
        result["member_of"] = member_of

    return result


# Inventory (CuratedHolding) JSON-LD


def inventory_to_jsonld(inventory) -> Dict[str, Any]:
    inv_id = f"https://data.globalise.huygens.nl/hdl:20.500.14722/inventory/{inventory.inventory_number}"

    # Title from earliest/latest dates or first title record
    title_content: Optional[str] = None
    if (getattr(inventory, "date_start", None) is not None) and (
        getattr(inventory, "date_end", None) is not None
    ):
        title_content = f"{inventory.date_start} - {inventory.date_end}"
    elif getattr(inventory, "date_start", None) is not None:
        title_content = f"From {inventory.date_start}"
    elif getattr(inventory, "date_end", None) is not None:
        title_content = f"Until {inventory.date_end}"
    if not title_content and getattr(inventory, "titles", None):
        if inventory.titles:
            title_content = inventory.titles[0].title

    title_obj = None
    if title_content:
        title_obj = {"type": "Title", "content": title_content}

    # Timespan
    timespan = None
    if (getattr(inventory, "date_start", None) is not None) or (
        getattr(inventory, "date_end", None) is not None
    ):
        timespan = {
            "type": "Timespan",
            "begin_of_the_begin": (
                str(inventory.date_start)
                if getattr(inventory, "date_start", None) is not None
                else None
            ),
            "end_of_the_end": (
                str(inventory.date_end)
                if getattr(inventory, "date_end", None) is not None
                else None
            ),
        }

    produced_by = {
        "type": "Production",
        # "classified_as": None,
        "took_place_at": {
            # "id": None,
            "type": "Place",
            "_label": "Place from our thesaurus",
        },
        "timespan": timespan,
        "carried_out_by": {
            # "id": None,
            "type": "Actor",
            "_label": "Polity or Person",
        },
    }

    # Parts: physical documents within inventory (conceptual documents -> physical id pattern)
    parts: List[Dict[str, Any]] = []
    if getattr(inventory, "documents", None):
        for doc in inventory.documents[:100]:  # limit to avoid huge payloads
            parts.append(
                {
                    "id": f"https://data.globalise.huygens.nl/hdl:20.500.14722/document/{doc.id}-physical",
                    "type": "PhysicalHumanMadeThing",
                    "_label": f"Physical Document {doc.id[:8]}",
                }
            )

    equivalent = inventory.handle if getattr(inventory, "handle", None) else None

    # Build member_of from real series relationships with nested parent chain per series
    def series_chain(s) -> Dict[str, Any]:
        # Build list from leaf to root
        chain = []
        current = s
        while current is not None:
            chain.append(current)
            current = current.part_of

        # Build nested structure from root to leaf (so root is last/outermost)
        node: Optional[Dict[str, Any]] = None
        for series_obj in reversed(chain):
            current_node: Dict[str, Any] = {
                # "id": f"https://data.globalise.huygens.nl/hdl:20.500.14722/series/{series_obj.id}",
                "type": "Set",
                "_label": series_obj.title,
            }
            if node is not None:
                current_node["member_of"] = node
            node = current_node

        assert node is not None
        return node

    member_of_list: List[Dict[str, Any]] = []
    if getattr(inventory, "member_of_series", None):
        for s in inventory.member_of_series:

            member_of_list.append(series_chain(s))

    result: Dict[str, Any] = {
        "@context": "https://linked.art/ns/v1/linked-art.json",
        "id": inv_id,
        "type": "CuratedHolding",
        "_label": f"Inventory {inventory.inventory_number}",
        # Classified_as as a single object (per provided example)
        "classified_as": [
            {
                "id": "http://vocab.getty.edu/aat/300027046",  # File unit
                "type": "Type",
                "_label": "File unit",
            }
        ],
        "identified_by": [
            {
                "type": "Identifier",
                "classified_as": [
                    {
                        "id": "http://vocab.getty.edu/aat/300312355",
                        "type": "Type",
                        "_label": "Accession number",
                    }
                ],
                "content": inventory.inventory_number,
            },
            {
                "type": "Identifier",
                "classified_as": [
                    {
                        "id": "http://vocab.getty.edu/aat/300445023",
                        "type": "Type",
                        "_label": "Entry number",
                    }
                ],
                "content": inventory.id,
            },
        ],
        "title": title_obj,
        "produced_by": produced_by,
        "member_of": member_of_list,
        "part": parts,
        "equivalent": equivalent,
        # IIIF
        "subject_of": [
            {
                "type": "LinguisticObject",
                "digitally_carried_by": [
                    {
                        "type": "DigitalObject",
                        "access_point": [
                            # {
                            #     "id": manifest_uri,
                            #     "type": "DigitalObject",  # Also Manifest?
                            #     "_label": f"IIIF Manifest for Inventory {inventory.inventory_number}",
                            # }
                            inventory_to_manifest_jsonld(
                                inventory,
                                manifest_uri=f"https://data.globalise.huygens.nl/iiif/inventory/{inventory.inventory_number}/manifest",
                            )
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
            }
        ],
    }

    return result


def inventory_to_manifest_jsonld(inventory, manifest_uri: str) -> Dict[str, Any]:
    """
    Generate a IIIF Presentation 3.0 Manifest for the Inventory
    """

    manifest: Dict[str, Any] = {
        "@context": [
            "http://iiif.io/api/extension/navplace/context.json",
            "http://iiif.io/api/presentation/3/context.json",
        ],
        "id": manifest_uri,
        "type": "Manifest",
        "label": {"en": [f"Inventory {inventory.inventory_number}"]},
        "requiredStatement": {
            "label": {"en": ["Attribution"]},
            "value": {
                "en": [
                    '<span>GLOBALISE Project. <a href="https://creativecommons.org/publicdomain/zero/1.0/"> <img src="https://licensebuttons.net/l/zero/1.0/88x31.png" alt="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"/> </a> </span>'
                ]
            },
        },
        "rights": "http://creativecommons.org/publicdomain/zero/1.0/",
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
                        "id": "https://globalise-huygens.github.io/document-view-sandbox/globalise.png",
                        "type": "Image",
                        "height": 182,
                        "width": 1200,
                        "format": "image/png",
                    }
                ],
            }
        ],
        "items": [],
        "seeAlso": f"https://data.globalise.huygens.nl/hdl:20.500.14722/inventory/{inventory.inventory_number}",
    }

    # Add navDate if inventory has date information
    if getattr(inventory, "date_start", None):
        manifest["navDate"] = f"{inventory.date_start}T00:00:00+00:00"
    elif getattr(inventory, "date_end", None):
        manifest["navDate"] = f"{inventory.date_end}T00:00:00+00:00"

    # Add Canvas for each Inventory's Scan (avoid document linkage)
    if getattr(inventory, "scans", None):
        # Sort scans by filename for consistent ordering
        sorted_scans = sorted(inventory.scans, key=lambda s: s.filename or "")
        for scan in sorted_scans:
            canvas_id = f"{manifest_uri}/canvas/{scan.id}"
            # Determine recto/verso from related pages, if present
            rv_label = None
            if getattr(scan, "pages", None):
                rv_values = [
                    p.recto_verso.value
                    for p in scan.pages
                    if getattr(p, "recto_verso", None)
                ]
                if len(rv_values) == 1:
                    rv_label = rv_values[0]
                elif len(rv_values) > 1:
                    # Combine unique values
                    uniq = sorted(set(rv_values))
                    rv_label = ", ".join(uniq)

            image_id = scan.get_image_url(size="max") or ""
            # IIIF Image service id from info.json, if available
            service_id = None
            if getattr(scan, "iiif_image_info", None):
                service_id = scan.iiif_image_info.replace("/info.json", "")

            # Metadata entries similar to the example (Filename, Web)
            # Use scan.filename directly; only include Web if `na_identifier` is a URL
            web_url = None
            if getattr(scan, "na_identifier", None):
                nai = str(scan.na_identifier)
                if nai.startswith("http://") or nai.startswith("https://"):
                    web_url = nai

            label_text = (
                scan.filename if not rv_label else f"{scan.filename} ({rv_label})"
            )

            canvas_obj: Dict[str, Any] = {
                "id": canvas_id,
                "type": "Canvas",
                "label": {"en": [label_text]},
                "height": scan.height,
                "width": scan.width,
                "metadata": [
                    {
                        "label": {"en": ["Filename"]},
                        "value": {"none": [scan.filename]},
                    },
                ],
                "items": [
                    {
                        "id": f"{canvas_id}/annotation_page/1",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": f"{canvas_id}/annotation/1",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": image_id,
                                    "type": "Image",
                                    "format": "image/jpeg",
                                    "height": scan.height,
                                    "width": scan.width,
                                    "service": [
                                        {
                                            "id": service_id,
                                            "type": "ImageService2",
                                            "profile": "http://iiif.io/api/image/2/level1.json",
                                        }
                                    ],
                                },
                                "target": canvas_id,
                            }
                        ],
                    }
                ],
                "annotations": [],
            }

            # Optional metadata: Web link and Recto/Verso info
            if web_url:
                canvas_obj["metadata"].append(
                    {
                        "label": {"en": ["Web"]},
                        "value": {"none": [f'<a href="{web_url}">{web_url}</a>']},
                    }
                )
            if rv_label:
                canvas_obj["metadata"].append(
                    {
                        "label": {"en": ["Recto/Verso"]},
                        "value": {"none": [rv_label]},
                    }
                )

            manifest["items"].append(canvas_obj)

    # Add thumbnail from first scan if available
    if manifest["items"] and getattr(inventory, "scans", None):
        first_scan = sorted(inventory.scans, key=lambda s: s.filename or "")[0]
        if first_scan:
            thumb_id = first_scan.get_image_url(size="982,") or ""
            service_id = None
            if getattr(first_scan, "iiif_image_info", None):
                service_id = first_scan.iiif_image_info.replace("/info.json", "")

            if thumb_id and service_id:
                manifest["thumbnail"] = [
                    {
                        "id": thumb_id,
                        "type": "Image",
                        "height": first_scan.height,
                        "width": first_scan.width,
                        "service": [
                            {
                                "@id": service_id,
                                "@type": "ImageService2",
                                "profile": "http://iiif.io/api/image/2/level1",
                                "format": "image/jpeg",
                            }
                        ],
                        "format": "image/jpeg",
                    }
                ]

    # Add Range for each Document in Inventory
    # TODO

    return manifest
