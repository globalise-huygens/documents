"""
Flask web application for inspecting GLOBALISE documents.
Provides a web interface to browse inventories, documents, scans, and pages.
"""

from flask import Flask, render_template, request, abort, Response, redirect, url_for
from flask_cors import CORS
from datetime import datetime
from sqlalchemy import create_engine, desc, func, event
from sqlalchemy import case
from collections import Counter
from sqlalchemy.orm import sessionmaker, scoped_session, selectinload
from models import (
    Base,
    Inventory,
    Document,
    Document2DocumentType,
    DocumentIdentificationMethod,
    DocumentType,
    Scan,
    Page,
    Page2Document,
    Series,
    Settlement,
    SettlementLabel,
    LayoutElement,
    EntityMention,
    LayoutElement2EntityMention,
)
from export import (  # type: ignore[import-not-found]
    inventory_to_manifest_jsonld,
    scan_to_jsonld,
    page_to_jsonld,
    document_physical_to_jsonld,
    inventory_to_jsonld,
    series_to_jsonld,
)
import gzip
import json
import logging
import os
import urllib.request
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Annotation page fetching (on-the-fly from object store)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=512)
def _fetch_annotation_page(url: str) -> dict:
    """Fetch and cache an annotation page JSON from the object store.

    The annotation pages are gzip-compressed JSON served from the GLOBALISE
    object store.  Results are cached in-process so repeated look-ups for the
    same scan page are free.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()
        try:
            data = json.loads(gzip.decompress(raw))
        except OSError:
            data = json.loads(raw)
    return data


def fetch_annotation_detail(annotation_identifier: str) -> dict | None:
    """Resolve a single annotation item from its annotation page.

    ``annotation_identifier`` is a full URL with a ``#fragment`` that
    identifies the item within the page, e.g.::

        https://data.globalise.huygens.knaw.nl/…:transcriptions:SCAN#region_abc

    The part before ``#`` is the annotation page URL; the fragment selects
    the item within the ``items`` list.
    """
    page_url = (
        annotation_identifier.split("#")[0]
        if "#" in annotation_identifier
        else annotation_identifier
    )
    try:
        page_data = _fetch_annotation_page(page_url)
        for item in page_data.get("items", []):
            if item.get("id") == annotation_identifier:
                return item
    except Exception as e:
        logger.warning("Failed to fetch annotation page %s: %s", page_url, e)
    return None


def _extract_entity_display(item: dict) -> dict:
    """Extract display-friendly fields from a raw entity annotation item."""
    bodies = item.get("body", [])
    body = bodies[0] if bodies else {}

    text = body.get("label", "")

    subject_uri = None
    subject = body.get("has_appellative_subject") or body.get(
        "has_classificatory_subject"
    )
    if isinstance(subject, dict):
        subject_uri = subject.get("id")

    concept_uri = None
    # concept_uri can come from classificatory subject when subject_uri is appellative
    cls_subject = body.get("has_classificatory_subject")
    if isinstance(cls_subject, dict) and cls_subject.get("id") != subject_uri:
        concept_uri = cls_subject.get("id")

    timespan_begin = None
    timespan_end = None
    timespan = body.get("timespan", {})
    if isinstance(timespan, dict):
        timespan_begin = timespan.get("end_of_the_begin")
        timespan_end = timespan.get("begin_of_the_end")

    return {
        "text": text,
        "subject_uri": subject_uri,
        "concept_uri": concept_uri,
        "timespan_begin": timespan_begin,
        "timespan_end": timespan_end,
    }


def _extract_layout_text(item: dict, page_data: dict) -> str | None:
    """Extract concatenated line text for a layout element block.

    Walks the transcription annotation page to find child lines that
    reference this block and joins their text.
    """
    block_id = item.get("id", "")
    lines = []
    for child in page_data.get("items", []):
        if child.get("textGranularity") != "line":
            continue
        for target in child.get("target", []):
            if isinstance(target, dict) and target.get("type") == "Annotation":
                if target.get("id") == block_id:
                    for body in child.get("body", []):
                        if body.get("type") == "TextualBody" and "value" in body:
                            lines.append(body["value"])
                            break
                break
    return " ".join(lines) if lines else None


def enrich_layout_element(le) -> dict:
    """Fetch text for a LayoutElement from the annotation page."""
    item = fetch_annotation_detail(le.annotation_identifier)
    if not item:
        return {"text": None}
    page_url = le.annotation_identifier.split("#")[0]
    try:
        page_data = _fetch_annotation_page(page_url)
    except Exception:
        return {"text": None}
    return {"text": _extract_layout_text(item, page_data)}


def enrich_entity_mention(em) -> dict:
    """Fetch display fields for an EntityMention from the annotation page."""
    item = fetch_annotation_detail(em.annotation_identifier)
    if not item:
        return {
            "text": None,
            "subject_uri": None,
            "concept_uri": None,
            "timespan_begin": None,
            "timespan_end": None,
        }
    return _extract_entity_display(item)


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-secret-key-change-in-production"
)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///globalise_documents.db"
)

# CORS

CORS(app)

# Initialize database
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], echo=False)


# Register a SQLite UDF to sort inventory numbers naturally by numeric prefix then suffix
# Example: 999 < 1000 < 1053 < 1053A < 1053B < 9999
@event.listens_for(engine, "connect")
def register_inv_sortkey(dbapi_connection, connection_record):
    def inv_sortkey(value):
        if value is None:
            return "0000000000|"
        s = str(value)
        i = 0
        L = len(s)
        while i < L and s[i].isdigit():
            i += 1
        num = int(s[:i] or 0)
        suffix = s[i:].upper()
        # zero-pad number to 10 digits so lexicographic order matches numeric
        return f"{num:010d}|{suffix}"

    try:
        dbapi_connection.create_function("inv_sortkey", 1, inv_sortkey)
    except Exception:
        # If not SQLite or function already registered, ignore
        pass


Base.metadata.create_all(engine)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Remove database session at the end of the request."""
    Session.remove()


def get_or_404(query):
    """Helper function to get first result or abort with 404."""
    result = query.first()
    if result is None:
        abort(404)
    return result


# slugify helper moved to export.py


@app.route("/")
def index():
    """Home page showing statistics."""
    db_session = Session()

    stats = {
        "inventories": db_session.query(Inventory).count(),
        "documents": db_session.query(Document).count(),
        "scans": db_session.query(Scan).count(),
        "pages": db_session.query(Page).count(),
    }

    recent_inventories = (
        db_session.query(Inventory)
        .order_by(func.inv_sortkey(Inventory.inventory_number))
        .limit(10)
        .all()
    )

    return render_template(
        "index.html", stats=stats, recent_inventories=recent_inventories
    )


@app.route("/inventories")
def inventories():
    """List all inventories."""
    db_session = Session()
    page = request.args.get("page", 1, type=int)
    per_page = 20

    # Natural ascending sort by numeric prefix and then suffix, e.g., 999 < 1000 < 1053 < 1053A < 9999
    inventory_query = db_session.query(Inventory).order_by(
        func.inv_sortkey(Inventory.inventory_number)
    )
    total = inventory_query.count()
    inventories_list = (
        inventory_query.offset((page - 1) * per_page).limit(per_page).all()
    )

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "inventories.html",
        inventories=inventories_list,
        page=page,
        total_pages=total_pages,
    )


@app.route("/inventory/<inventory_number>")
def inventory_detail(inventory_number):
    """Show details of a specific inventory."""
    from collections import Counter

    db_session = Session()
    inventory = get_or_404(
        db_session.query(Inventory).filter_by(inventory_number=inventory_number)
    )

    # Get documents in this inventory with scan information
    documents = db_session.query(Document).filter_by(inventory_id=inventory.id).all()

    # Add scan filename information to each document
    for doc in documents:
        if doc.pages:
            # Sort pages by index to get first and last
            sorted_pages = sorted(doc.pages, key=lambda p: p.index)
            if sorted_pages:
                first_page = (
                    db_session.query(Page).filter_by(id=sorted_pages[0].page_id).first()
                )
                last_page = (
                    db_session.query(Page)
                    .filter_by(id=sorted_pages[-1].page_id)
                    .first()
                )

                doc.first_scan_filename = (
                    first_page.scan.filename if first_page and first_page.scan else None
                )
                doc.last_scan_filename = (
                    last_page.scan.filename if last_page and last_page.scan else None
                )
            else:
                doc.first_scan_filename = None
                doc.last_scan_filename = None
        else:
            doc.first_scan_filename = None
            doc.last_scan_filename = None

    # Get scans in this inventory
    scans = (
        db_session.query(Scan)
        .filter_by(inventory_id=inventory.id)
        .order_by(Scan.filename)
        .limit(50)
        .all()
    )
    scan_count = db_session.query(Scan).filter_by(inventory_id=inventory.id).count()

    # Add detailed page information for each scan
    for scan in scans:
        pages_on_scan = (
            db_session.query(Page)
            .filter_by(scan_id=scan.id)
            .order_by(
                Page.recto_verso.desc()  # Verso (V) before Recto (R) alphabetically
            )
            .all()
        )

        scan.pages_info = []
        for page in pages_on_scan:
            # Get features for this page
            features = []
            if page.is_blank:
                features.append("Blank")
            if page.has_marginalia:
                features.append("Marginalia")
            if page.has_table:
                features.append("Table")
            if page.has_illustration:
                features.append("Illustration")
            if page.has_print:
                features.append("Print")
            if page.signatures:
                features.append("Signature")

            # Check if page is part of any document
            is_in_document = (
                db_session.query(Page2Document).filter_by(page_id=page.id).first()
                is not None
            )

            scan.pages_info.append(
                {
                    "page": page,
                    "features": ", ".join(features) if features else "No features",
                    "recto_verso": page.recto_verso.value if page.recto_verso else None,
                    "is_in_document": is_in_document,
                }
            )

    # Get pages in this inventory
    page_count = db_session.query(Page).filter_by(inventory_id=inventory.id).count()

    # Build series paths (breadcrumbs) for this inventory
    # Path goes from root (biggest) to leaf (most specific)
    def build_series_path(series_obj):
        nodes = []
        current = series_obj
        while current is not None:
            nodes.insert(0, current)
            current = current.part_of
        return nodes

    series_paths = []
    if getattr(inventory, "member_of_series", None):
        for s in inventory.member_of_series:
            series_paths.append(build_series_path(s))

    # Prepare timeline data for document identification visualization
    timeline_data = prepare_timeline_data(db_session, inventory.id)

    # Get layout element and entity mention counts for this inventory
    inv_page_ids = (
        db_session.query(Page.id).filter(Page.inventory_id == inventory.id).all()
    )
    inv_page_id_list = [pid for (pid,) in inv_page_ids]

    layout_type_counts = Counter()
    entity_type_counts = Counter()
    layout_element_count = 0
    entity_mention_count = 0
    if inv_page_id_list:
        layout_rows = (
            db_session.query(LayoutElement.layout_type, func.count(LayoutElement.id))
            .filter(LayoutElement.page_id.in_(inv_page_id_list))
            .group_by(LayoutElement.layout_type)
            .all()
        )
        for layout_type, cnt in layout_rows:
            layout_type_counts[layout_type] = cnt
            layout_element_count += cnt

        entity_rows = (
            db_session.query(EntityMention.entity_type, func.count(EntityMention.id))
            .filter(EntityMention.page_id.in_(inv_page_id_list))
            .group_by(EntityMention.entity_type)
            .all()
        )
        for entity_type, cnt in entity_rows:
            entity_type_counts[entity_type] = cnt
            entity_mention_count += cnt

    return render_template(
        "inventory_detail.html",
        inventory=inventory,
        documents=documents,
        scans=scans,
        scan_count=scan_count,
        page_count=page_count,
        series_paths=series_paths,
        timeline_data=timeline_data,
        layout_type_counts=dict(layout_type_counts),
        entity_type_counts=dict(entity_type_counts),
        layout_element_count=layout_element_count,
        entity_mention_count=entity_mention_count,
    )


def prepare_timeline_data(db_session, inventory_id):
    """
    Prepare data for vis.js timeline visualization of document identification methods.

    Returns a dictionary with:
    - groups: list of identification methods
    - items: list of documents with their page ranges
    """
    from sqlalchemy import func

    # Get all pages for this inventory, ordered by scan filename and page id
    pages = (
        db_session.query(Page)
        .filter(Page.inventory_id == inventory_id)
        .join(Scan)
        .order_by(Scan.filename, Page.id)
        .all()
    )

    if not pages:
        return {"groups": [], "items": []}

    # Create page index mapping
    page_to_index = {page.id: idx for idx, page in enumerate(pages)}

    # Get all documents for this inventory grouped by method
    documents = (
        db_session.query(Document)
        .filter(Document.inventory_id == inventory_id)
        .join(DocumentIdentificationMethod)
        .all()
    )

    # Build groups (one per identification method)
    methods = {}
    for doc in documents:
        if doc.method_id not in methods:
            methods[doc.method_id] = {
                "id": doc.method_id,
                "content": doc.method.name,
                "order": len(methods),
            }

    groups = list(methods.values())

    # Build items (one per document)
    items = []
    for doc in documents:
        # Get page indices for this document
        if doc.pages:
            page_indices = [
                page_to_index[p.page_id]
                for p in doc.pages
                if p.page_id in page_to_index
            ]
            if page_indices:
                start_idx = min(page_indices)
                end_idx = max(page_indices)

                items.append(
                    {
                        "id": doc.id,
                        "group": doc.method_id,
                        "start": start_idx + 1,  # Shift to 1-based for display
                        "end": end_idx + 2,  # vis.js uses exclusive end, +2 for 1-based
                        "content": (
                            doc.title
                            if doc.title
                            else f"Document ({len(page_indices)} pages)"
                        ),
                        "title": f"{doc.method.name}: {doc.title if doc.title else 'Untitled'}<br>Pages: {start_idx+1}-{end_idx+1}",
                    }
                )

    return {
        "groups": groups,
        "items": items,
        "total_pages": len(pages) + 1,  # +1 for 1-based display range
    }


@app.route("/documents")
def documents():
    """List all documents."""
    db_session = Session()
    page = request.args.get("page", 1, type=int)
    per_page = 20

    # Search functionality
    search = request.args.get("search", "").strip()

    doc_query = db_session.query(Document).join(Inventory)

    if search:
        doc_query = doc_query.filter(
            (Document.title.ilike(f"%{search}%"))
            | (Inventory.inventory_number.ilike(f"%{search}%"))
        )

    doc_query = doc_query.order_by(desc(Document.date_earliest_begin))

    total = doc_query.count()
    documents_list = doc_query.offset((page - 1) * per_page).limit(per_page).all()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "documents.html",
        documents=documents_list,
        page=page,
        total_pages=total_pages,
        search=search,
    )


@app.route("/document/<document_id>")
def document_detail(document_id):
    """Show details of a specific document."""
    from sqlalchemy import case
    from sqlalchemy.orm import joinedload

    db_session = Session()
    document = get_or_404(
        db_session.query(Document)
        .options(
            joinedload(Document.document_types_linked).joinedload(
                Document2DocumentType.document_type
            ),
        )
        .filter_by(id=document_id)
    )

    # Get pages for this document
    # Order by scan filename, then by recto_verso (Verso before Recto for double spreads)
    # For double page spreads: verso (left) comes before recto (right)
    page_docs = (
        db_session.query(Page2Document)
        .join(Page)
        .join(Scan)
        .filter(Page2Document.document_id == document_id)
        .order_by(
            Scan.filename,
            case(
                (Page.recto_verso == "Verso", 1),
                (Page.recto_verso == "Recto", 2),
                else_=3,
            ),
            Page.id,
        )
        .all()
    )

    # Get sub-documents
    sub_documents = db_session.query(Document).filter_by(part_of_id=document_id).all()

    # Add scan filename information
    first_scan_filename = None
    last_scan_filename = None
    if page_docs:
        first_page = db_session.query(Page).filter_by(id=page_docs[0].page_id).first()
        last_page = db_session.query(Page).filter_by(id=page_docs[-1].page_id).first()

        if first_page and first_page.scan:
            first_scan_filename = first_page.scan.filename
        if last_page and last_page.scan:
            last_scan_filename = last_page.scan.filename

    # Collect page IDs for this document
    doc_page_ids = [pd.page_id for pd in page_docs]

    # Get layout elements for this document's pages
    layout_elements = []
    layout_type_counts = Counter()
    if doc_page_ids:
        layout_elements = (
            db_session.query(LayoutElement)
            .filter(LayoutElement.page_id.in_(doc_page_ids))
            .order_by(LayoutElement.page_id, LayoutElement.layout_type)
            .all()
        )
        for elem in layout_elements:
            layout_type_counts[elem.layout_type] += 1

    # Get entity mention counts for this document's pages
    entity_type_counts = Counter()
    entity_mention_count = 0
    if doc_page_ids:
        entity_rows = (
            db_session.query(EntityMention.entity_type, func.count(EntityMention.id))
            .filter(EntityMention.page_id.in_(doc_page_ids))
            .group_by(EntityMention.entity_type)
            .all()
        )
        for entity_type, cnt in entity_rows:
            entity_type_counts[entity_type] = cnt
            entity_mention_count += cnt

    return render_template(
        "document_detail.html",
        document=document,
        page_docs=page_docs,
        sub_documents=sub_documents,
        first_scan_filename=first_scan_filename,
        last_scan_filename=last_scan_filename,
        layout_elements=layout_elements,
        layout_type_counts=dict(layout_type_counts),
        entity_type_counts=dict(entity_type_counts),
        entity_mention_count=entity_mention_count,
    )


@app.route("/scans")
def scans():
    """List all scans."""
    db_session = Session()
    page = request.args.get("page", 1, type=int)
    per_page = 30

    inventory_id = request.args.get("inventory_id")

    scan_query = db_session.query(Scan)

    if inventory_id:
        scan_query = scan_query.filter(Scan.inventory_id == inventory_id)

    scan_query = scan_query.order_by(Scan.filename)

    # Use faster count on indexed column without joins
    total = db_session.query(func.count(Scan.id))
    if inventory_id:
        total = total.filter(Scan.inventory_id == inventory_id)
    total = total.scalar()

    scans_list = scan_query.offset((page - 1) * per_page).limit(per_page).all()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "scans.html",
        scans=scans_list,
        page=page,
        total_pages=total_pages,
        inventory_id=inventory_id,
    )


@app.route("/scan/<filename>")
def scan_detail(filename):
    """Show details of a specific scan."""
    db_session = Session()
    scan = get_or_404(db_session.query(Scan).filter_by(filename=filename))

    # Get pages for this scan
    pages = db_session.query(Page).filter_by(scan_id=scan.id).all()

    return render_template("scan_detail.html", scan=scan, pages=pages)


@app.route("/pages")
def pages():
    """List all pages."""
    db_session = Session()
    page_num = request.args.get("page", 1, type=int)
    per_page = 20

    inventory_id = request.args.get("inventory_id")

    # Build efficient query: only join what's needed for display
    page_query = db_session.query(Page)

    if inventory_id:
        page_query = page_query.filter(Page.inventory_id == inventory_id)

    # Order by scan filename and page id
    page_query = page_query.join(Scan).order_by(Scan.filename, Page.id)

    # Use faster count on indexed column without joins
    total = db_session.query(func.count(Page.id))
    if inventory_id:
        total = total.filter(Page.inventory_id == inventory_id)
    total = total.scalar()

    pages_list = page_query.offset((page_num - 1) * per_page).limit(per_page).all()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "pages.html",
        pages=pages_list,
        page=page_num,
        total_pages=total_pages,
        inventory_id=inventory_id,
    )


@app.route("/page/<page_id>")
def page_detail(page_id):
    """Show details of a specific page."""
    db_session = Session()
    page = get_or_404(db_session.query(Page).filter_by(id=page_id))

    # Get documents for this page
    page_docs = db_session.query(Page2Document).filter_by(page_id=page_id).all()

    # Get layout elements for this page
    layout_elements = (
        db_session.query(LayoutElement)
        .filter_by(page_id=page_id)
        .order_by(LayoutElement.layout_type)
        .all()
    )

    # Get entity mentions for this page (eager-load linked layout elements)
    entity_mentions = (
        db_session.query(EntityMention)
        .filter_by(page_id=page_id)
        .options(
            selectinload(EntityMention.layout_elements).selectinload(
                LayoutElement2EntityMention.layout_element
            )
        )
        .order_by(EntityMention.entity_type)
        .all()
    )

    # Enrich layout elements and entity mentions from annotation pages
    enriched_layouts = {}
    for elem in layout_elements:
        enriched_layouts[elem.id] = enrich_layout_element(elem)

    enriched_entities = {}
    for mention in entity_mentions:
        enriched_entities[mention.id] = enrich_entity_mention(mention)

    # Get previous and next pages in the same inventory
    prev_page = None
    next_page = None

    if page.inventory_id:
        # Get all pages in this inventory ordered by scan filename and page id
        all_pages = (
            db_session.query(Page)
            .filter(Page.inventory_id == page.inventory_id)
            .join(Scan)
            .order_by(Scan.filename, Page.id)
            .all()
        )

        # Find current page index
        try:
            current_index = next(i for i, p in enumerate(all_pages) if p.id == page.id)
            if current_index > 0:
                prev_page = all_pages[current_index - 1]
            if current_index < len(all_pages) - 1:
                next_page = all_pages[current_index + 1]
        except StopIteration:
            pass

    return render_template(
        "page_detail.html",
        page=page,
        page_docs=page_docs,
        layout_elements=layout_elements,
        entity_mentions=entity_mentions,
        enriched_layouts=enriched_layouts,
        enriched_entities=enriched_entities,
        prev_page=prev_page,
        next_page=next_page,
    )


@app.route("/settlements")
def settlements():
    """List all canonical settlements imported from location_index.csv (step 6)."""
    db_session = Session()
    page = request.args.get("page", 1, type=int)
    per_page = 50
    search = request.args.get("search", "").strip()

    q = db_session.query(Settlement)

    if search:
        # Match against glob_id or any of the settlement's labels
        q = q.filter(
            Settlement.glob_id.ilike(f"%{search}%")
            | Settlement.labels.any(SettlementLabel.label.ilike(f"%{search}%"))
        )

    q = q.order_by(Settlement.glob_id)
    total = q.count()

    settlements_list = q.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "settlements.html",
        settlements=settlements_list,
        page=page,
        total_pages=total_pages,
        search=search,
    )


@app.route("/entity-mentions")
def entity_mentions_list():
    """Pageable, filterable list of entity mentions. Supports filtering by document, inventory, page, and entity type."""
    db_session = Session()
    page_num = request.args.get("page", 1, type=int)
    per_page = 50

    document_id = request.args.get("document_id")
    inventory_id = request.args.get("inventory_id")
    page_id = request.args.get("page_id")
    entity_type = request.args.get("entity_type")
    layout_type = request.args.get("layout_type")
    sort = request.args.get("sort", "text")  # text, type, page

    # Build query
    query = db_session.query(EntityMention)

    # Determine scope and build page ID filter
    scope_label = "All"
    scope_params = {}
    doc_page_ids = []
    inv_page_ids = []

    if page_id:
        query = query.filter(EntityMention.page_id == page_id)
        scope_label = f"Page {page_id[:8]}"
        scope_params["page_id"] = page_id
        # Get the page for breadcrumb
        scope_page = db_session.query(Page).filter_by(id=page_id).first()
    elif document_id:
        # Get all page IDs for this document
        doc_page_ids = [
            pd.page_id
            for pd in db_session.query(Page2Document)
            .filter_by(document_id=document_id)
            .all()
        ]
        if doc_page_ids:
            query = query.filter(EntityMention.page_id.in_(doc_page_ids))
        else:
            query = query.filter(EntityMention.page_id == None)  # noqa: E711
        scope_doc = db_session.query(Document).filter_by(id=document_id).first()
        scope_label = (
            f"Document: {scope_doc.title or document_id[:8]}"
            if scope_doc
            else f"Document {document_id[:8]}"
        )
        scope_params["document_id"] = document_id
    elif inventory_id:
        inv_page_ids = [
            pid
            for (pid,) in db_session.query(Page.id)
            .filter(Page.inventory_id == inventory_id)
            .all()
        ]
        if inv_page_ids:
            query = query.filter(EntityMention.page_id.in_(inv_page_ids))
        else:
            query = query.filter(EntityMention.page_id == None)  # noqa: E711
        scope_inv = db_session.query(Inventory).filter_by(id=inventory_id).first()
        scope_label = (
            f"Inventory: {scope_inv.inventory_number}"
            if scope_inv
            else f"Inventory {inventory_id[:8]}"
        )
        scope_params["inventory_id"] = inventory_id

    # Filter by entity type
    if entity_type:
        query = query.filter(EntityMention.entity_type == entity_type)
        scope_params["entity_type"] = entity_type

    # Filter by layout type (join through junction table)
    if layout_type:
        query = (
            query.join(
                LayoutElement2EntityMention,
                LayoutElement2EntityMention.entity_mention_id == EntityMention.id,
            )
            .join(
                LayoutElement,
                LayoutElement.id == LayoutElement2EntityMention.layout_element_id,
            )
            .filter(LayoutElement.layout_type == layout_type)
        )
        scope_params["layout_type"] = layout_type

    # Build a base scope filter for counts (without entity_type / layout_type filters)
    def _scope_filter(q):
        if page_id:
            q = q.filter(EntityMention.page_id == page_id)
        elif document_id and doc_page_ids:
            q = q.filter(EntityMention.page_id.in_(doc_page_ids))
        elif inventory_id and inv_page_ids:
            q = q.filter(EntityMention.page_id.in_(inv_page_ids))
        return q

    # Get available entity types for filter badges (within scope, respecting layout_type filter)
    type_counts_query = db_session.query(
        EntityMention.entity_type, func.count(EntityMention.id)
    )
    type_counts_query = _scope_filter(type_counts_query)
    if layout_type:
        type_counts_query = (
            type_counts_query.join(
                LayoutElement2EntityMention,
                LayoutElement2EntityMention.entity_mention_id == EntityMention.id,
            )
            .join(
                LayoutElement,
                LayoutElement.id == LayoutElement2EntityMention.layout_element_id,
            )
            .filter(LayoutElement.layout_type == layout_type)
        )
    type_counts = (
        type_counts_query.group_by(EntityMention.entity_type)
        .order_by(EntityMention.entity_type)
        .all()
    )

    # Get available layout types for filter badges (within scope, respecting entity_type filter)
    layout_type_counts_query = (
        db_session.query(
            LayoutElement.layout_type, func.count(func.distinct(EntityMention.id))
        )
        .join(
            LayoutElement2EntityMention,
            LayoutElement2EntityMention.entity_mention_id == EntityMention.id,
        )
        .join(
            LayoutElement,
            LayoutElement.id == LayoutElement2EntityMention.layout_element_id,
        )
    )
    layout_type_counts_query = _scope_filter(layout_type_counts_query)
    if entity_type:
        layout_type_counts_query = layout_type_counts_query.filter(
            EntityMention.entity_type == entity_type
        )
    layout_type_counts = (
        layout_type_counts_query.group_by(LayoutElement.layout_type)
        .order_by(LayoutElement.layout_type)
        .all()
    )

    # Sort
    if sort == "type":
        query = query.order_by(
            EntityMention.entity_type, EntityMention.annotation_identifier
        )
    elif sort == "page":
        query = query.order_by(EntityMention.page_id, EntityMention.entity_type)
    else:
        query = query.order_by(
            EntityMention.entity_type, EntityMention.annotation_identifier
        )

    # Eager-load linked layout elements for display
    query = query.options(
        selectinload(EntityMention.layout_elements).selectinload(
            LayoutElement2EntityMention.layout_element
        )
    )

    # Count and paginate
    total = query.count()
    mentions = query.offset((page_num - 1) * per_page).limit(per_page).all()
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "entity_mentions.html",
        mentions=mentions,
        page=page_num,
        total_pages=total_pages,
        total=total,
        scope_label=scope_label,
        scope_params=scope_params,
        entity_type=entity_type,
        layout_type=layout_type,
        sort=sort,
        type_counts=type_counts,
        layout_type_counts=layout_type_counts,
        document_id=document_id,
        inventory_id=inventory_id,
        page_id=page_id,
    )


@app.route("/layout-element/<layout_element_id>")
def layout_element_detail(layout_element_id):
    """Show details of a specific layout element."""
    db_session = Session()
    layout_elem = get_or_404(
        db_session.query(LayoutElement)
        .options(
            selectinload(LayoutElement.entity_mentions).selectinload(
                LayoutElement2EntityMention.entity_mention
            )
        )
        .filter_by(id=layout_element_id)
    )

    # Enrich with text from annotation page (on-the-fly)
    enriched = enrich_layout_element(layout_elem)

    # Enrich linked entity mentions
    enriched_entities = {}
    for link in layout_elem.entity_mentions:
        em = link.entity_mention
        enriched_entities[em.id] = enrich_entity_mention(em)

    # Get documents for this page
    page_docs = []
    if layout_elem.page_id:
        page_docs = (
            db_session.query(Page2Document).filter_by(page_id=layout_elem.page_id).all()
        )

    # Find other layout elements on the same page
    same_page_elements = (
        db_session.query(LayoutElement)
        .filter(
            LayoutElement.page_id == layout_elem.page_id,
            LayoutElement.id != layout_elem.id,
        )
        .order_by(LayoutElement.layout_type)
        .all()
    )

    return render_template(
        "layout_element_detail.html",
        layout_elem=layout_elem,
        enriched=enriched,
        enriched_entities=enriched_entities,
        page_docs=page_docs,
        same_page_elements=same_page_elements,
    )


@app.route("/settlement/<glob_id>")
def settlement_detail(glob_id):
    """Show details and linked documents for a single settlement."""
    db_session = Session()
    settlement = get_or_404(db_session.query(Settlement).filter_by(glob_id=glob_id))

    page = request.args.get("page", 1, type=int)
    per_page = 20

    doc_query = (
        db_session.query(Document)
        .filter_by(location_id=settlement.id)
        .order_by(Document.date_earliest_begin)
    )
    total = doc_query.count()
    documents = doc_query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "settlement_detail.html",
        settlement=settlement,
        documents=documents,
        page=page,
        total_pages=total_pages,
        total_docs=total,
    )


@app.route("/entity-mention/<entity_mention_id>")
def entity_mention_detail(entity_mention_id):
    """Show details of a specific entity mention."""
    db_session = Session()
    mention = get_or_404(
        db_session.query(EntityMention)
        .options(
            selectinload(EntityMention.layout_elements).selectinload(
                LayoutElement2EntityMention.layout_element
            )
        )
        .filter_by(id=entity_mention_id)
    )

    # Enrich with text/subject/timespan from annotation page (on-the-fly)
    enriched = enrich_entity_mention(mention)

    # Enrich linked layout elements
    enriched_layouts = {}
    for link in mention.layout_elements:
        le = link.layout_element
        enriched_layouts[le.id] = enrich_layout_element(le)

    # Get documents for this page
    page_docs = []
    if mention.page_id:
        page_docs = (
            db_session.query(Page2Document).filter_by(page_id=mention.page_id).all()
        )

    return render_template(
        "entity_mention_detail.html",
        mention=mention,
        enriched=enriched,
        enriched_layouts=enriched_layouts,
        page_docs=page_docs,
    )


@app.route("/api/annotation/<entity_or_layout>/<item_id>")
def api_annotation_detail(entity_or_layout, item_id):
    """API endpoint: fetch enriched data for a layout element or entity mention.

    Used by AJAX calls in list views to lazy-load text and metadata.
    Returns JSON with the enriched fields.
    """
    db_session = Session()
    if entity_or_layout == "entity":
        em = db_session.query(EntityMention).filter_by(id=item_id).first()
        if not em:
            return (
                json.dumps({"error": "not found"}),
                404,
                {"Content-Type": "application/json"},
            )
        data = enrich_entity_mention(em)
    elif entity_or_layout == "layout":
        le = db_session.query(LayoutElement).filter_by(id=item_id).first()
        if not le:
            return (
                json.dumps({"error": "not found"}),
                404,
                {"Content-Type": "application/json"},
            )
        data = enrich_layout_element(le)
    else:
        return (
            json.dumps({"error": "invalid type"}),
            400,
            {"Content-Type": "application/json"},
        )
    return json.dumps(data), 200, {"Content-Type": "application/json"}


@app.route("/search")
def search():
    """Global search across documents, inventories, scans, and settlements."""
    db_session = Session()
    query = request.args.get("q", "").strip()

    if not query:
        return render_template("search.html", query=query, results=None)

    # Search in documents
    docs = (
        db_session.query(Document)
        .filter(Document.title.ilike(f"%{query}%"))
        .limit(20)
        .all()
    )

    # Search in inventories
    invs = (
        db_session.query(Inventory)
        .filter(Inventory.inventory_number.ilike(f"%{query}%"))
        .limit(20)
        .all()
    )

    # Search in scans
    scans_list = (
        db_session.query(Scan).filter(Scan.filename.ilike(f"%{query}%")).limit(20).all()
    )

    # Search in settlements (glob_id or any label)
    settlement_list = (
        db_session.query(Settlement)
        .filter(
            Settlement.glob_id.ilike(f"%{query}%")
            | Settlement.labels.any(SettlementLabel.label.ilike(f"%{query}%"))
        )
        .limit(20)
        .all()
    )

    results = {
        "documents": docs,
        "inventories": invs,
        "scans": scans_list,
        "settlements": settlement_list,
    }

    return render_template("search.html", query=query, results=results)


@app.route("/methods")
def methods():
    """List all document identification methods."""
    db_session = Session()
    page = request.args.get("page", 1, type=int)
    per_page = 50

    method_query = db_session.query(DocumentIdentificationMethod).order_by(
        DocumentIdentificationMethod.name
    )

    total = method_query.count()
    methods_list = method_query.offset((page - 1) * per_page).limit(per_page).all()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "methods.html",
        methods=methods_list,
        page=page,
        total_pages=total_pages,
    )


def _parse_date_field(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except Exception:
        return None


@app.route("/methods/new", methods=["POST"])
def method_create():
    db_session = Session()
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip()
    date_val = _parse_date_field(request.form.get("date", ""))
    url_val = (request.form.get("url") or "").strip()

    if not name:
        abort(400, description="Name is required")

    method = DocumentIdentificationMethod(
        name=name,
        description=description or None,
        date=date_val,
        url=url_val or None,
    )
    db_session.add(method)
    db_session.commit()
    return redirect(url_for("method_detail", method_id=method.id))


@app.route("/method/<method_id>")
def method_detail(method_id):
    """Show details of a specific document identification method."""
    db_session = Session()
    method = get_or_404(
        db_session.query(DocumentIdentificationMethod).filter_by(id=method_id)
    )

    # Get documents using this method (paginated)
    page = request.args.get("page", 1, type=int)
    per_page = 20

    doc_query = db_session.query(Document).filter_by(method_id=method_id)
    total = doc_query.count()
    documents = doc_query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "method_detail.html",
        method=method,
        documents=documents,
        page=page,
        total_pages=total_pages,
        total_docs=total,
    )


@app.route("/method/<method_id>/edit", methods=["POST"])
def method_edit(method_id):
    db_session = Session()
    method = get_or_404(
        db_session.query(DocumentIdentificationMethod).filter_by(id=method_id)
    )

    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip()
    date_val = _parse_date_field(request.form.get("date", ""))
    url_val = (request.form.get("url") or "").strip()

    if not name:
        abort(400, description="Name is required")

    method.name = name
    method.description = description or None
    method.date = date_val
    method.url = url_val or None

    db_session.commit()
    return redirect(url_for("method_detail", method_id=method.id))


@app.route("/method/<method_id>/delete", methods=["POST"])
def method_delete(method_id):
    db_session = Session()
    method = get_or_404(
        db_session.query(DocumentIdentificationMethod).filter_by(id=method_id)
    )
    db_session.delete(method)
    db_session.commit()
    return redirect(url_for("methods"))


@app.route("/document-types")
def document_types():
    """List all document types from the SKOS thesaurus."""
    db_session = Session()

    scheme = request.args.get("scheme", "").strip()

    type_query = db_session.query(DocumentType).order_by(
        DocumentType.scheme, DocumentType.pref_label_en, DocumentType.pref_label_nl
    )
    if scheme:
        type_query = type_query.filter(DocumentType.scheme == scheme)

    all_types = type_query.all()

    # Count documents per type in a single query
    count_rows = (
        db_session.query(
            Document2DocumentType.document_type_id,
            func.count(Document2DocumentType.id),
        )
        .group_by(Document2DocumentType.document_type_id)
        .all()
    )
    type_doc_counts = {row[0]: row[1] for row in count_rows}

    schemes = [
        r[0]
        for r in db_session.query(DocumentType.scheme)
        .distinct()
        .order_by(DocumentType.scheme)
        .all()
    ]

    return render_template(
        "document_types.html",
        document_types=all_types,
        type_doc_counts=type_doc_counts,
        schemes=schemes,
        selected_scheme=scheme,
    )


@app.route("/document-type/<type_id>")
def document_type_detail(type_id):
    """Show all documents linked to a specific document type."""
    from sqlalchemy.orm import joinedload

    db_session = Session()
    doc_type = get_or_404(db_session.query(DocumentType).filter_by(id=type_id))

    page = request.args.get("page", 1, type=int)
    per_page = 20

    doc_query = (
        db_session.query(Document)
        .join(Document2DocumentType)
        .filter(Document2DocumentType.document_type_id == type_id)
        .order_by(Document.date_earliest_begin)
    )
    total = doc_query.count()
    documents = (
        doc_query.options(
            joinedload(Document.document_types_linked).joinedload(
                Document2DocumentType.document_type
            ),
        )
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "document_type_detail.html",
        doc_type=doc_type,
        documents=documents,
        page=page,
        total_pages=total_pages,
        total_docs=total,
    )


@app.route("/scan/<filename>/jsonld")
def scan_jsonld(filename):
    db_session = Session()
    scan = get_or_404(db_session.query(Scan).filter_by(filename=filename))
    data = scan_to_jsonld(scan)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


@app.route("/page/<page_id>/jsonld")
def page_jsonld(page_id):
    db_session = Session()
    page = get_or_404(db_session.query(Page).filter_by(id=page_id))
    data = page_to_jsonld(page)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


@app.route("/document/<document_id>/physical/jsonld")
def document_physical_jsonld(document_id):
    db_session = Session()
    document = get_or_404(db_session.query(Document).filter_by(id=document_id))
    data = document_physical_to_jsonld(document)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


@app.route("/inventory/<inventory_number>/jsonld")
def inventory_jsonld(inventory_number):
    db_session = Session()
    inventory = get_or_404(
        db_session.query(Inventory).filter_by(inventory_number=inventory_number)
    )
    data = inventory_to_jsonld(inventory)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


@app.route("/inventory/<inventory_number>/manifest")
def inventory_manifest(inventory_number):
    db_session = Session()
    inventory = get_or_404(
        db_session.query(Inventory).filter_by(inventory_number=inventory_number)
    )

    manifest_uri = f"https://data.globalise.huygens.knaw.nl/hdl:20.500.14722/inventory:{inventory_number}.manifest"

    data = inventory_to_manifest_jsonld(inventory, manifest_uri)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


@app.route("/series/<series_id>/jsonld")
def series_jsonld(series_id):
    db_session = Session()
    series = get_or_404(db_session.query(Series).filter_by(id=series_id))
    data = series_to_jsonld(series)

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/ld+json"
    )


# Template filters
@app.template_filter("date_range")
def date_range_filter(doc):
    """Format document date range."""
    if doc.date_text:
        return doc.date_text

    parts = []
    if doc.date_earliest_begin and doc.date_latest_begin:
        if doc.date_earliest_begin == doc.date_latest_begin:
            parts.append(str(doc.date_earliest_begin))
        else:
            parts.append(f"{doc.date_earliest_begin} - {doc.date_latest_begin}")
    elif doc.date_earliest_begin:
        parts.append(f"From {doc.date_earliest_begin}")

    if doc.date_earliest_end and doc.date_latest_end:
        if doc.date_earliest_end == doc.date_latest_end:
            if not parts:
                parts.append(f"Until {doc.date_earliest_end}")
        else:
            parts.append(f"to {doc.date_earliest_end} - {doc.date_latest_end}")

    return " ".join(parts) if parts else "Unknown date"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
