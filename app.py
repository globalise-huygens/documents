"""
Flask web application for inspecting GLOBALISE documents.
Provides a web interface to browse inventories, documents, scans, and pages.
"""

from flask import Flask, render_template, request, abort, Response, redirect, url_for
from flask_cors import CORS
from datetime import datetime
from sqlalchemy import create_engine, desc, func, event
from sqlalchemy.orm import sessionmaker, scoped_session
from models import (
    Base,
    Inventory,
    Document,
    DocumentIdentificationMethod,
    Scan,
    Page,
    Page2Document,
    Series,
)
from export import (  # type: ignore[import-not-found]
    inventory_to_manifest_jsonld,
    scan_to_jsonld,
    page_to_jsonld,
    document_physical_to_jsonld,
    inventory_to_jsonld,
    series_to_jsonld,
)
import json
import os

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

    return render_template(
        "inventory_detail.html",
        inventory=inventory,
        documents=documents,
        scans=scans,
        scan_count=scan_count,
        page_count=page_count,
        series_paths=series_paths,
        timeline_data=timeline_data,
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

    db_session = Session()
    document = get_or_404(db_session.query(Document).filter_by(id=document_id))

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

    return render_template(
        "document_detail.html",
        document=document,
        page_docs=page_docs,
        sub_documents=sub_documents,
        first_scan_filename=first_scan_filename,
        last_scan_filename=last_scan_filename,
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
        prev_page=prev_page,
        next_page=next_page,
    )


@app.route("/search")
def search():
    """Global search across documents, inventories, and scans."""
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

    results = {"documents": docs, "inventories": invs, "scans": scans_list}

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

    manifest_uri = request.url

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
