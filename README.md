# GLOBALISE Document Archive

Flask app for identifying, storing and browsing historical documents from the GLOBALISE corpus.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with pipx
pipx install uv
```

### Install Project Dependencies

```bash
# Install all dependencies (creates .venv automatically)
uv sync
```

This will automatically:

- Create a virtual environment in `.venv/`
- Install all required Python packages
- Set up the project for development

## Data Requirements

The application requires several data files that are too large to include in the repository. These files must be obtained separately and placed in the `data/` directory before running the import scripts.

### Required Data Files

Place the following files in the `data/` directory:

1. **documents_for_django.csv** - Scan metadata and inventory information (original dataset)
2. **documents_for_django_2025.csv** - Additional scan metadata (2025 dataset)
3. **page_metadata.csv** - Page-level metadata including folio numbers and scan types
4. **page_metadata_new_inventories.csv** - Page metadata for newly added inventories
5. **inventory2dates.json** - Date ranges for each inventory
6. **inventory2dates_extra.json** - Extra dates missing in EAD for inventories
7. **inventory2handle.json** - Handle URLs for inventories
8. **inventory2titles.json** - Titles for inventories
9. **inventory2uuid.json** - UUID mappings for inventories
10. **inventories.json** - Complete inventory information
11. **archival_hierarchy.json** - Archival series and hierarchy structure
12. **pp_project_globalisethesaurus.ttl** - SKOS thesaurus with GLOBALISE and TANAP document types
13. **location_index.csv** - Settlement/location index with GLOB IDs and spelling variants
14. **globalise_digitized_indexes.csv** - TANAP-digitized catalog records (OBP index)

Your `data/` directory should look like:

```
data/
├── documents_for_django.csv
├── documents_for_django_2025.csv
├── page_metadata.csv
├── page_metadata_new_inventories.csv
├── inventory2dates.json
├── inventory2dates_extra.json
├── inventory2handle.json
├── inventory2titles.json
├── inventory2uuid.json
├── inventories.json
├── overview_general_missives.csv
├── archival_hierarchy.json
├── pp_project_globalisethesaurus.ttl
├── location_index.csv
└── globalise_digitized_indexes.csv
```

## Database Setup

Run the three import scripts sequentially to create and populate the SQLite database:

### Step 1: Import Scans and Inventories

```bash
uv run python 1_import_scans_and_inventories.py
```

This script:

- Creates the database tables
- Imports inventory records from JSON files
- Imports scan metadata from CSV files
- Links scans to their respective inventories
- Expected runtime: 2-5 minutes depending on data size

### Step 2: Import Pages

```bash
uv run python 2_import_pages.py
```

This script:

- Updates scan types (single/double page)
- Creates page records with detailed metadata
- Links pages to scans
- Maps folio numbers and recto/verso positions
- Expected runtime: 5-10 minutes

### Step 3: Import Archival Hierarchy

```bash
uv run python 3_import_hierarchy.py data/archival_hierarchy.json
```

This script:

- Imports archival series (sets) and subseries
- Establishes parent-child relationships
- Updates inventory records with series information
- Expected runtime: 1-2 minutes

### Step 4: Identify Documents (Optional)

```bash
uv run python 4_identify_documents_baseline.py
```

This script implements a baseline document identification method for early modern archival documents:

- Creates a document identification method record
- Skips empty pages at the beginning of inventories (covers, archival covers)
- Identifies document boundaries based on:
  - Empty page sequences (is_blank=True)
  - Pages with signatures (indicating document end)
- Creates Document records and links them to pages
- Expected runtime: Varies based on inventory size

**Note:** This is a baseline implementation. More sophisticated document identification methods can be added as additional scripts that create different DocumentIdentificationMethod records.

### Step 5: Add Document Types

```bash
uv run python 5_import_document_types.py
# or with explicit paths:
uv run python 5_import_document_types.py --ttl /path/to/thesaurus.ttl --database sqlite:///globalise_documents.db
```

This script looks for document types in a `pp_project_globalisethesaurus.ttl` file and adds their UUID, the English and Dutch preflabels and whether it is a GLOBALISE or TANAP document type.

### Step 6: Import Settlements

```bash
uv run python 6_import_settlements.py
# or with explicit paths:
uv run python 6_import_settlements.py --csv /path/to/location_index.csv --database sqlite:///globalise_documents.db
```

This script:

- Imports settlement (location) data from `location_index.csv`
- Creates one Settlement per unique GLOB ID
- Creates multiple SettlementLabel records per settlement for spelling variants and alternative names
- Skips already existing settlements and labels on re-runs

### Step 7: Import OBP Index

```bash
uv run python 7_import_obp_index.py
# or with explicit paths:
uv run python 7_import_obp_index.py --csv /path/to/globalise_digitized_indexes.csv --database sqlite:///globalise_documents.db
```

This script:

- Imports TANAP-digitized catalog records (OBP index) from CSV
- Creates Document records with titles, dates, folio ranges, and locations
- Links documents to document types extracted from PoolParty URIs
- Resolves settlement labels to settlement records
- Creates external ID records for OBP_INDEX, TANAP, and DIGITIZED TYPOSCRIPTS contexts
- Creates a "TANAP Digitized Index" document identification method
- Depends on steps 1–6 (requires inventories, document types, and settlements in the database)

### Step 8: Add General Missives documents

```bash
uv run python 8_import_GM.py [--dry-run]
```

Uses the Ground Truth for General Missives to add documents. Requires the file `overview_general_missives.csv` to be in data folder.

### Verify Database

After running the import scripts, you should have a populated `globalise_documents.db` file.

## Running the Application

Start the Flask web server:

```bash
uv run python app.py
```

Then open your browser to: **http://localhost:5000**

### Container alternative

Alternatively, you can run the application using Docker:

```bash
docker pull ghcr.io/globalise-huygens/documents:latest
```

```bash
docker run -p 8000:8000 -v ./globalise_documents.db:/app/globalise_documents.db --rm globalisedocuments:latest
```

Then open your browser to: **http://localhost:8000**

## Usage

### Web Interface

- **Home** - Overview and statistics
- **Inventories** - Browse all archive inventories
- **Documents** - Search and filter documents
- **Scans** - View document scans with IIIF images
- **Pages** - Explore individual pages with metadata
- **Series** - Browse archival series hierarchy
- **Settlements** - Browse settlement locations
- **Document Types** - Browse document type classifications (GLOBALISE and TANAP)
- **Methods** - View document identification methods with timeline visualization
- **Search** - Full-text search across all content

## Exporting Data

### IIIF Collection

```bash
uv run python export_collection.py
```

Exports a top-level IIIF 3.0 Collection JSON file (`objects/inventory/collection.json`) that references all inventory manifests. Output is gzipped and ready for S3 upload.

### IIIF Manifests

```bash
uv run python export_manifests.py
```

Exports individual IIIF 3.0 Manifest JSON files for every inventory (`objects/inventory/<number>.manifest.json`). Output is gzipped and ready for S3 upload.

## Database Schema

The application uses SQLite with the following main tables:

- **Inventory** - Archive inventory records
- **InventoryTitle** - Titles for inventories
- **Series** - Archival series hierarchy (sets and subsets)
- **Scan** - Digital scans with IIIF URLs
- **Page** - Individual pages with folio numbers and metadata
- **Document** - Document records with date ranges
- **DocumentIdentificationMethod** - Methods used to identify documents
- **Page2Document** - Many-to-many relationships between pages and documents
- **DocumentType** - GLOBALISE and TANAP document type classifications
- **Document2DocumentType** - Many-to-many relationships between documents and types
- **Settlement** - Settlement/location entities with GLOB IDs
- **SettlementLabel** - Spelling variants and alternative names for settlements
- **ExternalID** - External identifiers (OBP, TANAP, etc.)
- **Document2ExternalID** - Many-to-many relationships between documents and external IDs

## Development

### Project Structure

```

documents/
├── app.py # Main Flask application
├── models.py # SQLAlchemy database models
├── db_utils.py # Database management utilities
├── 1_import_scans_and_inventories.py # Import script (step 1)
├── 2_import_pages.py # Import script (step 2)
├── 3_import_hierarchy.py # Import script (step 3)
├── 4_identify_documents_baseline.py # Document identification (baseline method)
├── 5_import_document_types.py       # Import document types from thesaurus (step 5)
├── 6_import_settlements.py          # Import settlements (step 6)
├── 7_import_obp_index.py            # Import OBP index records (step 7)
├── 8_import_GM.py                   # Import GM data (step 8)
├── export.py                        # Linked Art JSON-LD serialization helpers
├── export_collection.py             # Export IIIF Collection
├── export_manifests.py              # Export IIIF Manifests
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies (for pip)
├── pyproject.toml                   # UV/project configuration
├── data/                            # Data files (not in repo)
└── templates/                       # HTML templates
    ├── base.html
    ├── index.html
    ├── inventories.html
    ├── inventory_detail.html
    ├── documents.html
    ├── document_detail.html
    ├── document_types.html
    ├── document_type_detail.html
    ├── scans.html
    ├── scan_detail.html
    ├── pages.html
    ├── page_detail.html
    ├── settlements.html
    ├── settlement_detail.html
    ├── methods.html
    ├── method_detail.html

```

### Adding Dependencies

```bash
# Add a new package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update all packages
uv sync --upgrade
```

## License

**[TODO: Add license information]**
