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
└── archival_hierarchy.json
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

### Verify Database

After running all three scripts, you should have a populated `globalise_documents.db` file.

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
- **Search** - Full-text search across all content

## Database Schema

The application uses SQLite with the following main tables:

- **Inventory** - Archive inventory records
- **InventoryTitle** - Titles for inventories
- **Series** - Archival series hierarchy (sets and subsets)
- **Scan** - Digital scans with IIIF URLs
- **Page** - Individual pages with folio numbers and metadata
- **Document** - Document records with date ranges
- **Page2Document** - Many-to-many relationships between pages and documents

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
├── requirements.txt # Python dependencies (for pip)
├── pyproject.toml # UV/project configuration
├── data/ # Data files (not in repo)
└── templates/ # HTML templates
├── base.html
├── index.html
├── inventories.html
├── inventory_detail.html
├── documents.html
├── document_detail.html
├── scans.html
├── scan_detail.html
├── pages.html
├── page_detail.html
└── search.html

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
