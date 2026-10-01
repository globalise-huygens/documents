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
15. **annotationpages.csv** - Per-scan flags indicating availability of transcription, entity, and event annotation pages

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
├── annotationpages.csv
└── globalise_digitized_indexes.csv
```

## Database Setup

Run the import scripts sequentially to create and populate the SQLite database:

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

### Step 3.5: Mark Blank Pages from Normalized Text

```bash
uv run python 3.5_import_empty_pages.py
# optional overrides:
uv run python 3.5_import_empty_pages.py --threshold 30
uv run python 3.5_import_empty_pages.py --parquet data/normalized_texts.parquet
```

This script:

- Reads normalized text per scan from parquet (`filename`, `normalized_text`)
- Computes character length of `normalized_text`
- Sets `page.is_blank=True` when text length is below the threshold (default: 20)
- Sets `page.is_blank=False` otherwise
- Updates all pages linked to each matched scan filename

Use this step before baseline document identification so empty-page boundaries are available.

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

### Step 7.5: Fix OBP Inventory Numbers

```bash
uv run python 7.5_fix_obp_inventories.py [--dry-run]
```

The OBP CSV stores inventory numbers as integers, so documents of suffixed inventories (9014A, 1430A, …) end up under the base number: step 7 either skips them (base number not in the database) or puts them in the wrong inventory (e.g. 1430 instead of 1430A). This script uses the corrected inventory numbers in `data/OBP NT_gecorrigeerd.xlsx` to create the missing documents and move the misplaced ones (dropping their page links to pages of the old inventory). Page links for these documents are not recomputed.

### Step 8: Add General Missives documents

```bash
uv run python 8_import_GM.py [--dry-run]
```

Uses the Ground Truth for General Missives to add documents. Requires the file `overview_general_missives.csv` to be in data folder.

### Step 9: Import Annotation Page Availability

```bash
uv run python 9_import_annotation_pages_exist.py [--dry-run]
```

Sets `has_transcriptions`, `has_entities`, and `has_events` flags on Scan records based on `annotationpages.csv`. These flags control whether annotation page links are included in IIIF manifest exports.

### Step 9.5: Migrate Confidence Column

```bash
uv run python 9.5_backfill_confidence.py
```

Migrates the `page2document.confidence` column from a Float to a String-based Enum (`LinkConfidence`). This ensures compatibility with the refined matching scripts.

### Step 10: Match Folios

```bash
uv run python 10_match_folios.py
```

Matches pages to OBP documents by folio range. It assigns the `FOLIO_RANGE` confidence tier to every page whose folio number falls within a document's start/end range.

### Step 11: Interpolate pages

```bash
uv run python 11_interpolate_pages.py
```

Fills gaps for unlinked pages using scan-order neighbors.

An unlinked page is only interpolated if:

- it has a linked neighbour on both sides
- both neighbours resolve to exactly one document
- both neighbours belong to the same document

This avoids guessing at document boundaries or using ambiguous matches.

By default, interpolation is strict (propagation depth = 1):

- only original, confidently linked pages are used as neighbours
- gaps are filled one step at a time

You can increase interpolation depth to allow propagation across multiple gaps:

```bash
uv run python 11_interpolate_pages.py --propagation-depth 2
```

Example:

```
[Doc A] — [gap] — [gap] — [Doc A]
```

- depth = 1 → no interpolation (gap too large)
- depth = 2 → both gaps are filled

The script is safe to rerun:

- it will only fill previously unlinked pages
- useful after improving earlier matching steps (e.g. folio matching)

### Step 16: Add Titles to Baseline Documents

```bash
uv run python 16_add_titles_to_documents.py
# inspect without writing:
uv run python 16_add_titles_to_documents.py --dry-run
# replace existing non-empty titles as well:
uv run python 16_add_titles_to_documents.py --overwrite
```

This script:

- Targets documents created by `Baseline: Empty Pages & Signatures`
- Looks at pages linked to each baseline document in sequential order (`page2document.index`)
- Assigns the first non-empty `page.header` as the document title
- By default, only fills missing titles (does not overwrite existing non-empty titles)

### Step 18: Import ToC Structure

```bash
uv run python 18_import_toc_sections.py
# inspect matching without writing:
uv run python 18_import_toc_sections.py --dry-run
```

Combines three ToC sources (see `toc_sources.py`): the OBP CSV (its `ID` is the leading identifier, stored as `OBP_INDEX`), `data/TANAP VOC OBP Nationaal Archief.xlsx` and `data/OBP NT_gecorrigeerd.xlsx`. NT_gecorrigeerd lists the TANAP index in the physical order of the volumes but has its own numbering; its entries are matched to CSV ids by inventory, description and start folio (99.97% matched). For every document linked to an `OBP_INDEX` id it sets:

- `toc_order` – position within the inventory: NT order where available, else CSV id order (e.g. typoscript inventories)
- `toc_folio_sequence` – foliation sequence (physical section), numbered from folio restarts in `toc_order`
- `toc_katern` – katern label (settlement + DEEL, e.g. `Ternate 3`); not necessarily a physical section
- `toc_deel` – the CSV's `SECTION` (= DEEL)
- `toc_folio_start_side` / `toc_folio_end_side` – `Recto`/`Verso` when the index page range specifies it (e.g. `14v-16`)
- `nt_index` – the entry's ID in NT_gecorrigeerd

## Document Segmentation (`segmentation/`)

Finds, for every inventory, the best-scoring sequence of documents given all predictors and the ToC (when there is one).

```bash
uv run python -m segmentation split-texts                       # one-time: data/normalized_texts.parquet → data/texts/<inv>
SEGMENTATION_PAGEXML_DIR=/Volumes/HDE0090 \
  uv run python -m segmentation cache-layout                    # one-time: page layout from the PageXML zips → data/layout/<inv>
uv run python -m segmentation evaluate --cache /tmp/items.pkl   # cross-validated evaluation
uv run python -m segmentation fit --cache /tmp/items.pkl        # fit on all ground truth → segmentation/model.json
uv run python -m segmentation run 1120 1557 --out segments.csv   # segment inventories (add --no-toc to ignore the ToC)
uv run python -m segmentation import segments.csv [--dry-run]    # store them as method "Segmentation model"
```

`import` creates one document per ToC entry, subdocument and unindexed document (non-document runs are skipped). ToC documents take their title, dates and page range from the ToC entry and are linked to its index ids (OBP_INDEX, TANAP, DIGITIZED TYPOSCRIPTS and NT, the entry's id in OBP NT_gecorrigeerd). Subdocuments and nested ToC entries point to their parent with `part_of_id`, and a parent's pages include its subdocuments'. All pages of the scans are linked (source `SEGMENTATION_MODEL`, confidence `CANDIDATE`), and how the model found each document is stored as JSON in `document_evidence`. Re-importing an inventory replaces its earlier "Segmentation model" documents.

In the app, the inventory timeline shows the model's documents with their subdocuments in a separate row, a document's page explains how the model found it (ToC placement, page numbers, dates, start/end probabilities and their main reasons), a scan's page shows what the model read on it (header, header date and how it compares with the previous headers and across the scan, page numbers, signature, probabilities), and `/inventory/<inv>/evidence` lists this scan by scan.

How it works:

1. **Predictors** (`predictors.py`) turn each scan into features: the full text (`texts.py`; opening formulas such as *Copia*, *Extract*, *Register* or a salutation, closing formulas such as *Accordeert*, *onderstond*, *was geteekent* or place and date, a closing followed by a new opening on the same page, dotted leaders of table-of-contents pages, and how well the text matches a ToC description), blank pages, text length, position in the inventory, signature marks (collation / signed / quire letter), page/folio numbers (cleaned, with numbering restarts = sections), page numbers the ToC mentions (an entry starts, ends, or one ends and the next starts on this number), running-header wording and header dates (`header_dates.py`, compared noise-aware), language changes, marginalia, and the page layout from PageXML (`pagexml.py`: where text starts and stops, gaps between blocks, a closing formula halfway down with text below it, an opening or centred heading below the top, catch-words). Missing layout counts as neutral, but the model works best when the layout of all inventories has been cached. A Double scan (two pages) is one unit; its pages share their metadata.
2. **Scan models** (`model.py`): logistic regressions, using each scan's features and its neighbours', give per-scan log-odds that a document *starts* on the scan, that one *ends* on it, that a start is on the *same scan* where the previous document ends, and that the scan is a *non-document* page (cover, blank, table of contents, title page, …). Weights are fitted on the validated inventories; a length prior (empirical document lengths) completes the model.
3. **ToC alignment** (`segmenter.py`): every ToC entry (in `toc_order`) is placed on a start scan, in order. A monotone dynamic program places the entries it can on page numbers; Candidate positions come from exact page/folio matches, interpolation within a numbering run, or scans whose text matches the entry's description (the opening words, or words after a closing formula where a document starts mid-page; a matching short title page counts for the next scan only if that scan opens a document, since title pages also end documents); the number-to-scan rate (pagination vs foliation, single vs double scans) is estimated from the observed numbers. the score adds the start log-odds, header-date agreement and the fit between the span implied by the numbers and the actual gap. The remaining entries are placed between their placed neighbours on the best start evidence (`evidence`, or `forced` when it is weak).
4. **Segmentation**: a second dynamic program over all scans, with the placed ToC starts forced, decides every document's start and end and the type of every boundary: the next document starts on the *same scan* (letters copied one after another), on the *next scan*, or after a *gap* of non-document scans. Documents are labelled `toc`, `subdoc` (within the page range of the preceding ToC entry, e.g. enclosures or appendices) or `unindexed`; gaps are reported as `non-document`. Title pages are outside documents (as in the ToC and the validated inventories), whether they precede a document or follow it.

**Court records** (`court_records.py`): inventories without a ToC that have cases in `data/EMDCCR*.xlsx` (Early Modern Dutch Colonial Court Records: Raad van Justitie of Batavia and the Cape, one row per accused) are segmented on those cases instead (`run --no-court` to ignore them). A case with a start and end scan is a fixed document (kind `case`), and the ranges of individual accused within it are its subdocuments. A case with only a start scan (the Cape volumes) runs to the next known case, and covers and blanks at the open end are left out. References to another inventory (`656 (9354)`) continue a case across volumes, and starts or ends inside another case's range are dropped. The model segments everything else as usual. Titles are Dutch: claimant contra defendants; court, place, date (charge), e.g. *Advocaat-fiscaal van India mr. Adrianus Bergsma contra Jacobus Bunnegam en Pieter Hildernisse; Raad van Justitie, Batavia, 1733 (plichtsverzuim (fugie))*. The prosecutor's name is taken from the case's text when it names a known advocaat-fiscaal or fiscaal, and in civil disputes the claimant is unknown (*Onbekende eiser*). `import` stores title, dates and place and links each case to its id (ExternalID context `EMDCCR`).

**Versions and derived ToCs** (`versions.py`): much of the archive exists in several versions (a letter to the Heren XVII and the same letter to the kamer Zeeland, enclosures copied into the Batavia *overgekomen brieven en papieren*, …). Which version is the copy is often unclear, so versions are symmetric.

```bash
uv run python -m segmentation match-versions          # same text elsewhere, for every inventory without ToC entries or court cases (~2.5 h, resumable)
uv run python -m segmentation derive-tocs             # → version_blocks.csv and derived_toc.csv
uv run python -m segmentation run <inv ...> --out derived_segments.csv   # uses derived_toc.csv for those inventories
```

- `match-versions` compares every scan with every scan of the inventories within ±2 years. The measure is containment: the share of its word 3-grams found in the other scan, with formulaic 3-grams that occur in more than 100 scans ignored. A containment ≥ 0.3 means the same text. The shingles of each inventory are cached in `data/shingles/` and the matches are written to `data/versions/<inv>.parquet`.
- `derive-tocs` chains the matches into version blocks: runs of the same text in both volumes, in the same order. It then maps the titled documents of the other volume through each block, keeping titles verbatim. These are ToC entries from OBP/NT/typoscripts, or court cases. The same document found in several versions becomes one entry, and the others are listed as its versions.
- `run` uses the derived entries as forced starts, each moved by at most one scan to the best start evidence. Documents found inside an entry become its subdocuments. `import` stores the titles and dates, and keeps the source and other versions in the document's evidence.

The output of `run` has one row per segment: kind, boundary type, first / last scan, ToC id and title, parent ToC id, and the start/end log-odds.

New predictors (e.g. the text-embedding first/last-page model) are added as a `Predictor` subclass; per-scan probabilities in a CSV/parquet file (`filename, p_first, p_last`) can be plugged in without code via `SEGMENTATION_EXTERNAL_SCORES=path`. Refit the model afterwards.

Ground truth: the validated inventories are read from `data/<inv> - Document Segmentation.csv` (`ground_truth.py`: documents, subdocuments, same-scan boundaries and non-document page types; step 15's import mis-reads same-scan rows such as `END/START` with `id1/id2`, e.g. in 1388). The General Missives (from the database) are used for evaluation only: starts and missive interiors, since a missive's ToC entry may include appendices. Where their annotation starts on the missive's title page (55 of 924: a short scan, usually followed by a blank verso), the start is moved to the text. The model is fitted on the validated inventories only.

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

Sync with:

```bash
aws s3 sync objects/inventory/ s3://globalise-data/objects/inventory --acl=public-read --content-encoding gzip
```

### IIIF Manifests

```bash
uv run python export_manifests.py
```

Exports individual IIIF 3.0 Manifest JSON files for every inventory (`objects/inventory/<number>.manifest.json`). Output is gzipped and ready for S3 upload.

### IIIF Collection

```bash
uv run python export_collection.py
```

Exports a top-level IIIF 3.0 Collection JSON file (`objects/inventory/collection.json`) that references all inventory manifests. Output is gzipped and ready for S3 upload.

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
├── 9_import_annotation_pages_exist.py # Import annotation page flags (step 9)
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
