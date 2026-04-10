"""
Import annotation pages from GLOBALISE annotation zips.
Processes transcription and entity annotations to populate:
- LayoutElement: layout elements like headers, signatures, paragraphs
- EntityMention: recognized entity mentions (dates, places, persons)
- LayoutElement2EntityMention: links entity mentions found within layout elements
"""

import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import (
    Base,
    Page,
    Scan,
    LayoutElement,
    EntityMention,
    LayoutElement2EntityMention,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default layout types to import (keeps the DB small)
DEFAULT_LAYOUT_TYPES = {"header", "signature-mark"}


class AnnotationPageImporter:
    """Import annotation pages from zips containing transcriptions and entities."""

    def __init__(self, db_url: str, layout_types: Optional[Set[str]] = None):
        """
        Args:
            db_url: SQLAlchemy database URL.
            layout_types: Set of layout types to import. None means import ALL types.
                          Default at CLI level is DEFAULT_LAYOUT_TYPES.
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine, checkfirst=True)
        self.session = Session(self.engine)
        self.layout_types = layout_types  # None = all

        # Pre-load existing annotation identifiers for fast duplicate checking
        self._existing_layout_ids: Set[str] = set()
        self._existing_entity_ids: Set[str] = set()
        self._existing_links: Set[Tuple[str, str]] = set()
        self._load_existing_ids()

    def _load_existing_ids(self):
        """Pre-load existing annotation identifiers for fast duplicate checking."""
        t0 = time.time()
        self._existing_layout_ids = set(
            self.session.scalars(select(LayoutElement.annotation_identifier)).all()
        )
        self._existing_entity_ids = set(
            self.session.scalars(select(EntityMention.annotation_identifier)).all()
        )
        rows = self.session.execute(
            select(
                LayoutElement2EntityMention.layout_element_id,
                LayoutElement2EntityMention.entity_mention_id,
            )
        ).all()
        self._existing_links = {(r[0], r[1]) for r in rows}
        logger.info(
            f"Loaded {len(self._existing_layout_ids)} layout, "
            f"{len(self._existing_entity_ids)} entity, "
            f"{len(self._existing_links)} link existing IDs in {time.time()-t0:.1f}s"
        )

    def find_zip_files(self, data_dir: str = "data/ap") -> List[str]:
        """Find all zip files in the annotation pages directory."""
        ap_dir = Path(data_dir)
        if not ap_dir.exists():
            logger.warning(f"Annotation pages directory not found: {ap_dir}")
            return []

        zips = sorted(ap_dir.glob("*.zip"))
        logger.info(f"Found {len(zips)} annotation page zips")
        return [str(z) for z in zips]

    def extract_zip_contents(self, zip_path: str) -> Tuple[Dict, Dict]:
        """Extract transcription and entity annotation pages from a zip.

        Returns:
            Tuple of (transcriptions_dict, entities_dict) where keys are filenames
            without extension and values are parsed JSON.
        """
        transcriptions = {}
        entities = {}

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract transcriptions
                for name in zf.namelist():
                    if name.startswith("transcriptions/") and name.endswith(".json"):
                        filename = Path(name).stem
                        with zf.open(name) as f:
                            transcriptions[filename] = json.load(f)

                # Extract entities
                for name in zf.namelist():
                    if name.startswith("entities/") and name.endswith(".json"):
                        filename = Path(name).stem
                        with zf.open(name) as f:
                            entities[filename] = json.load(f)

            logger.info(
                f"Extracted {len(transcriptions)} transcription and {len(entities)} entity annotation pages"
            )
            return transcriptions, entities
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return {}, {}

    def parse_layout_elements(self, transcription_data: Dict) -> Dict[str, Dict]:
        """Parse layout elements (blocks) from transcription annotation page.

        Returns dict mapping block id to element info:
        {
            'region_id_123': {
                'annotation_id': 'full_url_id',
                'layout_type': 'header',
                'x_center': 1234,  # SVG centroid x-coordinate (for page assignment)
            }
        }
        """
        layout_elements = {}

        items = transcription_data.get("items", [])

        # Collect blocks and their layout type
        for item in items:
            if item.get("textGranularity") == "block":
                anno_id = item.get("id", "")
                block_id = anno_id.split("#")[-1] if "#" in anno_id else anno_id

                # Get layout type from body
                layout_type = None
                for body in item.get("body", []):
                    if body.get("type") == "SpecificResource":
                        source = body.get("source", {})
                        if isinstance(source, dict):
                            layout_type = source.get("label")
                        break

                # Get x_center from SVG selector for left/right page determination
                x_center = None
                for target in item.get("target", []):
                    if (
                        isinstance(target, dict)
                        and target.get("type") == "SpecificResource"
                    ):
                        selector = target.get("selector", {})
                        if (
                            isinstance(selector, dict)
                            and selector.get("type") == "SvgSelector"
                        ):
                            svg_value = selector.get("value", "")
                            coords = re.findall(r"(\d+),\d+", svg_value)
                            if coords:
                                xs = [int(x) for x in coords]
                                x_center = (min(xs) + max(xs)) // 2
                        break

                if layout_type:
                    # Skip if layout_type filtering is active and type not in allowed set
                    if (
                        self.layout_types is not None
                        and layout_type not in self.layout_types
                    ):
                        continue

                    layout_elements[block_id] = {
                        "annotation_id": anno_id,
                        "layout_type": layout_type,
                        "x_center": x_center,
                    }

        return layout_elements

    def parse_entities(self, entity_data: Dict) -> List[Dict]:
        """Parse entities from entity annotation page.

        Returns list of entity dicts with only identifiers and types:
        [
            {
                'annotation_id': 'url_id',
                'entity_type': 'DATE',
            }
        ]
        """
        entities = []
        items = entity_data.get("items", [])

        for item in items:
            bodies = item.get("body", [])
            if not bodies:
                continue

            entity_body = bodies[0]  # Main entity definition

            # Extract entity type from classified_as
            entity_type = None
            classified_as = entity_body.get("classified_as", {})
            if isinstance(classified_as, dict):
                entity_type_id = classified_as.get("id", "")
                # Extract type from "gan:DATE" -> "DATE"
                if ":" in entity_type_id:
                    entity_type = entity_type_id.split(":")[-1]
                else:
                    entity_type = classified_as.get("_label", "UNKNOWN")

            entities.append(
                {
                    "annotation_id": item.get("id", ""),
                    "entity_type": entity_type or "UNKNOWN",
                }
            )

        return entities

    def link_entities_to_layout_elements(
        self,
        transcription_data: Dict,
        layout_elements: Dict[str, Dict],
        entity_list: List[Dict],
        entity_data: Dict,
    ) -> Dict[str, Set[str]]:
        """Link entities to layout elements by tracing word -> line -> block.

        Returns dict mapping entity_annotation_id to set of block_ids
        that contain that entity.
        """
        if not entity_data:
            return {}

        # Build map of word annotation ids to their parent line ids
        word_to_line = {}
        line_to_block = {}

        items = transcription_data.get("items", [])

        # Pass 1: Map lines to their parent blocks
        for item in items:
            if item.get("textGranularity") == "line":
                line_anno_id = item.get("id", "")

                # Find parent block in targets
                for target in item.get("target", []):
                    if isinstance(target, dict) and target.get("type") == "Annotation":
                        block_anno_id = target.get("id", "")
                        if block_anno_id:
                            block_id = (
                                block_anno_id.split("#")[-1]
                                if "#" in block_anno_id
                                else block_anno_id
                            )
                            line_to_block[line_anno_id] = block_id
                        break

        # Pass 2: Map words to their parent lines
        for item in items:
            if item.get("textGranularity") == "word":
                word_anno_id = item.get("id", "")

                # Find parent line in targets
                for target in item.get("target", []):
                    if isinstance(target, dict) and target.get("type") == "Annotation":
                        line_anno_id = target.get("id", "")
                        if line_anno_id:
                            word_to_line[word_anno_id] = line_anno_id
                        break

        # Pass 3: For each entity, find which blocks it belongs to
        entity_to_blocks = {}
        entity_items = entity_data.get("items", [])

        for entity_item in entity_items:
            entity_anno_id = entity_item.get("id", "")
            blocks = set()

            # Look at entity targets to find word annotations
            targets = entity_item.get("target", [])
            for target in targets:
                if isinstance(target, dict):
                    target_id = target.get("id", "")
                    # Check if this is a word annotation
                    if target_id and "word" in target_id:
                        # Trace word -> line -> block
                        if target_id in word_to_line:
                            line_anno_id = word_to_line[target_id]
                            if line_anno_id in line_to_block:
                                block_id = line_to_block[line_anno_id]
                                blocks.add(block_id)

            entity_to_blocks[entity_anno_id] = blocks

        return entity_to_blocks

    def find_scan_for_page(self, page_number_str: str) -> Optional[Page]:
        """Find a page by scan filename pattern.

        For single scans, returns the one page.
        For double scans, returns the first page (caller should use find_pages_for_scan instead).
        """
        scan = self._find_scan(page_number_str)
        if scan and scan.pages:
            return scan.pages[0]
        return None

    def _find_scan(self, page_number_str: str) -> Optional[Scan]:
        """Find a scan by filename pattern."""
        stmt = select(Scan).where(Scan.filename.like(f"%{page_number_str}%")).limit(1)
        return self.session.scalars(stmt).first()

    def _get_page_for_layout_element(self, scan: Scan, x_center: Optional[int]) -> Page:
        """Determine which page a layout element belongs to on a scan.

        For double scans (2 pages), uses x_center to assign:
        - left half (x_center < width/2) -> verso page
        - right half (x_center >= width/2) -> recto page
        For single scans, returns the only page.
        """
        pages = scan.pages
        if len(pages) == 1:
            return pages[0]

        if len(pages) == 2 and x_center is not None:
            midpoint = scan.width // 2
            # Sort so we can reliably pick verso/recto
            verso_page = next(
                (p for p in pages if p.recto_verso and p.recto_verso.value == "Verso"),
                None,
            )
            recto_page = next(
                (p for p in pages if p.recto_verso and p.recto_verso.value == "Recto"),
                None,
            )

            if verso_page and recto_page:
                return verso_page if x_center < midpoint else recto_page

        # Fallback: return first page
        return pages[0]

    def import_zip(self, zip_path: str) -> bool:
        """Import a single annotation zip file."""
        logger.info(f"\nProcessing {Path(zip_path).name}...")

        transcriptions, entities_data = self.extract_zip_contents(zip_path)

        if not transcriptions:
            logger.warning(f"No transcriptions found in {zip_path}")
            return False

        # Process each scan's annotations
        processed_count = 0
        for page_id, trans_data in transcriptions.items():
            try:
                logger.info(f"  Processing scan {page_id}...")

                # Find the corresponding scan in database
                scan = self._find_scan(page_id)
                if not scan or not scan.pages:
                    logger.warning(f"  No scan/page found for {page_id}, skipping")
                    continue

                # Parse layout elements
                layout_elements = self.parse_layout_elements(trans_data)
                logger.info(f"    Found {len(layout_elements)} layout elements")

                # Parse entities
                entity_list = []
                entity_data_for_page = None
                if page_id in entities_data:
                    entity_data_for_page = entities_data[page_id]
                    entity_list = self.parse_entities(entity_data_for_page)
                    logger.info(f"    Found {len(entity_list)} entities")

                # Import layout elements and entities
                self.import_layout_elements_and_entities(
                    scan, layout_elements, entity_list, trans_data, entity_data_for_page
                )

                self.session.commit()
                processed_count += 1
            except Exception as e:
                self.session.rollback()
                logger.error(f"  Error processing page {page_id}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully processed {processed_count} pages from {zip_path}")
        return processed_count > 0

    def import_layout_elements_and_entities(
        self,
        scan: Scan,
        layout_elements: Dict[str, Dict],
        entity_list: List[Dict],
        transcription_data: Optional[Dict] = None,
        entity_data: Optional[Dict] = None,
    ):
        """Import layout elements and entities for a scan.

        For double-page scans, assigns each layout element and entity mention
        to the correct page (verso/recto) based on x_center coordinates.

        Only entity mentions linked to at least one imported layout element
        are imported (when layout_types filtering is active).
        """
        # Link entities to layout elements FIRST to know which entities to import
        entity_to_blocks: Dict[str, Set[str]] = {}
        if transcription_data and entity_data:
            entity_to_blocks = self.link_entities_to_layout_elements(
                transcription_data, layout_elements, entity_list, entity_data
            )

        # Create layout elements (skip existing via in-memory set)
        layout_element_map = {}  # Maps block_id to db entity

        for block_id, block_info in layout_elements.items():
            anno_id = block_info["annotation_id"]

            if anno_id in self._existing_layout_ids:
                # Need to fetch existing for linking
                stmt = select(LayoutElement).where(
                    LayoutElement.annotation_identifier == anno_id
                )
                layout_elem = self.session.scalars(stmt).first()
            else:
                page = self._get_page_for_layout_element(
                    scan, block_info.get("x_center")
                )
                layout_elem = LayoutElement(
                    page_id=page.id,
                    annotation_identifier=anno_id,
                    layout_type=block_info["layout_type"],
                )
                self.session.add(layout_elem)
                self._existing_layout_ids.add(anno_id)

            if layout_elem:
                layout_element_map[block_id] = layout_elem

        self.session.flush()

        # Determine which entities to import: only those linked to imported layout elements
        linked_entity_ids = set()
        for entity_anno_id, block_ids in entity_to_blocks.items():
            if block_ids & set(layout_element_map.keys()):
                linked_entity_ids.add(entity_anno_id)

        # If no layout type filter, import all entities; otherwise only linked ones
        if self.layout_types is None:
            entities_to_import = entity_list
        else:
            entities_to_import = [
                e for e in entity_list if e["annotation_id"] in linked_entity_ids
            ]

        # Create entity mentions (skip existing via in-memory set)
        entity_map = {}

        for entity_info in entities_to_import:
            anno_id = entity_info["annotation_id"]

            if anno_id in self._existing_entity_ids:
                stmt = select(EntityMention).where(
                    EntityMention.annotation_identifier == anno_id
                )
                entity = self.session.scalars(stmt).first()
            else:
                # Determine page from linked layout element (for double scans)
                linked_blocks = entity_to_blocks.get(anno_id, set())
                page = scan.pages[0]  # fallback
                for block_id in linked_blocks:
                    if block_id in layout_element_map:
                        page = self.session.get(
                            Page, layout_element_map[block_id].page_id
                        )
                        break

                entity = EntityMention(
                    page_id=page.id,
                    annotation_identifier=anno_id,
                    entity_type=entity_info["entity_type"],
                )
                self.session.add(entity)
                self._existing_entity_ids.add(anno_id)

            if entity:
                entity_map[anno_id] = entity

        self.session.flush()

        # Create layout_element <-> entity_mention links (skip existing via in-memory set)
        for entity_anno_id, block_ids in entity_to_blocks.items():
            if entity_anno_id not in entity_map:
                continue
            entity = entity_map[entity_anno_id]
            for block_id in block_ids:
                if block_id not in layout_element_map:
                    continue
                layout_elem = layout_element_map[block_id]
                link_key = (layout_elem.id, entity.id)
                if link_key not in self._existing_links:
                    self.session.add(
                        LayoutElement2EntityMention(
                            layout_element_id=layout_elem.id,
                            entity_mention_id=entity.id,
                        )
                    )
                    self._existing_links.add(link_key)

    def import_all_zips(self, data_dir: str = "data/ap"):
        """Import all annotation zips from the data directory."""
        zip_files = self.find_zip_files(data_dir)

        if not zip_files:
            logger.warning("No zip files found")
            return

        layout_desc = (
            "all types"
            if self.layout_types is None
            else ", ".join(sorted(self.layout_types))
        )
        logger.info(
            f"Starting import of {len(zip_files)} annotation zips (layout types: {layout_desc})"
        )
        t0 = time.time()

        for i, zip_path in enumerate(zip_files, 1):
            try:
                zt = time.time()
                self.import_zip(zip_path)
                elapsed = time.time() - zt
                logger.info(f"  [{i}/{len(zip_files)}] Done in {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"Error importing {zip_path}: {e}", exc_info=True)

        total = time.time() - t0
        logger.info(f"Import complete in {total:.1f}s")
        self.session.close()


def main():
    """Main entry point for importing all annotation pages."""
    parser = argparse.ArgumentParser(
        description="Import annotation pages (layout elements & entities) from GLOBALISE zips."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Import ALL layout types (paragraph, marginalia, catch-word, page-number, etc.). "
            "Default imports only header and signature-mark elements."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/ap",
        help="Directory containing annotation zip files (default: data/ap)",
    )
    parser.add_argument(
        "--db-url",
        default="sqlite:///globalise_documents.db",
        help="Database URL (default: sqlite:///globalise_documents.db)",
    )
    args = parser.parse_args()

    layout_types = None if args.all else DEFAULT_LAYOUT_TYPES
    importer = AnnotationPageImporter(args.db_url, layout_types=layout_types)
    importer.import_all_zips(args.data_dir)


if __name__ == "__main__":
    main()
