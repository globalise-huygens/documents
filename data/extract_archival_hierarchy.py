"""
Extract archival hierarchy from EAD files for database import.

This script parses EAD XML files and extracts:
- Fonds information
- Series and subseries hierarchy
- File groups
- Individual files (inventory numbers) with metadata

The data is structured for import into a database with Series and Inventory models.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
from lxml import etree as ET


@dataclass(kw_only=True)
class Base:
    code: str
    title: str


@dataclass(kw_only=True)
class Collection(Base):
    hasPart: list = field(default_factory=list)
    uri: str = field(default_factory=str)
    level: str = ""


@dataclass(kw_only=True)
class Fonds(Collection):
    level: str = "fonds"


@dataclass(kw_only=True)
class Series(Collection):
    level: str = "series"


@dataclass(kw_only=True)
class FileGroup(Collection):
    date: str = ""
    level: str = "filegrp"


@dataclass(kw_only=True)
class File(Base):
    uri: str
    date: str
    metsid: str = ""
    level: str = "file"
    parent_code: str = ""  # To track which series/filegrp this belongs to


def normalize_id(s: str) -> str:
    """
    Normalize an identifier so that it can be used in a URI.

    This function replaces white spaces, apostrophes, slashes
    and colons with a dash. It also removes all non-alphanumeric
    characters except for a dot and a dash.

    Args:
        s (str): Identifier to normalize.

    Returns:
        str: Normalized identifier.

    >>> normalize_id("7.27A, 7.37A")
    '7.27A-7.37A'
    """
    s = s.replace(" ", "-")
    s = s.replace("'", "-")
    s = s.replace("/", "-")
    s = s.replace(":", "-")

    s = "".join([c for c in s if c.isalnum() or c in "-."])

    while "--" in s:
        s = s.replace("--", "-")

    return s


def normalize_title(s: str) -> str:
    """
    Normalize a title so that it can be used in a URI.

    This function can be used when no identifier is available.
    It substitutes any diacritics in a title and converts it
    to lowercase. The same normalization as in normalize_id
    is applied.

    Args:
        s (str): Title to normalize.

    Returns:
        str: Normalized title.

    >>> normalize_title("Condé-sur-l'Escaut")
    'conde-sur-l-escaut'
    """
    from unidecode import unidecode

    s = normalize_id(s)
    s = unidecode(s).lower().strip()

    return s


def parse_ead(ead_file_path: str, filter_codes: set = set()) -> Fonds:
    """
    Parse an EAD file and extract the archival hierarchy.

    Args:
        ead_file_path (str): Path to the EAD file.
        filter_codes (set, optional): Set of inventory numbers to include. Defaults to empty set (all).

    Returns:
        Fonds: The parsed fonds with its hierarchy.
    """
    tree = ET.parse(ead_file_path)

    fonds_code = tree.find("eadheader/eadid").text
    fonds_title = tree.find("eadheader/filedesc/titlestmt/titleproper").text
    permalink_el = tree.find("eadheader/eadid[@url]")
    permalink = permalink_el.attrib["url"] if permalink_el is not None else ""

    fonds = Fonds(
        code=fonds_code,
        title=fonds_title,
        uri=permalink,
    )

    series_els = tree.findall(".//c[@level='series']")
    subseries_els = tree.findall(".//dsc[@type='combined']/c[@level='subseries']")
    dsc_el = tree.find(".//dsc[@type='combined']")

    if series_els:
        for series_el in series_els:
            s = get_series(series_el, parent_code=fonds_code, filter_codes=filter_codes)
            if s:
                fonds.hasPart.append(s)
    elif subseries_els:
        for subseries_el in subseries_els:
            s = get_series(
                subseries_el, parent_code=fonds_code, filter_codes=filter_codes
            )
            if s:
                fonds.hasPart.append(s)
    elif dsc_el is not None:
        parts = get_file_and_filegrp_els(
            dsc_el, parent_code=fonds_code, filter_codes=filter_codes
        )
        fonds.hasPart += parts

    return fonds


def get_series(
    series_el, parent_code: str = "", filter_codes: set = set()
) -> Optional[Series]:
    """
    Extract a series or subseries from an EAD element.

    Args:
        series_el: The XML element representing the series.
        parent_code (str): The code of the parent collection.
        filter_codes (set, optional): Set of inventory numbers to include.

    Returns:
        Series or None: The parsed series, or None if empty.
    """
    series_code_el = series_el.find("did/unitid[@type='series_code']")
    series_title = "".join(series_el.find("did/unittitle").itertext()).strip()

    while "  " in series_title:  # double space
        series_title = series_title.replace("  ", " ")

    if series_code_el is not None:
        series_code = normalize_id(series_code_el.text)
    else:
        series_code = normalize_title(series_title)

    s = Series(code=series_code, title=series_title)

    parts = get_file_and_filegrp_els(
        series_el, parent_code=series_code, filter_codes=filter_codes
    )
    s.hasPart += parts

    # Only return series if it has content
    if parts:
        return s
    return None


def get_file_and_filegrp_els(
    series_el, parent_code: str = "", filter_codes: set = set()
):
    """
    Extract files and file groups from a series element.

    Args:
        series_el: The XML element to process.
        parent_code (str): The code of the parent collection.
        filter_codes (set, optional): Set of inventory numbers to include.

    Returns:
        list: List of File, FileGroup, or Series objects.
    """
    parts = []

    file_and_filegrp_els = series_el.xpath("child::*")
    for el in file_and_filegrp_els:
        if el.get("level") == "file":
            i = get_file(el, parent_code, filter_codes)

        elif el.get("otherlevel") == "filegrp":
            i = get_filegrp(el, parent_code, filter_codes)

        elif el.get("level") == "subseries":
            i = get_series(el, parent_code, filter_codes)
        else:
            continue

        if i:
            parts.append(i)

    return parts


def get_filegrp(
    filegrp_el, parent_code: str = "", filter_codes: set = set()
) -> Optional[FileGroup]:
    """
    Extract a file group from an EAD element.

    Args:
        filegrp_el: The XML element representing the file group.
        parent_code (str): The code of the parent collection.
        filter_codes (set, optional): Set of inventory numbers to include.

    Returns:
        FileGroup or None: The parsed file group, or None if empty.
    """
    filegrp_code = normalize_id(filegrp_el.find("did/unitid").text)

    # Title
    filegrp_title = "".join(filegrp_el.find("did/unittitle").itertext()).strip()
    while "  " in filegrp_title:  # double space
        filegrp_title = filegrp_title.replace("  ", " ")

    if filegrp_code == "div.nrs.":
        filegrp_code += normalize_title(filegrp_title)

    # Date
    date_el = filegrp_el.find("did/unitdate")
    if date_el is not None:
        date = date_el.attrib.get("normal", date_el.attrib.get("text", ""))
    else:
        date = ""

    filegrp = FileGroup(
        code=filegrp_code,
        title=filegrp_title,
        date=date,
    )

    parts = get_file_and_filegrp_els(
        filegrp_el, parent_code=filegrp_code, filter_codes=filter_codes
    )
    filegrp.hasPart += parts

    # Only return if it has content
    if parts:
        return filegrp
    return None


def get_file(
    file_el, parent_code: str = "", filter_codes: set = set()
) -> Optional[File]:
    """
    Extract a file (inventory number) from an EAD element.

    Args:
        file_el: The XML element representing the file.
        parent_code (str): The code of the parent collection.
        filter_codes (set, optional): Set of inventory numbers to include.

    Returns:
        File or None: The parsed file, or None if filtered out or invalid.
    """
    did = file_el.find("did")

    # Inventory number
    inventorynumber_el = did.xpath("unitid[@identifier or @type='BD']")
    if inventorynumber_el:
        inventorynumber = inventorynumber_el[0].text
    else:
        return None

    # Filter on selection
    if filter_codes and inventorynumber not in filter_codes:
        return None

    # URI
    permalink_el = did.find("unitid[@type='handle']")
    permalink = permalink_el.text if permalink_el is not None else ""

    # Title
    title = "".join(did.find("unittitle").itertext()).strip()
    while "  " in title:  # double space
        title = title.replace("  ", " ")

    # Date
    date_el = did.find("unitdate")
    if date_el is not None:
        date = date_el.attrib.get("normal", date_el.attrib.get("text", ""))
    else:
        date = ""

    # METS id
    metsid_el = did.find("dao")
    if metsid_el is not None:
        metsid = metsid_el.attrib["href"].split("/")[-1]
    else:
        metsid = ""

    f = File(
        code=inventorynumber,
        title=title,
        uri=permalink,
        date=date,
        metsid=metsid,
        parent_code=parent_code,
    )

    return f


def parse_date_range(date_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse EAD date string into start and end dates.

    Args:
        date_str (str): Date string from EAD (e.g., "1607/1613", "1608", "1795-04-20/1795-10-10")

    Returns:
        tuple: (date_start, date_end) as ISO date strings or None
    """
    if not date_str:
        return None, None

    # Split on slash for date ranges
    if "/" in date_str:
        parts = date_str.split("/")
        start = parts[0].strip()
        end = parts[1].strip() if len(parts) > 1 else start
    else:
        start = end = date_str.strip()

    def normalize_date(d: str) -> Optional[str]:
        """Convert various date formats to ISO date (YYYY-MM-DD)."""
        if not d:
            return None

        # Already ISO format (YYYY-MM-DD)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            return d

        # Year only (YYYY) -> use January 1st for start, December 31st for end
        if re.match(r"^\d{4}$", d):
            return d  # Return year only, will be handled in conversion

        # Partial dates like YYYY-MM -> add day
        if re.match(r"^\d{4}-\d{2}$", d):
            return f"{d}-01"

        return None

    date_start = normalize_date(start)
    date_end = normalize_date(end)

    return date_start, date_end


def flatten_hierarchy(collection: Collection) -> dict:
    """
    Flatten the hierarchical structure into lists suitable for database import.

    Maps to Series (sets) and Inventory tables with proper relationships.
    Generates UUIDs for Series since they don't have unique identifiers in EAD.

    Args:
        collection (Collection): The root collection (Fonds).

    Returns:
        dict: Dictionary with 'series' (sets) and 'inventories' lists, plus 'inventory_series' relationships.
    """
    series_list = []
    inventories = []
    inventory_series_relations = []

    # Track series by their path to generate consistent UUIDs
    series_uuid_map = {}

    def get_or_create_series_id(path: str) -> str:
        """Get or create a UUID for a series based on its path."""
        if path not in series_uuid_map:
            series_uuid_map[path] = str(uuid.uuid4())
        return series_uuid_map[path]

    def traverse(item, parent_path="", series_ids_in_path=None):
        """Recursively traverse the hierarchy."""
        if series_ids_in_path is None:
            series_ids_in_path = []

        if isinstance(item, File):
            # This is an inventory number
            date_start, date_end = parse_date_range(item.date)

            inventory_data = {
                "id": str(uuid.uuid4()),
                "inventory_number": item.code,
                "na_identifier": item.metsid if item.metsid else None,
                "handle": item.uri if item.uri else None,
                "date_start": date_start,
                "date_end": date_end,
            }
            inventories.append(inventory_data)

            # Create InventoryTitle entry (separate table)
            # Note: You'll need to handle this in the import script
            inventory_data["titles"] = [item.title] if item.title else []

            # Link to only the last (most specific) series in the path
            if series_ids_in_path:
                inventory_series_relations.append(
                    {
                        "inventory_id": inventory_data["id"],
                        "series_id": series_ids_in_path[-1],
                    }
                )
        else:
            # This is a series (Fonds, Series, or FileGroup)
            current_path = f"{parent_path}/{item.code}" if parent_path else item.code
            series_id = get_or_create_series_id(current_path)

            # Determine parent series ID
            parent_series_id = None
            if parent_path:
                parent_series_id = get_or_create_series_id(parent_path)

            series_data = {
                "id": series_id,
                "title": f"{item.code} - {item.title}",
                "part_of_id": parent_series_id,
                "code": item.code,  # Extra field for reference
                "level": item.level,  # Extra field for reference
                "path": current_path,  # Extra field for reference
            }
            series_list.append(series_data)

            # Add this series to the path for children
            new_series_path = series_ids_in_path + [series_id]

            # Traverse children
            for child in item.hasPart:
                traverse(
                    child, parent_path=current_path, series_ids_in_path=new_series_path
                )

    traverse(collection)

    return {
        "series": series_list,
        "inventories": inventories,
        "inventory_series": inventory_series_relations,
    }


def export_to_json(fonds: Fonds, output_path: str):
    """
    Export the parsed hierarchy to a JSON file.

    Args:
        fonds (Fonds): The parsed fonds.
        output_path (str): Path to the output JSON file.
    """
    data = flatten_hierarchy(fonds)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"Exported {len(data['series'])} series, {len(data['inventories'])} inventories, "
        f"and {len(data['inventory_series'])} relationships to {output_path}"
    )


def main(
    ead_file_path: str,
    output_json_path: str = "",
    filter_codes_path: str = "",
) -> dict:
    """
    Extract archival hierarchy from an EAD file and prepare for database import.

    Args:
        ead_file_path (str): Path to the EAD file.
        output_json_path (str, optional): Path to output JSON file. If empty, no file is written.
        filter_codes_path (str, optional): Path to a JSON file with a list of inventory numbers to include.

    Returns:
        dict: Dictionary with 'sets' and 'inventory_numbers' lists.
    """

    # Restrict to a selection of inventory numbers
    if filter_codes_path and os.path.exists(filter_codes_path):
        with open(filter_codes_path, "r") as infile:
            code_selection = set(json.load(infile))
    else:
        code_selection = set()

    # Parse EAD, filter on relevant inventory numbers
    print(f"Parsing EAD file: {ead_file_path}")
    fonds = parse_ead(ead_file_path, filter_codes=code_selection)

    # Flatten hierarchy for database import
    data = flatten_hierarchy(fonds)

    print(
        f"Found {len(data['series'])} series and {len(data['inventories'])} inventories"
    )

    # Export to JSON if path provided
    if output_json_path:
        export_to_json(fonds, output_json_path)

    return data


if __name__ == "__main__":

    ead_file = "data/1.04.02.xml"
    output_file = "data/archival_hierarchy.json"

    filter_codes_file = "data/inventories.json"

    if os.path.exists(ead_file):
        result = main(
            ead_file_path=ead_file,
            output_json_path=output_file,
            filter_codes_path=filter_codes_file,
        )

        print("\nSample series:")
        for s in result["series"][:5]:
            print(f"  - {s['code']}: {s['title'][:60]}... (ID: {s['id'][:8]}...)")

        print("\nSample inventories:")
        for inv in result["inventories"][:5]:
            print(
                f"  - {inv['inventory_number']}: {inv.get('titles', [''])[0][:60]}... (ID: {inv['id'][:8]}...)"
            )

        print(
            f"\nTotal inventory-series relationships: {len(result['inventory_series'])}"
        )
    else:
        print(f"EAD file not found: {ead_file}")
        print("Please provide a valid EAD file path.")
