"""
Loaders for the three ToC sources of the OBP inventories, and the matching
between them.

1. The GLOBALISE Digitized Indexes CSV (basis of script 7). Its ID is the
   LEADING identifier: it is what script 7 stored as ExternalID(OBP_INDEX).
     - ID 1–170132:      the TANAP index
     - ID 170133–227526: OCR'd TANAP typoscripts (noisy, no section info)
   SECTION is the DEEL (part number) and nothing more. INVENTORY NUMBER is
   numeric, so suffixed inventories (9014A, 1430A, …) are collapsed onto
   their base number.

2. "TANAP VOC OBP Nationaal Archief.xlsx": same IDs as the CSV's TANAP part.
   LINK-2 holds the katern label (settlement + DEEL, e.g. "Ternate 3") and a
   page range with recto/verso markers ("14v-16").

3. "OBP NT_gecorrigeerd.xlsx": the TANAP index in the physical order of the
   volumes, with corrected (suffixed) inventory numbers. Its ID numbering does
   NOT match the other two; it has one row per (document × document type),
   and consecutive rows of the same document are collapsed here. NT entries
   are matched to CSV IDs by inventory, description and start folio.
"""

import logging
import os
import re
import zipfile

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_CANDIDATES = [
    os.path.join(DATA_DIR, "globalise_digitized_indexes_enriched.csv"),
    os.path.join(DATA_DIR, "globalise_digitized_indexes.csv"),
    os.path.join(
        DATA_DIR,
        "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799).csv.zip",
    ),
]
TANAP_XLSX = os.path.join(DATA_DIR, "TANAP VOC OBP Nationaal Archief.xlsx")
NT_XLSX = os.path.join(DATA_DIR, "OBP NT_gecorrigeerd.xlsx")

FUZZY_THRESHOLD = 80
FOLIO_BONUS = 10  # added to the text score when the start folios agree

LINK_RE = re.compile(r"^Katern: (?P<katern>.*?)\s*(?:\s+Pagina (?P<pagina>.*))?$")
PAGINA_RE = re.compile(r"^(\d+)([rv]?)(?:-(\d+)([rv]?))?$", re.IGNORECASE)
SIDES = {"r": "Recto", "v": "Verso"}


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"[^\w]+", " ", regex=True).str.strip()


def base_inventory(s: pd.Series) -> pd.Series:
    """'9014A' → '9014'; '1088a' → '1088'; '1053' → '1053'."""
    return s.astype(str).str.upper().str.replace(r"[A-Z]+$", "", regex=True)


# ── loaders ───────────────────────────────────────────────────────────────────


def load_obp_csv(path: str | None = None) -> pd.DataFrame:
    """The CSV behind script 7, with csv_id, inventory_number and folio_start added."""
    path = path or next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"OBP CSV not found; tried {CSV_CANDIDATES}")
    logger.info("Reading %s …", path)
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(
                n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")
            )
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    df["csv_id"] = df["ID"].astype(int)
    df["inventory_number"] = df["INVENTORY NUMBER"].map(
        lambda v: str(int(v)) if pd.notna(v) else None
    )
    df["folio_start"] = pd.to_numeric(df["FOLIONUMBER (START OF DOCUMENT)"], errors="coerce")
    df["deel"] = pd.to_numeric(df["SECTION"], errors="coerce")
    logger.info("Loaded %d CSV rows", len(df))
    return df


def parse_link(link) -> tuple[str | None, str | None]:
    """Return (katern, pagina) from a LINK-2 value."""
    if not isinstance(link, str):
        return None, None
    rest = re.sub(r"^NL-HaNA/1\.04\.02/[^/]*///", "", link.strip()).rstrip("/")
    m = LINK_RE.match(rest)
    if not m:
        return None, None
    return (m.group("katern") or "").strip() or None, (m.group("pagina") or "").strip() or None


def parse_sides(pagina: str | None) -> tuple[str | None, str | None]:
    """'14v-16' → ('Verso', None); '14' → (None, None); unparseable → (None, None)."""
    if not pagina:
        return None, None
    m = PAGINA_RE.match(pagina.replace(" ", ""))
    if not m:
        return None, None
    start = SIDES.get(m.group(2).lower()) if m.group(2) else None
    end = SIDES.get(m.group(4).lower()) if m.group(4) else None
    return start, end


def load_tanap_excel(path: str = TANAP_XLSX) -> pd.DataFrame:
    """csv_id, katern and folio sides from the TANAP Excel."""
    logger.info("Reading %s …", path)
    df = pd.read_excel(path, usecols=["ID", "LINK-2"])
    parsed = df["LINK-2"].map(parse_link)
    sides = parsed.str[1].map(parse_sides)
    return pd.DataFrame(
        {
            "csv_id": df["ID"].astype(int),
            "tanap_katern": parsed.str[0],
            "folio_start_side": sides.str[0],
            "folio_end_side": sides.str[1],
        }
    )


def load_nt(path: str = NT_XLSX) -> pd.DataFrame:
    """
    One row per NT document, in physical order. nt_index is the NT ID of the
    document's first (document × type) row; NT IDs increase with physical
    order, so nt_index doubles as an order key.
    """
    logger.info("Reading %s …", path)
    df = pd.read_excel(path, sheet_name=0)
    df = df.sort_values("ID")
    df["nt_inventory"] = df["INVENTARIS NUMMER"].astype(str).str.strip()
    key = df["nt_inventory"] + "|" + df["ORIGINEEL"].astype(str).str.rsplit("@", n=1).str[0]
    first = key != key.shift()
    docs = df[first].copy()
    logger.info("NT: %d rows collapsed into %d documents", len(df), len(docs))
    katern = docs["KATERN"].map(lambda k: "" if pd.isna(k) else f" {int(k)}")
    return pd.DataFrame(
        {
            "nt_index": docs["ID"].astype(int).values,
            "nt_inventory": docs["nt_inventory"].values,
            "nt_katern": (docs["VESTIGING"].astype(str).str.strip() + katern).values,
            "nt_folio_start": pd.to_numeric(docs["PAGINA VAN"], errors="coerce").values,
            "nt_text": normalize_text(docs["INHOUD"]).values,
        }
    )


# ── matching ──────────────────────────────────────────────────────────────────


def _exact(nt: pd.DataFrame, csv: pd.DataFrame, nt_inv: str, csv_inv: str) -> pd.DataFrame:
    """Match on (inventory, text, folio, occurrence); returns nt_index → csv_id."""
    a = nt.rename(columns={nt_inv: "k_inv", "nt_text": "k_text", "nt_folio_start": "k_fs"})
    b = csv.rename(columns={csv_inv: "k_inv", "text": "k_text", "folio_start": "k_fs"})
    keys = ["k_inv", "k_text", "k_fs"]
    a = a.assign(k_n=a.groupby(keys, dropna=False).cumcount())
    b = b.assign(k_n=b.groupby(keys, dropna=False).cumcount())
    return a[keys + ["k_n", "nt_index"]].merge(b[keys + ["k_n", "csv_id"]], on=keys + ["k_n"])[
        ["nt_index", "csv_id"]
    ]


def _fuzzy(nt: pd.DataFrame, csv: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Greedy best-match within base inventory; returns (nt_index, csv_id, score)."""
    out = []
    by_inv = {inv: g for inv, g in csv.groupby("base_inv")}
    for inv, grp in nt.groupby("base_inv"):
        cand = by_inv.get(inv)
        if cand is None:
            continue
        cands = list(zip(cand["csv_id"], cand["text"], cand["folio_start"]))
        used: set[int] = set()
        for nt_index, text, fs in zip(grp["nt_index"], grp["nt_text"], grp["nt_folio_start"]):
            best = None
            for csv_id, ctext, cfs in cands:
                if csv_id in used:
                    continue
                score = fuzz.token_sort_ratio(text, ctext) + (FOLIO_BONUS if fs == cfs else 0)
                if best is None or score > best[1]:
                    best = (csv_id, score)
            if best and best[1] >= FUZZY_THRESHOLD:
                used.add(best[0])
                out.append((nt_index, best[0], best[1]))
    return out


def match_nt_to_csv(nt: pd.DataFrame, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Map every NT document to a CSV id, one-to-one. Tries, in order:
      1. same inventory number, identical description and start folio
      2. same base inventory number (NT '9014A' ↔ CSV '9014'), same text/folio
      3. fuzzy description match within the base inventory
    Returns nt with an added csv_id column (NaN when unmatched).
    """
    csv = csv[["csv_id", "inventory_number", "folio_start"]].assign(
        text=normalize_text(csv["DESCRIPTION"]),
        base_inv=base_inventory(csv["inventory_number"]),
    )
    nt = nt.assign(base_inv=base_inventory(nt["nt_inventory"]))

    pairs = []
    for nt_inv, csv_inv, label in (
        ("nt_inventory", "inventory_number", "exact"),
        ("base_inv", "base_inv", "exact (base inventory)"),
    ):
        done_nt = {p for p, _ in pairs}
        done_csv = {c for _, c in pairs}
        m = _exact(
            nt[~nt["nt_index"].isin(done_nt)], csv[~csv["csv_id"].isin(done_csv)], nt_inv, csv_inv
        )
        pairs += list(m.itertuples(index=False, name=None))
        logger.info("NT ↔ CSV %s: %d", label, len(m))

    done_nt = {p for p, _ in pairs}
    done_csv = {c for _, c in pairs}
    fz = _fuzzy(nt[~nt["nt_index"].isin(done_nt)], csv[~csv["csv_id"].isin(done_csv)])
    pairs += [(n, c) for n, c, _ in fz]
    logger.info("NT ↔ CSV fuzzy: %d", len(fz))

    mapping = pd.DataFrame(pairs, columns=["nt_index", "csv_id"])
    assert mapping["csv_id"].is_unique and mapping["nt_index"].is_unique
    nt = nt.drop(columns="base_inv").merge(mapping, on="nt_index", how="left")
    logger.info("NT documents matched to a CSV id: %d of %d", nt["csv_id"].notna().sum(), len(nt))
    return nt
