import sqlite3
import re
from tqdm import tqdm

DB_PATH = "globalise_documents.db"

FILENAME_PATTERN = re.compile(r"_(\d+)[A-Za-z]*$")


def extract_scan_order(filename: str):
    """
    Extract scan order from:
      - ..._0001   -> 1
      - ..._0571A  -> 571
    Returns None if no numeric suffix exists (e.g. ..._P01)
    """
    match = FILENAME_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def add_scan_order_column(conn):
    cursor = conn.cursor()
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(scan)")}

    if "scan_order" not in columns:
        cursor.execute("ALTER TABLE scan ADD COLUMN scan_order INTEGER")
        conn.commit()


def populate_scan_order(conn):
    cursor = conn.cursor()

    scans = cursor.execute(
        "SELECT id, filename FROM scan"
    ).fetchall()

    updates = []
    skipped = []

    for scan_id, filename in tqdm(scans, desc="Processing scans"):
        order = extract_scan_order(filename)

        if order is None:
            skipped.append(filename)
            continue

        updates.append((order, scan_id))

    cursor.executemany(
        "UPDATE scan SET scan_order = ? WHERE id = ?",
        updates
    )

    conn.commit()

    print(f"\nUpdated scans: {len(updates)}")
    print(f"Skipped scans: {len(skipped)}")

    if skipped:
        print("\nExamples of skipped filenames:")
        for f in skipped[:20]:
            print("  ", f)


def main():
    with sqlite3.connect(DB_PATH) as conn:
        add_scan_order_column(conn)
        populate_scan_order(conn)


if __name__ == "__main__":
    main()