"""
Versions of the same text elsewhere in the archive, and ToC entries derived
from them.

Much of the archive exists in several versions (a letter sent to the Heren
XVII and to the kamer Zeeland, enclosures copied into the Batavia
'overgekomen brieven en papieren', ...). Which version is the copy is often
unclear, so versions are symmetric: neither is 'the original'.

1. Scan matches (`match_inventory`): every scan is compared with every scan of
   the inventories within `window` years (inventory dates), by the share of
   its word 3-grams (texts.normalize_tokens, hashed) that also occur in the
   other scan ('containment'). 3-grams that occur in many scans of the pool
   (formulas such as salutations) are ignored. A containment >= 0.3 means the
   same text (two independent HTR transcriptions of two handwritten copies);
   around 0.2 it is usually a related text.
2. Version blocks (`version_blocks`): chains of matches in which both volumes
   advance together (gaps of a few scans allowed: copies differ in page size,
   blank pages have no text) are runs of the same text in both volumes.
3. Derived ToC (`derive_toc`): the titled documents of the other volume
   (ToC entries of OBP / NT / typoscripts, court cases) whose start lies in a
   block are mapped onto this volume through the block, titles verbatim. When
   several versions give an entry for the same document, one is kept (a title
   that does not lean on the entry before it, such as 'Een dito ...', from the
   strongest block) and the others are listed as its versions.
"""

import logging
import os
import sqlite3
import time
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .texts import TEXT_DIR, normalize_tokens

logger = logging.getLogger("segmentation")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MATCH_DIR = os.path.join(DATA_DIR, "versions")
SHINGLE_DIR = os.path.join(DATA_DIR, "shingles")  # per-inventory cache of the hashed shingles
SEGMENTATION_METHOD = "Segmentation model"

SHINGLE = 3  # words per shingle
HASH_SPACE = 1 << 26
MIN_SHINGLES = 15  # scans with fewer (after dropping formulas) are not compared
MAX_DF = 100  # shingles in more scans of the pool are formulas, not text
MIN_CONTAINMENT = 0.15  # matches kept for the alignment
TOP_K = 5  # matches kept per scan
SAME_TEXT = 0.3  # containment from which two scans hold the same text
DITO_TITLE = r"(?i)^\s*(een\s+|enige\s+)?(dito|d[:\.]?o\b|idem|als\s*voren|item)"  # titles that depend on the entry before


# ── scan matches ──────────────────────────────────────────────────────────────


def scan_shingles(text: str) -> np.ndarray:
    w = normalize_tokens(text or "")
    return np.unique(np.fromiter((zlib.crc32(" ".join(w[i : i + SHINGLE]).encode()) % HASH_SPACE for i in range(len(w) - SHINGLE + 1)), dtype=np.int64))


@dataclass
class InventoryShingles:
    filenames: np.ndarray
    matrix: sparse.csr_matrix  # scans × HASH_SPACE, 1 where the scan has the shingle


def load_shingles(inventory_number: str, cache_dir: str = SHINGLE_DIR) -> InventoryShingles:
    """Hashed shingles of every scan with text; computed once, then read from cache_dir."""
    path = os.path.join(cache_dir, f"{inventory_number}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        filenames, indptr, indices = z["filenames"], z["indptr"], z["indices"]
    else:
        folder = os.path.join(TEXT_DIR, f"inv={inventory_number}")
        if os.path.isdir(folder):
            df = pd.read_parquet(folder, columns=["filename", "text"]).sort_values("filename")
            sets = [scan_shingles(t) for t in df["text"]]
            filenames = df["filename"].to_numpy(dtype=object)
        else:
            sets, filenames = [], np.array([], dtype=object)
        indptr = np.concatenate([[0], np.cumsum([len(x) for x in sets])]).astype(np.int64)
        indices = (np.concatenate(sets) if sets else np.array([], dtype=np.int64)).astype(np.int32)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, filenames=filenames, indptr=indptr, indices=indices)
    m = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(filenames), HASH_SPACE))
    return InventoryShingles(filenames, m)


class ShingleCache:
    """Shingles per inventory, kept while they are in the window of the targets
    being matched (targets are processed in date order)."""

    def __init__(self, max_scans: int = 2_000_000):
        self.items: OrderedDict[str, InventoryShingles] = OrderedDict()
        self.max_scans = max_scans

    def get(self, inv: str) -> InventoryShingles:
        if inv in self.items:
            self.items.move_to_end(inv)
        else:
            self.items[inv] = load_shingles(inv)
            while sum(len(v.filenames) for v in self.items.values()) > self.max_scans and len(self.items) > 1:
                self.items.popitem(last=False)
        return self.items[inv]


def inventory_dates(conn: sqlite3.Connection) -> pd.DataFrame:
    inv = pd.read_sql("SELECT inventory_number AS inv, date_start, date_end FROM inventory", conn)
    # the year from the text: pandas timestamps cannot hold dates before 1677
    inv["y0"] = pd.to_numeric(inv["date_start"].str[:4], errors="coerce")
    inv["y1"] = pd.to_numeric(inv["date_end"].str[:4], errors="coerce")
    inv = inv.set_index("inv")
    # undated: the years of the nearest dated inventory numbers (same series as a rule)
    num = pd.to_numeric(inv.index.str.extract(r"^(\d+)")[0], errors="coerce").to_numpy()
    dated = inv["y0"].notna().to_numpy() & inv["y1"].notna().to_numpy()
    for k in np.flatnonzero(~dated & ~np.isnan(num)):
        near = np.flatnonzero(dated & (np.abs(num - num[k]) <= 2))
        if len(near):
            inv.iloc[k, inv.columns.get_loc("y0")] = inv["y0"].iloc[near].min()
            inv.iloc[k, inv.columns.get_loc("y1")] = inv["y1"].iloc[near].max()
    return inv


def pool_for(target: str, dates: pd.DataFrame, window: int) -> list[str]:
    """Inventories whose years overlap the target's ± window (undated ones always; an undated
    target is not matched: comparing it with the whole archive does not fit in memory)."""
    y0, y1 = dates.at[target, "y0"], dates.at[target, "y1"]
    if pd.isna(y0) or pd.isna(y1):
        return []
    ok = dates["y0"].isna() | dates["y1"].isna() | ((dates["y0"] <= y1 + window) & (dates["y1"] >= y0 - window))
    return [i for i in dates.index[ok] if i != target]


def match_inventory(target: str, pool: list[str], cache: ShingleCache) -> pd.DataFrame:
    """For each scan of `target`, its best matches (containment >= MIN_CONTAINMENT) in the pool."""
    t = cache.get(target)
    cols = ["scan", "other_scan", "containment", "other_containment"]
    if not len(t.filenames):
        return pd.DataFrame(columns=cols)
    parts = [cache.get(i) for i in pool]
    parts = [p for p in parts if len(p.filenames)]
    P = sparse.vstack([p.matrix for p in parts], format="csr")
    other = np.concatenate([p.filenames for p in parts])
    # formulas: shingles in many scans of the pool
    df = np.bincount(P.indices, minlength=HASH_SPACE)
    keep = df[t.matrix.indices] <= MAX_DF
    # copies: eliminate_zeros works in place and would corrupt the cached matrix
    T = sparse.csr_matrix((t.matrix.data * keep, t.matrix.indices.copy(), t.matrix.indptr.copy()), shape=t.matrix.shape)
    T.eliminate_zeros()
    t_size = np.diff(T.indptr)
    p_rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    p_size = np.bincount(p_rows, weights=(df[P.indices] <= MAX_DF), minlength=P.shape[0])  # sizes without formulas
    overlap = (T @ P.T).tocsr()
    rows = []
    for i in range(T.shape[0]):
        if t_size[i] < MIN_SHINGLES:
            continue
        a, b = overlap.indptr[i], overlap.indptr[i + 1]
        if a == b:
            continue
        j, o = overlap.indices[a:b], overlap.data[a:b]
        c = o / t_size[i]
        ok = (c >= MIN_CONTAINMENT) & (p_size[j] >= MIN_SHINGLES)
        if not ok.any():
            continue
        j, o, c = j[ok], o[ok], c[ok]
        top = np.argsort(-c)[:TOP_K]
        for k in top:
            rows.append((t.filenames[i], other[j[k]], round(float(c[k]), 3), round(float(o[k] / max(p_size[j[k]], 1)), 3)))
    return pd.DataFrame(rows, columns=cols)


def untitled_inventories(conn: sqlite3.Connection) -> list[str]:
    """Inventories without a titled document of the segmentation method (no ToC entries, no court cases)."""
    rows = conn.execute(
        "SELECT i.inventory_number FROM inventory i WHERE NOT EXISTS (SELECT 1 FROM document d "
        "JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? WHERE d.inventory_id = i.id AND d.title IS NOT NULL)",
        (SEGMENTATION_METHOD,),
    ).fetchall()
    return sorted((r[0] for r in rows), key=lambda n: (len(n), n))


def unindexed_inventories(conn: sqlite3.Connection) -> list[str]:
    """Inventories without an index of their own: no OBP/TANAP ToC entries and no court
    cases (EMDCCR). Unlike untitled_inventories, this does not change when derived ToC
    entries are imported."""
    rows = conn.execute(
        "SELECT i.inventory_number FROM inventory i WHERE NOT EXISTS (SELECT 1 FROM document d "
        "JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' WHERE d.inventory_id = i.id) "
        "AND NOT EXISTS (SELECT 1 FROM document d JOIN document2external_id de ON de.document_id = d.id "
        "JOIN external_id e ON e.id = de.external_id AND e.context = 'EMDCCR' WHERE d.inventory_id = i.id)"
    ).fetchall()
    return sorted((r[0] for r in rows), key=lambda n: (len(n), n))


def _cache_one(inv: str) -> int:
    return len(load_shingles(inv).filenames)


def cache_shingles(inventories: list[str], workers: int = 8):
    """Tokenize the inventories not yet in SHINGLE_DIR, in parallel, with progress."""
    from multiprocessing import Pool

    todo = [i for i in inventories if not os.path.exists(os.path.join(SHINGLE_DIR, f"{i}.npz"))]
    logger.info("Shingles: %d of %d inventories to tokenize (%d already cached)", len(todo), len(inventories), len(inventories) - len(todo))
    t0, scans = time.time(), 0
    with Pool(workers) as pool:
        for k, n in enumerate(pool.imap_unordered(_cache_one, todo, chunksize=4), 1):
            scans += n
            if k % 100 == 0 or k == len(todo):
                eta = (time.time() - t0) / k * (len(todo) - k) / 60
                logger.info("  tokenized %d/%d inventories (%d scans, %.0fs; about %.0f min to go)", k, len(todo), scans, time.time() - t0, eta)


def match_all(targets: list[str], conn: sqlite3.Connection, window: int = 2, out_dir: str = MATCH_DIR, resume: bool = True):
    """Scan matches of every target, one parquet per target in out_dir; progress with an ETA."""
    os.makedirs(out_dir, exist_ok=True)
    dates = inventory_dates(conn)
    counts = dict(conn.execute("SELECT i.inventory_number, count(*) FROM scan s JOIN inventory i ON i.id = s.inventory_id GROUP BY 1").fetchall())
    targets = [t for t in targets if t in dates.index]
    done = {t for t in targets if resume and os.path.exists(os.path.join(out_dir, f"{t}.parquet"))}
    todo = sorted((t for t in targets if t not in done), key=lambda t: (dates.at[t, "y0"] if not pd.isna(dates.at[t, "y0"]) else 9999, t))
    if done:
        logger.info("Resuming: %d of %d inventories already matched", len(done), len(targets))
    needed = sorted({i for t in todo for i in pool_for(t, dates, window)} | set(todo))
    cache_shingles(needed)
    total = sum(counts.get(t, 0) for t in todo)
    cache, t0, scans_done = ShingleCache(), time.time(), 0
    for k, target in enumerate(todo, 1):
        t1 = time.time()
        pool = pool_for(target, dates, window)
        if not pool:
            logger.warning("%d/%d %s: no dates; skipped", k, len(todo), target)
            continue
        m = match_inventory(target, pool, cache)
        m.to_parquet(os.path.join(out_dir, f"{target}.parquet"), index=False)
        scans_done += counts.get(target, 0)
        n_scans = counts.get(target, 0)
        same = m[m["containment"] >= SAME_TEXT]["scan"].nunique() if len(m) else 0
        eta = (time.time() - t0) / max(scans_done, 1) * (total - scans_done) / 60
        logger.info("%d/%d %s: %d scans, %d with the same text elsewhere (%.0f%%); pool %d inventories; %.0fs (about %.0f min to go)",
                    k, len(todo), target, n_scans, same, 100 * same / max(n_scans, 1), len(pool), time.time() - t1, eta)


def load_matches(target: str, out_dir: str = MATCH_DIR) -> pd.DataFrame:
    path = os.path.join(out_dir, f"{target}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["scan", "other_scan", "containment", "other_containment"])


# ── version blocks ────────────────────────────────────────────────────────────


MAX_GAP = 8  # scans skipped in either volume within one block
MAX_PARTIAL_GAP = 10  # pieces of one document further apart stay separate entries
MIN_BLOCK_SCORE = 0.6  # summed containment of a block's matches


_positions: dict[str, dict[str, int]] = {}


def scan_positions(conn: sqlite3.Connection, inventories: list[str]) -> dict[str, int]:
    """Position (0..n-1 in scan order) of every scan of these inventories (cached per inventory)."""
    out = {}
    for inv in inventories:
        if inv not in _positions:
            fns = [r[0] for r in conn.execute(
                "SELECT s.filename FROM scan s JOIN inventory i ON i.id = s.inventory_id WHERE i.inventory_number = ? ORDER BY s.scan_order, s.filename", (inv,))]
            _positions[inv] = {f: k for k, f in enumerate(fns)}
        out.update(_positions[inv])
    return out


def _inv(filename: str) -> str:
    return filename.split("_")[2]


def version_blocks(matches: pd.DataFrame, pos: dict[str, int]) -> pd.DataFrame:
    """
    Runs of the same text in the target and one other volume: chains of matches
    with both positions non-decreasing and gaps <= MAX_GAP, highest summed
    containment first; each match is used once. One row per block.
    """
    cols = ["other_inventory", "start", "end", "other_start", "other_end", "n_matches", "score", "pairs"]
    if matches.empty:
        return pd.DataFrame(columns=cols)
    m = matches.copy()
    m["t"] = m["scan"].map(pos)
    m["o"] = m["other_scan"].map(pos)
    m["oinv"] = m["other_scan"].map(_inv)
    m = m.dropna(subset=["t", "o"])
    blocks = []
    for oinv, g in m.groupby("oinv"):
        g = g.sort_values(["t", "o"])
        t, o, c = g["t"].to_numpy(int), g["o"].to_numpy(int), g["containment"].to_numpy(float)
        used = np.zeros(len(g), dtype=bool)
        while True:
            best = np.where(used, -np.inf, c)
            prev = np.full(len(g), -1)
            for i in range(len(g)):
                if used[i]:
                    continue
                for j in range(i - 1, -1, -1):
                    if t[i] - t[j] > MAX_GAP:
                        break
                    if used[j] or t[j] == t[i] or not (0 <= o[i] - o[j] <= MAX_GAP):
                        continue
                    if best[j] + c[i] > best[i]:
                        best[i], prev[i] = best[j] + c[i], j
            i = int(np.argmax(best))
            if not np.isfinite(best[i]) or best[i] < MIN_BLOCK_SCORE and not (c[i] >= 0.5):
                break
            chain = []
            while i >= 0:
                chain.append(i)
                i = prev[i]
            chain.reverse()
            used[chain] = True
            # a block needs two matches, or one strong one
            if len(chain) >= 2 or c[chain[0]] >= 0.5:
                blocks.append((oinv, t[chain[0]], t[chain[-1]], o[chain[0]], o[chain[-1]], len(chain), float(c[chain].sum()),
                               [(int(t[k]), int(o[k])) for k in chain]))
    return pd.DataFrame(blocks, columns=cols).sort_values(["start", "score"], ascending=[True, False]).reset_index(drop=True)


# ── derived ToC ───────────────────────────────────────────────────────────────


def titled_documents(conn: sqlite3.Connection, inventory: str, pos: dict[str, int]) -> pd.DataFrame:
    """Documents of the segmentation method with a title (ToC entries, court
    cases) in `inventory`, with their first/last scan position and index ids."""
    docs = pd.read_sql(
        "SELECT d.id AS doc_id, d.title, d.part_of_id, d.date_earliest_begin, d.date_latest_end, "
        "       min(s.filename) AS first_scan, max(s.filename) AS last_scan "
        "FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? "
        "JOIN inventory i ON i.id = d.inventory_id AND i.inventory_number = ? "
        "JOIN page2document p2d ON p2d.document_id = d.id JOIN page p ON p.id = p2d.page_id JOIN scan s ON s.id = p.scan_id "
        "WHERE d.title IS NOT NULL GROUP BY d.id",
        conn, params=(SEGMENTATION_METHOD, inventory),
    )
    if docs.empty:
        return docs
    ids = pd.read_sql(
        "SELECT de.document_id AS doc_id, e.context || ':' || e.identifier AS ext FROM document2external_id de "
        "JOIN external_id e ON e.id = de.external_id WHERE de.document_id IN (SELECT d.id FROM document d "
        "JOIN inventory i ON i.id = d.inventory_id AND i.inventory_number = ?)", conn, params=(inventory,),
    ).groupby("doc_id")["ext"].apply(lambda s: ";".join(sorted(s)))
    docs["index_ids"] = docs["doc_id"].map(ids).fillna("")
    docs["start"] = docs["first_scan"].map(pos)
    docs["end"] = docs["last_scan"].map(pos)
    return docs.dropna(subset=["start", "end"])


def _map(pairs: list[tuple[int, int]], o: float, side: str) -> int:
    """Target position of other-volume position `o` through a block's matched pairs."""
    ts, os_ = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if o <= os_[0]:
        return int(ts[0] - (os_[0] - o)) if side == "start" else int(ts[0])
    if o >= os_[-1]:
        return int(ts[-1] + (o - os_[-1])) if side == "end" else int(ts[-1])
    k = int(np.searchsorted(os_, o, side="left"))
    (t1, o1), (t2, o2) = (ts[k - 1], os_[k - 1]), (ts[k], os_[k])
    return int(round(t1 + (o - o1) * (t2 - t1) / (o2 - o1))) if o2 > o1 else int(t1 if side == "start" else t2)


def derive_toc(blocks: pd.DataFrame, others: dict[str, pd.DataFrame], n_scans: int, slack: int = 2) -> pd.DataFrame:
    """
    ToC entries of this volume from the titled documents of the volumes it
    shares text with. An entry is taken when its start lies within a block
    (or up to `slack` scans before it: a title page has little text) and at
    least half of it, or 3 scans, is covered by the block.
    """
    rows = []
    for b in blocks.itertuples():
        docs = others.get(b.other_inventory)
        if docs is None or docs.empty:
            continue
        inside = docs[(docs["start"] >= b.other_start - slack) & (docs["start"] <= b.other_end)]
        # the block starts inside a document of the other volume: the innermost one it continues
        around = docs[(docs["start"] < b.other_start - slack) & (docs["end"] >= b.other_start + 1)]
        around = around.assign(span=around["end"] - around["start"]).sort_values("span").head(1)
        for partial, group in ((False, inside), (True, around)):
            for d in group.itertuples():
                covered = min(d.end, b.other_end) - max(d.start, b.other_start) + 1
                if covered < min(3, (d.end - d.start + 1) / 2):
                    continue
                s = b.start if partial else max(0, _map(b.pairs, d.start, "start"))
                e = min(n_scans - 1, max(s, _map(b.pairs, min(d.end, b.other_end + slack), "end")))
                rows.append(_entry_row(b, d, s, e, partial))
    if not rows:
        return pd.DataFrame()
    return _merge_versions(pd.DataFrame(rows))


def _entry_row(b, d, s: int, e: int, partial: bool) -> dict:
    return {
        "start": s, "end": e, "title": d.title, "date_begin": d.date_earliest_begin, "date_end": d.date_latest_end,
        "source_inventory": b.other_inventory, "source_doc_id": d.doc_id, "source_part_of_id": d.part_of_id,
        "source_index_ids": d.index_ids, "source_start": int(d.start), "source_end": int(d.end),
        "block_score": round(b.score, 2), "block_matches": b.n_matches,
        "partial": partial,  # this volume has the text from the middle of the document on: it began before the shared text
    }


def _merge_versions(toc: pd.DataFrame) -> pd.DataFrame:
    # an entry reached through several blocks: the strongest block's mapping; the pieces of a
    # partial one (a long document continuing through several blocks) are joined when they
    # are at most MAX_PARTIAL_GAP scans apart
    full = toc[~toc["partial"]].sort_values("block_score", ascending=False).drop_duplicates("source_doc_id")
    pieces = []
    for _, g in toc[toc["partial"] & ~toc["source_doc_id"].isin(full["source_doc_id"])].groupby("source_doc_id"):
        g = g.sort_values("start")
        run = [g.iloc[0]]
        for _, r in list(g.iterrows())[1:] + [(None, None)]:
            if r is not None and r["start"] <= max(x["end"] for x in run) + MAX_PARTIAL_GAP:
                run.append(r)
                continue
            best = max(run, key=lambda x: x["block_score"]).copy()
            best["start"], best["end"] = min(x["start"] for x in run), max(x["end"] for x in run)
            pieces.append(best)
            if r is not None:
                run = [r]
    toc = pd.concat([full, pd.DataFrame(pieces)]) if pieces else full
    toc = toc.sort_values(["start", "end"])
    # the same document in several versions (start within 2 scans, similar end): one entry
    out, cluster = [], []

    def same_document(a, b) -> bool:
        """b is another version of cluster lead a: from another volume, at about the same scans."""
        return (b["source_inventory"] not in {c["source_inventory"] for c in cluster}
                and abs(a["start"] - b["start"]) <= 2 and abs(a["end"] - b["end"]) <= max(2, 0.2 * (a["end"] - a["start"] + 1)))

    def flush():
        if not cluster:
            return
        g = pd.DataFrame(cluster)
        g["dito"] = g["title"].astype(str).str.match(DITO_TITLE)
        g = g.sort_values(["dito", "block_score"], ascending=[True, False])
        lead = g.iloc[0].drop("dito").copy()
        lead["versions"] = ";".join(f"{r.source_inventory}:{r.source_index_ids or r.source_doc_id}" for r in g.iloc[1:].itertuples())
        lead["n_versions"] = len(g)
        out.append(lead)
        cluster.clear()

    pending = [r for _, r in toc.iterrows()]
    while pending:  # entries of one source volume stay separate, also when they start on the same scan
        rest = []
        for r in pending:
            if not cluster or same_document(cluster[0], r):
                cluster.append(r)
            elif abs(r["start"] - cluster[0]["start"]) <= 2:
                rest.append(r)  # same place, but not this document: its own cluster later
            else:
                flush()
                cluster.append(r)
        flush()
        pending = rest
    return pd.DataFrame(out).sort_values(["start", "end"], ascending=[True, False]).reset_index(drop=True)


ANCHOR_WINDOW = 3  # scans around the mapped start searched for the source's start text
ANCHOR_MIN = 0.2  # containment of the source's start scan that counts as found


def anchor_starts(toc: pd.DataFrame, target: str, fn_of: dict, n_scans: int) -> pd.DataFrame:
    """
    Put each (non-partial) entry's start on the target scan, within ANCHOR_WINDOW
    of the mapped start, that holds most of the text of the source document's
    start scan (containment of its shingles; formulas are rare enough not to
    decide this, the whole scan is compared). verified: that text was found
    (containment >= ANCHOR_MIN). Partial entries start where the shared text
    starts and are verified by their block.
    """
    if toc.empty:
        return toc
    rows: dict[str, dict] = {}

    def shingles(inv: str) -> dict:
        if inv not in rows:
            x = load_shingles(inv)
            rows[inv] = {f: x.matrix[k].indices for k, f in enumerate(x.filenames)}
        return rows[inv]

    toc = toc.copy()
    toc["anchor"], toc["verified"] = np.nan, toc["partial"].astype(bool)
    tgt = shingles(target)
    for i, r in toc[~toc["partial"].astype(bool)].iterrows():
        src = shingles(r["source_inventory"]).get(fn_of.get((r["source_inventory"], int(r["source_start"]))))
        if src is None or len(src) < MIN_SHINGLES:
            continue
        best, best_k = 0.0, None
        for k in range(max(0, r["start"] - ANCHOR_WINDOW), min(n_scans, r["start"] + ANCHOR_WINDOW + 1)):
            t = tgt.get(fn_of.get((target, k)))
            if t is not None:
                c = len(np.intersect1d(src, t, assume_unique=True)) / len(src)
                if c > best or (c == best and best_k is not None and abs(k - r["start"]) < abs(best_k - r["start"])):
                    best, best_k = c, k
        toc.at[i, "anchor"] = round(best, 3)
        if best >= ANCHOR_MIN:
            toc.at[i, "verified"] = True
            shift = best_k - r["start"]
            toc.at[i, "start"] = best_k
            toc.at[i, "end"] = max(best_k, r["end"] + shift if r["end"] + shift < n_scans else r["end"])
    return toc.sort_values(["start", "end"], ascending=[True, False]).reset_index(drop=True)


def derive_all(targets: list[str], conn: sqlite3.Connection, out: str | None, blocks_out: str):
    """Version blocks and (unless out is None) derived ToC entries of every matched target, with progress."""
    t0, all_toc, all_blocks = time.time(), [], []
    titled_cache: dict[str, pd.DataFrame] = {}
    for k, target in enumerate(targets, 1):
        m = load_matches(target)
        invs = sorted({target} | set(m["other_scan"].map(_inv)))
        pos = scan_positions(conn, invs)
        fn_of = {}
        for f, p in pos.items():
            fn_of[(_inv(f), p)] = f
        n = sum(1 for f in pos if _inv(f) == target)
        b = version_blocks(m, pos)
        if out is None:
            toc = pd.DataFrame()
        else:
            for o in b["other_inventory"].unique():
                if o not in titled_cache:
                    titled_cache[o] = titled_documents(conn, o, pos)
            toc = derive_toc(b, titled_cache, n)
            toc = anchor_starts(toc, target, {(_inv(f), p): f for f, p in pos.items()}, n)
        if len(b):
            bb = b.drop(columns="pairs").copy()
            bb.insert(0, "inventory", target)
            for c, inv_col in (("start", None), ("end", None), ("other_start", "other_inventory"), ("other_end", "other_inventory")):
                bb[c + "_scan"] = [fn_of.get((r[inv_col] if inv_col else target, int(r[c]))) for _, r in bb.iterrows()]
            all_blocks.append(bb)
        if len(toc):
            toc.insert(0, "inventory", target)
            toc["start_scan"] = [fn_of.get((target, int(x))) for x in toc["start"]]
            toc["end_scan"] = [fn_of.get((target, int(x))) for x in toc["end"]]
            all_toc.append(toc)
        covered = set()
        for r in b.itertuples():
            covered |= set(range(r.start, r.end + 1))
        if k % 25 == 0 or k == len(targets):
            logger.info("%d/%d %s: %d blocks covering %d of %d scans, %d derived entries (%.0fs; about %.0f min to go)",
                        k, len(targets), target, len(b), len(covered), n, len(toc), time.time() - t0, (time.time() - t0) / k * (len(targets) - k) / 60)
    if all_toc and out:
        pd.concat(all_toc).to_csv(out, index=False)
    if all_blocks:
        pd.concat(all_blocks).to_csv(blocks_out, index=False)
    logger.info("Wrote %d derived entries to %s and %d version blocks to %s",
                sum(len(t) for t in all_toc), out, sum(len(b) for b in all_blocks), blocks_out)


# ── segmentation with a derived ToC ───────────────────────────────────────────


def load_derived_toc(path: str) -> dict[str, pd.DataFrame]:
    """Derived ToC entries per inventory (the CSV of `derive-tocs`)."""
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_csv(path, dtype={"inventory": str, "source_inventory": str}, keep_default_na=False, na_values=[""])
    return {inv: g.reset_index(drop=True) for inv, g in d.groupby("inventory")}


def segment_with_derived(inv, model, entries: pd.DataFrame):
    """
    Segment an inventory whose ToC comes from other versions of its text: each
    entry's start (moved by at most one scan to the best start evidence) is a
    forced document start; the model decides the ends and finds the other
    documents. Documents starting inside an entry's span are its subdocuments;
    entries starting on the same scan as an earlier one are part of it.
    """
    from .predictors import compute_features
    from .segmenter import Result, Segment, next_text_scan, scan_scores, segment_scans

    if inv.features is None:
        compute_features(inv)
    sc = scan_scores(inv, model, use_toc=False)
    blank = inv.scans["is_blank"].to_numpy(dtype=bool)
    text_at = next_text_scan(blank)
    pos = {f: k for k, f in enumerate(inv.scans["filename"])}
    e = entries.copy()
    e["s"] = e["start_scan"].map(pos)
    e["e"] = e["end_scan"].map(pos)
    e = e.dropna(subset=["s", "e"]).astype({"s": int, "e": int})
    # an unverified start (its text not found in this volume) moves at most one scan to the best start evidence;
    # a start on a blank scan moves to the next scan with text
    verified = e["verified"].astype(str).str.lower().eq("true") if "verified" in e else pd.Series(False, index=e.index)
    e["placed"] = [int(text_at[s if v else max(range(max(0, s - 1), min(inv.n, s + 2)), key=lambda c: sc.start[c] + (0.5 if c == s else 0) - 1e6 * blank[c])])
                   for s, v in zip(e["s"], verified)]
    e = e.sort_values(["placed", "e"], ascending=[True, False]).reset_index(drop=True)
    forced = set(e["placed"])
    use_prior = np.ones(inv.n, dtype=bool)
    for r in e.itertuples():
        use_prior[r.placed : max(r.placed, r.e) + 1] = False  # an entry's length is given by its source
    raw, _ = segment_scans(sc, model.length_prior, model.max_length, forced, use_prior, blank)

    by_scan = {p: g for p, g in e.groupby("placed")}
    next_entry = sorted(by_scan)
    segments, prev_end, spans = [], -1, []  # spans: (start, end, entry_id) of placed entries, end as derived
    for a, b, bt in raw:
        if a > prev_end + 1:
            segments.append(Segment(prev_end + 1, a - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[a - 1])))
        prev_end = max(prev_end, b)
        if a in by_scan:
            first_id = None
            for k, r in enumerate(by_scan[a].itertuples()):
                entry_id = f"{inv.inventory_number}:{r.Index}"
                end = max(a, r.e)
                # nested: within an earlier entry's span (another entry on the same scan: part of the first)
                parent = first_id if k else next((sid for s0, s1, sid in reversed(spans) if s0 < a and end <= s1), None)
                sg = Segment(a, b, "toc", bt if k == 0 else "shared", float(sc.start[a]), float(sc.end[b]))
                sg.derived = _derived_info(r, entry_id, parent)
                segments.append(sg)
                spans.append((a, end, entry_id))
                first_id = first_id or entry_id
            continue
        # a subdocument of the innermost entry whose span it starts in (before the next entry)
        parent = next((sid for s0, s1, sid in reversed(spans) if s0 < a <= s1), None)
        sg = Segment(a, b, "subdoc" if parent else "unindexed", bt, float(sc.start[a]), float(sc.end[b]))
        if parent:
            sg.derived = {"parent_entry_id": parent}
        segments.append(sg)
    if prev_end < inv.n - 1:
        segments.append(Segment(prev_end + 1, inv.n - 1, "non-document", "gap", float(sc.start[prev_end + 1]), float(sc.end[-1])))
    return Result(inv.inventory_number, segments, [], 0.0)


def _derived_info(r, entry_id: str, parent: str | None) -> dict:
    def val(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else x

    return {
        "entry_id": entry_id,
        "parent_entry_id": parent,
        "title": val(r.title),
        "date_begin": val(r.date_begin),
        "date_end": val(r.date_end),
        "derived_start": r.start_scan,
        "moved": int(r.placed - r.s),
        "source": {"inventory": r.source_inventory, "index_ids": val(r.source_index_ids), "scans": [int(r.source_start), int(r.source_end)]},
        "versions": [v for v in str(val(r.versions) or "").split(";") if v],
        "block_score": float(r.block_score),
        "verified": str(getattr(r, "verified", "")).lower() == "true",
        "anchor": None if pd.isna(getattr(r, "anchor", np.nan)) else float(r.anchor),
    }


# ── the text_version table ────────────────────────────────────────────────────


METHOD = f"shingle-containment-v1 ({SHINGLE}-grams, containment >= {MIN_CONTAINMENT}, max df {MAX_DF})"


def _scan_number(filename) -> int | None:
    try:
        return int(str(filename).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _inv_key(n: str):
    import re

    m = re.match(r"(\d+)(.*)", n)
    return (int(m.group(1)), m.group(2)) if m else (10**9, n)


def text_version_rows(blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Blocks (version_blocks.csv) as text_version rows: scan numbers, the lower
    inventory as a, and one row where the same run was found from both sides
    (when both volumes were targets): a row is dropped when a stronger row of
    the same pair overlaps it on both sides.
    """
    b = blocks.copy()
    for c in ("start_scan", "end_scan", "other_start_scan", "other_end_scan"):
        b[c] = b[c].map(_scan_number)
    b = b.dropna(subset=["start_scan", "end_scan", "other_start_scan", "other_end_scan"])
    swap = [_inv_key(str(i)) > _inv_key(str(o)) for i, o in zip(b["inventory"], b["other_inventory"])]
    swap = pd.Series(swap, index=b.index)
    r = pd.DataFrame({
        "inventory_a": np.where(swap, b["other_inventory"], b["inventory"]).astype(str),
        "scan_start_a": np.where(swap, b["other_start_scan"], b["start_scan"]).astype(int),
        "scan_end_a": np.where(swap, b["other_end_scan"], b["end_scan"]).astype(int),
        "inventory_b": np.where(swap, b["inventory"], b["other_inventory"]).astype(str),
        "scan_start_b": np.where(swap, b["start_scan"], b["other_start_scan"]).astype(int),
        "scan_end_b": np.where(swap, b["end_scan"], b["other_end_scan"]).astype(int),
        "n_matches": b["n_matches"].astype(int),
        "score": b["score"].astype(float).round(3),
    })
    keep = []
    for _, g in r.sort_values("score", ascending=False).groupby(["inventory_a", "inventory_b"], sort=False):
        kept = []
        for row in g.itertuples():
            if any(row.scan_start_a <= k.scan_end_a and k.scan_start_a <= row.scan_end_a and
                   row.scan_start_b <= k.scan_end_b and k.scan_start_b <= row.scan_end_b for k in kept):
                continue
            kept.append(row)
        keep += [k.Index for k in kept]
    r = r.loc[keep].sort_values(["inventory_a", "scan_start_a"]).reset_index(drop=True)
    r["method"] = METHOD
    return r


def load_text_versions(csv_paths: list[str], database_url: str):
    """Replace the text_version rows of this method with the blocks in csv_paths."""
    from sqlalchemy import create_engine, text

    from models import Base

    blocks = pd.concat([pd.read_csv(p, dtype={"inventory": str, "other_inventory": str}) for p in csv_paths])
    rows = text_version_rows(blocks)
    engine = create_engine(database_url)
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["text_version"]])
    with engine.begin() as con:
        removed = con.execute(text("DELETE FROM text_version WHERE method = :m"), {"m": METHOD}).rowcount
        rows.to_sql("text_version", con, if_exists="append", index=False, chunksize=5000)
    logger.info("text_version: %d rows from %d blocks (%d replaced), %d inventory pairs",
                len(rows), len(blocks), removed, rows.groupby(["inventory_a", "inventory_b"]).ngroups)
