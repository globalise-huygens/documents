"""
Full text per scan, from data/texts/inv=<inventory>/*.parquet: the
normalized_texts.parquet dump split by inventory (see README). Missing
inventories or scans give empty texts, so every text-based predictor
degrades to "no information".
"""

import os
import re
import unicodedata

import numpy as np
import pandas as pd

TEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "texts")


def split_texts(source: str, target: str = TEXT_DIR, memory_limit: str = "6GB"):
    """Split the corpus-wide text dump (columns filename, normalized_text,
    token_length) into one parquet per inventory under `target` (one-time)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(
        f"""COPY (SELECT regexp_extract(filename, '^NL-HaNA_1\\.04\\.02_([^_]+)_', 1) AS inv, filename,
                         normalized_text AS text, token_length
                  FROM read_parquet('{source}'))
            TO '{target}' (FORMAT parquet, PARTITION_BY (inv), OVERWRITE_OR_IGNORE)"""
    )


def load_texts(inventory_number: str, filenames: pd.Series) -> pd.Series:
    """Text of each scan (aligned with `filenames`), '' where unavailable."""
    folder = os.path.join(TEXT_DIR, f"inv={inventory_number}")
    if not os.path.isdir(folder):
        return pd.Series("", index=filenames.index)
    df = pd.read_parquet(folder, columns=["filename", "text"])
    texts = dict(zip(df["filename"], df["text"].fillna("")))
    return filenames.map(lambda f: texts.get(f, ""))


def normalize_tokens(text: str, prefix: int = 5) -> list[str]:
    """
    Crude, noise-tolerant word keys for matching early modern Dutch HTR text:
    accents stripped, lower case, ij→y, letters only, words of 3+ letters,
    truncated to `prefix` letters (absorbs inflection and spelling variants
    such as missive/missiven, generaal/generale).
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().replace("ij", "y")
    return [w[:prefix] for w in re.findall(r"[a-z]{3,}", text)]


# frequent function words and generic ToC vocabulary that say little about a specific entry
STOPWORDS = {
    w[:5]
    for w in "van den der het een aan aen door met voor ende and dat dit die des deze dese zyn syn als ter tot uyt uit "
    "over naar nae ook mede dato gedateerd gedateert anno sedert zoo soo hun haer haar zyne syne zynde".split()
}


class TextIndex:
    """
    IDF-weighted word overlap between short queries (ToC descriptions) and
    passages of the scans of one inventory. A scan can have several passages
    (its opening words, and the words after a closing formula, where a next
    document may start halfway down the page); a scan scores its best passage.
    """

    def __init__(self, passages: list[tuple[int, str]], n_scans: int):
        self.n = n_scans
        self.scan_of = np.array([i for i, _ in passages], dtype=int)
        self.postings: dict[str, list[int]] = {}
        for p, (_, t) in enumerate(passages):
            for w in set(normalize_tokens(t)) - STOPWORDS:
                self.postings.setdefault(w, []).append(p)
        n_docs = max(len(passages), 1)
        self.idf = {w: np.log((n_docs + 1) / (len(p) + 0.5)) for w, p in self.postings.items()}
        self.default_idf = np.log(n_docs + 1)

    def scores(self, query: str) -> np.ndarray:
        """Per scan: share of the query's (IDF-weighted) words in its best passage, 0..1."""
        words = set(normalize_tokens(query)) - STOPWORDS
        total = sum(self.idf.get(w, self.default_idf) for w in words)
        out = np.zeros(self.n)
        if not total or not len(self.scan_of):
            return out
        per_passage = np.zeros(len(self.scan_of))
        for w in words:
            if w in self.postings:
                per_passage[self.postings[w]] += self.idf[w]
        np.maximum.at(out, self.scan_of, per_passage / total)
        return out
