#!/usr/bin/env python3
"""
Step 26: clean spacing and hyphenation in the titles of the index entries.

The titles of the Digitized Indexes (step 19: the "- no linebreaks.xlsx" version)
were read by OCR from typoscripts, and some still carry its spacing errors. This
step corrects only those, and leaves the wording alone (no spelling corrections):

  spacing     no space before , ; . or : and inside brackets ("1709 .", "Eds .tot",
              "( ontvangen"): the space moves behind the mark ("Eds. tot"), a space
              after ; before a word (";item"), and leading, trailing and double spaces
  split words "gouver neur", "Am sterdam", "Brie fje": joined where one part hardly
              ever occurs as a word in the index titles (at most 3 times) and the
              joined form does (at least 10 times), never with a short function word
              ("Bossch en Hoven" stays)
  hyphens     "No-vember", "Cop-ie", "residen-t": joined where the hyphen cuts off one
              or two letters and the joined form occurs at least 10 times; not between
              equal vowels ("be-eedigde" is right)

Left as they are: the editors' ellipses ("uijt ... aen"), the VOC's abbreviations
with a colon ("ult:o", "Ed:s"), initials ("J;F. Gobius"), hyphenated compounds
("gesag-hebber"), = and „ inside words, and a ")" before a digit (a misread digit).

`review` writes every change to title_review.csv. `apply` sets the new titles on the
index entries and on the segmentation documents that copied them, where they still
have the reviewed title, and records old and new titles in title_corrections.csv
(the database keeps no history: only the titles change). Running it again changes
nothing.

  uv run python 26_clean_titles.py review [--out title_review.csv]
  uv run python 26_clean_titles.py apply [--review title_review.csv] [--dry-run] [--log title_corrections.csv]
"""

import argparse
import collections
import datetime
import os
import re
import sqlite3

import pandas as pd

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
TOC_METHOD, SEG_METHOD = "TANAP Digitized Index", "Segmentation model"
RARE, COMMON = 3, 10  # a part of a split word occurs at most RARE times as a word; the joined form at least COMMON times

PUNCT = re.compile(r"(?<=\w)\s+([,;]|:(?![A-Za-zæ])|\.(?!\.))(?=([A-Za-zæ])?)")  # "1709 .", "Eds .tot"; not "uijt ... aen"
OPEN = re.compile(r"(?<=\w)\s*\(\s+")  # "aangebracht( 20", "papieren ( ontvangen"
CLOSE = re.compile(r"\s+\)(?!\w)")  # not "item )17": there ")" is a misread digit
SPLIT = re.compile(r"\b([A-Za-zæ]+) ([a-zæ]+)\b")
HYPHEN = re.compile(r"\b([A-Za-zæ]+)-([a-zæ]+)\b")
STOP = {"en", "de", "het", "van", "in", "op", "te", "of", "die", "dat", "den", "der", "ter", "tot", "met", "voor", "na", "uit",
        "uijt", "bij", "aen", "aan", "is", "als"}
WORD = re.compile(r"[A-Za-zæ]+")


def connect(readonly=True):
    path = DATABASE_URL.removeprefix("sqlite:///")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True) if readonly else sqlite3.connect(path)


class Cleaner:
    def __init__(self, titles):
        self.words = collections.Counter(w for s in titles for w in WORD.findall(s.lower()))

    def spacing(self, s):
        s = re.sub(r"\s{2,}", " ", s.strip())
        s = PUNCT.sub(lambda m: m.group(1) + (" " if m.group(2) else ""), s)  # the space moves behind the mark
        s = re.sub(r"(?<=\w);(?=[a-zæ]{2})", "; ", s)  # ";item"; not "J;F." (initials)
        return CLOSE.sub(")", OPEN.sub(" (", s))

    def split(self, m):
        a, b = m.group(1), m.group(2)
        if a.lower() in STOP or b.lower() in STOP:
            return m.group(0)
        n = self.words
        return a + b if n[(a + b).lower()] >= COMMON and (n[a.lower()] <= RARE or n[b.lower()] <= RARE) else m.group(0)

    def hyphen(self, m):
        a, b = m.group(1), m.group(2)
        if a[-1].lower() == b[0].lower() and b[0].lower() in "aeiou":
            return m.group(0)
        return a + b if (len(a) <= 2 or len(b) <= 2) and self.words[(a + b).lower()] >= COMMON else m.group(0)

    def clean(self, s) -> tuple[str, list[str]]:
        rules = []
        for name, f in (("spacing", self.spacing), ("split words", lambda x: SPLIT.sub(self.split, x)),
                        ("hyphens", lambda x: HYPHEN.sub(self.hyphen, x))):
            new = f(s)
            if new != s:
                rules.append(name)
                s = new
        return s, rules


def entries(conn) -> pd.DataFrame:
    return pd.read_sql("""SELECT d.id document_id, d.title,
          (SELECT min(CAST(x.identifier AS INTEGER)) FROM document2external_id de JOIN external_id x ON x.id = de.external_id
            WHERE de.document_id = d.id AND x.context = 'OBP_INDEX') obp_index_id
        FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = ? WHERE d.title IS NOT NULL""",
                       conn, params=(TOC_METHOD,))


def review(out: str):
    t = entries(connect())
    cleaner = Cleaner(t.title)
    rows = []
    for r in t.itertuples():
        new, rules = cleaner.clean(r.title)
        if rules:
            rows.append({"document_id": r.document_id, "obp_index_id": r.obp_index_id, "rules": "; ".join(rules),
                         "old_title": r.title, "new_title": new})
    df = pd.DataFrame(rows)
    df["obp_index_id"] = df["obp_index_id"].astype("Int64")
    df.to_csv(out, index=False)
    print(f"{len(t):,} index titles; {len(df):,} to change → {out}")
    print(df.rules.str.split("; ").explode().value_counts().to_string())


def apply(path: str, dry_run: bool, log: str):
    rv = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    conn = connect(readonly=dry_run)
    # the segmentation documents that copied an entry's title, by index id (one query, not one per entry)
    copies = collections.defaultdict(list)
    for doc, ident in conn.execute("""SELECT d.id, x.identifier FROM document d
            JOIN document_identification_method m ON m.id = d.method_id AND m.name = ?
            JOIN document2external_id de ON de.document_id = d.id JOIN external_id x ON x.id = de.external_id AND x.context = 'OBP_INDEX'""",
                                   (SEG_METHOD,)):
        copies[ident].append(doc)
    stats = collections.Counter()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    changes = []
    for r in rv.itertuples():
        targets = [r.document_id] + (copies.get(str(r.obp_index_id), []) if isinstance(r.obp_index_id, str) else [])
        for doc in targets:
            row = conn.execute("SELECT title FROM document WHERE id = ?", (doc,)).fetchone()
            if row is None:
                continue
            if row[0] == r.new_title:
                stats["already cleaned"] += 1
                continue
            if row[0] != r.old_title:
                continue  # another title (a General Missive's, a merged entry's): not a copy of this one
            stats["index entries changed" if doc == r.document_id else "segmentation documents changed"] += 1
            changes.append({"document_id": doc, "obp_index_id": r.obp_index_id, "rules": r.rules,
                            "old_title": r.old_title, "new_title": r.new_title, "applied": now})
            if not dry_run:
                conn.execute("UPDATE document SET title = ? WHERE id = ?", (r.new_title, doc))
    if not dry_run:
        conn.commit()
        if changes:  # the record of what changed (the database keeps no history)
            pd.DataFrame(changes).to_csv(log, mode="a", header=not os.path.exists(log), index=False)
    print(f"reviewed changes: {len(rv):,}")
    for k, v in sorted(stats.items()):
        print(f"{k}: {v:,}")
    print("dry run: nothing changed" if dry_run else f"changes recorded in {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("--out", default="title_review.csv")
    a = sub.add_parser("apply")
    a.add_argument("--review", default="title_review.csv")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--log", default="title_corrections.csv", help="old and new titles of every document changed (appended)")
    args = ap.parse_args()
    if args.cmd == "review":
        review(args.out)
    else:
        apply(args.review, args.dry_run, args.log)


if __name__ == "__main__":
    main()
