#!/usr/bin/env python3
"""
Step 25: check suspect dates of the index entries against the documents' own
text, and correct the ones that are wrong.

The dates of the index entries (step 7: the enriched Digitized Indexes; step 23:
the descriptions) are derived from the descriptions, so reading the description
again cannot tell whether they are right. The witnesses here are independent of it:

  - the text: the date of the closing formula on the document's last scan or the
    one before, the last full date on the last scan, the running header of the
    first scans, the first date on the first scan (GLOBALISE entity annotations
    and page headers; tested on the General Missives, whose dates were checked by
    hand: a year for 87% of them, right within a year for 93.5%);
  - the previous index entry, for entries whose description refers to it ("Dito
    … dato 17 dito", "onder dien zelfden datum", "in 't voornoemde boekjaar");
  - the typoscript years (YEAR (EARLIEST), YEAR (LATEST)) and the inventory's years.

A date is suspect when it lies more than 2 years outside the inventory's or the
typoscript's years (1,499 of the 225,000 index entries' documents). For each,
`review` decides (date_review.csv):

  keep   the description names the year (periods covered, quoted older acts),
         or the text agrees
  fix    the date of the previous entry, for "Dito" entries, when the text agrees
         (or gives no year) and the inventory or typoscript years fit;
         else the year of the text when it agrees with the inventory or typoscript
         years and is 10 or more years from the date. A year fix keeps the day and
         month when the description names both (the day, and the month or "dito")
  clear  the enriched index's spurious 1739-01-01 – 1740-12-31 (given to entries
         whose description names no year), or a year that is a quantity in the
         description ("1773 1/2 pond"), when the text gives no year
  flag   the text disagrees, or gives no date, and there is no safe fix

`apply` sets the fixed and cleared dates on the index entries and on the
segmentation documents that copied them, where they still have the reviewed
dates, and records old and new values in date_corrections.csv (the database
itself keeps no history: only the dates change). With
--approved-only only the rows marked `y` in the approve column are applied.
Running it again changes nothing. The segmentation itself is not run again: it
read the old dates (the deposit's inputs keep them); later imports copy the
corrected dates from the entries.

  uv run python 25_correct_index_dates.py review [--documents deposit/results/documents.csv.gz] [--out date_review.csv]
  uv run python 25_correct_index_dates.py apply [--review date_review.csv] [--approved-only] [--dry-run] [--log date_corrections.csv]
"""

import argparse
import calendar
import datetime
import io
import json
import os
import re
import sqlite3
import zipfile

import pandas as pd

from segmentation.doc_titles import closing, entities, fetch_all, page_dates
from segmentation.header_dates import parse_header_date
from segmentation.inventory import _parse_list
from segmentation.texts import load_texts
from segmentation.title_dates import title_dates

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///globalise_documents.db")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_ZIP = os.path.join(DATA, "GLOBALISE - Digitized Indexes of the Dutch East India Company OBP (1602-1799).csv.zip")
DATES = ("date_earliest_begin", "date_latest_begin", "date_earliest_end", "date_latest_end")
SLACK = 2  # years outside the inventory or typoscript years before a date is suspect
FAR = 10  # a year from the text replaces a date only this far from it

DITTO = re.compile(r"\b(dito|ditto|do\.|idem|deser|deses|voorschreven|als ?vooren|als ?voren)|\bdo\b", re.I)  # refers to the previous entry
SAME_DATE = re.compile(r"(\b(zelfden|selfden|zelve|selve|dien) dat(um|o)|\beodem( dato)?|\bonder dato als)(?!\s*(in )?\d)", re.I)
SAME_YEAR = re.compile(r"\b(voornoemd|evengenoemd|evengemeld|gemeld|voorschreven|ditto|dito|zelve|selve|dit)e? boekja+r", re.I)
DITTO_DAYS = re.compile(r"\b(?:dato|van|den|de|op)\s+(\d{1,2})(?:\s*(?:en|&)\s*(\d{1,2}))?\s+(?:dito|do\.?|deser|deses|voorschreven)(?!\w)", re.I)
UNITS = (r"pond|ponden|ellen|el|kisten|kist|stux|stuks|stuk|ropias|rupias|ropijen|guldens|realen|rijksdaalders|lasten|last|picols|pikols|"
         r"balen|baal|kannen|vaten|vat|man|mannen|koppen|bomen|specerijbomen|zielen|persoonen|schepels|mudden|ankers|pakken|packen|corgies|"
         r"marken|taels|thails|catties|leggers|ton|tonnen")
QUANTITY = re.compile(rf"(?<!\d)1[5-8]\d\d\s*(?:\d+/\d+\s*|½\s*)?(?:{UNITS})\b", re.I)  # "1773 1/2 pond", "1670 ropias"
SPURIOUS = ("1739-01-01", "1740-12-31")  # the enriched index's 1739/40 for entries whose description names no year
MONTHS = [r"jan", r"feb", r"ma[ae]r", r"apr", r"m[ae][iy]|maij", r"jun", r"jul", r"aug", r"sep|7ber", r"o[ck]t|8ber", r"nov|9ber", r"dec|10ber|xber"]


def connect(readonly=True):
    path = DATABASE_URL.removeprefix("sqlite:///")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True) if readonly else sqlite3.connect(path)


def typoscript_years() -> dict[str, tuple[int, int]]:
    with zipfile.ZipFile(INDEX_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__"))
        t = pd.read_csv(io.BytesIO(z.read(name)), usecols=["ID", "YEAR (EARLIEST)", "YEAR (LATEST)"], low_memory=False)
    return {str(r[0]): (r[1], r[2]) for r in t.itertuples(index=False)}


def text_years(texts, heads, first, second, last, before) -> dict[str, int]:
    """The years the document's own text gives, by witness."""
    ev = {}
    for f in (last, before):
        if f and (c := closing(entities(f), texts.get(f, ""))) and "closing" not in ev:
            ev["closing"] = c[1].date.year
    full = [d.date.year for d, _, _ in page_dates(entities(last)) if d.precision == "day"]
    if full:
        ev["last date"] = full[-1]
    for f in (first, second):
        h = parse_header_date(heads.get(f, "")) if f else None
        if h is not None:
            ev["header"] = h.date.year
            break
    first_dates = [d.date.year for d, _, _ in page_dates(entities(first))]
    if first_dates:
        ev["first page"] = first_dates[0]
    return ev


def in_year(year: int, begin: str, end: str, title: str) -> tuple[str, str]:
    """A year fix: the whole year, or the current day and month in it when the description names them."""
    whole = (f"{year}-01-01", f"{year}-12-31")
    if not isinstance(begin, str) or not isinstance(end, str):
        return whole
    b, e = datetime.date.fromisoformat(begin), datetime.date.fromisoformat(end)
    month_named = re.search(MONTHS[b.month - 1], title, re.I) or DITTO.search(title)
    if b == e and re.search(rf"(?<!\d)0?{b.day}(?!\d)", title) and month_named and b.day <= calendar.monthrange(year, b.month)[1]:
        d = datetime.date(year, b.month, b.day).isoformat()
        return d, d
    if (b.day == 1 and b.month == e.month and b.year == e.year and e.day == calendar.monthrange(e.year, e.month)[1]
            and re.search(MONTHS[b.month - 1], title, re.I)):
        return f"{year}-{b.month:02d}-01", f"{year}-{b.month:02d}-{calendar.monthrange(year, b.month)[1]:02d}"
    return whole


def review(documents: str, out: str):
    conn = connect()
    docs = pd.read_csv(documents, dtype=str)
    docs = docs.merge(pd.read_sql("SELECT inventory_number inventory, date_start, date_end FROM inventory", conn), on="inventory", how="left")
    ty = typoscript_years()
    yr = lambda s: pd.to_numeric(s.str[:4], errors="coerce")
    docs["mx"] = pd.concat([yr(docs[k]) for k in DATES], axis=1).max(axis=1)
    docs["mn"] = pd.concat([yr(docs[k]) for k in DATES], axis=1).min(axis=1)
    docs["i_lo"], docs["i_hi"] = yr(docs.date_start), yr(docs.date_end)
    obp = docs.obp_index_ids.str.split(";").str[0]
    docs["t_lo"], docs["t_hi"] = obp.map(lambda i: ty.get(i, (None, None))[0]), obp.map(lambda i: ty.get(i, (None, None))[1])
    off_typoscript = ((docs.mx > docs.t_hi + SLACK) | (docs.mn < docs.t_lo - SLACK)).fillna(False)
    sus = docs[docs.title_source.isin(["VOC", "Resilient Diversity"]) & docs.mx.notna() & ((docs.mx > docs.i_hi + SLACK) | off_typoscript)]

    # the nearest preceding index entry (ToC order) that is dated and not itself suspect
    idx = pd.read_sql("""SELECT i.inventory_number inventory, d.toc_order, d.date_earliest_begin b, d.date_latest_end e,
          (SELECT min(CAST(x.identifier AS INTEGER)) FROM document2external_id de JOIN external_id x ON x.id = de.external_id
            WHERE de.document_id = d.id AND x.context = 'OBP_INDEX') obp
        FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index'
        JOIN inventory i ON i.id = d.inventory_id WHERE d.toc_order IS NOT NULL""", conn).sort_values(["inventory", "toc_order"])
    suspect = set(pd.to_numeric(sus.obp_index_ids.str.split(";").str[0], errors="coerce").dropna().astype(int))
    previous = {}
    for _, g in idx.groupby("inventory"):
        last = None
        for r in g.itertuples():
            if pd.notna(r.obp):
                previous[int(r.obp)] = last
            if isinstance(r.b, str) and not (pd.notna(r.obp) and int(r.obp) in suspect):
                last = (r.b, int(r.obp) if pd.notna(r.obp) else None, r.e if isinstance(r.e, str) else r.b)

    rows = []
    for inv, g in sus.groupby("inventory"):
        scans = [r[0] for r in conn.execute("SELECT s.filename FROM scan s JOIN inventory i ON i.id = s.inventory_id "
                                            "WHERE i.inventory_number = ? ORDER BY s.scan_order, s.filename", (inv,))]
        pos = {f: k for k, f in enumerate(scans)}
        heads = {f: _parse_list(h) for f, h in conn.execute(
            "SELECT s.filename, p.header FROM scan s JOIN page p ON p.scan_id = s.id JOIN inventory i ON i.id = s.inventory_id "
            "WHERE i.inventory_number = ?", (inv,)) if h}
        jobs = []
        for r in g.itertuples():
            a, b = pos[r.start_scan], pos[r.end_scan]
            jobs.append((r, scans[a], scans[a + 1] if a + 1 <= b else None, scans[b], scans[b - 1] if b - 1 >= a else None))
        files = sorted({f for j in jobs for f in j[1:] if f})
        fetch_all(files)
        texts = dict(zip(files, load_texts(inv, pd.Series(files))))
        for r, first, second, last, before in jobs:
            ev = text_years(texts, heads, first, second, last, before)
            w = next((ev[k] for k in ("closing", "last date", "header", "first page") if k in ev), None)  # the most reliable witness
            bad = int(r.mx) if r.mx > r.i_hi + SLACK or (pd.notna(r.t_hi) and r.mx > r.t_hi + SLACK) else int(r.mn)
            title = str(r.title).strip("[]")
            quantities = {int(m.group(0)[:4]) for m in QUANTITY.finditer(title)}
            desc_years = {d.year for x, y in title_dates(QUANTITY.sub(" ", title)) for d in (x, y)}
            fits = lambda y: (pd.notna(r.t_lo) and r.t_lo - 1 <= y <= r.t_hi + 1) or (r.i_lo - 1 <= y <= r.i_hi + 1)
            entry = pd.to_numeric(str(r.obp_index_ids).split(";")[0], errors="coerce")
            prev = previous.get(int(entry)) if pd.notna(entry) else None
            spurious = (r.date_earliest_begin, r.date_latest_end) == SPURIOUS and not desc_years & {1739, 1740}
            begin = end = None
            refers = DITTO.search(title) or SAME_DATE.search(title) or SAME_YEAR.search(title)
            if not desc_years and refers and prev and (w is None or abs(int(prev[0][:4]) - w) <= 1) and fits(int(prev[0][:4])):
                y, m = int(prev[0][:4]), int(prev[0][5:7])
                days, last_day = DITTO_DAYS.search(title), calendar.monthrange(y, m)[1]
                if SAME_DATE.search(title):
                    begin, end = prev[0], prev[0]
                elif SAME_YEAR.search(title):
                    begin, end = prev[0], prev[2]
                elif days and int(days.group(1)) <= last_day:
                    d1 = int(days.group(1))
                    d2 = int(days.group(2)) if days.group(2) and int(days.group(2)) <= last_day else d1
                    begin, end = datetime.date(y, m, d1).isoformat(), datetime.date(y, m, d2).isoformat()
                else:
                    begin, end = in_year(y, r.date_earliest_begin, r.date_latest_end, title)
                verdict = "fix: date of the previous index entry (Dito), " + ("the text agrees" if w is not None else "no date in the text")
            elif spurious or (bad in quantities and bad not in desc_years):
                why = "the enriched index's spurious 1739–1740" if spurious else "the year is a quantity in the description"
                if w is not None and fits(w):
                    verdict = f"fix: year from the text ({why})"
                    begin, end = f"{w}-01-01", f"{w}-12-31"  # the day and month are as doubtful as the year
                else:
                    verdict = f"clear: {why}"
            elif bad in desc_years:
                verdict = "keep: the description has this year"
            elif w is not None and abs(w - bad) <= 1:
                verdict = "keep: the text agrees"
            elif w is not None and not desc_years and fits(w) and abs(w - bad) >= FAR:
                verdict = "fix: year from the text, which agrees with the inventory years"
                begin, end = in_year(w, r.date_earliest_begin, r.date_latest_end, title)
            elif w is None:
                verdict = "flag: no date in the text"
            else:
                verdict = "flag: the text disagrees, no safe fix"
            rows.append({"verdict": verdict, "approve": "", "proposed_begin": begin, "proposed_end": end,
                         "document_id": r.document_id, "inventory_number": inv, "obp_index_id": None if pd.isna(entry) else int(entry),
                         "inventory_years": f"{int(r.i_lo)}-{int(r.i_hi)}",
                         "typoscript_years": f"{int(r.t_lo)}-{int(r.t_hi)}" if pd.notna(r.t_lo) else "",
                         "date_earliest_begin": r.date_earliest_begin, "date_latest_end": r.date_latest_end, "suspect_year": bad,
                         "text_year": w, "text_evidence": json.dumps(ev), "previous_entry": f"{prev[1]}: {prev[0]} – {prev[2]}" if prev else "",
                         "title": r.title})
    out_df = pd.DataFrame(rows).sort_values(["verdict", "inventory_number", "date_earliest_begin"])
    out_df["obp_index_id"] = out_df["obp_index_id"].astype("Int64")
    out_df.to_csv(out, index=False)
    print(f"{len(out_df):,} suspect dates (more than {SLACK} years outside the inventory or typoscript years) → {out}")
    print(out_df.verdict.value_counts().to_string())


def new_dates(begin, end) -> dict:
    """The four date columns for a corrected date (None: cleared). A whole year or month is one date of that precision."""
    if not isinstance(begin, str):
        return dict.fromkeys(DATES)
    b, e = datetime.date.fromisoformat(begin), datetime.date.fromisoformat(end)
    one = b == e or (b.year == e.year and ((b.month, b.day, e.month, e.day) == (1, 1, 12, 31)
                                            or (b.month == e.month and b.day == 1 and e.day == calendar.monthrange(e.year, e.month)[1])))
    if one:
        return {"date_earliest_begin": begin, "date_latest_begin": end, "date_earliest_end": begin, "date_latest_end": end}
    return {"date_earliest_begin": begin, "date_latest_begin": begin, "date_earliest_end": end, "date_latest_end": end}


def apply(path: str, approved_only: bool, dry_run: bool, log: str):
    rv = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    rv = rv[rv.verdict.str.startswith(("fix", "clear"))]
    if approved_only:
        rv = rv[rv.approve.fillna("").str.lower().eq("y")]
    conn = connect(readonly=dry_run)
    stats = {"rows": len(rv), "documents changed": 0, "already corrected": 0, "dates changed since the review": 0}
    now = datetime.datetime.now().isoformat(timespec="seconds")
    changes = []
    for r in rv.itertuples():
        new = new_dates(r.proposed_begin, r.proposed_end)
        targets = [r.document_id]  # the segmentation document and the index entry it took its dates from
        if pd.notna(r.obp_index_id):
            targets += [x[0] for x in conn.execute(
                "SELECT d.id FROM document d JOIN document_identification_method m ON m.id = d.method_id AND m.name = 'TANAP Digitized Index' "
                "JOIN document2external_id de ON de.document_id = d.id JOIN external_id x ON x.id = de.external_id "
                "WHERE x.context = 'OBP_INDEX' AND x.identifier = ?", (str(r.obp_index_id),))]
        reviewed = (r.date_earliest_begin if isinstance(r.date_earliest_begin, str) else None,
                    r.date_latest_end if isinstance(r.date_latest_end, str) else None)
        for doc in dict.fromkeys(targets):
            old = conn.execute(f"SELECT {', '.join(DATES)} FROM document WHERE id = ?", (doc,)).fetchone()
            if old is None:
                continue
            if list(old) == [new[c] for c in DATES]:
                stats["already corrected"] += 1
                continue
            if (old[0], old[3]) != reviewed:
                stats["dates changed since the review"] += 1
                continue
            stats["documents changed"] += 1
            changes.append({"document_id": doc, "obp_index_id": r.obp_index_id, "verdict": r.verdict,
                            **{f"old_{c.removeprefix('date_')}": v for c, v in zip(DATES, old)},
                            **{f"new_{c.removeprefix('date_')}": new[c] for c in DATES},
                            "evidence": json.dumps({"text": r.text_evidence, "previous_entry": r.previous_entry,
                                                    "typoscript_years": r.typoscript_years}), "applied": now})
            if not dry_run:
                conn.execute(f"UPDATE document SET {', '.join(f'{c} = ?' for c in DATES)} WHERE id = ?", (*[new[c] for c in DATES], doc))
    if not dry_run:
        conn.commit()
        if changes:  # the record of what changed: old and new dates (the database keeps no history)
            pd.DataFrame(changes).to_csv(log, mode="a", header=not os.path.exists(log), index=False)
    for k, v in stats.items():
        print(f"{k}: {v:,}")
    print("dry run: nothing changed" if dry_run else f"changes recorded in {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("--documents", default="deposit/results/documents.csv.gz", help="the segmentation documents (export_results.py)")
    rv.add_argument("--out", default="date_review.csv")
    ap_ = sub.add_parser("apply")
    ap_.add_argument("--review", default="date_review.csv")
    ap_.add_argument("--approved-only", action="store_true", help="only the rows marked y in the approve column")
    ap_.add_argument("--dry-run", action="store_true")
    ap_.add_argument("--log", default="date_corrections.csv", help="old and new dates of every document changed (appended)")
    args = ap.parse_args()
    if args.cmd == "review":
        review(args.documents, args.out)
    else:
        apply(args.review, args.approved_only, args.dry_run, args.log)


if __name__ == "__main__":
    main()
