"""
Command line for the segmentation metric.

  uv run python -m segmentation evaluate [--gm-limit N] [--cache FILE]
      cross-validated evaluation on the validated and General Missives inventories
  uv run python -m segmentation fit [--out segmentation/model.json]
      fit the model on all ground truth and save it
  uv run python -m segmentation run [1120 1557 ...] [--model FILE] [--out segments.csv] [--no-toc] [--no-court]
          [--no-gm] [--derived derived_toc.csv] [--resume]
      segment inventories (all when none are given) and write one row per segment;
      the General Missives identified by hand override the model and the ToC alignment
      (their start, span, title and dates; --no-gm to ignore them);
      inventories without a ToC that have court cases in data/EMDCCR*.xlsx are
      segmented on those cases (court_records.py; --no-court to ignore them); inventories
      with neither use the ToC entries derived from other versions of their text, when
      --derived (default derived_toc.csv) has them (versions.py; --derived "" to ignore);
      --resume continues an interrupted run, skipping inventories already in --out
  uv run python -m segmentation split-texts [data/normalized_texts.parquet]
      split the full-text dump into one file per inventory (data/texts/; one-time)
  SEGMENTATION_PAGEXML_DIR=/Volumes/HDE0090 uv run python -m segmentation cache-layout [inv ...]
      read the PageXML zips once and cache the page layout per inventory (data/layout/;
      all inventories when none are given; already cached ones are skipped)
  uv run python -m segmentation match-versions [inv ...] [--window 2] [--all]
      find the same text elsewhere in the archive for every scan of the inventories
      without ToC entries or court cases (all such when none are given, every inventory
      with --all; versions.py);
      one parquet per inventory in data/versions/, resumable
  uv run python -m segmentation derive-tocs [inv ...] [--out derived_toc.csv] [--blocks version_blocks.csv]
      version blocks and the ToC entries derived from the other versions' ToCs
  uv run python -m segmentation version-blocks [inv ...] [--out version_blocks_all.csv]
      version blocks of every matched inventory (all when none are given), no ToC entries
  uv run python -m segmentation load-versions version_blocks_all.csv
      (re)fill the text_version table: runs of scans with the same text in two inventories
  uv run python -m segmentation register-features [inv ...]
      text counts and PageXML layout per scan for the register classifier (registers.py; data/registers/features)
  uv run python -m segmentation register-fit
      train the register-page classifier on the validated inventories (+ reviewed scans)
  uv run python -m segmentation register-ranges [inv ...] [--threshold 0.9] [--continue-threshold 0.3] [--out register_ranges.csv]
      score every scan and write the ranges of register pages
  uv run python -m segmentation register-entries [--min-score 0.9] [--out register_entries.csv]
      rebuild the rows of the register ranges into entries: folio range / item number, date, title (rows.py)
  uv run python -m segmentation register-evaluate
      precision / recall of register ranges, estimated from the reviewed scans (cross-validated over inventories)
  uv run python -m segmentation register-match [inv ...] [--out register_candidates.csv]
      candidate start scans per register entry, by folio, date and title words (register_match.py)
  uv run python -m segmentation register-link-sample [--n 80]
      a stratified sample of entries to link at /review/register-links in the app
  uv run python -m segmentation register-sample [--n 200]
      draw a stratified sample of scans to label at /review/registers in the app
  uv run python -m segmentation import segments.csv [--dry-run] [--resume]
  uv run python -m segmentation review-split [segments.csv]
      split segments.csv into data/segments/<inventory>.parquet for /review/segments
      (the app also does this by itself when segments.csv has changed)
  uv run python -m segmentation review-evaluate [--out segment_reviews.csv]
      the boundaries checked in /review/segments, per placement method
      store the segments as documents of the method "Segmentation model"
      (re-importing an inventory replaces its earlier segmentation documents);
      --resume skips the inventories already imported from this CSV (<csv>.imported)
"""

import argparse
import json
import logging
import os
import time

import pandas as pd

from .evaluation import cross_validate, fit_model, format_counts, load_items
from .inventory import connect, load_gm_overrides, load_inventory
from .model import DEFAULT_MODEL_PATH, SegmentationModel
from .court_records import load_court_cases, segment_court_inventory
from .explain import document_evidence, scan_evidence
from .segmenter import scan_rule_violations, segment_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("segmentation")


def result_rows(inv, result, model) -> list[dict]:
    ev = scan_evidence(inv, model)
    rows = []
    fn = inv.scans["filename"]
    toc = inv.toc
    for k, sg in enumerate(result.segments):
        entries = [toc.loc[r] for r in sg.toc_rows]
        info = (sg.gm[0] if sg.gm else None) or sg.court or sg.derived or {}  # a hand-identified missive wins
        rows.append(
            {
                "inventory": inv.inventory_number,
                "segment": k,
                "kind": sg.kind,
                "start_scan": fn.iat[sg.start],
                "end_scan": fn.iat[sg.end],
                "n_scans": sg.end - sg.start + 1,
                "boundary": sg.boundary,
                "start_logit": round(sg.start_logit, 3),
                "end_logit": round(sg.end_logit, 3),
                "toc_csv_ids": ";".join(str(int(e.csv_id)) for e in entries if not pd.isna(e.csv_id)),
                "toc_title": entries[0].title if entries else None,
                "parent_csv_id": None if sg.parent_row is None or pd.isna(toc.at[sg.parent_row, "csv_id"]) else int(toc.at[sg.parent_row, "csv_id"]),
                "align_score": None if sg.align_score is None else round(sg.align_score, 3),
                "placed_by": ";".join(pl.how for pl in result.placements if pl.entry in sg.toc_rows),
                "evidence": json.dumps(document_evidence(inv, result, sg, ev), ensure_ascii=False) if sg.kind != "non-document" else None,
                "court_case": sg.court["case_id"] if sg.court else None,
                "parent_court_case": sg.court.get("parent_case") if sg.court else None,
                "title": info.get("title"),
                "date_begin": info.get("date_begin"),
                "date_end": info.get("date_end"),
                "place": sg.court["place"] if sg.court else None,
                "entry_id": sg.derived.get("entry_id") if sg.derived else None,
                "parent_entry_id": sg.derived.get("parent_entry_id") if sg.derived else None,
                "gm_ids": ";".join(g["gm_id"] for g in sg.gm) or None,
                "parent_gm_id": sg.parent_gm,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--gm-limit", type=int, default=None)
    ev.add_argument("--cache", default=None, help="pickle of prepared inventories (created if missing)")
    ev.add_argument("--l2", type=float, default=10.0)
    ft = sub.add_parser("fit")
    ft.add_argument("--out", default=DEFAULT_MODEL_PATH)
    ft.add_argument("--cache", default=None)
    rn = sub.add_parser("run")
    rn.add_argument("inventories", nargs="*", help="inventory numbers (default: all)")
    rn.add_argument("--model", default=DEFAULT_MODEL_PATH)
    rn.add_argument("--out", default=None)
    rn.add_argument("--no-toc", action="store_true")
    rn.add_argument("--no-court", action="store_true", help="ignore the court cases of data/EMDCCR*.xlsx")
    rn.add_argument("--no-gm", action="store_true", help="ignore the General Missives identified by hand")
    rn.add_argument("--derived", default="derived_toc.csv", help="ToC entries derived from other versions (derive-tocs)")
    rn.add_argument("--resume", action="store_true", help="skip inventories already in --out")
    st = sub.add_parser("split-texts")
    st.add_argument("source", nargs="?", default="data/normalized_texts.parquet")
    cl = sub.add_parser("cache-layout")
    cl.add_argument("inventories", nargs="*")
    mv = sub.add_parser("match-versions")
    mv.add_argument("inventories", nargs="*")
    mv.add_argument("--window", type=int, default=2)
    mv.add_argument("--all", action="store_true", help="all inventories, not only those without ToC entries or court cases")
    dt = sub.add_parser("derive-tocs")
    dt.add_argument("inventories", nargs="*")
    dt.add_argument("--out", default="derived_toc.csv")
    dt.add_argument("--blocks", default="version_blocks.csv")
    vb = sub.add_parser("version-blocks")
    vb.add_argument("inventories", nargs="*")
    vb.add_argument("--out", default="version_blocks_all.csv")
    lv = sub.add_parser("load-versions")
    lv.add_argument("csv", nargs="+")
    rf = sub.add_parser("register-features")
    rf.add_argument("inventories", nargs="*")
    sub.add_parser("register-fit")
    rr = sub.add_parser("register-ranges")
    rr.add_argument("inventories", nargs="*")
    rr.add_argument("--threshold", type=float, default=0.9)
    rr.add_argument("--continue-threshold", type=float, default=0.3)
    rr.add_argument("--out", default="register_ranges.csv")
    re_ = sub.add_parser("register-entries")
    re_.add_argument("--ranges", default="register_ranges.csv")
    re_.add_argument("--min-score", type=float, default=0.9)
    re_.add_argument("--out", default="register_entries.csv")
    sub.add_parser("register-evaluate")
    rm = sub.add_parser("register-match")
    rm.add_argument("inventories", nargs="*")
    rm.add_argument("--out", default="register_candidates.csv")
    rls = sub.add_parser("register-link-sample")
    rls.add_argument("--n", type=int, default=80)
    rls.add_argument("--candidates", default="register_candidates.csv")
    rs = sub.add_parser("register-sample")
    rs.add_argument("--n", type=int, default=200)
    rs.add_argument("--scores", default="data/registers/scores.parquet")
    rsp = sub.add_parser("review-split")
    rsp.add_argument("source", nargs="?", default=None)
    rev = sub.add_parser("review-evaluate")
    rev.add_argument("--out", default=None, help="also write the current reviews to this CSV")
    im = sub.add_parser("import")
    im.add_argument("csv")
    im.add_argument("--dry-run", action="store_true")
    im.add_argument("--resume", action="store_true", help="skip the inventories listed in <csv>.imported")
    args = ap.parse_args()

    if args.cmd == "evaluate":
        t = time.time()
        items = load_items(cache=args.cache, gm_limit=args.gm_limit)
        logger.info("Prepared %d inventories in %.0fs", len(items), time.time() - t)
        counts = cross_validate(items, l2=args.l2)
        print(format_counts(counts))
    elif args.cmd == "fit":
        items = load_items(sources=("validated",), cache=args.cache)
        model = fit_model(items)
        model.save(args.out)
        logger.info("Saved model to %s", args.out)
        for name, w in sorted(model.start.weights.items(), key=lambda kv: -abs(kv[1])):
            print(f"  start  {name:<26} {w:+.2f}")
        print(f"  start  {'(bias)':<26} {model.start.bias:+.2f}")
    elif args.cmd == "split-texts":
        from .texts import split_texts

        split_texts(args.source)
        logger.info("Split %s into data/texts/", args.source)
    elif args.cmd == "review-split":
        from .review import SEGMENTS_CSV, split_segments

        n = split_segments(args.source or SEGMENTS_CSV)
        logger.info("Split %s into %d inventories", args.source or SEGMENTS_CSV, n)
    elif args.cmd == "review-evaluate":
        from .review import evaluate, export_rows

        rows = export_rows(connect())
        print(evaluate(rows.to_dict("records")).to_string() if len(rows) else "No reviews yet")
        if args.out:
            rows.to_csv(args.out, index=False)
    elif args.cmd == "cache-layout":
        from .pagexml import CACHE_DIR, PAGEXML_DIR, load_layout

        if not PAGEXML_DIR:
            ap.error("set SEGMENTATION_PAGEXML_DIR to the folder with the <inventory>.zip files")
        conn = connect()
        numbers = args.inventories or [r[0] for r in conn.execute("SELECT inventory_number FROM inventory ORDER BY inventory_number")]
        skipped = 0
        for k, n in enumerate(numbers, 1):
            if os.path.exists(os.path.join(CACHE_DIR, f"{n}.parquet")):
                skipped += 1
                continue
            files = pd.read_sql("SELECT filename FROM scan s JOIN inventory i ON i.id = s.inventory_id WHERE i.inventory_number = ?", conn, params=(n,))["filename"]
            t = time.time()
            layout = load_layout(n, files)
            logger.info("%d/%d %s: %d of %d scans with layout (%.1fs)", k, len(numbers), n, int(layout["has_layout"].sum()), len(files), time.time() - t)
        logger.info("Done; %d inventories were already cached", skipped)
    elif args.cmd.startswith("register-"):
        from . import registers
        from .versions import unindexed_inventories

        conn = connect()
        targets = getattr(args, "inventories", None) or unindexed_inventories(conn)
        if args.cmd == "register-features":
            registers.cache_features(targets)
        elif args.cmd == "register-fit":
            registers.fit(conn)
        elif args.cmd == "register-match":
            from .register_match import match_all
            from .register_toc import load_register_entries

            res = match_all(targets, load_register_entries(), args.out)
            logger.info("Wrote %d candidates for %d entries to %s", len(res), res.groupby(["register_first_scan", "entry"]).ngroups, args.out)
        elif args.cmd == "register-link-sample":
            from sqlalchemy import create_engine

            from models import Base
            from .inventory import DATABASE_URL
            from .register_match import link_sample

            sample = link_sample(pd.read_csv(args.candidates, dtype={"inventory": str, "item": str}), args.n)
            engine = create_engine(DATABASE_URL)
            Base.metadata.create_all(engine, tables=[Base.metadata.tables["register_link_review"]])
            have = set(pd.read_sql("SELECT entry_key FROM register_link_review", engine)["entry_key"])
            start = int(pd.read_sql("SELECT coalesce(max(position), -1) AS m FROM register_link_review", engine)["m"].iat[0]) + 1
            new = sample[~sample["entry_key"].isin(have)].copy()
            new["position"] = range(start, start + len(new))
            new.to_sql("register_link_review", engine, if_exists="append", index=False)
            logger.info("register_link_review: %d entries added; review them at /review/register-links", len(new))
        elif args.cmd == "register-evaluate":
            res = registers.evaluate(conn)
            res.to_csv("register_evaluation.csv", index=False)
            print(res.to_string(index=False))
        elif args.cmd == "register-entries":
            from .rows import entries_for_ranges

            e = entries_for_ranges(pd.read_csv(args.ranges, dtype={"inventory": str}), args.min_score)
            e.to_csv(args.out, index=False)
            logger.info("Wrote %d register entries (%d ranges) to %s", len(e), e["register_first_scan"].nunique() if len(e) else 0, args.out)
        elif args.cmd == "register-ranges":
            scores = registers.score(targets)
            os.makedirs(os.path.dirname(registers.MODEL_PATH), exist_ok=True)
            scores.to_parquet(os.path.join(os.path.dirname(registers.MODEL_PATH), "scores.parquet"), index=False)
            r = registers.ranges(scores, args.threshold, args.continue_threshold)
            r.to_csv(args.out, index=False)
            logger.info("Wrote %d register ranges (%d scans, %d inventories) to %s", len(r), r["n_scans"].sum(), r["inventory"].nunique(), args.out)
        else:
            from sqlalchemy import create_engine

            from models import Base
            from .inventory import DATABASE_URL

            sample = registers.review_sample(pd.read_parquet(args.scores), args.n)
            engine = create_engine(DATABASE_URL)
            Base.metadata.create_all(engine, tables=[Base.metadata.tables["register_review"]])
            have = set(pd.read_sql("SELECT filename FROM register_review", engine)["filename"])
            start = int(pd.read_sql("SELECT coalesce(max(position), -1) AS m FROM register_review", engine)["m"].iat[0]) + 1
            new = sample[~sample["filename"].isin(have)].copy()
            new["position"] = range(start, start + len(new))
            new.to_sql("register_review", engine, if_exists="append", index=False)
            logger.info("register_review: %d scans added (%d already there); label them at /review/registers", len(new), len(sample) - len(new))
    elif args.cmd == "load-versions":
        from .inventory import DATABASE_URL
        from .versions import load_text_versions

        load_text_versions(args.csv, DATABASE_URL)
    elif args.cmd in ("match-versions", "derive-tocs", "version-blocks"):
        from . import versions

        conn = connect()
        if args.inventories:
            targets = args.inventories
        elif getattr(args, "all", False):
            targets = [r[0] for r in conn.execute("SELECT inventory_number FROM inventory ORDER BY inventory_number")]
        else:
            targets = versions.untitled_inventories(conn)
        if args.cmd == "match-versions":
            versions.match_all(targets, conn, window=args.window)
        elif args.cmd == "version-blocks":
            if not args.inventories:
                targets = [r[0] for r in conn.execute("SELECT inventory_number FROM inventory ORDER BY inventory_number")]
            versions.derive_all([t for t in targets if os.path.exists(os.path.join(versions.MATCH_DIR, f"{t}.parquet"))], conn, None, args.out)
        else:
            versions.derive_all([t for t in targets if os.path.exists(os.path.join(versions.MATCH_DIR, f"{t}.parquet"))], conn, args.out, args.blocks)
    elif args.cmd == "import":
        from .db_import import import_segments
        from .inventory import DATABASE_URL

        import_segments(args.csv, DATABASE_URL, dry_run=args.dry_run, resume=args.resume)
    else:
        run(args)


def run(args):
    """Segment inventories; with --out, append each inventory's rows to the CSV as it is done."""
    model = SegmentationModel.load(args.model)
    conn = connect()
    cases, court_ranges = ({}, {}) if args.no_court else load_court_cases()
    if court_ranges:
        logger.info("Court records: %d cases in %d inventories", len(cases), len(court_ranges))
    from .versions import load_derived_toc, segment_with_derived

    derived = load_derived_toc(args.derived)
    if derived:
        logger.info("Derived ToC entries for %d inventories (%s)", len(derived), args.derived)
    numbers = args.inventories or [r[0] for r in conn.execute("SELECT inventory_number FROM inventory ORDER BY inventory_number")]
    done = set()
    if args.out and args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["inventory"], dtype=str)["inventory"])
        logger.info("Resuming: %d inventories already in %s", len(done), args.out)
    elif args.out and os.path.exists(args.out):
        os.remove(args.out)
    todo = [n for n in numbers if n not in done]
    scan_counts = dict(conn.execute("SELECT i.inventory_number, count(*) FROM scan s JOIN inventory i ON i.id = s.inventory_id GROUP BY 1").fetchall())
    total_scans = sum(scan_counts.get(n, 0) for n in todo)
    printed, t0, scans_done, n_segments = [], time.time(), 0, 0
    for k, n in enumerate(todo, 1):
        try:
            inv = load_inventory(conn, n)
        except KeyError:
            logger.warning("%s: inventory not found in the database; skipped", n)
            continue
        try:
            gm = None if args.no_gm else load_gm_overrides(conn, inv)
            if court_ranges.get(n) and inv.toc.empty:
                res = segment_court_inventory(inv, model, cases, court_ranges[n])
            elif n in derived and inv.toc.empty and not args.no_toc and (gm is None or gm.empty):
                res = segment_with_derived(inv, model, derived[n])
            else:
                res = segment_inventory(inv, model, use_toc=not args.no_toc, gm=gm)
            rows = result_rows(inv, res, model)
            broken = scan_rule_violations(inv, res)
            if broken:
                logger.warning("%s: %d breaches of the hard rules, e.g. %s", n, len(broken), "; ".join(broken[:3]))
        except Exception:
            logger.exception("%s: segmentation failed; skipped", n)
            continue
        scans_done += inv.n
        n_segments += len(rows)
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        else:
            printed += rows
        kinds = pd.Series([s.kind for s in res.segments]).value_counts().to_dict()
        eta = (time.time() - t0) / max(scans_done, 1) * (total_scans - scans_done) / 60
        logger.info("%d/%d %s: %d scans, %d ToC entries placed of %d, segments %s (about %.0f min to go)",
                    k, len(todo), n, inv.n, len(res.placements), len(inv.toc), kinds, eta)
    if args.out:
        logger.info("Wrote %d segments to %s", n_segments, args.out)
    else:
        print(pd.DataFrame(printed).to_string(max_colwidth=50))


if __name__ == "__main__":
    main()
