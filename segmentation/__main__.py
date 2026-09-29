"""
Command line for the segmentation metric.

  uv run python -m segmentation evaluate [--gm-limit N] [--cache FILE]
      cross-validated evaluation on the validated and General Missives inventories
  uv run python -m segmentation fit [--out segmentation/model.json]
      fit the model on all ground truth and save it
  uv run python -m segmentation run 1120 1557 [--model FILE] [--out segments.csv] [--no-toc]
      segment inventories and write one row per segment
  uv run python -m segmentation split-texts [data/normalized_texts.parquet]
      split the full-text dump into one file per inventory (data/texts/; one-time)
  SEGMENTATION_PAGEXML_DIR=/Volumes/HDE0090 uv run python -m segmentation cache-layout [inv ...]
      read the PageXML zips once and cache the page layout per inventory (data/layout/;
      all inventories when none are given; already cached ones are skipped)
  uv run python -m segmentation import segments.csv [--dry-run]
      store the segments as documents of the method "Segmentation model"
      (re-importing an inventory replaces its earlier segmentation documents)
"""

import argparse
import json
import logging
import os
import time

import pandas as pd

from .evaluation import cross_validate, fit_model, format_counts, load_items
from .inventory import connect, load_inventory
from .model import DEFAULT_MODEL_PATH, SegmentationModel
from .explain import document_evidence, scan_evidence
from .segmenter import segment_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("segmentation")


def result_rows(inv, result, model) -> list[dict]:
    ev = scan_evidence(inv, model)
    rows = []
    fn = inv.scans["filename"]
    toc = inv.toc
    for k, sg in enumerate(result.segments):
        entries = [toc.loc[r] for r in sg.toc_rows]
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
    rn.add_argument("inventories", nargs="+")
    rn.add_argument("--model", default=DEFAULT_MODEL_PATH)
    rn.add_argument("--out", default=None)
    rn.add_argument("--no-toc", action="store_true")
    st = sub.add_parser("split-texts")
    st.add_argument("source", nargs="?", default="data/normalized_texts.parquet")
    cl = sub.add_parser("cache-layout")
    cl.add_argument("inventories", nargs="*")
    im = sub.add_parser("import")
    im.add_argument("csv")
    im.add_argument("--dry-run", action="store_true")
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
    elif args.cmd == "import":
        from .db_import import import_segments
        from .inventory import DATABASE_URL

        import_segments(args.csv, DATABASE_URL, dry_run=args.dry_run)
    else:
        model = SegmentationModel.load(args.model)
        conn = connect()
        rows = []
        for n in args.inventories:
            try:
                inv = load_inventory(conn, n)
            except KeyError:
                logger.warning("%s: inventory not found in the database; skipped", n)
                continue
            try:
                res = segment_inventory(inv, model, use_toc=not args.no_toc)
            except Exception:
                logger.exception("%s: segmentation failed; skipped", n)
                continue
            rows += result_rows(inv, res, model)
            kinds = pd.Series([s.kind for s in res.segments]).value_counts().to_dict()
            logger.info("%s: %d scans, %d ToC entries placed of %d, segments %s", n, inv.n, len(res.placements), len(inv.toc), kinds)
        df = pd.DataFrame(rows)
        if args.out:
            df.to_csv(args.out, index=False)
            logger.info("Wrote %d segments to %s", len(df), args.out)
        else:
            print(df.to_string(max_colwidth=50))


if __name__ == "__main__":
    main()
