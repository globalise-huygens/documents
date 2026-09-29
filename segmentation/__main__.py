"""
Command line for the segmentation metric.

  uv run python -m segmentation evaluate [--gm-limit N] [--cache FILE]
      cross-validated evaluation on the validated and General Missives inventories
  uv run python -m segmentation fit [--out segmentation/model.json]
      fit the model on all ground truth and save it
  uv run python -m segmentation run 1120 1557 [--model FILE] [--out segments.csv] [--no-toc]
      segment inventories and write one row per segment
"""

import argparse
import logging
import time

import pandas as pd

from .evaluation import cross_validate, fit_model, format_counts, load_items
from .inventory import connect, load_inventory
from .model import DEFAULT_MODEL_PATH, SegmentationModel
from .segmenter import segment_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("segmentation")


def result_rows(inv, result) -> list[dict]:
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
    else:
        model = SegmentationModel.load(args.model)
        conn = connect()
        rows = []
        for n in args.inventories:
            inv = load_inventory(conn, n)
            res = segment_inventory(inv, model, use_toc=not args.no_toc)
            rows += result_rows(inv, res)
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
