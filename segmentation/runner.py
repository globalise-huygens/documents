"""
The work of `python -m segmentation run` for one inventory, in a worker process
(functions in __main__ cannot be sent to the processes `run` starts).
"""

import logging

import pandas as pd

from .court_records import load_court_cases, segment_court_inventory
from .inventory import connect, load_gm_overrides, load_inventory
from .model import SegmentationModel
from .segmenter import scan_rule_violations, segment_inventory

logger = logging.getLogger("segmentation")


_worker: dict = {}  # per process: model, connection, court cases, derived entries, options


def init_worker(args):
    from .versions import load_derived_toc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cases, court_ranges = ({}, {}) if args.no_court else load_court_cases()
    _worker.update(model=SegmentationModel.load(args.model), conn=connect(), cases=cases, court_ranges=court_ranges,
                   derived=load_derived_toc(args.derived), args=args)


def segment_one(n):
    """Rows of one inventory and what the log reports, or None when it is not in the database or fails."""
    from .versions import segment_with_derived

    w, args = _worker, _worker["args"]
    try:
        inv = load_inventory(w["conn"], n)
    except KeyError:
        logger.warning("%s: inventory not found in the database; skipped", n)
        return n, None
    try:
        gm = None if args.no_gm else load_gm_overrides(w["conn"], inv)
        if w["court_ranges"].get(n) and inv.toc.empty:
            res = segment_court_inventory(inv, w["model"], w["cases"], w["court_ranges"][n])
        elif n in w["derived"] and inv.toc.empty and not args.no_toc and (gm is None or gm.empty):
            res = segment_with_derived(inv, w["model"], w["derived"][n])
        else:
            res = segment_inventory(inv, w["model"], use_toc=not args.no_toc, gm=gm)
        from .__main__ import result_rows

        rows = result_rows(inv, res, w["model"])
        broken = scan_rule_violations(inv, res)
        if broken:
            logger.warning("%s: %d breaches of the hard rules, e.g. %s", n, len(broken), "; ".join(broken[:3]))
    except Exception:
        logger.exception("%s: segmentation failed; skipped", n)
        return n, None
    kinds = pd.Series([sg.kind for sg in res.segments]).value_counts().to_dict()
    return n, (rows, inv.n, len(res.placements), len(inv.toc), kinds)
