import datetime

import numpy as np
import pandas as pd
import pytest

from segmentation.header_dates import compare_dates, parse_header_date
from segmentation.inventory import InventoryData
from segmentation.model import AlignParams, LengthPrior, SegmentationModel
from segmentation.predictors import clean_numbers, compute_features, signature_kind
from segmentation.evaluation import flat_segments
from segmentation.segmenter import ScanScores, align_toc, segment_inventory, segment_scans


# ── header dates ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "header, expected, precision",
    [
        ("['Van Bengale onder dato 14:\\' Febr: 1743.']", datetime.date(1743, 2, 14), "day"),
        ("['Van Iavas Oost Cust den 15„e 9ber 1714.']", datetime.date(1714, 11, 15), "day"),
        ("['141. van Cabo de goede Hoop ult:o meij 1720.']", datetime.date(1720, 5, 31), "day"),
        ("['van Jappan den 7:\\' Navember 1683. ']", datetime.date(1683, 11, 7), "day"),
        ("['Van Maccasser den 10: IJunij 1764. ']", datetime.date(1764, 6, 10), "day"),
        ("['Den 12. Februarij In de Stad Cochim ', 'Ao. 1774 ']", datetime.date(1774, 2, 12), "day"),
        ("['A„o 1693, Junij banda int Cast:s nass=r ']", datetime.date(1693, 6, 1), "month"),
        ("['A:o 1741. ']", datetime.date(1741, 1, 1), "year"),
    ],
)
def test_parse_header_date(header, expected, precision):
    d = parse_header_date(header)
    assert d is not None and d.date == expected and d.precision == precision


def test_parse_header_date_none():
    assert parse_header_date("['P=r Transport ƒ82682: 18: 3 ƒ29077: 7: 8 E ']") is None
    assert parse_header_date("") is None


def test_compare_dates_noise():
    a = parse_header_date("den 14 Feb 1743")
    assert compare_dates(a, parse_header_date("den 14 Febr 1743")) == "same"
    assert compare_dates(a, parse_header_date("den 11 Feb 1743")) == "uncertain"  # one-character misread
    assert compare_dates(a, parse_header_date("den 24 Maart 1743")) == "changed"
    assert compare_dates(a, parse_header_date("Maart 1743")) == "changed"
    assert compare_dates(a, parse_header_date("A:o 1743")) == "same"
    assert compare_dates(a, None) == "unknown"


def test_signature_kind():
    assert signature_kind("Accordeert Jaabe Roeder") == "collation"
    assert signature_kind("Gerrit Ris") == "signed"
    assert signature_kind("E") == "quire"
    assert signature_kind("„ 6162. 5- – -") == "number"
    assert signature_kind("") == "none"


# ── numbers ───────────────────────────────────────────────────────────────────


def test_clean_numbers_misread_and_restart():
    # page numbers on every other scan; 1691 is a misread of 169; numbering restarts at 1
    raw = [[165], [], [167], [], [1691], [], [171], [], [173], [], [1], [], [3], [], [5]]
    value, run = clean_numbers(pd.Series(raw))
    assert np.isnan(value[4])
    assert value[6] == 171
    assert run[0] == 0 and run[8] == 0  # scan 9 (unnumbered, between the runs) may go either way
    assert run[10] == 1 and run[14] == 1


# ── dynamic programs ──────────────────────────────────────────────────────────


def _scores(start, end=None, shared=0.1, nondoc=None):
    n = len(start)
    p = np.full(n, shared)
    return ScanScores(
        start=np.asarray(start, dtype=float),
        end=np.zeros(n) if end is None else np.asarray(end, dtype=float),
        log_shared=np.log(p),
        log_not_shared=np.log(1 - p),
        nondoc=np.full(n, -5.0) if nondoc is None else np.asarray(nondoc, dtype=float),
    )


def test_segment_scans_starts_follow_logits():
    segs, _ = segment_scans(_scores([0, -3, -3, 2, -3, -3, 3, -3]), LengthPrior(), 100)
    assert [(a, e, t) for a, e, t in segs] == [(0, 2, "first"), (3, 5, "adjacent"), (6, 7, "adjacent")]


def test_segment_scans_shared_boundary():
    shared = np.full(8, 0.1)
    shared[5] = 0.95  # strong evidence that the document starting on scan 5 shares it with the previous one
    sc = _scores([0, -5, -5, -5, -5, 3, -5, -5])
    sc.log_shared, sc.log_not_shared = np.log(shared), np.log(1 - shared)
    segs, _ = segment_scans(sc, LengthPrior(), 100)
    assert segs == [(0, 5, "first"), (5, 7, "shared")]


def test_segment_scans_gap_and_forced():
    # scans 3-4 look like non-document pages (e.g. a cover), scan 6 is a forced ToC start
    nondoc = np.array([-5, -5, -5, 4, 4, -5, -5, -5], dtype=float)
    start = np.array([0, -5, -5, -5, -5, 2, -5, -5], dtype=float)
    segs, _ = segment_scans(_scores(start, nondoc=nondoc), LengthPrior(), 100, forced={6})
    assert segs == [(0, 2, "first"), (5, 5, "gap"), (6, 7, "adjacent")]


def test_flat_segments_boundary_types():
    truth = pd.DataFrame(
        {"start": [0, 5, 9, 12], "end": [5, 8, 20, 14], "part_of_id": [None, None, None, "p"], "csv_id": [1, 2, 3, None]}
    )
    flat = flat_segments(truth, 25)
    assert flat[["start", "end", "boundary"]].values.tolist() == [
        [0, 5, "first"], [5, 8, "shared"], [9, 11, "adjacent"], [12, 14, "adjacent"]
    ]


def _toy_inventory():
    n = 24
    folios = [[] for _ in range(n)]
    for i in range(2, n, 2):  # foliated single scans: number on every other scan
        folios[i] = [i // 2]
    scans = pd.DataFrame(
        {
            "scan_id": [f"s{i}" for i in range(n)],
            "filename": [f"x_{i:04d}" for i in range(n)],
            "scan_order": range(n),
            "scan_type": "Single",
            "languages": "nld",
            "n_pages": 1,
            "is_blank": False,
            "marginalia": False,
            "layout": "single",
            "header": "",
            "signatures": "",
            "folios": folios,
            "text_len": 3000,
        }
    )
    toc = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "csv_id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "folio_start": [1, 4, 8],
            "folio_end": [3, 7, 11],
            "start_side": [None, None, None],
            "end_side": [None, None, None],
            "toc_order": [1, 2, 3],
            "folio_sequence": [1, 1, 1],
            "katern": [None, None, None],
            "date_begin": [None, None, None],
            "date_end": [None, None, None],
        }
    )
    toc["end_eff"] = toc["folio_end"]
    return InventoryData("toy", "toy", scans, toc)


def test_align_toc_places_entries_on_numbered_scans():
    inv = _toy_inventory()
    compute_features(inv)
    placements = align_toc(inv, np.zeros(inv.n), AlignParams())
    assert [(inv.toc.at[p.entry, "csv_id"], p.scan) for p in placements] == [(1, 2), (2, 8), (3, 16)]


def test_segment_inventory_labels_segments():
    inv = _toy_inventory()
    model = SegmentationModel()
    model.start.bias = -5.0
    model.nondoc.bias = -1.0  # the two unnumbered scans before the first entry become a non-document run
    res = segment_inventory(inv, model)
    kinds = {sg.start: sg.kind for sg in res.segments}
    assert kinds[2] == kinds[8] == kinds[16] == "toc"
    assert res.segments[0].kind == "non-document" and res.segments[0].end == 1


# ── texts ─────────────────────────────────────────────────────────────────────

from segmentation.predictors import CLOSING_RE, GENRE_RE, SALUTATION_RE
from segmentation.texts import TextIndex, normalize_tokens


def test_formula_patterns():
    assert GENRE_RE.search("Copia Missive van den gouverneur")
    assert GENRE_RE.search("Translaet van een Javaanse brief")
    assert SALUTATION_RE.search("Hoog Edle gestrenge groot agtb: Erntfeste heeren")
    assert CLOSING_RE.search("(onderstont) U: hoog Edelhed:s nedrige dinaren (was getekent) W=m Bolton")
    assert CLOSING_RE.search("Accordeert. H„k Geerling")
    assert not CLOSING_RE.search("den 12 April 1694 is het schip vertrokken")


def test_normalize_tokens_absorbs_spelling():
    assert normalize_tokens("Missiven") == normalize_tokens("missive")
    assert normalize_tokens("Batavia's") == ["batav"]


def test_text_index_prefers_rare_words_and_best_passage():
    passages = [
        (0, "Copia missive van den gouverneur aan Batavia"),
        (1, "Copia missive van den gouverneur uit Palembang over peper"),
        (2, "rekening van de peper"),
        (2, "Instructie voor den commandeur naar Palembang"),  # a passage after a closing formula
    ]
    idx = TextIndex(passages, 3)
    s = idx.scores("Missive uit Palembang over peper")
    assert s.argmax() == 1
    assert idx.scores("Instructie voor den commandeur").argmax() == 2
