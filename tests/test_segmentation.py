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


# ── court records ─────────────────────────────────────────────────────────────

from segmentation.court_records import Case, _scan_ref, case_title, person_name  # noqa: E402


@pytest.mark.parametrize(
    "value, expected",
    [(99, ("9350", 99)), ("99 [9353]", ("9353", 99)), ("656 (9354)", ("9354", 656)), (np.nan, None), ("P", None)],
)
def test_scan_ref(value, expected):
    assert _scan_ref(value, "9350") == expected


def test_person_name():
    assert person_name({"Voornaam": "David Julius", "Tussenvoegsel": "van", "Achternaam": "Titsma"}) == "David Julius van Titsma"
    assert person_name({"Voornaam": "Tamboe", "Herkomst": "Madagascar"}) == "Tamboe van Madagascar"
    assert person_name({"Voornaam": "Gerrit Schaagen", "Herkomst": "Hoorn"}) == "Gerrit Schaagen"
    assert person_name({"Voornaam": "anoniem"}) == ""


def _case(persons, civil=False, place="Batavia"):
    return Case("BR-1", "Raad van Justitie", place, persons, datetime.date(1734, 10, 20), datetime.date(1734, 10, 20),
                "20 oktober 1734", ["Geweld"], civil, {})


def test_case_title():
    persons = [{"Voornaam": n, "Achternaam": "X", "Aanklacht_Datum": "1734-10-20", "Aanklacht_Standaard": "Geweld"} for n in "ABCDE"]
    assert case_title(_case(persons[:2]), persons[:2], "Adrianus Bergsma") == (
        "Advocaat-fiscaal van India mr. Adrianus Bergsma contra A X en B X; Raad van Justitie, Batavia, 20 oktober 1734 (geweld)"
    )
    assert "contra A X, B X, C X en 2 anderen;" in case_title(_case(persons), persons, None)
    assert case_title(_case(persons[:1], civil=True), persons[:1], None).startswith("Onbekende eiser contra A X;")
    # the prosecutor entered as an accused without a charge is not a defendant
    fiscal = {"Voornaam": "Adrianus", "Achternaam": "Bergsma"}
    assert "contra A X;" in case_title(_case(persons[:1]), persons[:1] + [fiscal], None, ["Adrianus Bergsma"])


# ── versions ──────────────────────────────────────────────────────────────────

from segmentation.versions import derive_toc, version_blocks  # noqa: E402


def _fn(inv, k):
    return f"NL-HaNA_1.04.02_{inv}_{k:04d}"


def test_version_blocks_chain():
    # target scans 0..9 match other scans 100..109 in order; one stray match elsewhere
    rows = [(_fn("1", t), _fn("2", 100 + t), 0.6, 0.6) for t in range(10)] + [(_fn("1", 5), _fn("3", 7), 0.2, 0.2)]
    m = pd.DataFrame(rows, columns=["scan", "other_scan", "containment", "other_containment"])
    pos = {f: int(f[-4:]) for f in set(m["scan"]) | set(m["other_scan"])}
    b = version_blocks(m, pos)
    assert len(b) == 1
    r = b.iloc[0]
    assert (r.other_inventory, r.start, r.end, r.other_start, r.other_end, r.n_matches) == ("2", 0, 9, 100, 109, 10)


def _block(inv, pairs, score):
    return {"other_inventory": inv, "start": pairs[0][0], "end": pairs[-1][0], "other_start": pairs[0][1], "other_end": pairs[-1][1],
            "n_matches": len(pairs), "score": score, "pairs": pairs}


def _docs(rows):
    return pd.DataFrame([{"doc_id": d, "title": t, "part_of_id": None, "date_earliest_begin": None, "date_latest_end": None,
                          "index_ids": f"OBP_INDEX:{d}", "start": s, "end": e} for d, t, s, e in rows])


def test_derive_toc_maps_and_merges_versions():
    blocks = pd.DataFrame([_block("A", [(t, 50 + t) for t in range(20)], 15.0), _block("B", [(t, 200 + t) for t in range(20)], 10.0)])
    others = {
        "A": _docs([("a1", "Een dito dato 5 Meij 1705.", 50, 59), ("a2", "Register.", 60, 69), ("a3", "Bijlage", 60, 62)]),
        "B": _docs([("b1", "Missive van Colombo dato 5 Meij 1705", 201, 209)]),
    }
    toc = derive_toc(blocks, others, n_scans=20)
    first = toc[toc["start"] <= 1].iloc[0]
    # the same letter in A and B: one entry, the title that does not lean on the entry before, the other as version
    assert first["title"] == "Missive van Colombo dato 5 Meij 1705" and first["n_versions"] == 2 and first["versions"] == "A:OBP_INDEX:a1"
    # two entries of one volume on the same scan stay separate
    assert set(toc[toc["start"] == 10]["title"]) == {"Register.", "Bijlage"}


def test_match_inventory_leaves_cache_intact(tmp_path, monkeypatch):
    from scipy import sparse

    from segmentation import versions as v

    def shingles(rows):
        indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
        return v.InventoryShingles(np.array([_fn("9", k) for k in range(len(rows))], dtype=object),
                                   sparse.csr_matrix((np.ones(indptr[-1], dtype=np.float32), np.concatenate(rows), indptr), shape=(len(rows), v.HASH_SPACE)))

    common = list(range(1000, 1200))  # in every scan of the pool: formulas, dropped
    target = shingles([np.array(sorted(set(range(20)) | set(common[:150])))] * 2)
    others = {str(k): shingles([np.array(sorted(set(range(20)) | set(common)))]) for k in range(v.MAX_DF + 1)}
    cache = v.ShingleCache()
    cache.items.update({"T": target, **others})
    before = (target.matrix.indptr.copy(), target.matrix.indices.copy())
    v.match_inventory("T", list(others), cache)
    assert np.array_equal(target.matrix.indptr, before[0]) and np.array_equal(target.matrix.indices, before[1])


def test_text_version_rows_symmetric():
    from segmentation.versions import text_version_rows

    def blk(i, o, s, e, os_, oe, score):
        return {"inventory": i, "other_inventory": o, "start": 0, "end": 0, "other_start": 0, "other_end": 0, "n_matches": 3,
                "score": score, "start_scan": _fn(i, s), "end_scan": _fn(i, e), "other_start_scan": _fn(o, os_), "other_end_scan": _fn(o, oe)}

    # the same run found from both sides, and a separate run
    b = pd.DataFrame([blk("10668", "4319", 7, 68, 233, 262, 33.4), blk("4319", "10668", 233, 260, 7, 66, 30.0), blk("10668", "4319", 85, 95, 268, 280, 5.0)])
    r = text_version_rows(b)
    assert len(r) == 2
    first = r.iloc[0]
    assert (first.inventory_a, first.scan_start_a, first.scan_end_a, first.inventory_b, first.scan_start_b, first.scan_end_b) == ("4319", 233, 262, "10668", 7, 68)


# ── register rows ─────────────────────────────────────────────────────────────


def _line(x0, x1, y, text):
    return {"x0": x0, "x1": x1, "y": y, "y0": y - 0.006, "y1": y + 0.004, "text": text, "region": "paragraph"}


def test_register_entries_rows_and_dittos():
    from segmentation.rows import register_entries

    lines = [
        _line(0.20, 0.60, 0.05, "Brieven & Papieren van Bengaalen"),
        _line(0.10, 0.69, 0.10, "fo. 69. a 112 Een brief door den Direct & Raad Aan de 17e: in dato 10 Jan 1775."),
        # one row split into three fragments, out of order
        _line(0.42, 0.65, 0.122, "„„ „ „ 26 Maert 1771"),
        _line(0.11, 0.31, 0.120, "123 „ 124. „ „"),
        _line(0.34, 0.42, 0.121, "„ _o"),
        _line(0.12, 0.18, 0.140, "129"),
    ]
    heading, entries = register_entries(lines)
    assert [r.text for r in heading] == ["Brieven & Papieren van Bengaalen"]
    assert [(e.folio_start, e.folio_end) for e in entries] == [(69, 112), (123, 124), (129, 129)]
    assert entries[0].resolved == "Een brief door den Direct & Raad Aan de 17e: in dato 10 Jan 1775"
    assert entries[1].resolved == "Een brief door den Direct & Raad Aan de 17e: in dato 26 Maert 1771"


def test_register_two_columns():
    from segmentation.rows import split_columns

    left = [_line(0.15, 0.52, 0.1 + 0.04 * k, f"{k + 1}. Copia missive van als boven nummer {k}") for k in range(6)]
    right = [_line(0.56, 0.90, 0.1 + 0.04 * k, f"{k + 7}. Copia resolutien genomen in rade {k}") for k in range(6)]
    cols = split_columns(left + right)
    assert len(cols) == 2 and {l["x0"] for l in cols[0]} == {0.15} and {l["x0"] for l in cols[1]} == {0.56}
    single = [_line(0.10, 0.70, 0.1 + 0.04 * k, f"{k} „ {k + 3} Een brief aan de 17e dito dito") for k in range(6)] + \
             [_line(0.75, 0.92, 0.1 + 0.04 * k, f"in dato {k + 1} Maart 1771") for k in range(6)]
    assert len(split_columns(single)) == 1  # a date column is not a second column of entries


def test_monotone_assignment():
    from segmentation.register_match import _monotone

    # entry 0 fits document 2 best, entry 1 document 1 (out of order) or 3 (a bit less): order wins
    scores = np.array([[0, 0, 5, 0], [0, 6, 0, 4], [0, 0, 0, 0]], dtype=float)
    assert _monotone(scores, [0, 1, 2]) == {0: 2, 1: 3}
