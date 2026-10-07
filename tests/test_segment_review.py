import pandas as pd
import pytest

from segmentation.review import build_rows, evaluate, insert_unplaced, split_segments


def F(n):
    return f"NL-HaNA_1.04.02_9_{n:04d}"


def item(key, start, end, boundary="adjacent", kind="toc", **kw):
    return {"key": key, "kind": kind, "depth": 0, "title": key, "start": F(start), "end": F(end),
            "model_start": F(start), "model_end": F(end), "boundary": boundary, "placed_by": "exact", "review": None, **kw}


def data(*items):
    return {"inventory": "9", "scans": [{"f": F(n)} for n in range(1, 41)], "items": list(items)}


def test_moved_start_moves_previous_end_along():
    d = data(item("toc:1", 1, 5), item("toc:2", 6, 10, "adjacent"), item("toc:3", 10, 14, "shared"))
    rows = build_rows(d, {"key": "toc:2", "start": F(8), "end": F(10), "link": True})
    assert [r["item_key"] for r in rows] == ["toc:2", "toc:1"]
    assert rows[0]["start_status"] == "corrected" and rows[0]["end_status"] == "confirmed"
    assert rows[1]["end_scan"] == F(7) and rows[1]["end_status"] == "implied" and rows[1]["start_status"] is None


def test_moved_end_moves_next_start_along_on_shared_boundary():
    d = data(item("toc:1", 1, 5), item("toc:2", 6, 10), item("toc:3", 10, 14, "shared"))
    rows = build_rows(d, {"key": "toc:2", "start": F(6), "end": F(12), "link": True})
    assert rows[1]["item_key"] == "toc:3" and rows[1]["start_scan"] == F(12)


def test_entries_on_the_same_span_and_reviewed_neighbours_stay():
    reviewed = item("toc:0", 1, 5)
    reviewed["review"] = {"end_status": "confirmed", "end_scan": F(5)}
    d = data(reviewed, item("toc:1", 6, 20), item("toc:2", 6, 20, "shared"), item("toc:3", 20, 25, "shared"))
    assert len(build_rows(d, {"key": "toc:2", "start": F(15), "end": F(20), "link": True})) == 1
    assert len(build_rows(d, {"key": "toc:1", "start": F(7), "end": F(14), "link": True})) == 1
    assert len(build_rows(d, {"key": "toc:1", "start": F(7), "end": F(14), "link": False})) == 1


def test_unsure_and_verdicts_do_not_count_as_checked():
    d = data(item("toc:1", 1, 5), item("toc:2", 6, 10))
    unsure = build_rows(d, {"key": "toc:2", "start": F(7), "end": F(10), "verdict": "unsure", "link": True})
    assert len(unsure) == 1 and unsure[0]["start_status"] is None and unsure[0]["start_scan"] == F(7)
    assert evaluate(unsure).empty is False and "start exact" not in evaluate(unsure).columns
    with pytest.raises(ValueError):
        build_rows(d, {"key": "toc:2", "start": F(9), "end": F(7)})
    with pytest.raises(ValueError):
        build_rows(d, {"key": "toc:2", "verdict": "bogus"})


def test_side_on_a_double_scan_is_a_correction():
    d = data(item("toc:1", 1, 5))
    row = build_rows(d, {"key": "toc:1", "start": F(1), "start_side": "right", "end": F(5)})[0]
    assert row["start_status"] == "corrected" and row["start_side"] == "right" and row["end_status"] == "confirmed"


def test_added_documents_get_numbered_keys():
    d = data(item("toc:1", 1, 5), {**item("added:1", 7, 8, kind="added")})
    row = build_rows(d, {"key": "added:new", "start": F(10), "end": F(12), "title": "Lijst"})[0]
    assert row["item_key"] == "added:2" and row["title"] == "Lijst" and row["start_status"] == "corrected"


def test_unplaced_entries_follow_the_preceding_index_entry():
    items = [item("toc:1", 1, 5, csv_id="1", toc_order=1.0), item("subdoc:x", 3, 5, kind="subdoc", depth=1),
             item("toc:3", 6, 9, csv_id="3", toc_order=3.0)]
    toc = pd.DataFrame({"csv_id": [1, 2, 3], "title": ["a", "b", "c"], "folio_start": [1, 4, 7], "folio_end": [3, 6, 9],
                        "toc_order": [1, 2, 3], "date_begin": [None, None, None]})
    out = insert_unplaced(items, toc)
    assert [x["key"] for x in out] == ["toc:1", "subdoc:x", "toc:2", "toc:3"]
    assert out[2]["kind"] == "unplaced" and out[2]["index_folios"] == "fol. 4–6"


def test_evaluate_buckets_by_method():
    rows = [
        {"placed_by": "interpolated", "kind": "toc", "model_start": F(10), "start_scan": F(12), "start_status": "corrected",
         "model_end": F(20), "end_scan": F(20), "end_status": "confirmed"},
        {"placed_by": "exact", "kind": "toc", "model_start": F(1), "start_scan": F(1), "start_status": "confirmed",
         "model_end": F(5), "end_scan": F(6), "end_status": "implied"},
        {"placed_by": "not placed", "kind": "unplaced", "start_scan": F(30), "start_status": "corrected", "end_status": None},
    ]
    t = evaluate(rows)
    assert t.loc["interpolated", "start ±2–5"] == 1 and t.loc["interpolated", "end exact"] == 1
    assert t.loc["exact", "start exact"] == 1 and "end ±1" not in t.columns  # implied ends are not counted
    assert t.loc["not placed", "found"] == 1 and t.loc["all", "start exact"] == 1


def test_split_segments_writes_one_file_per_inventory(tmp_path):
    src = tmp_path / "segments.csv"
    pd.DataFrame({"inventory": ["1", "1", "2", "3", "3"], "segment": ["0", "1", "0", "0", "1"],
                  "evidence": ['{"a": "x\ny"}', "", "", "", ""]}).to_csv(src, index=False)
    assert split_segments(str(src), str(tmp_path / "out"), chunksize=2) == 3
    back = pd.read_parquet(tmp_path / "out" / "1.parquet")
    assert len(back) == 2 and back["evidence"].iat[0] == '{"a": "x\ny"}'
    assert len(pd.read_parquet(tmp_path / "out" / "3.parquet")) == 2
