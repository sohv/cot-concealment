import json

from src.utils.io import append_jsonl, read_jsonl, round_floats, write_json


def test_round_floats_recurses_through_nested_containers():
    obj = {"a": 0.123456, "b": [1.987654, {"c": 2.5}], "d": "text", "e": 3}
    assert round_floats(obj) == {"a": 0.1235, "b": [1.9877, {"c": 2.5}], "d": "text", "e": 3}


def test_round_floats_leaves_bools_and_ints_alone():
    assert round_floats({"ok": True, "n": 7}) == {"ok": True, "n": 7}


def test_append_jsonl_appends_without_overwriting(tmp_path):
    path = tmp_path / "out.jsonl"
    append_jsonl(path, [{"id": 1, "score": 0.123456}])
    append_jsonl(path, [{"id": 2, "score": 0.5}])
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["score"] == 0.1235


def test_write_json_indents_and_rounds(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"seed": 42, "temp": 0.600001})
    text = path.read_text()
    assert "\n  " in text
    assert json.loads(text)["temp"] == 0.6
