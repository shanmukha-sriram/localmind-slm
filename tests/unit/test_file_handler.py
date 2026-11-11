import json

import pandas as pd

from app.utils.file_handler import atomic_write_text, read_jsonl, write_csv, write_json


def test_atomic_write_creates_parents_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a" / "b.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert not list(target.parent.glob("*.tmp"))


def test_atomic_write_overwrites_existing_file(tmp_path):
    target = tmp_path / "f.txt"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_read_jsonl_skips_truncated_last_line(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding="utf-8")  # interrupted write
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_json_and_csv_exports_keep_unicode(tmp_path):
    write_json({"text": "h₂o café"}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))["text"] == "h₂o café"
    df = pd.DataFrame({"model": ["a"], "v": [1.5]})
    write_csv(df, tmp_path / "x.csv")

    assert pd.read_csv(tmp_path / "x.csv").equals(df)
