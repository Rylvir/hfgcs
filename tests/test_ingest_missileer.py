import json
from pathlib import Path

from decoding.ingest_missileer import main, select, split_for, to_label

FIXTURE = Path(__file__).parent / "fixtures" / "missileer_archive_sample.json"


def feed():
    return json.loads(FIXTURE.read_text())


def test_fixture_is_cc0():
    assert feed()["license"] == "CC0-1.0"


def test_corrected_row_maps_to_silver_with_machine_reference():
    f = feed()
    row = next(r for r in f["messages"] if r["corrected"])
    lab = to_label(row, f)
    assert lab["label_status"] == "silver"
    assert lab["labels"]["body"] == row["text"].replace("?", ".")
    assert lab["reference_machine"]["text"] == row["machine_text"].replace("?", ".")
    assert lab["license"] == "CC0-1.0" and lab["attribution"].endswith("missileer.watch")
    assert lab["audio_format"] == "opus" and lab["type"] == "eam"


def test_machine_rows_never_become_labels_of_record():
    f = feed()
    for row in f["messages"]:
        lab = to_label(row, f)
        if not row["corrected"]:
            assert lab["label_status"] == "machine"


def test_radio_check_has_no_body_and_unheard_chars_are_dots():
    f = feed()
    labs = {r["id"]: to_label(r, f) for r in f["messages"]}
    assert labs["eam-8c9af94f"]["type"] == "radio_check"
    assert labs["eam-8c9af94f"]["labels"]["body"] is None
    unconf = labs["eam-52e408b3"]
    assert "?" not in unconf["labels"]["body"] and "has_unheard_chars" in unconf["flags"]


def test_select_prefers_corrected_then_complete():
    picked = select(feed()["messages"], 2)
    assert picked[0]["corrected"] is True
    assert picked[1]["grade"] == "message" and "?" not in picked[1]["text"]


def test_split_is_deterministic():
    assert split_for("eam-8c26a8bf") == split_for("eam-8c26a8bf")


def test_cli_offline(tmp_path):
    out = tmp_path / "s.jsonl"
    assert main(["--feed", str(FIXTURE), "--sample", "3", "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 3 and all(r["source"] == "missileer.watch" for r in rows)
