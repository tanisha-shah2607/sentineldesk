"""Data-layer tests. Split tests are skipped until data/ exists."""
import pandas as pd
import pytest

from src.config import KB_PATH, TEST_INPUT_PATH, TEST_LABEL_PATH
from src.data import build_text

needs_split = pytest.mark.skipif(
    not (KB_PATH.exists() and TEST_INPUT_PATH.exists() and TEST_LABEL_PATH.exists()),
    reason="run scripts/make_split.py first")


def test_null_subject_handled():
    assert build_text(None, "hello").startswith("[no subject]")
    assert build_text("", "hello").startswith("[no subject]")
    assert build_text("   ", "hello").startswith("[no subject]")
    assert build_text(float("nan"), None)


def test_normal_subject_kept():
    assert build_text("VPN down", "help") == "VPN down\nhelp"


@needs_split
def test_no_overlap_and_80_20():
    from src.data import load_kb, load_test_inputs
    kb, te = load_kb(), load_test_inputs()
    assert set(kb.ticket_id).isdisjoint(set(te.ticket_id))
    assert abs(len(kb) / (len(kb) + len(te)) - 0.8) < 0.001


@needs_split
def test_pinned_dataset():
    from src.config import EXPECTED_ROWS
    from src.data import load_kb, load_test_inputs
    assert len(load_kb()) + len(load_test_inputs()) == EXPECTED_ROWS


@needs_split
def test_holdout_answer_never_on_disk():
    """Strongest leakage guarantee: no holdout file has an answer column."""
    assert set(pd.read_parquet(TEST_INPUT_PATH).columns) == {"ticket_id", "subject", "body"}
    assert "answer" not in pd.read_parquet(TEST_LABEL_PATH).columns


@needs_split
def test_split_is_deterministic():
    from src.data import load_kb
    assert load_kb().ticket_id.iloc[0] == "TKT-00000"


@needs_split
def test_clean_views():
    from src.data import load_kb_clean, load_test_labels
    k = load_kb_clean()
    assert k["type"].notna().all() and set(k.priority) <= {"high", "medium", "low"}
    assert load_kb_clean(need_answer=True)["answer"].notna().all()
    assert set(load_test_labels().priority) <= {"high", "medium", "low"}


@needs_split
def test_shared_val_slice():
    from src.data import load_kb
    tr, va, al = load_kb("train"), load_kb("val"), load_kb()
    assert set(tr.ticket_id).isdisjoint(set(va.ticket_id))
    assert len(tr) + len(va) == len(al)
    assert 0.08 < len(va) / len(al) < 0.12
    assert load_kb("val").ticket_id.tolist() == va.ticket_id.tolist()  # deterministic


def test_literal_backslash_n_unescaped():
    from src.text import unescape
    t = build_text("Hallo", "Sehr geehrt,\n\nich schreibe")
    assert "\n" not in t and "\n\nich" in t
    assert unescape(None) == "" and unescape("a\nb") == "a\nb"
