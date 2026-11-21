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
    t = build_text("Hallo", r"Sehr geehrt,\n\nich schreibe")   # raw string: literal backslash + n
    assert "\\n" not in t and "\n\nich" in t
    assert unescape(None) == "" and unescape(r"a\nb") == "a\nb"


@needs_split
def test_holdout_seen_ids_are_holdout_ids_only():
    from src.data import holdout_seen_ids, load_test_inputs
    seen, te = holdout_seen_ids(), set(load_test_inputs().ticket_id)
    assert seen and seen <= te
    assert 0.15 < len(seen) / len(te) < 0.40   # ~22% of all holdout tickets have a twin in the KB


def test_norm_body_collapses_case_space_and_escapes():
    import pandas as pd
    from src.data import norm_body
    s = pd.Series([r"Hallo  Welt\n\nBitte", "hallo welt bitte"])   # raw: literal backslash-n
    assert norm_body(s).nunique() == 1
