"""Data-layer tests. Split tests are skipped until data/ exists."""
import pytest

from src.config import KB_PATH, TEST_PATH
from src.data import build_text

needs_split = pytest.mark.skipif(not (KB_PATH.exists() and TEST_PATH.exists()),
                                 reason="run scripts/make_split.py first")


def test_null_subject_handled():
    assert build_text(None, "hello").startswith("[no subject]")
    assert build_text("", "hello").startswith("[no subject]")
    assert build_text("   ", "hello").startswith("[no subject]")
    assert build_text(float("nan"), None)  # NaN subject + None body: no crash


def test_normal_subject_kept():
    assert build_text("VPN down", "help") == "VPN down\nhelp"


@needs_split
def test_no_overlap_and_80_20():
    from src.data import load_kb, load_test_inputs
    kb, te = load_kb(), load_test_inputs()
    assert set(kb.ticket_id).isdisjoint(set(te.ticket_id))
    ratio = len(kb) / (len(kb) + len(te))
    assert abs(ratio - 0.8) < 0.001


@needs_split
def test_test_inputs_expose_no_labels():
    from src.data import load_test_inputs
    cols = set(load_test_inputs().columns)
    assert cols == {"ticket_id", "subject", "body"}


@needs_split
def test_labels_exclude_answer():
    from src.data import load_test_labels
    assert "answer" not in load_test_labels().columns


@needs_split
def test_split_is_deterministic():
    """Same seed => same first KB id mapping; guards against accidental re-shuffles."""
    from src.data import load_kb
    kb = load_kb()
    assert kb.ticket_id.iloc[0] == "TKT-00000"


@needs_split
def test_clean_views():
    from src.data import load_kb_clean, load_test_labels
    k = load_kb_clean()
    assert k["type"].notna().all() and set(k.priority) <= {"high", "medium", "low"}
    assert load_kb_clean(need_answer=True)["answer"].notna().all()
    assert set(load_test_labels().priority) <= {"high", "medium", "low"}
