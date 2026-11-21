"""Leakage audit. Run before the demo and before submitting: python scripts/leakage_audit.py
Exit code 1 if any check FAILs. Checks:

 1. DISK     holdout files contain no `answer`; inputs file has only id/subject/body.
 2. STATIC   only allowed files touch holdout files/loaders or read raw data directly.
 3. ARTIFACT models/ and index/ files contain no holdout ticket ids (TKT-<id> >= KB size).
 4. RETRIEVAL every ticket_id returned for holdout queries is a KB id (needs S3's real retrieve()).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402

results: list[tuple[str, str, str]] = []   # (status, check, detail)


def rec(status, check, detail=""):
    results.append((status, check, detail))


# ---------------------------------------------------------------- 1. disk
def check_disk():
    if not (C.KB_PATH.exists() and C.TEST_INPUT_PATH.exists() and C.TEST_LABEL_PATH.exists()):
        rec("SKIP", "disk", "data/ missing; run scripts/make_split.py")
        return None
    cols_in = set(pd.read_parquet(C.TEST_INPUT_PATH).columns)
    cols_lab = set(pd.read_parquet(C.TEST_LABEL_PATH).columns)
    ok = cols_in == {"ticket_id", "subject", "body"} and "answer" not in cols_lab
    rec("PASS" if ok else "FAIL", "disk", f"inputs cols={sorted(cols_in)} labels cols={sorted(cols_lab)}")
    return len(pd.read_parquet(C.KB_PATH, columns=["ticket_id"]))


# ---------------------------------------------------------------- 2. static
# pattern -> (allowed path prefixes/names, severity if violated)
RULES = {
    # evaluate.py and baselines.py are scoring-only scripts: they load labels to SCORE predictions, never as input
    r"load_test_labels|TEST_LABEL_PATH|test_labels": (
        {"evaluate.py", "scripts/baselines.py", "src/data.py", "src/config.py", "scripts/leakage_audit.py"}, "FAIL"),
    r"load_test_inputs|TEST_INPUT_PATH|test_inputs": (
        {"evaluate.py", "scripts/baselines.py", "src/data.py", "src/config.py", "scripts/leakage_audit.py"}, "WARN"),
    r"read_parquet|read_csv|load_dataset|read_json\(|open\(.*data/": (
        # src/retrieval.py only reads its own index/ cache (kb_meta.parquet), never data/
        {"src/data.py", "src/retrieval.py", "scripts/leakage_audit.py", "scripts/run_adversarial.py"}, "WARN"),
}


def check_static():
    skip_dirs = {".venv", ".git", "data", "models", "index", "results", "__pycache__", ".pytest_cache"}
    n_files = 0
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).as_posix()
        if set(p.relative_to(ROOT).parts[:-1]) & skip_dirs or rel.startswith("tests/"):
            continue
        n_files += 1
        text = p.read_text(encoding="utf-8", errors="replace")
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        # strip docstrings so prose mentions don't count
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        for pat, (allowed, sev) in RULES.items():
            if rel in allowed:
                continue
            for m in re.finditer(pat, code):
                rec(sev, "static", f"{rel}: uses `{m.group(0)}` (allowed only in {sorted(allowed)})")
                break
    if not any(r[1] == "static" for r in results):
        rec("PASS", "static", f"{n_files} python files scanned, no holdout/loader misuse")


# ---------------------------------------------------------------- 3. artifacts
def check_artifacts(kb_size):
    if kb_size is None:
        rec("SKIP", "artifacts", "no KB size")
        return
    scanned, bad = 0, []
    for d in (C.MODEL_DIR, C.INDEX_DIR):
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            scanned += 1
            ids = {int(x) for x in re.findall(rb"TKT-(\d{5})", p.read_bytes())}
            leaked = sorted(i for i in ids if i >= kb_size)
            if leaked:
                bad.append(f"{p.relative_to(ROOT).as_posix()}: holdout ids e.g. {leaked[:3]}")
    if bad:
        for b in bad:
            rec("FAIL", "artifacts", b)
    elif scanned:
        rec("PASS", "artifacts", f"{scanned} files in models/ and index/ scanned, no holdout ids")
    else:
        rec("SKIP", "artifacts", "models/ and index/ are empty (nothing built yet)")


# ---------------------------------------------------------------- 4. retrieval
def check_retrieval(kb_size, n=200):
    if kb_size is None:
        rec("SKIP", "retrieval", "no data")
        return
    from src.data import load_test_inputs
    from src.retrieval import retrieve
    from src.text import build_text
    q = load_test_inputs().head(n)
    try:
        ids = {h["ticket_id"] for s, b in zip(q.subject, q.body) for h in retrieve(build_text(s, b))}
    except NotImplementedError:
        rec("SKIP", "retrieval", "retrieve() not implemented yet")
        return
    except Exception as e:  # noqa: BLE001  (e.g. index not built)
        rec("SKIP", "retrieval", f"retrieve() failed ({type(e).__name__}: {e}); build the index first")
        return
    if ids <= {"TKT-00000"}:
        rec("SKIP", "retrieval", "retrieve() is still the stub")
        return
    bad = sorted(i for i in ids if int(i[4:]) >= kb_size)
    rec("FAIL" if bad else "PASS", "retrieval",
        f"{len(ids)} distinct ids from {n} holdout queries" + (f"; holdout ids returned: {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    kb_size = check_disk()
    check_static()
    check_artifacts(kb_size)
    check_retrieval(kb_size)
    width = max(len(c) for _, c, _ in results)
    for status, check, detail in results:
        print(f"[{status}] {check:<{width}}  {detail}")
    fails = sum(s == "FAIL" for s, _, _ in results)
    warns = sum(s == "WARN" for s, _, _ in results)
    print(f"\n{fails} FAIL, {warns} WARN" + ("  -> LEAKAGE RISK" if fails else "  -> OK"))
    sys.exit(1 if fails else 0)
