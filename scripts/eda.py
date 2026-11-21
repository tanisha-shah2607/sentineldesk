"""EDA on the KB split only (never inspect holdout labels for modelling decisions).
Usage: python scripts/eda.py  -> prints stats, writes results/eda.md"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_kb  # noqa: E402

kb = load_kb()
out = [f"# EDA (KB split, n={len(kb)})\n"]

subj_null = kb["subject"].isna() | (kb["subject"].astype(str).str.strip() == "")
out.append(f"Null/empty subjects: {int(subj_null.sum())} ({subj_null.mean():.1%})")
out.append(f"Null bodies: {int(kb['body'].isna().sum())}")
out.append(f"Null answers: {int(kb['answer'].isna().sum())}\n")

for col in ["type", "queue", "priority", "language"]:
    out.append(f"## {col}\n```\n{kb[col].value_counts(dropna=False).to_string()}\n```")

out.append("## priority x type\n```\n" + pd.crosstab(kb["type"], kb["priority"]).to_string() + "\n```")
out.append("## queue x priority\n```\n" + pd.crosstab(kb["queue"], kb["priority"]).to_string() + "\n```")

blen = kb["body"].fillna("").str.len()
out.append(f"## body length (chars)\n```\n{blen.describe().to_string()}\n```")
tag_cols = [c for c in kb.columns if c.startswith("tag_")]
if tag_cols:
    out.append(f"## avg non-null tags per ticket: {kb[tag_cols].notna().sum(axis=1).mean():.2f}")

text = "\n\n".join(out)
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "eda.md").write_text(text, encoding="utf-8")
print(text)
