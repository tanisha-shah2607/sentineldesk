# SentinelDesk

Multilingual incident triage, smart escalation, and precedent-grounded reply copilot.

## Setup
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in LLM settings (S4)
python scripts/make_split.py    # S1 creates the split once; share data/ with the team
```
Run UI: `streamlit run app.py` · Tests: `pytest` · Evaluate: `python evaluate.py --n 500`

## Ownership and contracts
| Module | Owner | Contract |
|---|---|---|
| `src/data.py`, `evaluate.py` | S1 | `load_kb()`, `load_test_inputs()`, `load_test_labels()` (scoring only) |
| `src/triage.py` | S2 | `triage(subject, body) -> {language,type,queue,priority,conf,overall_conf}` |
| `src/retrieval.py` | S3 | `retrieve(text, k) -> [{ticket_id,score,subject,answer,queue,type,language}]` |
| `src/gate.py`, `generate.py`, `llm.py`, `pii.py` | S4 | `decide(...)`, `draft_reply(...)` |
| `src/pipeline.py`, `app.py`, `burst.py` | S5 | `run_ticket(subject, body)` |

Every stub already returns the correct shape, so the whole app runs from minute one.
Replace stub bodies; do not change signatures or return keys without telling the team.

## Rules
- **Zero leakage:** model/retrieval code may only use `load_kb()`. Holdout labels and answers never reach inference.
- Branch per person, small PRs, `main` must always pass `pytest` and start the UI.
- Never commit `.env` or `data/`.

## Results
_S1 fills in after `python evaluate.py`._

## Data notes (S1 findings)
Raw split: 49,412 KB / 12,353 holdout (seed 42). ~21% of rows (all German) have **no type, no answer,
5-level priority (critical/very_low) and 42 extra queues**. They are not in the task's label scheme.
- Train classifiers on `load_kb_clean()` (38,819 rows: 4 types, 3 priorities, 10 queues).
- Build the RAG index from `load_kb_clean(need_answer=True)` (38,810 rows).
- Evaluation scores `load_test_labels()` clean rows only (9,768). Pipeline still must not crash on any input.
- 8.7% of subjects are null: always go through `build_text()`.
- Language mix in clean rows: en 22,561 / de 16,258.
