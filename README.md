# SentinelDesk

Multilingual incident triage, smart escalation, and precedent-grounded reply copilot.

## Setup (Python 3.11 or 3.12; faiss/torch wheels are unreliable on 3.14)
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in LLM settings (S4)
python scripts/make_split.py    # prints row count + fingerprint; aborts if your dataset differs from the team's
pytest
```
Run UI: `streamlit run app.py` · Evaluate: `python evaluate.py --n-gen 50`
Do not `pip freeze > requirements.txt` in PowerShell (writes UTF-16). Edit requirements.txt by hand.

## Ownership and contracts
| Module | Owner | Contract |
|---|---|---|
| `src/data.py`, `src/text.py`, `evaluate.py` | S1 | data loaders, `build_text`, staged evaluation |
| `src/triage.py` | S2 | `triage(subject, body)`, `triage_batch([(subject, body)])` |
| `src/retrieval.py` | S3 | `retrieve(text, k)`, `retrieve_batch(texts, k)`, `build_index(part)` |
| `src/gate.py`, `generate.py`, `llm.py`, `pii.py` | S4 | `decide(...)`, `draft_reply(...)` |
| `src/pipeline.py`, `app.py`, `burst.py` | S5 | `run_ticket(subject, body)` |

Return shapes are documented at the top of each module; every stub already returns them, so the app runs from minute one.
- `triage` -> `{language, type, queue, priority, conf:{type,queue,priority}, overall_conf}`; `overall_conf = min(type conf, queue conf)`
  (priority excluded; high+Incident still escalates); the gate compares it to `CONF_THRESHOLD`; `conf.priority` is the probability
  of the label output, after the recall-tuned threshold.
- `build_text` / `unescape` (src/text.py) turn literal `\n` escapes (21% of bodies, 3% of answers) into real line breaks.
  Use `unescape(answer)` before showing or prompting with an answer.
- `retrieve` -> `[{ticket_id, score, subject, answer, snippet, queue, type, language}]`
- `decide` -> `{escalate, reasons}`; `draft_reply` -> `{text, cited_ids, warnings, used_fallback}`
- `run_ticket` -> `{triage, hits, gate, draft(+for_review), latency}`; latency keys (ms): `triage_ms, retrieve_ms, gate_ms, generate_ms, total_ms`
- Two-stage escalation: `decide()` runs before generation; any draft `warnings` afterwards also escalate. Escalated tickets still get a draft, labelled "for engineer review".

## Data rules (zero leakage)
- All data goes through `src/data.py`. The holdout `answer` column is never written to disk; holdout inputs
  (`test_inputs.parquet`) and labels (`test_labels.parquet`) are separate files; only `evaluate.py` calls `load_test_labels()`.
- Dataset pin: 61,765 rows, `config.EXPECTED_FINGERPRINT`; `make_split.py` aborts on mismatch so everyone has identical `TKT-` ids.
  Only `en`/`de` appear in the file.
- ~21% of raw rows (all German) have no type, no answer, 5-level priority and 42 extra queues. Use the clean views:
  `load_kb_clean()` for training (38,819 rows), `load_kb_clean(need_answer=True)` for the RAG index (38,810),
  `load_test_labels()` scores 9,768 clean holdout rows. The live pipeline must still not crash on any input.
- **One shared validation slice:** `load_kb(part="train"|"val")` / `load_kb_clean(part=...)` (10%, keyed on ticket_id).
  Train models and build the tuning index on `train`; calibrate and tune thresholds on `val`. Never plot confidence on rows
  the model trained on, and never tune `SIM_THRESHOLD` with val rows inside the index (each would retrieve itself at ~1.0).
  The final app index and final models may use `part="all"`.
- 8.7% of subjects are null: always build input with `src.text.build_text()`.
- Clean-row language mix: en 22,561 / de 16,258.

## Evaluation (three stages; never run the LLM over the whole holdout)
A. Classification: `triage_batch` on the full clean holdout: Macro-F1 (type, queue, priority).
B. Retrieval + gate: `retrieve_batch` + `decide`, no LLM: escalation rate, retrieval hit rates.
C. Generation: `run_ticket` on a stated sample (`--n-gen`, default 50): citation validity, warning/fallback rate, latency.

**High-Priority Incident Recall** (headline): of holdout tickets whose true label is Incident + high, the fraction the system
escalates to a human. We also report the classifier version: the fraction for which triage predicts both labels correctly.

## Checks to run before merging your module / before the demo
- `python scripts/run_adversarial.py`: 20 tricky tickets in `eval/adversarial.json` (empty subject, prompt injection,
  fake refund/SLA/price/integration asks, legal threat, PII, unsafe advice, French, garbage, 400x-long body). Checks:
  no crash, escalation expectation, language, no forbidden claims in the draft, cited ids were retrieved.
  Note: with the stubs everything escalates, so most cases pass trivially; it only becomes meaningful with real modules.
- `python scripts/leakage_audit.py`: disk (no holdout answers), static scan (who touches holdout files/loaders),
  models/ and index/ contain no holdout ticket ids, retrieval returns only KB ids. Exits 1 on any FAIL.

## Results
_S1 fills in after `python evaluate.py`._
