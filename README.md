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
- `decide` -> `{escalate, reasons, codes}`; `draft_reply` -> `{text, cited_ids, warnings, used_fallback, llm_error}` (`codes` and `llm_error` are additive)
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

## Gate and generation (S4)
**Escalation gate** (`src/gate.py`): deterministic rules, each a pure function. Any rule that fires sends the ticket to
"Escalate to On-Call Engineer". `decide()` returns `{escalate, reasons, codes}`, where `codes` are rule ids that evaluate.py can count.

| Code | Fires when | Notes |
|---|---|---|
| `HIGH_INCIDENT` | predicted `priority == high` and `type == Incident` | the brief's mandatory rule |
| `LOW_CONFIDENCE` | `overall_conf < CONF_THRESHOLD` | threshold in `config.py`, tuned on the `val` slice |
| `NO_PRECEDENT` | no hits, or top similarity `< SIM_THRESHOLD` | the score scale comes from S3's index |
| `UNSUPPORTED_ASK` | ticket asks for a refund/compensation, a price/discount, or a guarantee/SLA, and no retrieved answer covers that topic | en + de keyword patterns |
| `LEGAL` | legal threat or legal commitment (lawsuit, lawyer, *Anwalt*, *Klage* ...) | always a human |
| `PROMPT_INJECTION` | "ignore previous instructions", "system prompt", *"ignoriere alle vorherigen Anweisungen"* ... | |
| `EMPTY_TICKET` | body under 15 characters | |

After generation, any draft `warning` also escalates the ticket (see `pipeline.py`).

**Grounded generation** (`src/generate.py`, `src/llm.py`): hallucination control works in layers.
1. **PII masking** (`src/pii.py`): emails, phone numbers, IPs, IBANs, Luhn-valid card numbers, labelled account numbers
   and URL credentials are masked in the ticket *and* the precedents before anything is sent to the LLM.
2. **Prompt**: the model sees only the top-k retrieved precedents. The ticket is framed as untrusted customer text, and
   its `<`/`>` are escaped so it cannot close the frame. The model must cite `[TKT-xxxxx]` inline, reply in the
   detected language (translating cross-lingual precedents), and say "cannot confirm" for anything outside the evidence.
3. **Output validator** (`validate_draft`): strips citations not in the retrieved set; strips sentences with money amounts,
   percentages, deadlines ("within 4 hours"), or refund/guarantee/SLA/credit/discount/free-of-charge claims that the
   evidence does not support (refusal sentences such as "we cannot confirm a refund" are kept); flags a draft with no
   citation or in the wrong language. Every fix adds a `warning`, so the ticket escalates.
4. **Template fallback**: if the LLM is not configured, fails, times out or refuses, the reply quotes the top precedent
   verbatim in a localized template. That reply is grounded by construction (`used_fallback: true`, `llm_error` says why).

LLM setup (`.env`, see `.env.example`). The default model is `claude-opus-5-5` at `LLM_EFFORT=low`; set `LLM_MODEL=claude-haiku-5-5` for lower latency.
- `LLM_PROVIDER=anthropic`: direct Claude API with `ANTHROPIC_API_KEY`, plus the server-side refusal fallback.
- `LLM_PROVIDER=vertex`: Claude on Google Cloud Vertex AI, paid with Google Cloud credits. There is no API key (a Gemini key does not work).
  Enable the Claude model in Vertex AI Model Garden, run `gcloud auth application-default login`, and set `ANTHROPIC_VERTEX_PROJECT_ID`.
  Refusals retry on `LLM_FALLBACK_MODEL` through the SDK's client-side middleware.
- `LLM_PROVIDER=ollama`: a local model. With `LLM_PROVIDER=none` the app still works using template replies.
Tests: `pytest tests/test_gate.py tests/test_pii.py tests/test_generate.py tests/test_llm.py` (offline; the LLM is faked).

## Results
_S1 fills in after `python evaluate.py`._
