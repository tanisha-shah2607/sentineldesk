# SentinelDesk

Multilingual incident triage, smart escalation, and precedent-grounded reply copilot.

## Setup (Python 3.11 or 3.12; faiss/torch wheels are unreliable on 3.14)
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in LLM settings (S4)
python scripts/make_split.py    # prints row count + fingerprint; aborts if your dataset differs from the team's
python scripts/train_triage.py  # triage classifier (~1.5 min) -> models/triage.joblib, prints validation metrics
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
- `triage` -> `{language, type, queue, priority, conf:{type,queue,priority}, overall_conf, outage_terms}`; `overall_conf = min(type conf, queue conf)`
  (priority excluded; high+Incident still escalates); the gate compares it to `CONF_THRESHOLD`; `conf.priority` is the probability
  of the label output, after the recall-tuned threshold and the outage rule. `outage_terms` lists outage wording found
  (e.g. `["vpn down"]`, empty if none), usable as the escalation reason.
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

## Triage classifier (S2)
`triage(subject, body)` returns language, type, queue, priority, calibrated confidences and `outage_terms` in about 2 ms.
Each machine trains its own model once with `python scripts/train_triage.py`, so scikit-learn versions never clash on load.
Without `models/triage.joblib`, `triage()` returns placeholder labels with a warning, so contract tests run anywhere.

**How it works** (`src/triage_model.py`, shared by training and `triage()`)
- Input is `build_text(subject, body)`. One TF-IDF over word 1-2 grams (sublinear tf, min_df 2) feeds three logistic-regression
  heads (type, queue, priority) with `class_weight="balanced"`, since Macro-F1 counts small classes as much as big ones.
  Character 3-5 grams were tried and dropped: no better on val (type 0.895 vs 0.889, queue 0.682 vs 0.675, priority 0.714 vs 0.720)
  at about 8x the feature volume.
- Regularisation `C`: type 32, queue 16, priority 16, picked on val from {1, 4, 16, 32, 64, 128}; scores rise steeply up to 16, then flatten.
- Calibration: temperature scaling per head on val (one number per head; predictions unchanged). Expected calibration error,
  before -> after: type 0.030 -> 0.014, queue 0.079 -> 0.040, priority 0.031 -> 0.028.
- Priority bar: "high" when calibrated P(high) >= 0.31, not only when it is the most likely label.
- Outage rule: clear outage wording makes a ticket Incident + high (`OUTAGE_PATTERN`: "outage", "VPN/server/login ... down",
  "has been down", "offline", "unreachable", "ausgefallen/Ausfall", "nicht (mehr) erreichbar", "Cyberangriff"). Each marks a
  high-priority Incident in roughly half or more of the training tickets that use it (base rate 17%). Weak signals ("breach" 24%,
  "not working" 15%) are left out, and so is "crash" (45%, but common in ordinary app-crash tickets). Only 3 training tickets
  describe a VPN outage, so without the rule the English hospital VPN ticket came out as Problem / medium.
- The bar and the rule share one budget, set on val: the lowest bar that keeps priority Macro-F1 within 2 points of the model alone.
- Each confidence is the calibrated probability of the label actually output, so a "high" set by the bar or the rule can show a
  low `conf.priority`. `overall_conf = min(conf.type, conf.queue)`.
- Language: langdetect loaded with only its English and German profiles and a fixed seed, run on the raw subject + body without
  the "[no subject]" placeholder (those English words tipped short German tickets to English). The dataset's `language` label is
  wrong on ~11% of KB tickets: 4,236 labelled `de` are written in English, and so are their answers. Against the real language,
  the detector disagrees on about 20 of 38,810 tickets. Use `triage()["language"]`, not the label.

**Validation results** (34,955 train / 3,864 val tickets from the shared slice; `C`, temperatures, bar and rule were all tuned on val)

25.8% of val tickets have exactly the same body (and labels) as a training ticket, and the model gets those ~99% right. The
never-seen column (2,869 tickets) is closer to what new tickets will see. The holdout probably contains the same kind of copies.

| | all val | never-seen |
|---|---|---|
| type Macro-F1 | 0.884 | 0.848 |
| queue Macro-F1 | 0.683 | 0.565 |
| priority Macro-F1 | 0.696 | 0.593 |
| High-Priority Incident Recall (triage predicts Incident + high) | 0.812 | 0.753 |
| ... counting low-confidence escalation (`overall_conf` < 0.50) | 0.909 | 0.879 |

Model alone (most likely label, no bar, no rule): recall 0.738 / 0.656, priority Macro-F1 0.714 / 0.613.

`overall_conf` against routing accuracy (type and queue both right), all val:

| `overall_conf` | below 0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9 and up |
|---|---|---|---|---|---|
| share of tickets | 4% | 25% | 27% | 30% | 14% |
| routing right | 15% | 31% | 54% | 86% | 98% |

So `CONF_THRESHOLD = 0.50` sends 29% of tickets to a human, and the model's routing is right on only 28% of those.

Speed: 1.9 ms per ticket (p95 2.9 ms); `triage_batch` 0.9 ms per ticket, so the full clean holdout takes about 10 s.
The model file is 10 MB and loads in about 3 s.

**Leakage**: `scripts/train_triage.py` reads only `load_kb_clean(part="train")` and `load_kb_clean(part="val")`, never the holdout
files, and every threshold is tuned on val. `scripts/leakage_audit.py` passes with `models/triage.joblib` present.

**Known limits**
- The labels come from a synthetic dataset and the model learns its conventions; for example, pricing questions go to Billing and
  Payments, as 46% of training tickets mentioning prices do.
- Queue is the weakest label: General Inquiry 0.57, Product Support 0.64, Customer Service 0.65 F1 (model alone, val).
- Hand-written tickets unlike the dataset are harder; the English hospital VPN ticket still routes to Customer Service (the rule
  fixes type and priority, not the department).
- The lower bar labels many unusual inputs (prompt injections, garbage) "high". Those then escalate, which is the safe side.
- Two-word English tickets with no telltale word ("Printer broken") can be detected as German.

## Evaluation (three stages; never run the LLM over the whole holdout)
A. Classification: `triage_batch` on the full clean holdout: Macro-F1 (type, queue, priority).
B. Retrieval + gate: `retrieve_batch` + `decide`, no LLM: escalation rate, retrieval hit rates.
C. Generation: `run_ticket` on a stated sample (`--n-gen`, default 50): citation validity, warning/fallback rate, latency.

**High-Priority Incident Recall** (headline): of holdout tickets whose true label is Incident + high, the fraction the system
escalates to a human. We also report the classifier version: the fraction for which triage predicts both labels correctly.

**Duplicates (important for honest numbers).** The three source CSVs overlap (the 20k file is another version of the big file),
so the same ticket can appear on both sides of the split. 21.8% of all holdout bodies, and **27.5% of clean holdout tickets**,
have an exact-body twin in the KB (`data.holdout_seen_ids()`), and twins share their type/queue/priority labels 99.6-99.8% of
the time. Likewise 25.8% of val tickets have a twin in train. So every stage A/B metric is reported three ways by `evaluate.py`:
`all_*` (the brief's headline), `never_seen_*` (holdout tickets with no twin in the KB; the fair estimate for genuinely new
tickets), and Macro-F1 by **detected** language (`detected_en_*`, `detected_de_*`). The dataset's `language` column is wrong on
~11% of tickets, so never group or filter by it. When presenting, lead with `never_seen_*` and show `all_*` next to it.

`scripts/make_split.py` falls back to downloading the repo's CSV files over plain HTTPS if the `datasets` library fails
(or force it with `SENTINEL_FORCE_CSV=1`); the pinned fingerprint check guarantees the same split either way.
Windows machines where pyarrow's DLL is blocked need `fastparquet` (in requirements.txt).

## Checks to run before merging your module / before the demo
- `python scripts/run_adversarial.py`: 20 tricky tickets in `eval/adversarial.json` (empty subject, prompt injection,
  fake refund/SLA/price/integration asks, legal threat, PII, unsafe advice, French, garbage, 400x-long body). Checks:
  no crash, escalation expectation, language, no forbidden claims in the draft, cited ids were retrieved.
  Note: with the stubs everything escalates, so most cases pass trivially; it only becomes meaningful with real modules.
- `python scripts/leakage_audit.py`: disk (no holdout answers), static scan (who touches holdout files/loaders),
  models/ and index/ contain no holdout ticket ids, retrieval returns only KB ids. Exits 1 on any FAIL.

## Baseline floors (`python scripts/baselines.py`, trained on clean KB, scored on the 9,768-ticket clean holdout)
Macro-F1 (type / queue / priority) and High-Priority Incident recall from the classifier alone:

| model | all | never-seen (7,083) |
|---|---|---|
| majority class | 0.142 / 0.045 / 0.192, recall 0.000 | 0.142 / 0.044 / 0.193, recall 0.000 |
| plain TF-IDF + linear SVM | 0.897 / 0.673 / 0.714, recall 0.789 | 0.859 / 0.565 / 0.619, recall 0.721 |
| S2 calibrated model (`triage()`) | 0.892 / 0.672 / 0.703, recall 0.830 | 0.857 / 0.562 / 0.606, recall 0.775 |

Read this honestly: the dataset's labels cap accuracy (queue tops out near 0.67), and a plain SVM matches `triage()` on Macro-F1.
What `triage()` adds is calibrated confidence for the gate, the outage rule, and higher high-priority-incident recall (+4 to +5 points).
TF-IDF keyword retrieval (queue match of the top hit / any of top 5): all 0.802 / 0.909, never-seen **0.729 / 0.875**; type match
of any of top 5 is 0.99 either way. The dense (e5) retriever should beat the never-seen numbers, especially cross-lingually.

## Gate and generation (S4)
**Escalation gate** (`src/gate.py`): deterministic rules, each a pure function. Any rule that fires sends the ticket to
"Escalate to On-Call Engineer". `decide()` returns `{escalate, reasons, codes}`, where `codes` are rule ids that evaluate.py can count.

| Code | Fires when | Notes |
|---|---|---|
| `HIGH_INCIDENT` | predicted `priority == high` and `type == Incident` | the brief's mandatory rule |
| `LOW_CONFIDENCE` | `overall_conf < CONF_THRESHOLD` | threshold in `config.py`, tuned on the `val` slice |
| `NO_PRECEDENT` | no hits, or top similarity `< SIM_THRESHOLD` (0.85) | 0.85 comes from S3's `scripts/tune_threshold.py`: e5 scores sit in ~0.75-0.95, and 0.85 flags no hand-typed in-scope ticket while catching 12/17 off-topic ones |
| `UNSUPPORTED_ASK` | ticket asks for a refund/compensation, a price/discount, or a guarantee/SLA, and no retrieved answer covers that topic | en + de keyword patterns |
| `LEGAL` | legal threat or legal commitment (lawsuit, lawyer, *Anwalt*, *Klage* ...) | always a human |
| `PROMPT_INJECTION` | "ignore previous instructions", "system prompt", *"ignoriere alle vorherigen Anweisungen"* ... | |
| `EMPTY_TICKET` | body under 15 characters | |
| `UNREADABLE` | no recognisable English/German word or IT term (keyboard mash, symbols, digits only) | similarity can't catch garbage: it scores like real short tickets |

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
_S1 fills in after `python evaluate.py` with the real gate and retrieval._

## Data credit and license
Dataset: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face,
licensed CC BY-NC 4.0 (attribution, non-commercial). SentinelDesk is a hackathon project and is not for commercial use. The data
is not redistributed in this repository (`data/` is gitignored).
