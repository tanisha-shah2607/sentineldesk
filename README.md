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
python scripts/build_index.py   # embeds the KB once into index/ (~2 min on Apple GPU, longer on CPU)
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
- `retrieve` -> `[{ticket_id, score, subject, answer, snippet, queue, type, language}]`, sorted by score; `[]` for empty text.
  Optional `part="all"|"train"` and `xl_slots` (see Retrieval below).
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

## Retrieval (S3)
Finds the 5 most similar resolved tickets in the historical KB, for the gate (is there a real precedent?) and the reply
generator (what resolved it before?). Code: `src/retrieval.py`. Build: `python scripts/build_index.py`.

**Model: `intfloat/multilingual-e5-base`.** A sentence-embedding model (XLM-RoBERTa base, ~278M parameters) that maps
English and German text into one 768-dimension vector space, so similar tickets are close regardless of language. It reads
up to 512 tokens and runs locally (no API key, no per-call cost, no ticket text leaves the app). Picked over
`paraphrase-multilingual-MiniLM-L12-v2` with `scripts/bench_embed.py` (200 val queries vs 5,000 train tickets):

| | MiniLM-L12 | e5-base |
|---|---|---|
| Vector size / max input | 384 / 128 tokens | 768 / 512 tokens |
| Top-1 queue match | 43.0% | **46.5%** |
| Embed full KB, 38,810 tickets (Apple GPU) | ~0.8 min (estimated) | 2.2 min (measured) |
| One query | 8 ms | 9 ms |

Both are fast enough; e5 won on accuracy and on not truncating long tickets (MiniLM cuts at 128 tokens).
e5 needs the `query: ` / `passage: ` prefixes, which `embed_texts()` adds.

**What is indexed (leakage-safe).** Only `load_kb_clean(need_answer=True)`: 38,810 KB tickets with an answer, embedded as
`build_text(subject, body)`. Nothing from the holdout is read (`leakage_audit.py` check 4 confirms every returned id is a KB id).
- Duplicates: 5,384 tickets repeat another ticket's body word for word. Each body is kept once (lowest `ticket_id`), so the
  5 hits are 5 different precedents. Final index: **33,426** tickets.
- Two indexes from one embedding cache: `part="all"` (33,426) for the app and evaluation, and `part="train"` (30,607) for
  tuning on the validation slice, so a val ticket can never retrieve itself.
- Cache key = model name + hash of `src/text.py`. If `build_text()` changes, the cache is detected as stale and rebuilt
  instead of silently mismatching query-time text.

**Search: exact cosine similarity in NumPy, not FAISS.** Vectors are L2-normalised, so a dot product is the cosine;
this gives exactly the results of `faiss.IndexFlatIP`. FAISS was dropped because on macOS its wheel and PyTorch's wheel each
bundle `libomp`, and the process aborts (`OMP: Error #15`) once both are loaded. At ~33k vectors exact search costs a few ms.

**Cross-lingual precedents (bonus).** e5 gives same-language tickets a small score bonus (~0.02-0.03), and because its scores
sit in a narrow band, that bonus decides the ranking: the same ticket written in English and German shared on average 0.1
of its 5 hits, even though the two query vectors are very close (cosine 0.93). Yet for German tickets the best *English*
precedents are as useful as German ones (top-1 queue match 57.5% English-only vs 53.2% German-only). So `retrieve()` keeps
the 3 best hits overall and gives the last 2 slots (`CROSS_LINGUAL_SLOTS`, default 2) to the best tickets in the language
missing from those 3. Hits stay sorted by their real score, so the top hit, and with it the gate's `SIM_THRESHOLD`, never
changes. The reply is still written in the ticket's language. Set `CROSS_LINGUAL_SLOTS=0` to turn it off.
Example: a German hospital VPN outage now also retrieves TKT-26021 "Severe Service Disruption Detected in EMR Platform".

**Quality** (validation tickets without a word-for-word copy in train, n=2,869, searched against the train index;
`scripts/tune_threshold.py`, `scripts/cross_lingual.py`):

| | Top-1 queue | Queue in top 5 | Type in top 5 | Has a hit in the other language |
|---|---|---|---|---|
| All, without cross-lingual mix | 68.4% | 85.6% | 97.3% | 36% |
| **All, with mix (default)** | **68.4%** | **88.3%** | **98.2%** | **100%** |
| English (n=1,712), with mix | 74.6% | 89.6% | 98.5% | 100% |
| German (n=1,157), with mix | 59.1% | 86.3% | 97.8% | 100% |

Including the 995 val tickets that do have a copy in train raises top-1 queue match to 76.1%; that gap is the copies, not
better retrieval, so we quote the lower number. Higher scores mean better precedents: top-1 queue match rises from 43% in
the lowest tenth of scores to 98% in the highest.

**`SIM_THRESHOLD = 0.85`** (no real precedent → escalate). Tuned with `scripts/tune_threshold.py`:
- Dataset tickets are long and uniform in style and score high (99% above 0.88). Short tickets typed live score lower
  (0.856-0.901 for 19 hand-typed routine tickets) and overlap with off-topic text (0.785-0.874 for 17 off-topic tickets).
  A threshold tuned on dataset tickets alone (~0.89) would escalate 15 of the 19 routine tickets.
- Rule: the highest threshold that escalates none of the hand-typed routine tickets (0.855), minus a 0.005 margin.

| Threshold | Val tickets escalated | Hand-typed routine escalated | Off-topic caught |
|---|---|---|---|
| **0.85** | 0% | 0 / 19 | 12 / 17 |
| 0.875 | 0.6% | 6 / 19 | 17 / 17 |
| 0.89 | 3.4% | 15 / 19 | 17 / 17 |

Not caught at 0.85: "???" and keyboard mash (they score like real short tickets, so similarity cannot catch them; the
gate needs its own too-few-words rule, S4), and two off-topic questions (recipe, gardening) at 0.852, which the generation
prompt must refuse because no precedent addresses them.

**Speed** (100 holdout queries, `retrieve()` with the cross-lingual mix, including query embedding):

| | One query p50 / p95 | Cold start (model + index + first query) |
|---|---|---|
| Apple GPU (MPS) | 16 / 18 ms | 3.4 s |
| CPU (like Cloud Run) | 31 / 40 ms | 3.8 s |

`retrieve_batch` on 10,000 holdout tickets: 41 s on Apple GPU. The device is picked automatically (CUDA, then MPS,
then CPU; override with `EMBED_DEVICE`). Call `warmup()` at app start so the first ticket isn't slow.

**Deploying (Cloud Run):** build `index/` offline and ship it with the image or a bucket (embedding at container start
would be slow on CPU); download the e5 weights (~1.1 GB) during the Docker build; give the service at least 4 GiB of memory.

**Limitations**
- German is weaker than English (top-1 queue 59.1% vs 74.6%).
- Short live tickets score lower than dataset tickets, so the score scale differs between live input and the dataset.
- Some tickets with generic answers sit close to many vague queries. TKT-07586 (answer: "could you please provide more
  details…") often fills a cross-lingual slot for vague German tickets; it is not a real resolution.
- Similarity cannot tell garbage input from short real tickets; that needs a separate gate rule.
- Validation and holdout contain word-for-word copies of KB tickets (about a quarter), which inflates scores that include them.

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

## Results
_S1 fills in after `python evaluate.py`._
