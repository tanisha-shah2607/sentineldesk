# SentinelDesk demo script (~4 minutes)

**Before going on stage**
- `git pull`, then `./run.sh --rebuild` (Windows: `run.bat --rebuild`). It retrains/re-indexes, prints `READY`, and opens the console.
  The sidebar must show **Triage model: trained** and **Retrieval index: ready** before you start.
- `./run.sh` (or `run.bat`). Open http://localhost:8501 and press **Reset queue**.
- Check the sidebar **System status**: green = real stage, yellow = demo stand-in. Say which is which if asked; never present mock output as model results.
- If a real stage breaks on the day, rehearse with `SENTINELDESK_FORCE_MOCKS=all ./run.sh`.
- Have `results/results.md` (from `python evaluate.py`) open in another tab.

## 1. Problem (30 s)
"Support engineers get thousands of tickets in English and German. Most are routine, but a few high-priority incidents get buried. SentinelDesk triages every ticket, decides whether a human must take over, and drafts replies grounded only in past resolved tickets."

## 2. Routine ticket → auto-draft (45 s)
- Sidebar → **Wrong invoice (en)** → **Triage ticket**.
- Point at: type / queue / priority with confidence, the green **AUTO-DRAFT READY** banner.
- Point at **Retrieved precedents** and the `[TKT-…]` citation in the draft: "Every sentence comes from a real resolved ticket. Nothing is invented."

## 3. German ticket + cross-lingual precedents (45 s)
- **Rechnung falsch (de)**: language detected as DE, reply drafted in German.
- **VPN ausgefallen (de, cross-lingual)**: escalates with *Outage wording detected: “ausgefallen”*.
  Once S3's cross-lingual retrieval is merged, the precedents include English tickets (S3's demo pair: TKT-26021).
  "One multilingual model: a German ticket can be answered from an English precedent."

## 4. Escalation (45 s)
- **Production outage (en, high)**: red **ESCALATE** banner listing every rule that fired (e.g. high-priority Incident, no close precedent).
- "Escalation rules are deterministic and auditable. The draft is still prepared, labelled *for engineer review*."
- **No subject + PII (en)**: missing subject handled. "Emails, phone numbers and IPs are masked before anything reaches the LLM."

## 5. Major incident (45 s)
- **Reset queue**, then **Simulate outage**: five near-identical login-error tickets arrive.
- The 🚨 **MAJOR INCIDENT SUSPECTED** banner appears after the third, and the German report joins the same cluster. Open **Live queue** to see them grouped.
- "Instead of five engineers chasing five tickets, on-call is paged once."

## 6. Numbers (30 s)
- Show `results/results.md`. Lead with **High-Priority Incident Recall (escalated)**, then Macro-F1 per target, escalation rate and latency p50/p95.
- State the definitions exactly as in `evaluate.py` and that the holdout was never seen in training or the index.

## 7. Close (15 s)
"Triage, safe escalation, grounded drafts and outage detection, in one console, in English and German."

## Likely questions
- *Why not let the LLM decide escalation?* Rules are predictable and testable; the LLM only writes text.
- *What if the LLM is down?* A template reply that still cites the best precedent (`used_fallback`).
- *Leakage?* Holdout answers are never written to disk; the index is KB-only; thresholds are tuned on a separate validation slice.
