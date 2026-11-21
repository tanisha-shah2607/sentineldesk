"""Owner: Niti. SentinelDesk triage console.  Run: streamlit run app.py   (or ./run.sh / run.bat)"""
import os

import streamlit as st

from src import config as C
from src.burst import BurstDetector
from src.examples import EXAMPLES, OUTAGE_BURST
from src.pipeline import STAGES, run_ticket_verbose
from src.text import build_text, unescape

st.set_page_config(page_title="SentinelDesk", page_icon="🛡️", layout="wide")

# Streamlit >= 1.50 replaced use_container_width with width="stretch"; requirements allow >= 1.35.
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL_WIDTH = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}



@st.cache_resource(show_spinner="Loading models and retrieval index…")
def warm_up() -> dict:
    """Once per server process: load S2's model and S3's index/encoder so the first ticket is fast."""
    info = {"triage_model": None, "index_ready": False, "warmup_error": None}
    try:
        from src.triage import model_info
        info["triage_model"] = model_info()
    except Exception as e:  # noqa: BLE001
        info["warmup_error"] = f"triage: {type(e).__name__}"
    try:
        from src.retrieval import index_ready, warmup
        info["index_ready"] = index_ready()
        if info["index_ready"]:
            warmup()
    except Exception as e:  # noqa: BLE001
        info["warmup_error"] = f"retrieval: {type(e).__name__}: {e}"
    return info


@st.cache_data(max_entries=2000, show_spinner=False)
def answer_language(text: str) -> str:
    """The dataset's language column is wrong on ~11% of tickets (S2), so show the detected language."""
    try:
        from src.triage import detect_language
        return detect_language(unescape(text))
    except Exception:  # noqa: BLE001
        return "?"


WARM = warm_up()

# ---------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("detector", BurstDetector.auto())
ss.setdefault("history", [])          # newest first: dicts with result, status, burst, subject, body
ss.setdefault("subject", "")
ss.setdefault("body", "")
ss.setdefault("stage_status", {})


def process(subject, body) -> dict:
    r, status = run_ticket_verbose(subject, body)
    burst = ss.detector.add(build_text(subject, body))
    ss.stage_status = status
    item = {"id": burst["ticket_id"], "subject": subject, "body": body, "result": r,
            "status": status, "burst": burst}
    ss.history.insert(0, item)
    return item


def load_example():
    name = ss.example_pick
    ex = next((e for e in EXAMPLES if e["name"] == name), None)
    if ex:
        ss.subject, ss.body = ex["subject"] or "", ex["body"]


def simulate_burst():
    for e in OUTAGE_BURST:
        process(e["subject"], e["body"])


def reset():
    ss.detector.reset()
    ss.history = []


def effective_gate(item) -> dict:
    """Gate decision plus the major-incident rule (console-level, since run_ticket is stateless)."""
    g, b = item["result"]["gate"], item["burst"]
    if b["burst"]:
        return {"escalate": True,
                "reasons": g["reasons"] + [f"Possible major incident: {b['count']} similar tickets in the "
                                           f"last {ss.detector.window}"]}
    return g


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🛡️ SentinelDesk")
    st.caption("Multilingual incident triage · smart escalation · grounded replies")

    st.subheader("Try an example")
    st.selectbox("Example ticket", ["—"] + [e["name"] for e in EXAMPLES], key="example_pick",
                 on_change=load_example, label_visibility="collapsed")
    c1, c2 = st.columns(2)
    c1.button("Simulate outage", on_click=simulate_burst, **FULL_WIDTH,
              help=f"Sends {len(OUTAGE_BURST)} near-duplicate login-outage tickets")
    c2.button("Reset queue", on_click=reset, **FULL_WIDTH)

    st.subheader("System status")
    if ss.stage_status:
        for s in STAGES:
            v = ss.stage_status.get(s, "?")
            st.markdown(f"{'🟢' if v == 'live' else '🟡'} **{s}**: {v}")
    else:
        st.caption("Run a ticket to see which stages are live.")
    provider = os.getenv("LLM_PROVIDER") or "none (template replies)"
    st.markdown(f"**LLM:** {provider}")
    tm = WARM["triage_model"]
    st.markdown("**Triage model:** " + ("trained" if tm else "not trained, run `python scripts/train_triage.py`"))
    st.markdown("**Retrieval index:** " + ("ready" if WARM["index_ready"] else "not built, run `python scripts/build_index.py`"))
    st.markdown(f"**Outage detector:** {ss.detector.backend}")
    if WARM["warmup_error"]:
        st.caption(f"Warm-up issue: {WARM['warmup_error']}")

    with st.expander("Thresholds (src/config.py)"):
        st.markdown(f"- Confidence < **{C.CONF_THRESHOLD}** → escalate\n"
                    f"- Top similarity < **{C.SIM_THRESHOLD}** → escalate\n"
                    f"- Precedents retrieved: **{C.TOP_K}**\n"
                    f"- Burst: ≥ **{ss.detector.min_cluster}** similar in last **{ss.detector.window}** "
                    f"(sim ≥ {ss.detector.sim_threshold})")

# ---------------------------------------------------------------- major-incident banner
active = next((h for h in ss.history if h["burst"]["burst"]), None)
if active:
    st.error(f"🚨 **MAJOR INCIDENT SUSPECTED**: {active['burst']['count']} similar tickets "
             f"(latest {active['id']}): “{active['burst']['label']}”. Page on-call once; link duplicates.")

tab_console, tab_queue, tab_about = st.tabs(["Triage console", f"Live queue ({len(ss.history)})", "How it works"])

# ---------------------------------------------------------------- console
with tab_console:
    st.title("Triage console")
    with st.form("ticket"):
        st.text_input("Subject (optional)", key="subject")
        st.text_area("Ticket body", key="body", height=150,
                     placeholder="Paste a customer ticket (English or German)…")
        submitted = st.form_submit_button("Triage ticket", type="primary")

    if submitted:
        if not ss.body.strip():
            st.warning("Please enter a ticket body.")
        else:
            with st.spinner("Triaging…"):
                process(ss.subject, ss.body)

    if ss.history:
        item = ss.history[0]
        r, status = item["result"], item["status"]
        tri, lat, draft, hits = r["triage"], r["latency"], r["draft"], r["hits"]
        gate = effective_gate(item)

        if gate["escalate"]:
            st.error("### ⛔ ESCALATE TO ON-CALL ENGINEER")
            for reason in gate["reasons"]:
                st.markdown(f"- {reason}")
            if tri.get("outage_terms"):
                st.markdown("- Outage wording detected: " + ", ".join(f"“{t}”" for t in tri["outage_terms"]))
        else:
            st.success("### ✅ AUTO-DRAFT READY: agent can review and send")

        mocked = [s for s, v in status.items() if v != "live"]
        if mocked:
            st.info("Demo fallback in use for: " + ", ".join(f"**{s}** ({status[s]})" for s in mocked))

        m = st.columns(6)
        m[0].metric("Language", tri["language"].upper())
        m[1].metric("Type", tri["type"])
        m[2].metric("Queue", tri["queue"])
        m[3].metric("Priority", tri["priority"])
        m[4].metric("Confidence", f"{tri['overall_conf']:.2f}")
        m[5].metric("Latency", f"{lat['total_ms']:.0f} ms")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Draft reply (for engineer review)" if draft["for_review"] else "Draft reply")
            st.markdown(unescape(draft["text"]))
            cap = "Cited: " + (", ".join(draft["cited_ids"]) or "none")
            if draft["used_fallback"]:
                cap += " · template fallback"
            st.caption(cap)
            for w in draft["warnings"]:
                st.warning(w)
        with right:
            st.subheader("Confidence")
            for k in ("type", "queue", "priority"):
                v = float(tri["conf"][k])
                st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v:.2f}")
            st.caption(f"Overall = min(type, queue); escalate below {C.CONF_THRESHOLD}. "
                       "Priority is excluded: high-priority Incidents always escalate.")
            st.subheader("Latency")
            st.caption(" · ".join(f"{k.replace('_ms', '')} {v:.0f} ms" for k, v in lat.items()))

        st.subheader("Retrieved precedents")
        if hits:
            st.dataframe([{"ticket": h["ticket_id"], "similarity": round(float(h["score"]), 3),
                           "lang": answer_language(h["answer"]), "queue": h["queue"], "type": h["type"],
                           "subject": h["subject"], "answer": unescape(h["snippet"])} for h in hits],
                         hide_index=True, **FULL_WIDTH)
        else:
            st.caption("No precedents found.")
    else:
        st.caption("Pick an example in the sidebar or paste a ticket, then press **Triage ticket**.")

# ---------------------------------------------------------------- queue
with tab_queue:
    st.title("Live queue")
    if not ss.history:
        st.caption("No tickets yet. Try **Simulate outage** in the sidebar.")
    else:
        n_esc = sum(effective_gate(h)["escalate"] for h in ss.history)
        q = st.columns(3)
        q[0].metric("Tickets", len(ss.history))
        q[1].metric("Escalated", n_esc)
        q[2].metric("Auto-drafted", len(ss.history) - n_esc)
        rows = []
        for h in ss.history:
            t, g = h["result"]["triage"], effective_gate(h)
            rows.append({"id": h["id"], "decision": "ESCALATE" if g["escalate"] else "auto-draft",
                         "burst": "🚨" if h["burst"]["burst"] else "",
                         "lang": t["language"], "type": t["type"], "priority": t["priority"],
                         "queue": t["queue"], "conf": round(t["overall_conf"], 2),
                         "subject": h["subject"] or "[no subject]", "ms": round(h["result"]["latency"]["total_ms"])})
        st.dataframe(rows, hide_index=True, **FULL_WIDTH)

# ---------------------------------------------------------------- about
with tab_about:
    st.title("How it works")
    st.markdown(f"""
1. **Triage**: language detection plus calibrated classifiers for type, queue and priority (en/de).
2. **Retrieve**: the top {C.TOP_K} most similar *resolved historical tickets* (knowledge base only, never the holdout).
3. **Gate**: deterministic escalation rules: high-priority Incident, confidence < {C.CONF_THRESHOLD},
   or no precedent with similarity ≥ {C.SIM_THRESHOLD}.
4. **Draft**: the reply is grounded only in retrieved answers and cites `[TKT-…]` ids. PII is masked before any LLM call.
   A failed grounding check escalates the ticket. Escalated tickets still get a draft, marked *for engineer review*.
5. **Major-incident detection**: near-duplicate tickets in a rolling window trigger one outage banner.

Stages that aren't ready yet fall back to clearly labelled demo stand-ins (see *System status*).
""")
