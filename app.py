"""Owner: S5. Run: streamlit run app.py"""
import streamlit as st

from src.pipeline import run_ticket

st.set_page_config(page_title="SentinelDesk", layout="wide")
st.title("SentinelDesk: Triage Console")

subject = st.text_input("Subject (optional)")
body = st.text_area("Ticket body", height=160)

if st.button("Triage", type="primary") and body.strip():
    r = run_ticket(subject, body)
    tri, gate, lat, draft = r["triage"], r["gate"], r["latency"], r["draft"]

    if gate["escalate"]:
        st.error("ESCALATE TO ON-CALL ENGINEER: " + "; ".join(gate["reasons"]))
    else:
        st.success("AUTO-DRAFT READY")

    c = st.columns(5)
    c[0].metric("Type", tri["type"])
    c[1].metric("Queue", tri["queue"])
    c[2].metric("Priority", tri["priority"])
    c[3].metric("Confidence", f"{tri['overall_conf']:.2f}")
    c[4].metric("Latency", f"{lat['total_ms']:.0f} ms")
    st.caption(f"Language: {tri['language']} | " + ", ".join(f"{k}={v:.0f}" for k, v in lat.items()))

    st.subheader("Retrieved precedents")
    st.dataframe([{"ticket_id": h["ticket_id"], "similarity": round(h["score"], 3),
                   "answer": h["snippet"]} for h in r["hits"]], use_container_width=True)

    st.subheader("Draft reply (for engineer review)" if draft["for_review"] else "Draft reply")
    st.write(draft["text"])
    st.caption("Cited: " + ", ".join(draft["cited_ids"]) + (" | template fallback" if draft["used_fallback"] else ""))
    for w in draft["warnings"]:
        st.warning(w)
