"""Owner: S4. Deterministic, unit-testable escalation rules.

decide(triage_out, hits, text) -> {"escalate": bool, "reasons": [str]}
"""
from .config import CONF_THRESHOLD, SIM_THRESHOLD


def decide(triage_out: dict, hits: list[dict], text: str) -> dict:
    reasons = []
    if triage_out["priority"] == "high" and triage_out["type"] == "Incident":
        reasons.append("High-priority Incident: immediate human escalation")
    if triage_out["overall_conf"] < CONF_THRESHOLD:
        reasons.append(f"Low classifier confidence ({triage_out['overall_conf']:.2f} < {CONF_THRESHOLD})")
    top = hits[0]["score"] if hits else 0.0
    if top < SIM_THRESHOLD:
        reasons.append(f"No close historical precedent (top similarity {top:.2f} < {SIM_THRESHOLD})")
    # TODO(S4): unsupported-action detection (refund guarantees, prices, legal commitments)
    return {"escalate": bool(reasons), "reasons": reasons}
