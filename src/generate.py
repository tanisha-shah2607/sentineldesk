"""Owner: S4.

draft_reply(subject, body, lang, hits) -> {"text": str, "cited_ids": [str]}
Must: use only retrieved answers, cite [TKT-xxxxx], reply in `lang`, never invent
policies/prices/guarantees. Falls back to a template if the LLM fails.
"""


def template_fallback(lang: str, hits: list[dict]) -> dict:
    if not hits:
        msg = ("Danke für Ihre Nachricht. Wir leiten Ihr Anliegen an einen Mitarbeiter weiter."
               if lang == "de" else
               "Thank you for contacting us. We are forwarding your request to a human agent.")
        return {"text": msg, "cited_ids": []}
    top = hits[0]
    intro = "Basierend auf einem ähnlichen Fall" if lang == "de" else "Based on a similar past case"
    return {"text": f"{intro} [{top['ticket_id']}]: {top['answer']}", "cited_ids": [top["ticket_id"]]}


def draft_reply(subject, body, lang: str, hits: list[dict]) -> dict:
    # STUB: S4 adds prompt + llm_complete + citation/fabrication validation
    return template_fallback(lang, hits)
