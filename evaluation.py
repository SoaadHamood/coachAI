# evaluation.py
import json
import re
from typing import Any, Dict, List, Optional

from settings import client, COACH_MODEL, GRADER_MODEL
from prompts import COACH_SYSTEM_PROMPT, GRADER_RUBRIC, CHECKLIST_SYSTEM_PROMPT


# -------------------------
# Transcript helpers
# -------------------------
def _tail(s: str, n: int) -> str:
    return (s or "")[-n:]


def _last_nonempty_line(transcript: str) -> str:
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_recent_context(transcript: str, max_lines: int = 14) -> str:
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:])[-2400:]


def _agent_only(transcript: str) -> str:
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    return "\n".join([ln for ln in lines if ln.upper().startswith("AGENT:")])


def _has_customer(transcript: str) -> bool:
    return "CUSTOMER:" in (transcript or "").upper()


# -------------------------
# OpenAI Responses output extraction (works across SDK variants)
# -------------------------
def _response_to_text(r: Any) -> str:
    if r is None:
        return ""

    txt = getattr(r, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = getattr(r, "output", None)
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            content = getattr(item, "content", None) if item is not None else None
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
                        continue
                    if isinstance(part, dict):
                        t2 = part.get("text")
                        if isinstance(t2, str) and t2.strip():
                            chunks.append(t2.strip())
            elif isinstance(content, str) and content.strip():
                chunks.append(content.strip())
        if chunks:
            return "\n".join(chunks).strip()

    if isinstance(r, dict):
        t = r.get("output_text")
        if isinstance(t, str) and t.strip():
            return t.strip()

    return ""


def _responses_text(system_prompt: str, user_prompt: str, model: str, max_output_tokens: int) -> str:
    if client is None:
        return ""

    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # IMPORTANT: do NOT pass response_format=... here (your SDK rejects it)
    try:
        r = client.responses.create(
            model=model,
            input=payload,
            max_output_tokens=max_output_tokens,
        )
        return _response_to_text(r)
    except Exception as e:
        print("[evaluation] responses.create failed:", repr(e))
        return ""


# -------------------------
# JSON extraction (robust)
# -------------------------
def _extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    t = text.strip()

    # direct json
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # try first {...} range
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        cand = t[i : j + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


# -------------------------
# Coach heuristics (reduce nonsense + make tips appear at the right time)
# -------------------------
def _script_state(transcript: str) -> dict:
    a = _agent_only(transcript).lower()

    opening_done = bool(
        re.search(r"\b(my name is|this is)\b", a)
        and re.search(r"\b(team|support|from|company)\b", a)
        and re.search(r"\bhow can i help\b", a)
    )
    identification_done = bool(re.search(r"\b(name|last\s*(4|four)|id|phone|phone number|email)\b", a))
    empathy_done = bool(re.search(r"\b(i understand|i'm sorry|sorry to hear|that sounds|i can imagine|i appreciate)\b", a))
    restate_done = bool(re.search(r"\b(just to confirm|to confirm|to make sure i understand|if i understand|so you('re| are))\b", a))
    expectations_done = bool(re.search(r"\b(next step|what i('ll| will) do|within|today|tomorrow|minutes|hours|by (the end|eod))\b", a))
    close_done = bool(re.search(r"\b(to summarize|just to summarize|summary|recap)\b", a))
    feedback_done = bool(re.search(r"\b(survey|feedback|rate|rating)\b", a))
    near_closing = bool(re.search(r"\b(anything else|goodbye|bye|thank you for calling|have a (good|nice) day)\b", a))

    # clarify: we ONLY count it if it's after the customer spoke at least once
    clarify_done = bool(_has_customer(transcript) and ("?" in a or re.search(r"\b(can you|could you|what|when|where|which|how)\b", a)))

    return {
        "opening_done": opening_done,
        "identification_done": identification_done,
        "empathy_done": empathy_done,
        "clarify_done": clarify_done,
        "restate_done": restate_done,
        "expectations_done": expectations_done,
        "close_done": close_done,
        "feedback_done": feedback_done,
        "near_closing": near_closing,
    }


def _next_missing_step(state: dict, transcript: str) -> Optional[str]:
    if not state["opening_done"]:
        return "opening"
    if _has_customer(transcript) and not state["empathy_done"]:
        return "empathy"
    if _has_customer(transcript) and not state["clarify_done"]:
        return "clarify"
    if _has_customer(transcript) and not state["restate_done"]:
        return "restate"
    if _has_customer(transcript) and not state["expectations_done"]:
        return "expectations"
    if state["near_closing"]:
        if not state["close_done"]:
            return "close"
        if not state["feedback_done"]:
            return "feedback"
    return None


# -------------------------
# Public API
# -------------------------
def coach_tips(transcript: str) -> Dict[str, Any]:
    if client is None:
        return {"should_intervene": False, "tip": "", "reason_tag": "missing_key", "urgency": "low"}

    # Only show tips right after CUSTOMER spoke (best timing)
    last = _last_nonempty_line(transcript).upper()
    if last and not last.startswith("CUSTOMER:"):
        return {"should_intervene": False, "tip": "", "reason_tag": "timing", "urgency": "low"}

    state = _script_state(transcript)
    missing = _next_missing_step(state, transcript)
    if not missing:
        return {"should_intervene": False, "tip": "", "reason_tag": "other", "urgency": "low"}

    focus = _extract_recent_context(transcript)
    user_msg = (
        "Return STRICT JSON only.\n"
        f"Your single job: coach the NEXT missing checklist step: {missing}\n"
        "Tip must match the last 1–2 turns.\n\n"
        f"Transcript (recent):\n{focus}"
    )

    txt = _responses_text(COACH_SYSTEM_PROMPT, user_msg, COACH_MODEL, max_output_tokens=200)
    data = _extract_first_json_object(txt)
    if not data:
        return {"should_intervene": False, "tip": "", "reason_tag": "parse_error", "urgency": "low"}

    tip = str(data.get("tip") or "").strip()
    if not tip:
        return {"should_intervene": False, "tip": "", "reason_tag": "other", "urgency": "low"}

    # clamp length
    words = tip.split()
    if len(words) > 16:
        tip = " ".join(words[:16])

    reason = str(data.get("reason_tag") or missing or "other").strip().lower()
    if reason not in {"opening", "identification", "listening", "empathy", "clarify", "restate", "tone", "expectations", "close", "feedback", "other"}:
        reason = "other"

    urgency = str(data.get("urgency") or "low").strip().lower()
    if urgency not in {"low", "medium", "high"}:
        urgency = "low"

    should = bool(data.get("should_intervene", True))
    if not tip:
        should = False

    return {"should_intervene": should, "tip": tip, "reason_tag": reason, "urgency": urgency}


def grade_exam(transcript: str) -> Dict[str, Any]:
    if client is None:
        return {
            "score": 0,
            "pass": False,
            "summary": "Missing OPENAI_API_KEY",
            "strengths": [],
            "improvements": ["Set OPENAI_API_KEY and restart the server."],
        }

    payload = (_tail(transcript, 4500).strip() or "(empty transcript)")
    txt = _responses_text(GRADER_RUBRIC, payload, GRADER_MODEL, max_output_tokens=360)
    data = _extract_first_json_object(txt)

    if not data:
        # print raw to console for debugging
        print("[evaluation] grader raw (non-json):", (txt or "")[:800])
        return {
            "score": 0,
            "pass": False,
            "summary": "Could not parse grader output.",
            "strengths": [],
            "improvements": ["Try again."],
        }

    score = max(0, min(100, int(data.get("score", 0) or 0)))
    passed = bool(data.get("pass", score >= 70))
    strengths = [str(x).strip() for x in (data.get("strengths") or []) if str(x).strip()][:5]
    improvements = [str(x).strip() for x in (data.get("improvements") or []) if str(x).strip()][:7]

    return {
        "score": score,
        "pass": passed,
        "summary": str(data.get("summary") or "").strip(),
        "strengths": strengths,
        "improvements": improvements,
    }


def evaluate_checklist(transcript: str, customer_type: str = "", emotion_level: Optional[int] = None) -> Dict[str, Any]:
    if client is None:
        return {
            "checklist_score": 0,
            "items": [],
            "highlights": [],
            "improvements": ["Set OPENAI_API_KEY and restart the server."],
            "next_time_say": [],
        }

    payload = (_tail(transcript, 6500).strip() or "(empty transcript)")

    meta = []
    if customer_type:
        meta.append(f"customer_type={customer_type}")
    if emotion_level is not None:
        meta.append(f"emotion_level={emotion_level}")
    meta_txt = ("\nMeta: " + ", ".join(meta)) if meta else ""

    txt = _responses_text(CHECKLIST_SYSTEM_PROMPT, payload + meta_txt, GRADER_MODEL, max_output_tokens=900)
    data = _extract_first_json_object(txt)

    if not data:
        print("[evaluation] checklist raw (non-json):", (txt or "")[:800])
        return {
            "checklist_score": 0,
            "items": [],
            "highlights": [],
            "improvements": ["Could not parse checklist output."],
            "next_time_say": [],
        }

    score = max(0, min(100, int(data.get("checklist_score", 0) or 0)))

    items = data.get("items") or []
    clean_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        _id = str(it.get("id") or "").strip()[:40]
        title = str(it.get("title") or "").strip()[:60]
        status = str(it.get("status") or "missing").strip().lower()
        if status not in {"done", "partial", "missing"}:
            status = "missing"

        evidence = " ".join(str(it.get("evidence") or "").strip().split()[:12])
        note = " ".join(str(it.get("note") or "").strip().split()[:18])

        if _id and title:
            clean_items.append({"id": _id, "title": title, "status": status, "evidence": evidence, "note": note})

    highlights = [str(x).strip() for x in (data.get("highlights") or []) if str(x).strip()][:4]
    improvements = [str(x).strip() for x in (data.get("improvements") or []) if str(x).strip()][:6]
    next_time_say = [str(x).strip() for x in (data.get("next_time_say") or []) if str(x).strip()][:2]

    return {
        "checklist_score": score,
        "items": clean_items,
        "highlights": highlights,
        "improvements": improvements,
        "next_time_say": next_time_say,
    }
