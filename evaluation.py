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
    """
    NOTE: We intentionally do NOT pass response_format here because some SDK builds reject it.
    Instead, we enforce robust JSON extraction + parsing.
    """
    if client is None:
        return ""

    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

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
# JSON extraction (more robust for production)
# -------------------------
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_code_fences(t: str) -> str:
    """
    If the model wraps JSON in ```json ... ``` fences, extract the inner content.
    If multiple fenced blocks exist, we keep the first.
    """
    if not t:
        return t
    m = _CODE_FENCE_RE.search(t)
    if m:
        return (m.group(1) or "").strip()
    # also handle unclosed fences
    t = re.sub(r"^\s*```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t.strip())
    return t.strip()


def _extract_balanced_json_substring(t: str) -> Optional[str]:
    """
    Find the first balanced {...} object in a string.
    Works even if the model adds text before/after JSON.
    """
    if not t:
        return None

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start : i + 1]
    return None


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output:
    - strips ```json fences
    - tries direct json.loads
    - tries first balanced {...} substring
    """
    if not text:
        return None

    t = _strip_code_fences(text.strip())

    # Try direct JSON
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try first balanced JSON object inside noisy text
    cand = _extract_balanced_json_substring(t)
    if cand:
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

    # clarify: ONLY after the customer spoke at least once
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
# evaluation.py
import re
import time
from typing import Dict, Any

_LAST_TIP_AT = 0.0

_FILLERS = ["um", "uh", "erm", "like", "you know", "actually", "basically"]
_UNCONFIDENT = ["maybe", "i think", "probably", "not sure", "i guess", "kind of", "sort of"]


def _count_fillers(text: str) -> int:
    t = (text or "").lower()
    count = 0
    for f in _FILLERS:
        count += t.count(f)
    # stutter like "I I", "we we"
    count += len(re.findall(r"\b(\w+)\s+\1\b", t))
    return count


def _has_unconfident(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _UNCONFIDENT)


def coach_tips(transcript: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Returns:
    {
      should_intervene: bool,
      tip: str,
      reason_tag: str,
      urgency: low|medium|high
    }
    """
    global _LAST_TIP_AT
    meta = meta or {}

    # ------------------
    # Cooldown (anti-spam)
    # ------------------
    now = time.time()
    COOLDOWN_SEC = 12
    if now - _LAST_TIP_AT < COOLDOWN_SEC:
        return {
            "should_intervene": False,
            "tip": "",
            "reason_tag": "cooldown",
            "urgency": "low",
        }

    silence_ms = int(meta.get("silence_ms") or 0)
    agent_last = (meta.get("agent_last_utterance") or "").strip()
    recent_text = agent_last or transcript[-300:]

    fillers = _count_fillers(recent_text)
    unconfident = _has_unconfident(recent_text)

    trigger = None
    urgency = "low"

    # ------------------
    # Trigger detection
    # ------------------
    if silence_ms >= 3500:
        trigger = "long_silence"
        urgency = "high"
    elif silence_ms >= 2500:
        trigger = "silence"
        urgency = "medium"
    elif fillers >= 3:
        trigger = "fluency"
        urgency = "medium"
    elif unconfident:
        trigger = "confidence"
        urgency = "medium"

    if not trigger:
        return {
            "should_intervene": False,
            "tip": "",
            "reason_tag": "none",
            "urgency": "low",
        }

    # ------------------
    # Ask LLM for ONE short tip
    # ------------------
    try:
        user_msg = f"""
Return STRICT JSON only.

Trigger: {trigger}
Write ONE short coaching tip (max 16 words).

Rules:
- Silence → suggest 1 clear sentence + 1 direct question
- Fluency → slow down, remove fillers
- Confidence → stronger phrasing

Recent context:
{transcript[-400:]}

Agent last utterance:
{agent_last}
"""

        raw = _responses_text(
            COACH_SYSTEM_PROMPT,
            user_msg,
            COACH_MODEL,
            max_output_tokens=120,
        )

        data = _extract_first_json_object(raw)
        tip = (data or {}).get("tip", "").strip()

        if not tip:
            return {
                "should_intervene": False,
                "tip": "",
                "reason_tag": "empty",
                "urgency": "low",
            }

        # clamp length
        tip = " ".join(tip.split()[:16])

        _LAST_TIP_AT = now
        return {
            "should_intervene": True,
            "tip": tip,
            "reason_tag": trigger,
            "urgency": urgency,
        }

    except Exception:
        return {
            "should_intervene": False,
            "tip": "",
            "reason_tag": "error",
            "urgency": "low",
        }


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
        # This is the exact case you see on deploy:
        # model returns fenced JSON or adds preface text -> parser fails.
        print("[evaluation] checklist raw (non-json):", (txt or "")[:1200])
        return {
            "checklist_score": 0,
            "items": [],
            "highlights": [],
            "improvements": ["Could not parse checklist output2222."],
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
