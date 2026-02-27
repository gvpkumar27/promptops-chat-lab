import base64
import binascii
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List


SAFE_REFUSAL = "I can't help with that request. I can help with safe PromptOps tasks."
OUT_OF_SCOPE_REFUSAL = (
    "I can only help with approved PromptOps topics. "
    "Please rephrase your request within that scope."
)
HARMFUL_CONTENT_REFUSAL = (
    "I can't help with harmful or extremist content. "
    "If your goal is safety, I can help with defensive prompting and misuse prevention."
)

_POLICY_CACHE: dict | None = None

# Minimal public fallback policy. Keep this intentionally generic.
_PUBLIC_FALLBACK_POLICY = {
    "attack_patterns": [
        r"\bignore\s+previous\s+instructions\b",
        r"\breveal\s+(the\s+)?system\s+prompt\b",
        r"\bdeveloper\s+mode\b",
        r"\bjail\w*break\w*\b",
        r"\bprompt\s+injection\b",
    ],
    "refusal_markers": [
        "i can't",
        "i cannot",
        "cannot assist",
        "won't help",
        "not able to",
        "cannot provide",
        "can't share internal instructions",
    ],
    "allowed_topic_keywords": {
        "Prompt engineering": [
            "prompt engineering",
            "prompting",
            "zero-shot",
            "few-shot",
            "prompt evaluation",
            "guardrails",
            "rag",
            "agents",
        ]
    },
    "allowed_topic_patterns": {
        "Prompt engineering": [r"\bprom\w*\s+engin\w*\b"]
    },
    "blocked_broad_patterns": {
        "General coding help": [r"\bwrite\s+(python|java|javascript|c\+\+|code)\b"],
        "Medical/legal/financial advice": [
            r"\bmedical\s+advice\b",
            r"\blegal\s+advice\b",
            r"\bfinancial\s+advice\b",
        ],
        "Open-domain Q&A": [r"\bweather\b", r"\bnews\b", r"\bwho\s+is\b"],
    },
    "harmful_canonical_terms": [
        "alqaeda",
        "qaeda",
        "isis",
        "daesh",
        "terrorism",
        "terrorist",
        "extremism",
        "extremist",
    ],
    "misuse_intent_patterns": [
        r"\bhow\s+to\b",
        r"\bsteps?\s+to\b",
        r"\bhelp\s+me\b",
        r"\bbypass\b",
    ],
    "misuse_target_patterns": [r"\bhack\w*\b", r"\bphish\w*\b", r"\bmalware\b", r"\bexploit\w*\b"],
    "misuse_target_terms": [
        "hack",
        "phishing",
        "malware",
        "exploit",
        "credential",
        "password",
        "account",
    ],
    "safety_context_hints": [
        "defensive",
        "prevention",
        "mitigation",
        "awareness",
        "education",
        "safety",
    ],
}


LEET_MAP = str.maketrans(
    {
        "@": "a",
        "$": "s",
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "!": "i",
        "|": "i",
    }
)


def _strict_guardrails_enabled() -> bool:
    return os.getenv("PROMPTOPS_STRICT_GUARDRAILS", "true").strip().lower() == "true"


def _load_policy() -> dict:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    policy_path = os.getenv("PROMPTOPS_POLICY_PATH", "").strip()
    if policy_path:
        data = json.loads(Path(policy_path).read_text(encoding="utf-8-sig"))
        _validate_policy(data)
        _POLICY_CACHE = data
        return _POLICY_CACHE

    if _strict_guardrails_enabled():
        raise RuntimeError(
            "Strict guardrails enabled but PROMPTOPS_POLICY_PATH is not set. "
            "Provide a private policy JSON file."
        )

    _POLICY_CACHE = _PUBLIC_FALLBACK_POLICY
    return _POLICY_CACHE


def ensure_guardrails_ready() -> None:
    _load_policy()


def _validate_policy(data: dict) -> None:
    required_keys = [
        "attack_patterns",
        "refusal_markers",
        "allowed_topic_keywords",
        "allowed_topic_patterns",
        "blocked_broad_patterns",
        "harmful_canonical_terms",
        "misuse_intent_patterns",
        "misuse_target_patterns",
        "misuse_target_terms",
        "safety_context_hints",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RuntimeError(f"Invalid policy: missing keys: {', '.join(missing)}")


def _get_policy_or_fail_closed() -> dict | None:
    try:
        return _load_policy()
    except Exception:
        return None


def _normalize_alpha_token(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def _strip_zero_width(text: str) -> str:
    return re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)


def _basic_normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = _strip_zero_width(t)
    t = t.translate(LEET_MAP)
    return t


def _compact_alpha_chunks(text: str) -> str:
    return re.sub(r"\b(?:[a-z]\s+){2,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), text)


def _try_decode_base64(token: str) -> str:
    t = token.strip()
    if len(t) < 12 or len(t) > 512:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", t):
        return ""
    try:
        raw = base64.b64decode(t, validate=True)
    except (binascii.Error, ValueError):
        return ""
    try:
        out = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    if len(out.strip()) < 6:
        return ""
    return out


def _try_decode_hex(token: str) -> str:
    t = token.strip().lower()
    if len(t) < 12 or len(t) > 512 or len(t) % 2 != 0:
        return ""
    if not re.fullmatch(r"[0-9a-f]+", t):
        return ""
    try:
        raw = bytes.fromhex(t)
        out = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    if len(out.strip()) < 6:
        return ""
    return out


def _expand_obfuscated_candidates(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9+/=]{12,}|[0-9a-fA-F]{12,}", text)
    decoded = []
    for token in tokens:
        b64 = _try_decode_base64(token)
        if b64:
            decoded.append(b64)
        hx = _try_decode_hex(token)
        if hx:
            decoded.append(hx)
    if decoded:
        return text + "\n" + "\n".join(decoded)
    return text


def _normalized_text_and_tokens(text: str) -> tuple[str, List[str]]:
    normalized = _basic_normalize_text(text.strip())
    expanded = _expand_obfuscated_candidates(normalized)
    lower = _compact_alpha_chunks(expanded.lower())
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", lower)
    tokens = [_normalize_alpha_token(t) for t in raw_tokens]
    tokens = [t for t in tokens if t]
    return lower, tokens


def _has_misuse_signature(lower: str, tokens: List[str], policy: dict) -> bool:
    has_intent = any(re.search(p, lower) for p in policy["misuse_intent_patterns"])
    has_target = (
        any(re.search(p, lower) for p in policy["misuse_target_patterns"])
        or _token_near_terms(tokens, policy["misuse_target_terms"], max_distance=2)
    )
    has_safety_context = any(hint in lower for hint in policy["safety_context_hints"])
    return has_intent and has_target and not has_safety_context


def _blocked_scope_category(lower: str, policy: dict) -> str | None:
    for category, patterns in policy["blocked_broad_patterns"].items():
        if any(re.search(pattern, lower) for pattern in patterns):
            return category
    return None


def _matched_topics(lower: str, policy: dict) -> List[str]:
    matched: List[str] = []
    for topic, keywords in policy["allowed_topic_keywords"].items():
        if any(keyword in lower for keyword in keywords):
            matched.append(topic)
            continue
        patterns = policy["allowed_topic_patterns"].get(topic, [])
        if any(re.search(pattern, lower) for pattern in patterns):
            matched.append(topic)
    return matched


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def _token_near_terms(
    tokens: List[str], terms: List[str], max_distance: int = 2
) -> bool:
    for token in tokens:
        if len(token) < 4:
            continue
        for term in terms:
            if token == term:
                return True
            if (
                abs(len(token) - len(term)) <= max_distance
                and _levenshtein(token, term) <= max_distance
            ):
                return True
    return False


def detect_harmful_content(text: str) -> bool:
    policy = _get_policy_or_fail_closed()
    if policy is None:
        return True

    lower, tokens = _normalized_text_and_tokens(text)
    harmful_terms = policy["harmful_canonical_terms"]
    extremist_hit = _token_near_terms(tokens, harmful_terms, max_distance=2)
    if extremist_hit:
        return True

    if _has_misuse_signature(lower, tokens, policy):
        return True

    return False


def detect_injection_risk(text: str) -> Dict[str, object]:
    policy = _get_policy_or_fail_closed()
    if policy is None:
        return {
            "is_attack": True,
            "risk_score": 1.0,
            "pattern_hits": ["policy_not_loaded"],
        }

    normalized = _basic_normalize_text(text.strip())
    expanded = _expand_obfuscated_candidates(normalized)
    lower = _compact_alpha_chunks(expanded.lower())
    hits: List[str] = []
    for pattern in policy["attack_patterns"]:
        if re.search(pattern, lower):
            hits.append(pattern)
    score = len(hits) / max(len(policy["attack_patterns"]), 1)
    return {
        "is_attack": len(hits) > 0,
        "risk_score": round(score, 3),
        "pattern_hits": hits,
    }


def is_refusal(text: str) -> bool:
    policy = _get_policy_or_fail_closed()
    if policy is None:
        return True
    lower = text.strip().lower()
    return any(marker in lower for marker in policy["refusal_markers"])


def classify_topic_scope(text: str) -> Dict[str, object]:
    policy = _get_policy_or_fail_closed()
    if policy is None:
        return {
            "is_in_scope": False,
            "blocked_category": "Policy configuration required",
            "matched_topics": [],
        }

    lower = text.strip().lower()
    if detect_harmful_content(text):
        return {
            "is_in_scope": False,
            "blocked_category": "Harmful/extremist content",
            "matched_topics": [],
        }

    blocked_category = _blocked_scope_category(lower, policy)
    if blocked_category is not None:
        return {
            "is_in_scope": False,
            "blocked_category": blocked_category,
            "matched_topics": [],
        }

    matched_topics = _matched_topics(lower, policy)

    return {
        "is_in_scope": len(matched_topics) > 0,
        "blocked_category": None,
        "matched_topics": matched_topics,
    }
