import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promptcraft_lab.guardrails import (
    HARMFUL_CONTENT_REFUSAL,
    OUT_OF_SCOPE_REFUSAL,
    SAFE_REFUSAL,
    classify_topic_scope,
    detect_injection_risk,
    ensure_guardrails_ready,
    is_refusal,
)
from promptcraft_lab.ollama_client import OllamaClient
from promptcraft_lab.prompts import load_prompt_version, resolve_system_prompt
from promptcraft_lab.telemetry import init_metrics_server, record_event


DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "prompts/versions/v1_baseline.json")
ADMIN_MODE = os.getenv("PROMPTOPS_ADMIN_MODE", "false").strip().lower() == "true"
LOG_CHAT_TITLES = os.getenv("PROMPTOPS_LOG_CHAT_TITLES", "false").strip().lower() == "true"
METRICS_LOG_PATH = ROOT / "eval" / "results" / "internal_metrics.jsonl"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_stats():
    return {
        "total_turns": 0,
        "attack_attempts": 0,
        "blocked_precheck": 0,
        "out_of_scope_blocked": 0,
        "refused_postcheck": 0,
        "unsafe_passes": 0,
        "avg_latency_ms": 0.0,
    }


def new_chat_record(index: int):
    ts = int(time.time() * 1000)
    return {
        "id": f"chat-{ts}-{index}",
        "title": f"New Chat {index}",
        "messages": [],
        "stats": empty_stats(),
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
    }


def init_state():
    if "chats" not in st.session_state:
        # Migrate old single-chat state if present.
        old_messages = st.session_state.get("chat", [])
        old_stats = st.session_state.get("stats", empty_stats())
        chat0 = new_chat_record(1)
        chat0["messages"] = old_messages
        chat0["stats"] = old_stats
        if old_messages:
            first_user = old_messages[0].get("user", "").strip()
            if first_user:
                chat0["title"] = " ".join(first_user.split()[:5])
        st.session_state.chats = [chat0]
        st.session_state.active_chat_id = chat0["id"]

    if "active_chat_id" not in st.session_state and st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]


def get_active_chat():
    active_id = st.session_state.active_chat_id
    for chat in st.session_state.chats:
        if chat["id"] == active_id:
            return chat
    # Fallback to first chat
    if st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        return st.session_state.chats[0]
    chat = new_chat_record(1)
    st.session_state.chats = [chat]
    st.session_state.active_chat_id = chat["id"]
    return chat


def create_new_chat():
    index = len(st.session_state.chats) + 1
    chat = new_chat_record(index)
    st.session_state.chats.insert(0, chat)
    st.session_state.active_chat_id = chat["id"]


def update_latency(stats, latency_ms):
    n = stats["total_turns"]
    prev = stats["avg_latency_ms"]
    stats["avg_latency_ms"] = round(((prev * (n - 1)) + latency_ms) / max(n, 1), 2)


def show_stats(stats):
    attempts = stats["attack_attempts"]
    defended = stats["blocked_precheck"] + stats["refused_postcheck"]
    defense_rate = (defended / attempts) if attempts else 1.0

    st.sidebar.metric("Turns", stats["total_turns"])
    st.sidebar.metric("Attack Attempts", attempts)
    st.sidebar.metric("Blocked (Pre)", stats["blocked_precheck"])
    st.sidebar.metric("Blocked (Out-of-Scope)", stats["out_of_scope_blocked"])
    st.sidebar.metric("Refused (Post)", stats["refused_postcheck"])
    st.sidebar.metric("Unsafe Passes", stats["unsafe_passes"])
    st.sidebar.metric("Defense Success", f"{defense_rate:.2%}")
    st.sidebar.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")


def log_internal_metric(payload):
    event = {"ts_utc": now_utc_iso(), **payload}
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(event, ensure_ascii=True) + "\n")


def build_messages(system_prompt, few_shot, chat_pairs, user_text):
    messages = [{"role": "system", "content": system_prompt}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    for row in chat_pairs[-8:]:
        messages.append({"role": "user", "content": row["user"]})
        messages.append({"role": "assistant", "content": row["assistant"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def chat_matches_query(chat, query: str) -> bool:
    if not query:
        return True
    q = query.lower().strip()
    if q in chat["title"].lower():
        return True
    for row in chat.get("messages", []):
        if q in row.get("user", "").lower() or q in row.get("assistant", "").lower():
            return True
    return False


def sidebar_chat_navigation(active_chat):
    if st.sidebar.button("New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    search_text = st.sidebar.text_input(
        "Search Chats",
        value="",
        placeholder="Search by title or content",
    )
    st.sidebar.subheader("Your Chats")

    chats_sorted = sorted(
        st.session_state.chats,
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )
    shown = 0
    for chat in chats_sorted:
        if not chat_matches_query(chat, search_text):
            continue
        shown += 1
        label = chat["title"]
        if len(label) > 40:
            label = label[:37] + "..."
        if st.sidebar.button(label, key=f"chat_btn_{chat['id']}", use_container_width=True):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    if shown == 0:
        st.sidebar.caption("No chats found.")

    if st.sidebar.button("Clear Current Chat", use_container_width=True):
        active_chat["messages"] = []
        active_chat["stats"] = empty_stats()
        active_chat["updated_at"] = now_utc_iso()
        st.rerun()


def render_sticky_header():
    st.markdown(
        """
        <style>
        .promptops-header {
            position: sticky;
            top: 0;
            z-index: 999;
            background: rgba(255, 255, 255, 0.98);
            border-bottom: 1px solid #e5e7eb;
            padding: 0.75rem 0 0.5rem 0;
            margin-bottom: 0.5rem;
        }
        .promptops-header h1 {
            margin: 0;
            font-size: 1.7rem;
        }
        .promptops-header p {
            margin: 0.2rem 0 0 0;
            color: #4b5563;
        }
        </style>
        <div class="promptops-header">
            <h1>PromptOps Secure Chat Lab</h1>
            <p>Controlled chatbot for prompt engineering and defense testing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history(active_chat):
    for row in active_chat["messages"]:
        with st.chat_message("user"):
            st.write(row["user"])
        with st.chat_message("assistant"):
            st.write(row["assistant"])


def apply_precheck_controls(risk, scope, stats):
    if risk["is_attack"]:
        stats["blocked_precheck"] += 1
        return SAFE_REFUSAL, 0.0, "blocked_attack_precheck"

    if not scope["is_in_scope"]:
        stats["out_of_scope_blocked"] += 1
        if scope.get("blocked_category") == "Harmful/extremist content":
            return HARMFUL_CONTENT_REFUSAL, 0.0, "blocked_out_of_scope"
        return OUT_OF_SCOPE_REFUSAL, 0.0, "blocked_out_of_scope"

    return None, None, None


def query_model_response(client, system_prompt, few_shot, history, user_text):
    messages = build_messages(system_prompt, few_shot, history, user_text)
    start = time.perf_counter()
    try:
        reply = client.chat(messages, temperature=0.1)
    except Exception:
        return (
            "Service temporarily unavailable. Please try again "
            "after verifying Ollama is running.",
            0.0,
            "service_error",
            True,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return reply, latency_ms, "served", False


def recover_in_scope_false_refusal(
    client, system_prompt, few_shot, history, user_text, reply, scope, risk
):
    if not scope["is_in_scope"] or risk["is_attack"] or not is_refusal(reply):
        return reply, False

    recovery_note = (
        "The previous answer was an over-refusal. The user request is in approved PromptOps scope. "
        "Provide a safe, concise educational answer focused on defensive prompt engineering."
    )
    retry_messages = build_messages(system_prompt, few_shot, history, user_text)
    retry_messages.append({"role": "assistant", "content": reply})
    retry_messages.append({"role": "user", "content": recovery_note})
    try:
        recovered = client.chat(retry_messages, temperature=0.1)
    except Exception:
        return reply, False
    if recovered and not is_refusal(recovered):
        return recovered, True
    return reply, False


def process_turn(client, system_prompt, few_shot, active_chat, user_text, scope, risk):
    stats = active_chat["stats"]
    precheck_reply, precheck_latency, precheck_action = apply_precheck_controls(risk, scope, stats)
    if precheck_action:
        return precheck_reply, precheck_latency, precheck_action, False

    reply, latency_ms, action, error_flag = query_model_response(
        client=client,
        system_prompt=system_prompt,
        few_shot=few_shot,
        history=active_chat["messages"],
        user_text=user_text,
    )
    recovered_reply, recovered = recover_in_scope_false_refusal(
        client=client,
        system_prompt=system_prompt,
        few_shot=few_shot,
        history=active_chat["messages"],
        user_text=user_text,
        reply=reply,
        scope=scope,
        risk=risk,
    )
    if recovered:
        reply = recovered_reply
        action = "served_after_recovery"

    if risk["is_attack"]:
        if is_refusal(reply):
            stats["refused_postcheck"] += 1
        else:
            stats["unsafe_passes"] += 1
    return reply, latency_ms, action, error_flag


def persist_turn(active_chat, user_text, reply, latency_ms, scope, risk):
    if not active_chat["messages"]:
        active_chat["title"] = " ".join(user_text.split()[:5]) or active_chat["title"]

    active_chat["messages"].append(
        {
            "user": user_text,
            "assistant": reply,
            "latency_ms": latency_ms,
            "is_in_scope": scope["is_in_scope"],
            "is_attack": risk["is_attack"],
            "risk_score": risk["risk_score"],
        }
    )
    active_chat["updated_at"] = now_utc_iso()


def emit_metrics(active_chat, action, scope, risk, latency_ms, user_text, error_flag):
    metric_payload = {
        "chat_id": active_chat["id"],
        "model": DEFAULT_MODEL,
        "action": action,
        "is_in_scope": scope["is_in_scope"],
        "matched_topics": scope.get("matched_topics", []),
        "blocked_category": scope.get("blocked_category"),
        "is_attack": risk["is_attack"],
        "attack_hit_count": len(risk.get("pattern_hits", [])),
        "latency_ms": latency_ms,
        "user_text_len": len(user_text),
        "error": error_flag,
    }
    if LOG_CHAT_TITLES:
        metric_payload["chat_title"] = active_chat["title"]
    log_internal_metric(metric_payload)
    record_event(
        model=DEFAULT_MODEL,
        action=action,
        is_in_scope=scope["is_in_scope"],
        is_attack=risk["is_attack"],
        blocked_category=scope.get("blocked_category"),
        latency_ms=latency_ms,
        error=error_flag,
    )


def main():
    st.set_page_config(page_title="PromptOps Secure Chat Lab", layout="wide")
    render_sticky_header()
    try:
        ensure_guardrails_ready()
    except Exception as exc:
        st.error(f"Guardrail policy configuration error: {exc}")
        st.stop()

    init_metrics_server()
    init_state()
    active_chat = get_active_chat()

    if ADMIN_MODE:
        st.sidebar.header("Runtime")
        st.sidebar.text(f"Model: {DEFAULT_MODEL}")
        st.sidebar.text(f"Endpoint: {DEFAULT_BASE_URL}")

    sidebar_chat_navigation(active_chat)

    if ADMIN_MODE:
        show_stats(active_chat["stats"])

    cfg = load_prompt_version(str(ROOT / DEFAULT_PROMPT_VERSION))
    system_prompt = resolve_system_prompt(ROOT, cfg)
    few_shot = cfg.get("few_shot_examples", [])
    client = OllamaClient(base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)

    render_chat_history(active_chat)

    user_text = st.chat_input("Ask a PromptOps question...")
    if not user_text:
        return

    scope = classify_topic_scope(user_text)
    risk = detect_injection_risk(user_text)
    stats = active_chat["stats"]
    stats["total_turns"] += 1
    if risk["is_attack"]:
        stats["attack_attempts"] += 1

    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        reply, latency_ms, action, error_flag = process_turn(
            client=client,
            system_prompt=system_prompt,
            few_shot=few_shot,
            active_chat=active_chat,
            user_text=user_text,
            scope=scope,
            risk=risk,
        )
        update_latency(stats, latency_ms)
        st.write(reply)

    persist_turn(active_chat, user_text, reply, latency_ms, scope, risk)
    emit_metrics(active_chat, action, scope, risk, latency_ms, user_text, error_flag)


if __name__ == "__main__":
    main()
