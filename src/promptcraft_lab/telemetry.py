import os
import threading

from prometheus_client import Counter, Histogram, start_http_server

_LOCK = threading.Lock()
_STARTED = False

REQUESTS_TOTAL = Counter(
    "promptops_chat_requests_total",
    "Total chat requests processed",
    ["model", "action", "is_in_scope", "is_attack", "blocked_category"],
)

BLOCKED_ATTACK_TOTAL = Counter(
    "promptops_chat_blocked_attack_total",
    "Total requests blocked due to attack detection",
    ["model"],
)

BLOCKED_SCOPE_TOTAL = Counter(
    "promptops_chat_blocked_scope_total",
    "Total requests blocked due to out-of-scope policy",
    ["model", "blocked_category"],
)

ERRORS_TOTAL = Counter(
    "promptops_chat_errors_total",
    "Total chat processing errors",
    ["model", "action"],
)

LATENCY_MS = Histogram(
    "promptops_chat_latency_ms",
    "Chat response latency in milliseconds",
    ["model", "action"],
    buckets=(100, 250, 500, 1000, 2000, 5000, 10000, 20000, 60000),
)


def _bool_label(value: bool) -> str:
    return "true" if bool(value) else "false"


def init_metrics_server():
    global _STARTED
    enabled = os.getenv("PROM_METRICS_ENABLED", "true").strip().lower() == "true"
    if not enabled:
        return

    with _LOCK:
        if _STARTED:
            return
        port = int(os.getenv("PROM_METRICS_PORT", "9108"))
        bind_addr = os.getenv("PROM_METRICS_BIND", "")
        try:
            start_http_server(port, addr=bind_addr)
        except OSError:
            # Streamlit reruns can race on startup; if already bound, continue.
            pass
        _STARTED = True


def record_event(
    *,
    model: str,
    action: str,
    is_in_scope: bool,
    is_attack: bool,
    blocked_category: str | None,
    latency_ms: float,
    error: bool,
):
    blocked_label = blocked_category or "none"

    REQUESTS_TOTAL.labels(
        model=model,
        action=action,
        is_in_scope=_bool_label(is_in_scope),
        is_attack=_bool_label(is_attack),
        blocked_category=blocked_label,
    ).inc()

    if action == "blocked_attack_precheck":
        BLOCKED_ATTACK_TOTAL.labels(model=model).inc()

    if action == "blocked_out_of_scope":
        BLOCKED_SCOPE_TOTAL.labels(model=model, blocked_category=blocked_label).inc()

    if error:
        ERRORS_TOTAL.labels(model=model, action=action).inc()

    if latency_ms and latency_ms > 0:
        LATENCY_MS.labels(model=model, action=action).observe(float(latency_ms))
