import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promptcraft_lab.ollama_client import OllamaClient
from promptcraft_lab.prompts import load_prompt_version, resolve_system_prompt
from promptcraft_lab.guardrails import (
    HARMFUL_CONTENT_REFUSAL,
    OUT_OF_SCOPE_REFUSAL,
    SAFE_REFUSAL,
    classify_topic_scope,
    detect_injection_risk,
    ensure_guardrails_ready,
)


def main():
    parser = argparse.ArgumentParser(description="Interactive prompt playground")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--prompt-version", default="prompts/versions/v1_baseline.json")
    args = parser.parse_args()
    try:
        ensure_guardrails_ready()
    except Exception as exc:
        raise RuntimeError(f"Guardrail policy configuration error: {exc}") from exc

    cfg = load_prompt_version(str(ROOT / args.prompt_version))
    system_prompt = resolve_system_prompt(ROOT, cfg)
    few_shot = cfg.get("few_shot_examples", [])

    client = OllamaClient(base_url=args.base_url, model=args.model)

    print("Prompt playground started. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        scope = classify_topic_scope(user_input)
        risk = detect_injection_risk(user_input)
        if risk["is_attack"]:
            print(f"Assistant> {SAFE_REFUSAL}")
            continue

        if not scope["is_in_scope"]:
            if scope.get("blocked_category") == "Harmful/extremist content":
                print(f"Assistant> {HARMFUL_CONTENT_REFUSAL}")
            else:
                print(f"Assistant> {OUT_OF_SCOPE_REFUSAL}")
            continue

        messages = [{"role": "system", "content": system_prompt}]
        for ex in few_shot:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_input})

        reply = client.chat(messages)
        print(f"Assistant> {reply}")


if __name__ == "__main__":
    main()
