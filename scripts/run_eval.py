import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promptcraft_lab.evaluator import Evaluator
from promptcraft_lab.guardrails import ensure_guardrails_ready
from promptcraft_lab.ollama_client import OllamaClient
from promptcraft_lab.prompts import (
    load_jsonl,
    load_prompt_version,
    resolve_system_prompt,
)


def collect_samples(data_dir: Path):
    samples = []
    for path in sorted(data_dir.glob("*.jsonl")):
        samples.extend(load_jsonl(str(path)))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run prompt evaluation suite")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--prompt-version", default="prompts/versions/v1_baseline.json")
    args = parser.parse_args()
    ensure_guardrails_ready()

    version_cfg = load_prompt_version(str(ROOT / args.prompt_version))
    system_prompt = resolve_system_prompt(ROOT, version_cfg)
    few_shot = version_cfg.get("few_shot_examples", [])

    client = OllamaClient(base_url=args.base_url, model=args.model)
    evaluator = Evaluator(client=client, system_prompt=system_prompt, few_shot_examples=few_shot)

    samples = collect_samples(ROOT / "eval" / "data")
    results = [evaluator.score_sample(s) for s in samples]
    summary = evaluator.summarize(results)

    out_json = ROOT / "eval" / "results" / "latest_report.json"
    out_md = ROOT / "eval" / "results" / "latest_report.md"
    evaluator.write_report(summary, out_json, out_md)

    print("Evaluation complete")
    print(f"Model: {args.model}")
    print(f"Pass rate: {summary['overall_pass_rate']}")
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
