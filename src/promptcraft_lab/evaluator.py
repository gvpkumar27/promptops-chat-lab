import json
import time
from pathlib import Path

from .metrics import (
    exact_match,
    estimate_tokens,
    json_validity,
    keyword_coverage,
    refusal_score,
)


class Evaluator:
    def __init__(self, client, system_prompt: str, few_shot_examples=None):
        self.client = client
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples or []

    def build_messages(self, user_text: str):
        messages = [{"role": "system", "content": self.system_prompt}]
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": user_text})
        return messages

    def score_sample(self, sample: dict):
        messages = self.build_messages(sample["input"])
        prompt_blob = "\n".join(m["content"] for m in messages)
        start = time.perf_counter()
        output = self.client.chat(messages)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        metric = sample.get("metric", "exact_match")
        score = 0.0
        if metric == "exact_match":
            score = exact_match(output, sample.get("expected", ""))
        elif metric == "keyword_coverage":
            score = keyword_coverage(output, sample.get("expected_keywords", []))
        elif metric == "json_validity":
            score = json_validity(output, sample.get("expected_json_keys", []))
        elif metric == "safety_refusal":
            score = refusal_score(output, bool(sample.get("expect_refusal", False)))

        return {
            "id": sample.get("id"),
            "category": sample.get("category"),
            "metric": metric,
            "score": round(float(score), 4),
            "latency_ms": latency_ms,
            "estimated_prompt_tokens": estimate_tokens(prompt_blob),
            "response_length_chars": len(output),
            "input": sample.get("input"),
            "output": output,
        }

    def summarize(self, results):
        by_category = {}
        for r in results:
            by_category.setdefault(r["category"], []).append(r)

        category_summary = {}
        for category, rows in by_category.items():
            avg_score = sum(x["score"] for x in rows) / len(rows)
            avg_latency = sum(x["latency_ms"] for x in rows) / len(rows)
            category_summary[category] = {
                "samples": len(rows),
                "pass_rate": round(avg_score, 4),
                "avg_latency_ms": round(avg_latency, 2),
            }

        all_scores = [r["score"] for r in results]
        all_latency = [r["latency_ms"] for r in results]
        return {
            "total_samples": len(results),
            "overall_pass_rate": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
            "overall_avg_latency_ms": (
                round(sum(all_latency) / len(all_latency), 2)
                if all_latency
                else 0.0
            ),
            "by_category": category_summary,
            "results": results,
        }

    def write_report(self, summary: dict, out_json: Path, out_md: Path):
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        lines = [
            "# Evaluation Report",
            "",
            f"- Total samples: {summary['total_samples']}",
            f"- Overall pass rate: {summary['overall_pass_rate']}",
            f"- Overall avg latency (ms): {summary['overall_avg_latency_ms']}",
            "",
            "## Category summary",
            "",
        ]
        for cat, stats in summary["by_category"].items():
            lines.append(
                f"- {cat}: pass_rate={stats['pass_rate']} "
                f"samples={stats['samples']} "
                f"avg_latency_ms={stats['avg_latency_ms']}"
            )

        lines.extend(["", "## Sample outputs", ""])
        for row in summary["results"]:
            lines.append(f"### {row['id']} ({row['category']})")
            lines.append(f"- metric: {row['metric']}")
            lines.append(f"- score: {row['score']}")
            lines.append(f"- latency_ms: {row['latency_ms']}")
            lines.append(f"- output: {row['output'][:300]}")
            lines.append("")

        out_md.write_text("\n".join(lines), encoding="utf-8")
