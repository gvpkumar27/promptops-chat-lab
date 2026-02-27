# Hands-On Labs

## Lab 1: Zero-shot vs Few-shot
- Run baseline prompt on `eval/data/task_general.jsonl`.
- Enable few-shot exemplars in `prompts/versions/v1_fewshot.json`.
- Compare pass rate and latency.

## Lab 2: System vs User Prompt Separation
- Edit your configured system prompt source in your private deployment setup.
- Keep task in user prompt files.
- Observe consistency differences in report.

## Lab 3: Context Efficiency
- Add unnecessary context to `prompts/user/noisy_context.md`.
- Compare `estimated_prompt_tokens` and pass rate.

## Lab 4: Defensive Prompting
- Run `eval/data/task_security.jsonl`.
- Tune refusal and safe extraction behavior in your private prompt/policy setup.
- Track `refusal_rate_on_attacks` and false refusals.

## Lab 5: Prompt Versioning
- Clone `prompts/versions/v1_baseline.json` into `v2_candidate.json`.
- Change instruction style and decomposition strategy.
- Evaluate and select best candidate based on metrics.
