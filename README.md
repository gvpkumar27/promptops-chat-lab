# PromptOps Evaluation Lab

PromptOps Evaluation Lab is a local-first PromptOps platform for secure prompt experimentation, runtime guardrails, and evaluation reporting.

## Technology Stack
- Python 3.11+
- Streamlit (chat application)
- Ollama (local model runtime)
- Docker Compose
- Prometheus + Grafana (observability)

## Functional Scope
In scope:
- Prompt engineering, guardrails, and safety evaluation workflows.
- Runtime safety controls for prompt injection and out-of-scope requests.
- Evaluation reporting over curated benchmark datasets.

Out of scope:
- Open-domain assistant usage unrelated to PromptOps.
- Harmful/offensive guidance.
- Medical, legal, or financial advice.

## Runtime Architecture
- `chat-ui` (Streamlit) handles user interaction, guardrail checks, model calls, and metrics export.
- Ollama serves inference at `OLLAMA_BASE_URL`.
- Prometheus scrapes application metrics.
- Grafana visualizes operational dashboards.
- Evaluation jobs run via `scripts/run_eval.py` and publish reports to `eval/results/`.

## Prerequisites
- Python 3.11+
- Docker Desktop (for containerized deployment and dashboards)
- Ollama installed locally or available via Docker service

## Required Configuration
Set these environment variables before running the application:
- `PROMPTOPS_SYSTEM_PROMPT_PATH=<ABSOLUTE_PATH_TO_PRIVATE_SYSTEM_PROMPT_MD>`
- `PROMPTOPS_POLICY_PATH=<ABSOLUTE_PATH_TO_PRIVATE_POLICY_JSON>`
- `PROMPTOPS_STRICT_GUARDRAILS=true`

Optional:
- `OLLAMA_MODEL` (default: `llama3.2:1b`)
- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `GF_SECURITY_ADMIN_PASSWORD` (required when running Grafana)

## Local Execution
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run evaluation:
   - `python scripts/run_eval.py`
4. Run chat UI:
   - `streamlit run scripts/run_chat_ui.py`
5. Access:
   - Chat UI: `http://localhost:8501`

## Docker Execution
If local Ollama is already bound to `127.0.0.1:11434`:
- `docker compose up -d --no-deps chat-ui`

If using Docker-managed Ollama:
- `docker compose up -d ollama`
- `docker compose run --rm ollama-init`
- `docker compose up -d chat-ui`

Access:
- Chat UI: `http://localhost:8501`

## Observability
Prometheus + Grafana is the supported dashboard stack.

Start:
- If local Ollama is running:
  - `docker compose up -d --no-deps chat-ui prometheus grafana`
- Otherwise:
  - `docker compose up -d chat-ui prometheus grafana`

Access:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Evaluation Metrics
- `exact_match`
- `keyword_coverage`
- `json_validity`
- `refusal_rate_on_attacks`
- `latency_ms`
- `estimated_prompt_tokens`
- `response_length_chars`

## Security and Open-Source Hygiene
- Keep private prompt and policy files outside this repository.
- Never commit `.env` or private configuration assets.
- Publish only sanitized templates (for example, `policies/guardrails.public.example.json`).
- Run internal quality/security checks in private workflows before release.

## Project Structure
- `src/promptcraft_lab/`: core application and runtime logic
- `scripts/`: entry points for chat and evaluation
- `eval/data/`: benchmark datasets
- `prompts/`: prompt templates and versions
- `policies/`: public-safe guardrail template
- `monitoring/`: Prometheus and Grafana configuration
- `docs/`: lab and evaluation documentation
