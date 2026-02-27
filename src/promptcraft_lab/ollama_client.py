import requests
import time


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.2:1b"):
        self.base_url = self._normalize_base_url(base_url)
        self.model = model

    def chat(self, messages, temperature: float = 0.1):
        chat_url = f"{self.base_url}/api/chat"
        chat_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        last_error = None
        for attempt in range(2):
            try:
                response = requests.post(chat_url, json=chat_payload, timeout=120)
                if response.status_code == 404:
                    return self._generate_fallback(messages, temperature)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break
        raise RuntimeError(f"Ollama chat failed after retry: {last_error}") from last_error

    def _generate_fallback(self, messages, temperature: float = 0.1):
        generate_url = f"{self.base_url}/api/generate"
        prompt = self._messages_to_prompt(messages)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(generate_url, json=payload, timeout=120)
        if response.status_code == 404:
            models = self.list_models()
            available = ", ".join(models) if models else "none"
            raise RuntimeError(
                "Ollama returned 404 for chat/generate. "
                f"Check base URL ({self.base_url}) and pull model '{self.model}'. "
                f"Available models: {available}"
            )

        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def list_models(self):
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if res.status_code != 200:
                return []
            data = res.json()
            items = data.get("models", [])
            return [m.get("name", "") for m in items if m.get("name")]
        except Exception:
            return []

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base = (base_url or "").strip().rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        return base

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)
