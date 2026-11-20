"""OpenRouter chat client with free-model discovery, rotation and a circuit breaker.

- Model list: OPENROUTER_MODELS (comma-separated) if set, else discovered at runtime from OpenRouter's /models endpoint
  (ids ending in ':free' or with zero prompt+completion price), ranked by PREFERENCE. Free models come and go, so
  nothing is hard-coded as the only option.
- Each call tries models in order; rate limits / 5xx / timeouts / empty replies move on to the next model, and a model
  that failed is skipped for COOLDOWN seconds.
- Raises LLMUnavailable only when every model failed or no key is configured; callers turn that into a grounded
  template answer, so the API never fails because of the LLM.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx

BASE = "https://openrouter.ai/api/v1"
PREFERENCE = ("llama-3.3-70b", "deepseek-chat", "deepseek-v3", "qwen3-235b", "qwen-2.5-72b", "gpt-oss-120b", "gemma-3-27b",
              "mistral-small", "llama-4", "qwen3", "gemma", "mistral", "llama", "deepseek", "")
EXCLUDE = ("vision", "embed", "audio", "image", "-vl", "guard", "coder", "math", "r1-distill")
COOLDOWN = 90.0


class LLMUnavailable(RuntimeError):
    pass


class OpenRouterLLM:
    def __init__(self, api_key: str | None = None, models: list[str] | None = None, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        env_models = [m.strip() for m in os.environ.get("OPENROUTER_MODELS", "").split(",") if m.strip()]
        self._models = models or env_models or None
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, transport=transport) if transport else httpx.Client(timeout=timeout)
        self.failed_at: dict[str, float] = {}
        self.last_model: str | None = None
        self.last_attempts: list[dict] = []

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": os.environ.get("APP_URL", "https://localhost"), "X-Title": "Property RAG (DarGlobal + Wasalt)"}

    def models(self) -> list[str]:
        if self._models:
            return self._models
        try:
            data = self.client.get(f"{BASE}/models", headers=self._headers()).json().get("data", [])
        except Exception as e:
            raise LLMUnavailable(f"model discovery failed: {e}") from e
        free = []
        for m in data:
            mid = m.get("id", "")
            pr = m.get("pricing") or {}
            is_free = mid.endswith(":free") or (str(pr.get("prompt")) == "0" and str(pr.get("completion")) == "0")
            if is_free and not any(x in mid.lower() for x in EXCLUDE) and (m.get("context_length") or 0) >= 8000:
                free.append(m)
        rank = lambda m: (next(i for i, p in enumerate(PREFERENCE) if p in m["id"].lower()), -(m.get("context_length") or 0))  # noqa: E731
        self._models = [m["id"] for m in sorted(free, key=rank)][:8]
        if not self._models:
            raise LLMUnavailable("no free models found on OpenRouter")
        return self._models

    def chat(self, messages: list[dict], max_tokens: int = 800, temperature: float = 0.1, json_mode: bool = False,
             budget_s: float | None = None) -> str:
        if not self.configured:
            raise LLMUnavailable("OPENROUTER_API_KEY not set")
        self.last_attempts = []
        now = time.time()
        candidates = [m for m in self.models() if now - self.failed_at.get(m, 0) > COOLDOWN] or self.models()
        budget = budget_s if budget_s is not None else float(os.environ.get("LLM_BUDGET_SECONDS", "40"))
        for model in candidates:
            if time.time() - now > budget:                 # overall deadline across all attempts
                self.last_attempts.append({"model": model, "ok": False, "error": "skipped: time budget used"})
                break
            body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            t0 = time.time()
            try:
                r = self.client.post(f"{BASE}/chat/completions", headers=self._headers(), json=body)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                payload = r.json()
                if payload.get("error"):
                    raise RuntimeError(str(payload["error"])[:160])
                text = (payload["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                self.last_model = model
                self.last_attempts.append({"model": model, "ok": True, "ms": int((time.time() - t0) * 1000)})
                return text
            except Exception as e:
                self.failed_at[model] = time.time()
                self.last_attempts.append({"model": model, "ok": False, "error": str(e)[:160],
                                           "ms": int((time.time() - t0) * 1000)})
        raise LLMUnavailable(f"all models failed: {self.last_attempts}")

    def chat_json(self, messages: list[dict], max_tokens: int = 500) -> dict:
        try:
            text = self.chat(messages, max_tokens=max_tokens, temperature=0.0, json_mode=True)
        except LLMUnavailable:
            raise
        except Exception:
            text = self.chat(messages, max_tokens=max_tokens, temperature=0.0)
        return parse_json(text)


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group())
        raise


_LLM: OpenRouterLLM | None = None


def get_llm() -> OpenRouterLLM:
    global _LLM
    if _LLM is None:
        _LLM = OpenRouterLLM()
    return _LLM


if __name__ == "__main__":                                   # uv run python -m rag.llm  -> which free models answer now
    import sys

    from .env import load_env
    load_env()
    llm = OpenRouterLLM(timeout=25)
    if not llm.configured:
        sys.exit("OPENROUTER_API_KEY is not set (.env)")
    models = llm.models()
    print(f"{len(models)} candidate models ({'OPENROUTER_MODELS' if os.environ.get('OPENROUTER_MODELS') else 'discovered free models'}):")
    working = []
    for m in models:
        one = OpenRouterLLM(models=[m], timeout=25)
        t0 = time.time()
        try:
            out = one.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=5, budget_s=30)
            working.append(m)
            print(f"  OK    {m:55} {int((time.time() - t0) * 1000):>6} ms  -> {out[:20]!r}")
        except LLMUnavailable:
            err = (one.last_attempts or [{}])[-1].get("error", "?")
            print(f"  FAIL  {m:55} {int((time.time() - t0) * 1000):>6} ms  -> {err[:90]}")
    if working:
        print(f"\nTo pin the working ones, add to .env:\nOPENROUTER_MODELS={','.join(working)}")
    else:
        print("\nNo free model answered. Free models are rate-limited per minute and per day (higher daily limits once the "
              "account has credits); wait a few minutes and retry. Until then the app answers with grounded summaries.")
