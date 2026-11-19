from __future__ import annotations

from typing import Protocol


class ModelProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def generate(self, prompt: str) -> tuple[str, dict]: ...


class DashScopeProvider:
    """Cloud-backed provider. Document text and question context are sent to DashScope."""

    def __init__(self, embedding_model: str, generation_model: str):
        import os
        from pathlib import Path

        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        key_file = os.getenv("DASHSCOPE_API_KEY_FILE")
        if not key and key_file:
            try:
                key = Path(key_file).read_text(encoding="utf-8-sig").strip()
            except OSError as exc:
                raise RuntimeError("无法读取配置的 DashScope 密钥文件") from exc
        if not key:
            raise RuntimeError("尚未配置 DASHSCOPE_API_KEY，请在 .env 文件中填写")
        self.api_key = key
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1").rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def _post(self, route: str, payload: dict) -> dict:
        import requests
        try:
            response = requests.post(f"{self.base_url}/{route}", json=payload,
                                     headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError("无法连接百炼，请检查网络后重试") from exc
        if not response.ok:
            raise RuntimeError(f"百炼请求失败（HTTP {response.status_code}）")
        return response.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 10):
            data = self._post("embeddings", {
                "model": self.embedding_model, "input": texts[start:start + 10],
            })["data"]
            vectors.extend(row["embedding"] for row in sorted(data, key=lambda row: row["index"]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def rewrite(self, prompt: str, model: str) -> tuple[str, dict]:
        return self._generate(prompt, model)

    def generate(self, prompt: str) -> tuple[str, dict]:
        return self._generate(prompt, self.generation_model)

    def _generate(self, prompt: str, model: str) -> tuple[str, dict]:
        response = self._post("chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "enable_thinking": False,
        })
        usage = response.get("usage") or {}
        stats = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        return response["choices"][0]["message"]["content"] or "", stats

