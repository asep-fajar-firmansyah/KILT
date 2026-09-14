"""Chat backends used by the SynMeS annotation stage."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List


class TransformersChat:
    """Hugging Face text-generation backend, as used by ToT4ES."""

    def __init__(self, model_id: str, device_map: str = "auto", seed: int | None = None):
        import torch
        from transformers import pipeline

        self._torch = torch
        self.model_id = model_id
        self.seed = seed
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.tokenizer = self.pipe.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> List[str]:
        if self.seed is not None:
            self._torch.manual_seed(self.seed)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        do_sample = temperature > 0.0
        prompts = [prompt] * n if (n > 1 and do_sample) else [prompt]
        outputs = self.pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            batch_size=len(prompts),
        )
        texts = [item[0]["generated_text"].strip() for item in outputs]
        while len(texts) < n:
            texts.append(texts[-1])
        return texts[:n]

    def __repr__(self) -> str:
        return f"TransformersChat(model={self.model_id})"


class OllamaChat:
    """Ollama `/api/chat` backend for locally served models."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        seed: int | None = None,
        timeout: float = 300.0,
    ):
        scheme = urllib.parse.urlparse(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"Ollama base URL must be http(s): {base_url}")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> List[str]:
        outputs = []
        for offset in range(n):
            options = {"temperature": temperature, "num_predict": max_new_tokens}
            if self.seed is not None:
                options["seed"] = self.seed + offset
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            outputs.append(body.get("message", {}).get("content", "").strip())
        return outputs

    def __repr__(self) -> str:
        return f"OllamaChat(model={self.model_id}, base_url={self.base_url})"


def build_backend(spec: str, seed: int | None = None, ollama_url: str = "http://localhost:11434"):
    """Build a chat backend from `heuristic`, `ollama:<model>`, or `hf:<model>`."""
    if spec == "heuristic":
        return None
    if spec.startswith("ollama:"):
        return OllamaChat(spec[len("ollama:") :], base_url=ollama_url, seed=seed)
    model_id = spec[len("hf:") :] if spec.startswith("hf:") else spec
    return TransformersChat(model_id, seed=seed)
