"""
OpenAI-backed drop-in replacement for ModelManager.
Used in Demo Mode so interviewers can test LLM features without loading a local GPU model.
Requires OPENAI_API_KEY in environment.
"""

import os
import time
from typing import Optional, Tuple

_SYSTEM_BASE = (
    "You are a general-purpose language model. "
    "Answer questions clearly but without strong structure or instruction-following emphasis."
)

_SYSTEM_FINETUNED = (
    "You are a QLoRA fine-tuned assistant specialising in machine learning. "
    "Follow instructions precisely, use markdown formatting, provide structured answers "
    "with numbered lists, and demonstrate clear chain-of-thought reasoning. "
    "Your responses should be noticeably more structured and instruction-focused than a base model."
)


class OpenAIWrapper:
    """
    Drop-in replacement for ModelManager that calls gpt-4o-mini.

    - is_loaded is always True when OPENAI_API_KEY is set.
    - load_with_adapter() switches to a fine-tuned system prompt (simulates QLoRA adapter).
    - All VRAM / memory methods return zero.
    """

    load_time: Optional[float] = 0.0

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("OPENAI_API_KEY", "")
        self._mode: str = "base"
        self.model_id: str = "gpt-4o-mini"

    # ── Core interface ────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_chat_template: bool = True,
    ) -> Tuple[str, float]:
        """Call gpt-4o-mini. System prompt switches based on _mode (base vs finetuned)."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        system_prompt = _SYSTEM_FINETUNED if self._mode == "finetuned" else _SYSTEM_BASE

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(max_new_tokens, 1024),
            temperature=temperature,
        )
        elapsed = time.time() - t0
        text = response.choices[0].message.content or ""
        return text, elapsed

    def load_base_model(self, model_id: str):
        """No-op — resets to base mode."""
        self._mode = "base"
        return None, None

    def load_with_adapter(self, adapter_path: str):
        """Simulate adapter loading by switching system prompt to fine-tuned mode."""
        self._mode = "finetuned"
        return None

    def unload(self) -> None:
        """No-op — resets to base mode."""
        self._mode = "base"

    def get_vram_info(self) -> dict:
        return {"used_mb": 0, "free_mb": 0, "total_mb": 0, "pct": 0.0, "allocated_mb": 0}

    def get_memory_footprint(self) -> int:
        return 0
