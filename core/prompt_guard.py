"""
Prompt Guard Manager for Llama-Prompt-Guard-2-86M.
Supports both fp16 (Full FT baseline) and 4-bit NF4 (QLoRA) loading modes.
"""

import gc
import os
import time
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

# Label mapping from model config (SAFE=0, INJECTION=1)
LABEL_MAP = {0: "SAFE", 1: "INJECTION"}
LABEL_COLORS = {"SAFE": "#4caf50", "INJECTION": "#ff6b6b"}


class PromptGuardManager:
    """Manages loading and inference for Llama-Prompt-Guard-2-86M."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_quantized: bool = False
        self._load_time: Optional[float] = None

    def load(self, quantize: bool = False) -> None:
        """
        Load the Prompt Guard model.
        quantize=False → fp16, standard full fine-tuning baseline
        quantize=True  → 4-bit NF4 via bitsandbytes (QLoRA mode)
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        hf_token = os.environ.get("HF_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            **token_kwargs,
        )

        t0 = time.time()
        if quantize:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                **token_kwargs,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                **token_kwargs,
            )

        self.model.eval()
        self.is_quantized = quantize
        self._load_time = time.time() - t0

    def predict(self, text: str) -> dict:
        """
        Classify a single text as SAFE (0) or INJECTION (1).
        Returns {"label": str, "score": float, "label_id": int, "scores": {SAFE: x, INJECTION: y}}
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        label_id = int(probs.argmax())
        score = float(probs[label_id])

        return {
            "label": LABEL_MAP.get(label_id, str(label_id)),
            "label_id": label_id,
            "score": round(score, 4),
            "scores": {
                "SAFE": round(float(probs[0]), 4),
                "INJECTION": round(float(probs[1]), 4),
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Classify a list of texts. Returns list of dicts."""
        return [self.predict(t) for t in texts]

    def unload(self) -> None:
        """Free GPU and system memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_quantized = False

    def get_vram_mb(self) -> int:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved(0) // (1024 ** 2)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def load_time(self) -> Optional[float]:
        return self._load_time
