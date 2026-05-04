"""
Singleton model manager for QLoRA Demo Site.
Handles load/unload of 4-bit quantized models and LoRA adapters.
Designed for RTX 3050 4GB VRAM.
"""

import gc
import os
import time
from typing import Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SUPPORTED_MODELS = {
    "TinyLlama-1.1B-Chat (Recommended)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "LLaMA 3.2 1B Instruct (needs HF_TOKEN)": "meta-llama/Llama-3.2-1B-Instruct",
}

# Default chat template for TinyLlama
TINYLLAMA_TEMPLATE = (
    "<|system|>\nYou are a helpful AI assistant specialized in machine learning.</s>\n"
    "<|user|>\n{prompt}</s>\n<|assistant|>\n"
)

ALPACA_TEMPLATE = (
    "### Instruction:\n{prompt}\n\n### Response:\n"
)


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.is_4bit: bool = False
        self.has_adapter: bool = False
        self.adapter_path: Optional[str] = None
        self._load_time: Optional[float] = None

    def _get_bnb_config(self):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def load_base_model(self, model_id: str) -> Tuple[object, object]:
        """Load a 4-bit quantized model. Returns (model, tokenizer)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN")
        kwargs = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right",
            **kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self._get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            **kwargs,
        )
        self.model.config.use_cache = False
        self.model_id = model_id
        self.is_4bit = True
        self.has_adapter = False
        self.adapter_path = None
        self._load_time = time.time() - t0
        return self.model, self.tokenizer

    def load_with_adapter(self, adapter_path: str):
        """Attach a trained LoRA adapter to the already-loaded base model."""
        from peft import PeftModel
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.has_adapter = True
        self.adapter_path = adapter_path
        return self.model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_chat_template: bool = True,
    ) -> Tuple[str, float]:
        """Generate text. Returns (response_text, elapsed_seconds)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        if use_chat_template:
            formatted = TINYLLAMA_TEMPLATE.format(prompt=prompt)
        else:
            formatted = ALPACA_TEMPLATE.format(prompt=prompt)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, elapsed

    def unload(self):
        """Free VRAM completely."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_id = None
        self.is_4bit = False
        self.has_adapter = False

    def get_vram_info(self) -> dict:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"used_mb": 0, "free_mb": 0, "total_mb": 0, "pct": 0.0}
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        used = r  # reserved is what the driver reports as "used"
        return {
            "used_mb": used // (1024 ** 2),
            "free_mb": (t - r) // (1024 ** 2),
            "total_mb": t // (1024 ** 2),
            "pct": round(used / t * 100, 1),
            "allocated_mb": a // (1024 ** 2),
        }

    def get_memory_footprint(self) -> int:
        """Return model memory footprint in bytes (for loaded model)."""
        if self.model is None:
            return 0
        try:
            return self.model.get_memory_footprint()
        except Exception:
            return 0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def load_time(self) -> Optional[float]:
        return self._load_time
