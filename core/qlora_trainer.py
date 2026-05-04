"""
QLoRA Training Pipeline for RTX 3050 4GB.
Runs in a background thread, pushes progress via queue.Queue.
"""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path: str = "data/train_data/alpaca_tiny.json"
    output_dir: str = "adapters/qlora_checkpoint"
    lora_r: int = 8
    lora_alpha: int = 16           # auto = 2 * r
    lora_dropout: float = 0.05
    max_steps: int = 50
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    per_device_train_batch_size: int = 1   # FIXED: 4GB constraint
    gradient_accumulation_steps: int = 4


class VRAMLoggingCallback:
    """Custom callback compatible with HuggingFace TrainerCallback interface."""

    def __init__(self, progress_queue: queue.Queue):
        self._queue = progress_queue
        self._monitor_handle = None
        self._setup_nvml()

    def _setup_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._monitor_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def _get_vram_mb(self) -> int:
        if self._monitor_handle is None:
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved(0) // (1024 ** 2)
            return 0
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._monitor_handle)
            return info.used // (1024 ** 2)
        except Exception:
            return 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._queue.put({
            "step": state.global_step,
            "loss": round(float(logs.get("loss", 0.0)), 4),
            "vram_mb": self._get_vram_mb(),
            "epoch": round(float(state.epoch or 0.0), 2),
            "lr": float(logs.get("learning_rate", 0.0)),
            "status": "running",
            "ts": time.time(),
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._queue.put({"status": "done", "step": state.global_step})


class QLoRATrainer:
    def __init__(self, model, tokenizer, config: TrainingConfig, progress_queue: queue.Queue):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._queue = progress_queue
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._trainer = None

    def _build_lora_config(self):
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self._config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _build_training_args(self):
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=self._config.output_dir,
            per_device_train_batch_size=self._config.per_device_train_batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            max_steps=self._config.max_steps,
            learning_rate=self._config.learning_rate,
            fp16=True,
            bf16=False,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            save_strategy="no",
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,  # Windows compatibility
            remove_unused_columns=False,
        )

    def _load_dataset(self):
        import datasets

        path = self._config.dataset_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def format_sample(sample):
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            output = sample.get("output", "")
            if inp:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return {"text": text}

        formatted = [format_sample(s) for s in raw]
        ds = datasets.Dataset.from_list(formatted)
        split = ds.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]

    def _run_training(self):
        try:
            from peft import prepare_model_for_kbit_training
            from trl import SFTTrainer
            from transformers import TrainerCallback

            self._queue.put({"status": "preparing", "step": 0, "loss": 0.0, "vram_mb": 0, "epoch": 0.0})

            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(
                self._model,
                use_gradient_checkpointing=True,
            )

            lora_config = self._build_lora_config()
            training_args = self._build_training_args()
            train_ds, eval_ds = self._load_dataset()

            callback = VRAMLoggingCallback(self._queue)

            # Wrap callback to match HuggingFace TrainerCallback interface
            class HFCallbackWrapper(TrainerCallback):
                def __init__(self, cb):
                    self._cb = cb

                def on_log(self, args, state, control, logs=None, **kwargs):
                    self._cb.on_log(args, state, control, logs=logs, **kwargs)

                def on_train_end(self, args, state, control, **kwargs):
                    self._cb.on_train_end(args, state, control, **kwargs)

            self._trainer = SFTTrainer(
                model=model,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=lora_config,
                dataset_text_field="text",
                max_seq_length=self._config.max_seq_length,
                tokenizer=self._tokenizer,
                args=training_args,
                callbacks=[HFCallbackWrapper(callback)],
                packing=False,
            )

            self._trainer.train()

            # Update the original model reference to the trained PEFT model
            self._model.__class__ = self._trainer.model.__class__
            self._model.__dict__.update(self._trainer.model.__dict__)

        except Exception as e:
            self._queue.put({"status": "error", "message": str(e), "step": 0})

    def run(self):
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()

    def save_adapter(self, path: str = None):
        if path is None:
            path = self._config.output_dir
        if self._trainer is not None:
            self._trainer.model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            return path
        raise RuntimeError("Trainer not initialized or training not started.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float = None):
        if self._thread:
            self._thread.join(timeout=timeout)

    def get_trainable_param_info(self) -> str:
        """Return trainable parameter count info string."""
        try:
            from peft import get_peft_model
            total = sum(p.numel() for p in self._model.parameters())
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            pct = 100 * trainable / total if total > 0 else 0
            return f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)"
        except Exception:
            return "Parameter info unavailable"
