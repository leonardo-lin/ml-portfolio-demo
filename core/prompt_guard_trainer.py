"""
Prompt Guard Trainer — Full FT vs QLoRA comparison.
Trains Llama-Prompt-Guard-2-86M on safe/injection classification dataset.
Records VRAM + system RAM at every step for memory comparison charts.
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
class GuardTrainingConfig:
    mode: str = "qlora"          # "full_ft" | "qlora"
    max_steps: int = 30
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 4
    dataset_path: str = "data/train_data/prompt_guard_dataset.json"
    output_dir: str = "adapters/prompt_guard"


def _get_vram_mb() -> int:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_reserved(0) // (1024 ** 2)


def _get_ram_mb() -> int:
    try:
        import psutil
        return psutil.virtual_memory().used // (1024 ** 2)
    except Exception:
        return 0


class _MemoryCallback:
    """HuggingFace TrainerCallback that records VRAM + RAM each log step."""

    def __init__(self, progress_queue: queue.Queue, mode: str):
        self._queue = progress_queue
        self._mode = mode

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._queue.put({
            "step": state.global_step,
            "loss": round(float(logs.get("loss", 0.0)), 4),
            "vram_mb": _get_vram_mb(),
            "ram_mb": _get_ram_mb(),
            "mode": self._mode,
            "status": "running",
            "ts": time.time(),
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._queue.put({
            "status": "done",
            "step": state.global_step,
            "mode": self._mode,
            "peak_vram_mb": _get_vram_mb(),
            "peak_ram_mb": _get_ram_mb(),
        })


class PromptGuardTrainer:
    """
    Trains Prompt Guard model in either 'full_ft' or 'qlora' mode.
    Runs in background thread; pushes per-step memory records to progress_queue.
    """

    MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

    def __init__(self, config: GuardTrainingConfig, progress_queue: queue.Queue):
        self._config = config
        self._queue = progress_queue
        self._thread: Optional[threading.Thread] = None
        self._trainer = None

    def run(self) -> None:
        self._thread = threading.Thread(target=self._train, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _load_dataset(self):
        import datasets as hf_datasets

        path = self._config.dataset_path
        if not os.path.isabs(path):
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        ds = hf_datasets.Dataset.from_list(raw)
        split = ds.train_test_split(test_size=0.15, seed=42)
        return split["train"], split["test"]

    def _tokenize(self, tokenizer, dataset):
        def _tok(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
            )
        return dataset.map(_tok, batched=True, remove_columns=["text"])

    def _train(self) -> None:
        try:
            import os as _os
            hf_token = _os.environ.get("HF_TOKEN")
            token_kwargs = {"token": hf_token} if hf_token else {}

            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                TrainerCallback,
            )

            self._queue.put({
                "status": "preparing",
                "step": 0,
                "mode": self._config.mode,
                "vram_mb": _get_vram_mb(),
                "ram_mb": _get_ram_mb(),
            })

            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, **token_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_ds, eval_ds = self._load_dataset()
            train_ds = self._tokenize(tokenizer, train_ds)
            eval_ds = self._tokenize(tokenizer, eval_ds)

            # ── Load model ───────────────────────────────────────────────────
            if self._config.mode == "qlora":
                from transformers import BitsAndBytesConfig
                from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    **token_kwargs,
                )
                model = prepare_model_for_kbit_training(model)
                lora_cfg = LoraConfig(
                    r=self._config.lora_r,
                    lora_alpha=self._config.lora_alpha,
                    lora_dropout=self._config.lora_dropout,
                    task_type=TaskType.SEQ_CLS,
                    # RoBERTa-style attention projection names
                    target_modules=["query", "value"],
                )
                model = get_peft_model(model, lora_cfg)
                optim = "paged_adamw_8bit"
                fp16 = True
                bf16 = False

            else:  # full_ft
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **token_kwargs,
                )
                optim = "adamw_torch"
                fp16 = True
                bf16 = False

            # ── Training arguments ───────────────────────────────────────────
            training_args = TrainingArguments(
                output_dir=self._config.output_dir,
                per_device_train_batch_size=self._config.per_device_train_batch_size,
                max_steps=self._config.max_steps,
                learning_rate=self._config.learning_rate,
                fp16=fp16,
                bf16=bf16,
                optim=optim,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                dataloader_num_workers=0,
                remove_unused_columns=False,
                label_names=["labels"],
            )

            # ── Callback wrapper ─────────────────────────────────────────────
            mem_callback_instance = _MemoryCallback(self._queue, self._config.mode)

            class _HFWrapper(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    mem_callback_instance.on_log(args, state, control, logs=logs)

                def on_train_end(self, args, state, control, **kwargs):
                    mem_callback_instance.on_train_end(args, state, control)

            self._trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=[_HFWrapper()],
            )
            self._trainer.train()

        except Exception as e:
            self._queue.put({
                "status": "error",
                "message": str(e),
                "mode": self._config.mode,
                "step": 0,
            })
