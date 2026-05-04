# ML Portfolio Demo

A fully runnable Streamlit application demonstrating **QLoRA fine-tuning**, **Multimodal RAG**, **ReAct Agent**, and **Llama 4 Prompt Guard** — all on a consumer-grade **RTX 3050 4 GB** GPU.

Built as a portfolio demo to show end-to-end ML engineering skills: from quantized model training to retrieval-augmented generation and AI safety classification.

---

## Features

| Tab | Feature | GPU Required |
|-----|---------|:------------:|
| ⚡ QLoRA Training | Fine-tune TinyLlama-1.1B in 4-bit NF4 with live loss curve + VRAM monitor | ✓ |
| 🔍 Model Compare | Side-by-side base vs fine-tuned response comparison | ✓ |
| 📚 RAG Pipeline | Multimodal ingestion (Whisper + BLIP) → ChromaDB → Knowledge Graph query | — |
| 🧠 ReAct Agent | Step-by-step Thought / Action / Observation reasoning trace | — |
| 📊 Experiment Results | 5 Plotly charts: loss curves, VRAM comparison, quality radar, perplexity, Pareto frontier | — |
| 📖 Manual | Full Chinese/English user guide embedded in the app | — |
| 🛡️ Prompt Guard | Llama 4 `Llama-Prompt-Guard-2-86M` inference + Full FT vs QLoRA RAM comparison | ✓ |

---

## Hardware

| Item | Spec |
|------|------|
| GPU | NVIDIA RTX 3050 Laptop — **4 GB VRAM** |
| RAM | 15 GB |
| CPU | AMD Ryzen 6C / 12T |
| CUDA | 12.5 (PyTorch cu121) |
| Python | 3.9.12 |
| OS | Windows 11 |

> **Why 4 GB?** Most QLoRA demos assume 24–80 GB VRAM. This project intentionally targets consumer hardware to prove that 4-bit quantization + LoRA makes large-model fine-tuning accessible.

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Training | PyTorch | 2.3.1+cu121 |
| Training | Transformers | 4.44.2 |
| Training | PEFT | 0.12.0 |
| Training | bitsandbytes | 0.43.3 |
| Training | TRL (SFTTrainer) | 0.10.1 |
| RAG | ChromaDB | 0.5.5 |
| RAG | sentence-transformers | 3.0.1 |
| RAG | LangChain | 0.2.16 |
| RAG | NetworkX | 3.x |
| Audio | OpenAI Whisper | 20231117 |
| Image | Salesforce BLIP | via Transformers |
| UI | Streamlit | 1.38.0 |
| Charts | Plotly | 5.24.1 |
| Monitoring | pynvml + psutil | — |

---

## Installation

### Prerequisites
- NVIDIA GPU with CUDA 11.8+ driver
- Python 3.9–3.11
- [ffmpeg](https://ffmpeg.org/download.html) in PATH (for audio transcription)

### Steps

```bash
# 1. Clone
git clone https://github.com/<your-username>/ml-portfolio-demo.git
cd ml-portfolio-demo

# 2. Run install script (creates venv, installs all packages in correct order)
install.bat

# 3. (Optional) Set HuggingFace token for gated models
cp .env.example .env
# Edit .env → HF_TOKEN=hf_xxxxxxxxxxxx
```

> **Why a separate venv?** Installing into Anaconda base causes numpy ABI conflicts with scikit-learn. The install script creates an isolated `venv/` automatically.

### Manual install (if needed)

```bash
python -m venv venv && venv\Scripts\activate

# PyTorch CUDA (install first — sets C++ ABI)
pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# QLoRA stack
pip install bitsandbytes==0.43.3
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1

# RAG stack
pip install chromadb==0.5.5 sentence-transformers==3.0.1
pip install langchain==0.2.16 langchain-community==0.2.16

# App
pip install streamlit==1.38.0 plotly==5.24.1 pynvml==11.5.0 psutil==5.9.8 python-dotenv==1.0.1 networkx

# Audio (needs ffmpeg binary)
pip install openai-whisper==20231117 --no-build-isolation

# Generate pre-computed experiment charts
python utils/generate_experiment_data.py
```

---

## Usage

```bash
# Double-click run.bat, or:
venv\Scripts\activate
streamlit run app.py
```

Open http://localhost:8501

---

## Demo Walkthrough

Suggested order for a live demo or interview:

```
1. 📊 Charts       — Establish why QLoRA matters (VRAM 8× reduction, 99.3% quality retained)
2. ⚡ QLoRA        — Load TinyLlama → Start Training (runs in background thread, live loss chart)
3. 📚 RAG          — While model trains: upload docs, show knowledge graph, run semantic query
4. 🧠 ReAct        — Show step-by-step reasoning trace (works in demo mode without GPU)
5. ⚡ QLoRA        — Training complete → Save Adapter
6. 🔍 Compare      — Base vs fine-tuned side-by-side (the highlight)
7. 🛡️ Prompt Guard — Classify injection attempts + Full FT vs QLoRA RAM comparison chart
```

---

## Project Structure

```
demo_site/
├── app.py                              # Streamlit entry point (7 tabs)
├── requirements.txt
├── install.bat                         # One-click environment setup
├── run.bat                             # One-click app launch
├── .env.example                        # HF_TOKEN placeholder
│
├── core/
│   ├── model_manager.py                # 4-bit model load/unload/generate
│   ├── qlora_trainer.py                # Background-thread QLoRA training + queue
│   ├── rag_pipeline.py                 # Whisper + BLIP + ChromaDB + NetworkX
│   ├── react_agent.py                  # ReAct Thought/Action/Observation loop
│   ├── vram_monitor.py                 # pynvml + psutil background monitor
│   ├── prompt_guard.py                 # Llama-Prompt-Guard-2-86M wrapper
│   └── prompt_guard_trainer.py         # Full FT vs QLoRA training comparison
│
├── tabs/
│   ├── tab_qlora.py                    # Tab 1: live training UI
│   ├── tab_compare.py                  # Tab 2: base vs fine-tuned
│   ├── tab_rag.py                      # Tab 3: multimodal RAG
│   ├── tab_agent.py                    # Tab 4: ReAct agent
│   ├── tab_charts.py                   # Tab 5: experiment charts
│   ├── tab_manual.py                   # Tab 6: user guide
│   └── tab_prompt_guard.py             # Tab 7: Llama 4 Prompt Guard
│
├── utils/
│   ├── chart_builder.py                # Plotly figure factory (5 charts + sparkline)
│   └── generate_experiment_data.py     # One-time script: generate simulated data
│
└── data/
    ├── train_data/
    │   ├── alpaca_tiny.json            # 60 ML instruction-response pairs
    │   └── prompt_guard_dataset.json   # 121 safe/injection classification samples
    ├── rag_docs/knowledge_docs.txt     # QLoRA / RAG / ReAct knowledge base
    └── precomputed/
        └── experiment_results.json     # Simulated experiment data (anchored to papers)
```

---

## Key Design Decisions

**4-bit NF4 quantization (QLoRA)**
Weights stored in NormalFloat4 format — information-theoretically optimal for normally-distributed neural network weights. Reduces a 1.1B model from ~2.2 GB (fp16) to ~560 MB, enabling training on 4 GB VRAM.

**Background thread training**
`QLoRATrainer.run()` spawns a daemon thread so training never blocks the Streamlit event loop. Progress is pushed to a `queue.Queue`; the UI polls every 2 seconds and calls `st.rerun()` to update charts.

**Lazy loading**
All heavy models (Whisper, BLIP, sentence-transformers, Prompt Guard) are loaded on first use, not at startup. This keeps cold-start time under 3 seconds even when no GPU-heavy features are used.

**Simulated experiment data**
Charts for 8B/70B models are pre-computed and anchored to published benchmarks (Dettmers et al. 2023 QLoRA paper, MT-Bench Zheng et al. 2023). All charts are annotated with the data source.

---

## Acceptance Criteria

- [ ] `streamlit run app.py` starts without ImportError
- [ ] Tab 5 charts render without GPU
- [ ] Tab 6 manual renders without GPU
- [ ] Tab 1: Load TinyLlama → VRAM < 700 MB
- [ ] Tab 1: Train 50 steps → loss decreases, no OOM
- [ ] Tab 3: Index knowledge base → ChromaDB count > 0
- [ ] Tab 3: Query returns relevant chunks with scores
- [ ] Tab 4: ReAct demo mode completes 2-step trace
- [ ] Tab 7: Prompt Guard classifies injection examples correctly
- [ ] Tab 7: Full FT and QLoRA training both complete 30 steps
- [ ] Tab 7: QLoRA peak VRAM lower than Full FT peak VRAM

---

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629) — Yao et al., 2022
- [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023
- [Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) — Meta, 2025
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023
