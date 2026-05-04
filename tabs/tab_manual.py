"""
Tab 6: User Manual
"""

import streamlit as st


def render():
    st.header("User Manual")
    st.caption("QLoRA ML Portfolio Demo Site — Complete Operating Guide")

    # ── System Requirements ──────────────────────────────────────────────────
    with st.expander("1. System Requirements", expanded=False):
        st.markdown("""
| Item | Minimum Requirement | This Machine |
|------|---------|---------|
| GPU | 4GB VRAM (NVIDIA) | RTX 3050 4GB ✓ |
| RAM | 8GB | 15GB ✓ |
| Disk Space | 10GB (model cache) | To be confirmed |
| Python | 3.9–3.11 | 3.9.12 ✓ |
| CUDA Driver | 11.8+ | 12.5 ✓ |
| OS | Windows 10/11 | Windows 11 ✓ |
""")

    # ── Installation Steps ───────────────────────────────────────────────────
    with st.expander("2. Installation Steps (First-time Setup)", expanded=False):
        st.markdown("**2.1 Prerequisites** — Confirm `ffmpeg` is installed (required for audio features):")
        st.code("ffmpeg -version", language="bash")

        st.markdown("**2.2 Create a virtual environment and install packages** (use `venv` to avoid polluting Anaconda):")
        st.code("""cd "C:\\path\\to\\ml-portfolio-demo"

# Create isolated environment (run once)
python -m venv venv
venv\\Scripts\\activate

# PyTorch CUDA (most important, must be installed first, ~2GB)
pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# QLoRA packages
pip install bitsandbytes==0.43.3
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1 einops==0.8.0

# RAG packages
pip install chromadb==0.5.5 sentence-transformers==3.0.1
pip install langchain==0.2.16 langchain-community==0.2.16

# UI packages
pip install streamlit==1.38.0 plotly==5.24.1 pynvml==11.5.0 psutil==5.9.8 python-dotenv==1.0.1 networkx

# Audio package (requires ffmpeg binary)
pip install openai-whisper==20231117""", language="bash")

        st.markdown("**2.3 Generate experiment data (run once):**")
        st.code("python utils\\generate_experiment_data.py", language="bash")

    # ── Launch ────────────────────────────────────────────────────────────────
    with st.expander("3. Launch", expanded=False):
        st.code("""cd "C:\\path\\to\\ml-portfolio-demo"
venv\\Scripts\\activate
streamlit run app.py""", language="bash")
        st.info("Your browser should open automatically, or visit **http://localhost:8501** manually.")

    # ── Tab Guide ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("How to Use Each Tab")

    tab_info = {
        "⚡ Tab 1 — QLoRA Training (Core Feature)": {
            "desc": "Run real QLoRA fine-tuning on an RTX 3050 and monitor live loss and VRAM usage.",
            "steps": [
                "Select a model: TinyLlama-1.1B-Chat (recommended) or LLaMA 3.2 1B (requires HF token).",
                "Set parameters: Max Steps=50, Learning Rate=2e-4, LoRA Rank=8.",
                "Click [Load Model] and wait 30-60 seconds (first download is about 2.2GB).",
                "Sidebar shows VRAM around 560MB, meaning 4-bit quantization is loaded correctly.",
                "Click [Start Training]. The Loss Chart updates every 2 seconds, with VRAM around 1750MB.",
                "After training, click [Save Adapter] to save at `adapters/qlora_checkpoint/`.",
            ],
            "note": "If OOM occurs: reduce Max Steps to 20, reload the model, and try again.",
        },
        "🔍 Tab 2 — Model Compare (Result Validation)": {
            "desc": "Compare base and fine-tuned responses side by side. Prerequisite: finish training and save adapter in Tab 1.",
            "steps": [
                "Choose a sample question from the dropdown or enter your own prompt.",
                "Adjust Max New Tokens (100-400).",
                "Click [Generate Comparison].",
                "Left panel: Base Model response / Right panel: Fine-tuned response.",
                "Bottom metrics are auto-calculated: Response Length / Vocabulary Diversity / Avg Word Length.",
            ],
            "note": "Fine-tuned responses on ML topics should usually be more detailed and use more precise terminology.",
        },
        "📚 Tab 3 — RAG Pipeline (Multimodal Knowledge Base)": {
            "desc": "Demonstrates multimodal processing and hybrid retrieval (vector search + knowledge graph). Works without GPU.",
            "steps": [
                "Upload data (optional): audio -> Whisper transcription / image -> BLIP caption / text -> automatic chunking.",
                "Check [Also index built-in knowledge base] to include built-in QLoRA/RAG/ML documents.",
                "Click [Process & Index] until you see 'Index ready: XX chunks'.",
                "Explore the interactive knowledge graph (zoom + hover to inspect node relations).",
                "Enter a question and click [Run RAG Query].",
                "Review retrieved chunks (with similarity scores), graph-expanded context, and pipeline trace.",
            ],
            "note": "Embedding models run on CPU, so no GPU is required.",
        },
        "🧠 Tab 4 — ReAct Agent (Reasoning Trace)": {
            "desc": "Shows step-by-step reasoning flow (Thought -> Action -> Observation).",
            "steps": [
                "No model loaded = Demo Mode (prebuilt trace) / model loaded = Live Mode (real reasoning).",
                "Choose a sample question or input your own question.",
                "Set Max Reasoning Steps (2-8).",
                "Click [Run Agent].",
                "Blue = Thought / Orange = Action / Green = Observation / Bottom = Final Answer.",
            ],
            "note": "Available tools: `search_rag` / `calculate` / `lookup_graph`.",
        },
        "🛡️ Tab 7 — Prompt Guard (Llama 4 Safety Classifier)": {
            "desc": "Uses Llama-Prompt-Guard-2-86M (Llama family) to detect prompt injection/jailbreak and compare memory usage between Full FT and QLoRA.",
            "steps": [
                "Set `HF_TOKEN` in `.env` (required for Meta gated model) and restart the app.",
                "Accept access terms at `huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M`.",
                "Click [Load Model (fp16)] to download (~170MB) and display VRAM usage.",
                "Use Safe/Injection examples on the left, or input custom text, then click [Classify].",
                "Check SAFE/INJECTION label and confidence score.",
                "Click [Run Full Fine-Tuning] for 30 steps and log VRAM + RAM.",
                "Click [Run QLoRA] for another 30 steps to overlay a second curve.",
                "Review peak-memory table and bar chart: QLoRA VRAM should be about 75% lower than Full FT.",
            ],
            "note": "QLoRA uses 4-bit NF4 quantization + LoRA (`task_type=SEQ_CLS`) with `paged_adamw_8bit` optimizer.",
        },
        "📊 Tab 5 — Experiment Results (Charts)": {
            "desc": "Shows large-scale comparison charts for QLoRA vs Full Fine-Tuning. Always available without GPU.",
            "steps": [
                "A. Training Loss Curves - QLoRA 8B / 70B / Full FT curves (500 steps).",
                "B. VRAM Requirements - GPU memory needs across model scales and methods.",
                "C. Quality Radar - 5-dimension response quality radar chart (anchored to MT-Bench).",
                "D. Perplexity vs Steps - line chart of training progress vs perplexity.",
                "E. Time-Quality Tradeoff - training time vs quality Pareto frontier.",
            ],
            "note": "If Tab 1 training has run, chart A automatically overlays a gold 'Live Demo' curve.",
        },
    }

    for title, info in tab_info.items():
        with st.expander(title, expanded=False):
            st.markdown(f"**Purpose:** {info['desc']}")
            st.markdown("**Steps:**")
            for i, step in enumerate(info["steps"], 1):
                st.markdown(f"{i}. {step}")
            if info.get("note"):
                st.warning(f"Note: {info['note']}")

    # ── Suggested Demo Sequence ───────────────────────────────────────────────
    st.divider()
    st.subheader("Recommended Demo Sequence")
    demo_steps = [
        ("1", "📊 Charts", "Build context first: why QLoRA matters."),
        ("2", "⚡ QLoRA", "Click Load Model + Start Training while explaining 4-bit quantization."),
        ("3", "📚 RAG", "While training runs in background, show multimodal upload and knowledge graph."),
        ("4", "🧠 ReAct", "Show reasoning trace and emphasize chain-of-thought style execution."),
        ("5", "⚡ QLoRA", "Return to confirm training completion and save adapter."),
        ("6", "🔍 Compare", "Base vs Fine-tuned side-by-side comparison as the highlight."),
    ]
    cols = st.columns(6)
    for col, (num, tab, desc) in zip(cols, demo_steps):
        with col:
            st.markdown(
                f'<div style="background:#0d1b2a;padding:10px;border-radius:8px;'
                f'text-align:center;border:1px solid #00d4ff">'
                f'<b style="color:#00d4ff">{num}</b><br>{tab}<br>'
                f'<small style="color:#aaa">{desc}</small></div>',
                unsafe_allow_html=True,
            )

    # ── Common Troubleshooting ────────────────────────────────────────────────
    st.divider()
    st.subheader("Common Troubleshooting")
    qa = [
        ("OOM (Out of Memory)", "Insufficient VRAM", "Reduce Max Steps, restart, then try again."),
        ("Model download failed", "HuggingFace token is required", "Set `HF_TOKEN=hf_xxx` in `.env`."),
        ("cp950 encoding error", "Using system Python instead of venv", "Make sure `venv\\Scripts\\activate` is used."),
        ("sentence-transformers import failed", "NumPy ABI conflict (common in Anaconda base env)", "Use the project `venv`; do not run in Anaconda base environment."),
        ("Audio cannot be processed", "Missing ffmpeg binary", "Install ffmpeg and add it to system PATH."),
        ("Charts are blank", "Experiment data not generated", "Run `python utils\\generate_experiment_data.py`."),
    ]
    for problem, cause, fix in qa:
        with st.expander(f"❓ {problem}"):
            st.markdown(f"**Cause:** {cause}")
            st.markdown(f"**Fix:** {fix}")
