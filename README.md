# ML Portfolio Demo

可在本機實際執行的 **Streamlit** 作品集網站，展示 **QLoRA 微調**、**多模態 RAG**、**ReAct 推理代理**，以及 **Llama Prompt Guard** 等流程；目標硬體為消費級 **RTX 3050 4 GB VRAM**，證明小顯存也能跑通端到端 ML 工程。

---

## 這個專案在做什麼？

這是一份「可點、可看、可跑」的互動 Demo，把論文裡常見的概念（量化 + LoRA、RAG、ReAct）收成 **7 個分頁**，從訓練、檢索、代理推理到安全分類，都在同一個 App 裡完成。你不需要先讀完整程式碼，就能依序體驗：**為什麼 QLoRA 省 VRAM → 實際怎麼訓練 → 訓練完怎麼比對 → 知識庫與圖譜怎麼輔助回答 → ReAct 怎麼一步步查資料 → 圖表與手冊怎麼補齊敘事**。

---

## 你可以做到什麼？（功能一覽）

| 分頁 | 你能做什麼 | 是否需要 GPU |
|------|------------|:-------------:|
| QLoRA Training | 以 4-bit NF4 微調 TinyLlama-1.1B，即時看 **Loss 曲線** 與 **VRAM 監控**、調整步數／學習率／LoRA rank | ✓ |
| Model Compare | **基底模型 vs 微調後** 並排比對回答品質 | ✓ |
| RAG Pipeline | 多模態入庫（Whisper + BLIP）→ ChromaDB → 可查詢；並可檢視 **知識圖譜**（實體與關係） | 視功能而定 |
| ReAct Agent | 自訂或範例問題，觀看 **Thought → Action（如 search_rag）→ Observation** 的逐步 trace | 可 Demo 模式 |
| Experiment Results | **5 張 Plotly 圖**（含模擬的 8B／70B 對照），說明 VRAM、Loss、品質雷達等 | — |
| Manual | 內建中英使用說明 | — |
| Prompt Guard | Llama **Prompt Guard** 推論、安全／注入分類，以及 Full FT vs QLoRA 等對照 | ✓ |

> **實驗圖表**：8B／70B 等數值為 **預先計算、對齊文獻趨勢的模擬資料**（圖上會標註 Dettmers et al. 2023、Zheng et al. 2023）；**本機 Live 訓練** 以 TinyLlama 1.1B 為主，與圖中「Live」標記一致。

---

## 介面預覽（`demo_image/`）

以下截圖對應實際分頁與圖表，方便快速理解「打開網頁後會看到什麼」。

### 主畫面：QLoRA 訓練與即時監控

側欄顯示 GPU 記憶體與環境資訊；主區可選基底模型、調整 **Max Steps／Learning Rate／LoRA r**，並即時更新 **Loss** 與 **VRAM**。

![QLoRA 訓練主介面與即時 Loss／VRAM](demo_image/main%20page.png)

### 實驗結果：訓練 Loss 與 VRAM 需求（含模擬曲線）

與論文趨勢對齊的對照圖：QLoRA vs Full FT、不同參數量級，以及本專案 **Live Demo（1.1B）** 的定位。

![訓練 Loss 曲線對照](demo_image/training%20loss%20curves.png)

![VRAM 需求柱狀圖對照](demo_image/VRAM%20requirement.png)

### 實驗結果：回應品質雷達圖與關鍵洞察

從 Coherence、Instruction Following、Relevance、Fluency、Factuality 等面向比較 **Base／QLoRA／Full FT（模擬）**；並以摘要數字強調 **VRAM 約 8× 節省** 與 **品質保留約 99.3%** 等敘事（數值來自圖表預設敘事，仍請以圖上 footnote 為準）。

![回應品質雷達圖](demo_image/Response%20quality.png)

![關鍵洞察（VRAM 節省與品質保留）](demo_image/key%20insight.png)

### RAG：知識圖譜視覺化

將文件中的概念抽成節點與邊，可依 **Degree（連線數）** 著色，協助理解知識庫結構與 RAG 檢索上下文。

![知識圖譜（實體與關係）](demo_image/knowledge%20graph.png)

### ReAct Agent：Thought / Action / Observation

可輸入自訂問題、調整最大推理步數；每一步顯示 **思考、工具動作（例如 search_rag）、觀察結果**，示範代理如何依知識庫回答技術問題。

![ReAct 代理逐步推理範例](demo_image/ReAct_example.png)

---

## Features（英文摘要）

| Tab | Feature | GPU Required |
|-----|---------|:------------:|
| QLoRA Training | Fine-tune TinyLlama-1.1B in 4-bit NF4 with live loss curve + VRAM monitor | ✓ |
| Model Compare | Side-by-side base vs fine-tuned response comparison | ✓ |
| RAG Pipeline | Multimodal ingestion (Whisper + BLIP) → ChromaDB → Knowledge Graph query | — |
| ReAct Agent | Step-by-step Thought / Action / Observation reasoning trace | — |
| Experiment Results | 5 Plotly charts (loss, VRAM, quality radar, perplexity, Pareto) | — |
| Manual | Full Chinese/English user guide embedded in the app | — |
| Prompt Guard | Llama Prompt Guard inference + Full FT vs QLoRA RAM comparison | ✓ |

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

> **Why 4 GB?** 多數 QLoRA 示範假設 24–80 GB VRAM。本專案刻意鎖定消費級顯卡，展示 **4-bit 量化 + LoRA** 如何讓「在本機微調」變得可行。

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

建議示範或面試時的瀏覽順序：

```
1. Experiment Results — 先建立「為什麼 QLoRA」的直觀（VRAM、Loss、品質雷達）
2. QLoRA Training     — 載入 TinyLlama → 開始訓練（背景執行緒 + 即時 Loss）
3. RAG Pipeline       — 訓練進行中可處理文件、看知識圖譜、跑語意查詢
4. ReAct Agent        — 展示 Thought / Action / Observation（可用 Demo 模式）
5. QLoRA Training     — 訓練完成 → 儲存 Adapter
6. Model Compare      — 基底 vs 微調並排（亮點）
7. Prompt Guard       — 注入範例分類 + Full FT vs QLoRA 對照圖表
```

---

## Project Structure

```
ml-portfolio-demo/
├── app.py                              # Streamlit 入口（多分頁）
├── requirements.txt
├── install.bat                         # 一鍵建立環境
├── run.bat                             # 一鍵啟動
├── demo_image/                         # README 與說明用截圖
├── .env.example                        # HF_TOKEN 範本
│
├── core/
│   ├── model_manager.py                # 4-bit 載入／卸載／生成
│   ├── qlora_trainer.py                # 背景執行緒 QLoRA 訓練 + queue
│   ├── rag_pipeline.py                 # Whisper + BLIP + ChromaDB + NetworkX
│   ├── react_agent.py                  # ReAct Thought/Action/Observation
│   ├── vram_monitor.py                 # pynvml + psutil 背景監控
│   ├── prompt_guard.py                 # Prompt Guard 包裝
│   └── prompt_guard_trainer.py         # Full FT vs QLoRA 訓練對照
│
├── tabs/
│   ├── tab_qlora.py
│   ├── tab_compare.py
│   ├── tab_rag.py
│   ├── tab_agent.py
│   ├── tab_charts.py
│   ├── tab_manual.py
│   └── tab_prompt_guard.py
│
├── utils/
│   ├── chart_builder.py
│   └── generate_experiment_data.py
│
└── data/
    ├── train_data/
    │   ├── alpaca_tiny.json
    │   └── prompt_guard_dataset.json
    ├── rag_docs/knowledge_docs.txt
    └── precomputed/
        └── experiment_results.json
```

---

## Key Design Decisions

**4-bit NF4 quantization (QLoRA)**  
權重以 NormalFloat4 儲存，大幅壓縮顯存占用，使 4 GB VRAM 上仍能進行微調相關示範。

**Background thread training**  
`QLoRATrainer.run()` 以背景執行緒執行，避免阻塞 Streamlit 事件迴圈；進度經 `queue.Queue` 傳回 UI 定期刷新。

**Lazy loading**  
Whisper、BLIP、sentence-transformers、Prompt Guard 等皆在首次使用時載入，降低冷啟動時間。

**Simulated experiment data**  
8B／70B 等圖表數據為預先計算，並在圖上標註文獻依據（Dettmers et al. 2023、Zheng et al. 2023）。

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
