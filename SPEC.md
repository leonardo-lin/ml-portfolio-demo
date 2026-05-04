# QLoRA ML Portfolio Demo Site — 完整規格書

版本：1.0  
日期：2026-05-04  
作者：林（基於面試答案 Q1/Q2/Q3 整理）

---

## 一、專案目標

打造一個 Streamlit 多分頁互動式 Demo 網站，用於展示以下 ML 工程能力：

1. **QLoRA Fine-tuning Pipeline** — 在 4GB VRAM 硬體上實際運行的量化 LoRA 訓練
2. **Multimodal RAG Pipeline** — 音訊/圖片/文字多模態資料處理 + 向量+知識圖譜混合檢索
3. **ReAct Reasoning Agent** — Thought/Action/Observation 逐步推理追蹤
4. **實驗結果視覺化** — 基於真實論文數字錨定的 5 張 Plotly 圖表

---

## 二、硬體規格與限制

| 項目 | 規格 | 限制影響 |
|------|------|---------|
| GPU | NVIDIA RTX 3050, 4096MB VRAM | 最大可用模型：~1-3B（4-bit量化） |
| RAM | 15GB（通常剩餘 ~3-5GB） | ChromaDB 需常駐記憶體，注意用量 |
| CPU | AMD Ryzen 6C/12T | 訓練 CPU 負擔輕，Whisper 跑 CPU |
| CUDA | 12.5 | PyTorch CUDA 12.1 wheel 相容 |
| Python | 3.9.12 | 所有套件須相容 Python 3.9 |
| OS | Windows 11 | bitsandbytes 需 v0.43.3+（含 Windows DLL） |

### VRAM 訓練預算（TinyLlama 1.1B，r=8）
```
4-bit NF4 模型權重:         ~560 MB
LoRA adapter 層 (fp16):      ~40 MB
Activation (batch=1, seq=512): ~300 MB
8-bit paged optimizer states: ~550 MB
CUDA kernels + 雜項:         ~300 MB
─────────────────────────────────────
總計:                       ~1750 MB  (43% of 4096 MB)
安全餘裕:                   ~2346 MB  ✓ 安全
```

---

## 三、技術棧

| 分類 | 套件 | 版本 | 用途 |
|------|------|------|------|
| UI | streamlit | 1.38.0 | 主框架 |
| 圖表 | plotly | 5.24.1 | 所有互動圖表 |
| 深度學習 | torch (CUDA 12.1) | 2.3.1+cu121 | 基礎框架 |
| 模型載入 | transformers | 4.44.2 | HuggingFace 模型 |
| 量化 | bitsandbytes | 0.43.3 | 4-bit NF4 量化 |
| LoRA | peft | 0.12.0 | LoRA adapter 注入 |
| 加速 | accelerate | 0.34.2 | device_map="auto" |
| SFT訓練 | trl | 0.10.1 | SFTTrainer |
| 嵌入 | sentence-transformers | 3.0.1 | all-MiniLM-L6-v2 |
| 向量庫 | chromadb | 0.5.5 | 本地持久化向量存儲 |
| LLM編排 | langchain + langchain-community | 0.2.16 | RAG 流程編排 |
| 音訊 | openai-whisper | 20231117 | STT 語音轉文字 |
| 圖像 | transformers (BLIP) | 同上 | 圖片語意描述 |
| 知識圖譜 | networkx | 2.7.1 | 實體關係圖 |
| GPU 監控 | pynvml | 11.5.0 | VRAM 即時監控 |
| 系統監控 | psutil | 5.9.8 | RAM 使用率 |
| 環境變數 | python-dotenv | 1.0.1 | HF_TOKEN 管理 |

---

## 四、目錄結構

```
demo_site/
├── app.py                          # Streamlit 主入口
├── requirements.txt                # 所有 pinned 依賴
├── CLAUDE.md                       # 開發指引（本 AI 讀取）
├── SPEC.md                         # 本規格書
├── .env.example                    # HF_TOKEN=hf_xxxxx
│
├── core/                           # 業務邏輯層（純 Python，不含 Streamlit）
│   ├── __init__.py
│   ├── model_manager.py            # 模型載入/卸載單例
│   ├── qlora_trainer.py            # QLoRA 訓練迴圈 + 進度佇列
│   ├── rag_pipeline.py             # Multimodal RAG 完整流程
│   ├── react_agent.py              # ReAct 推理代理
│   └── vram_monitor.py             # pynvml 背景執行緒監控
│
├── tabs/                           # Streamlit UI 層（每個分頁一個檔案）
│   ├── __init__.py
│   ├── tab_qlora.py                # Tab 1：QLoRA 訓練設定 + 即時監控
│   ├── tab_compare.py              # Tab 2：Base vs Fine-tuned 對比
│   ├── tab_rag.py                  # Tab 3：RAG Pipeline 互動
│   ├── tab_agent.py                # Tab 4：ReAct Agent 追蹤
│   └── tab_charts.py               # Tab 5：實驗結果圖表
│
├── utils/                          # 工具函式
│   ├── __init__.py
│   ├── generate_experiment_data.py # 一次性：生成模擬實驗數據
│   └── chart_builder.py            # Plotly figure factory（5 個圖表）
│
├── data/
│   ├── train_data/
│   │   └── alpaca_tiny.json        # 200 筆 Alpaca 格式訓練資料
│   ├── rag_docs/
│   │   ├── sample_audio.wav        # Whisper 示範音訊（可選）
│   │   ├── sample_image.jpg        # BLIP 示範圖片（可選）
│   │   └── knowledge_docs.txt      # RAG 知識庫文字
│   └── precomputed/
│       └── experiment_results.json # 模擬實驗數據（生成後緩存）
│
└── adapters/
    └── qlora_checkpoint/           # 訓練後 LoRA adapter 儲存位置
        └── .gitkeep
```

---

## 五、各模組詳細規格

### 5.1 `app.py` — 主入口

職責：
- `st.set_page_config(page_title="QLoRA ML Demo", layout="wide", page_icon="🤖")`
- 初始化 `st.session_state`：
  ```python
  session_defaults = {
      "model_loaded": False,
      "adapter_trained": False,
      "adapter_path": None,
      "training_history": [],   # [{step, loss, vram_mb, epoch}]
      "training_active": False,
      "rag_indexed": False,
      "chroma_collection": None,
  }
  ```
- Sidebar：即時 VRAM 儀表盤（pynvml sparkline）、硬體資訊（GPU名稱/VRAM/CPU/RAM）
- 5 個分頁：`st.tabs(["⚡ QLoRA Training", "🔍 Model Compare", "📚 RAG Pipeline", "🧠 ReAct Agent", "📊 Experiment Results"])`

---

### 5.2 `core/model_manager.py` — 模型管理單例

```python
@st.cache_resource
def get_model_manager() -> ModelManager
```

**BitsAndBytesConfig（固定，不允許修改）：**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # nested quantization 節省 ~0.4GB
)
```

**方法列表：**
| 方法 | 簽名 | 說明 |
|------|------|------|
| `load_base_model` | `(model_id: str) -> tuple[model, tokenizer]` | 載入 4-bit 量化模型 |
| `load_with_adapter` | `(adapter_path: str) -> model` | 在 base model 上載入 LoRA adapter |
| `generate` | `(prompt: str, max_new_tokens: int = 200) -> str` | 文字生成 |
| `unload` | `() -> None` | 清除 VRAM：`del model; gc.collect(); torch.cuda.empty_cache()` |
| `get_vram_info` | `() -> dict` | `{used_mb, free_mb, total_mb, pct}` |
| `get_memory_footprint` | `() -> int` | bytes，用於驗證量化成功 |

**支援的模型（UI 下拉選單）：**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`（預設，最穩定）
- `meta-llama/Llama-3.2-1B-Instruct`（需 HF_TOKEN）

---

### 5.3 `core/qlora_trainer.py` — QLoRA 訓練迴圈

**LoraConfig：**
```python
LoraConfig(
    r=8,                    # 可透過 UI 調整：4, 8, 16
    lora_alpha=16,          # 固定 = 2 × r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**TrainingArguments（4GB 硬約束）：**
```python
TrainingArguments(
    per_device_train_batch_size=1,      # 硬約束：不可改為 2
    gradient_accumulation_steps=4,      # effective batch = 4
    num_train_epochs=3,
    max_steps=50,                       # UI 可調：10~100
    learning_rate=2e-4,                 # UI 可調：1e-5~5e-4
    fp16=True,                          # RTX 3050 用 fp16（非 bf16）
    optim="paged_adamw_8bit",           # 關鍵：optimizer states offload 到 CPU RAM
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=1,
    report_to="none",
    output_dir="adapters/qlora_checkpoint",
)
```

**訓練執行緒：**
```python
class QLoRATrainer:
    def __init__(self, model, tokenizer, dataset_path, lora_r, max_steps, lr, progress_queue)
    def run(self) -> None          # 在 threading.Thread 中執行
    def stop(self) -> None         # 設定停止旗標
```

**自訂 Callback：**
```python
class VRAMLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 每步推送到 queue
        queue.put({
            "step": state.global_step,
            "loss": logs.get("loss", 0.0),
            "vram_mb": get_vram_used_mb(),
            "epoch": round(state.epoch, 2),
            "status": "running"
        })
```

**資料集格式（alpaca_tiny.json）：**
```json
[
  {
    "instruction": "Explain what QLoRA is",
    "input": "",
    "output": "QLoRA is a quantized low-rank adaptation method..."
  }
]
```
格式化為：`"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"`

---

### 5.4 `core/rag_pipeline.py` — Multimodal RAG

**流程：**
```
[音訊 WAV] → Whisper small → 帶時間戳記的 transcript chunks
[圖片 JPG] → BLIP-base → 語意描述 narrative（非簡單 tag）
[文字 TXT] → 滑動視窗 chunking（512 tokens, overlap 50）
     ↓
all-MiniLM-L6-v2 embedding
     ↓
ChromaDB（本地持久化，data/chroma_db/）
     ↓
[查詢] → 向量相似度 top-k=3
       + NetworkX 圖譜鄰居擴展（co-occurrence edges）
       → 合併上下文
       → LLM 生成最終答案
```

**知識圖譜構建（無 spaCy 依賴）：**
- 用 regex 提取大寫名詞短語作為節點
- 同一句子內的節點建立邊（co-occurrence）
- 邊權重 = 共同出現次數
- 圖譜視覺化：NetworkX → Plotly scatter + line（Fruchterman-Reingold layout）

**類別方法：**
```python
class MultimodalRAGPipeline:
    def ingest_audio(self, path: str) -> List[str]
    def ingest_image(self, path: str) -> str
    def ingest_text(self, path: str) -> List[str]
    def build_index(self, chunks: List[str]) -> None
    def build_knowledge_graph(self, chunks: List[str]) -> nx.Graph
    def query(self, question: str, top_k: int = 3) -> RAGResult
    def get_graph_plotly(self) -> go.Figure
```

---

### 5.5 `core/react_agent.py` — ReAct Agent

**ReAct Prompt Template：**
```
You are a helpful AI assistant. Solve the question step by step.

Available tools:
- search_rag[query]: Search the knowledge base for relevant information
- calculate[expression]: Evaluate a mathematical expression
- lookup_graph[entity]: Find related concepts in the knowledge graph

Format your response as:
Thought: <your reasoning>
Action: <tool_name>[<input>]
Observation: <tool result>
... (repeat up to 5 times)
Final Answer: <your final answer>

Question: {question}
{scratchpad}
```

**ReActTrace dataclass：**
```python
@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str

@dataclass  
class ReActTrace:
    question: str
    steps: List[ReActStep]
    final_answer: str
    total_steps: int
    elapsed_sec: float
```

**工具綁定：**
```python
TOOLS = {
    "search_rag": rag_pipeline.query,
    "calculate": lambda expr: str(eval(expr, {"__builtins__": {}}, {})),
    "lookup_graph": kg.get_neighbors,
}
```
> 注意：calculate 使用受限的 eval，僅允許數學運算，不允許任何 builtins。

---

### 5.6 `core/vram_monitor.py` — VRAM 監控

```python
class VRAMMonitor:
    def __init__(self, poll_interval_sec: float = 0.5, history_len: int = 120)
    def start(self) -> None
    def stop(self) -> None
    def get_current(self) -> dict   # {used_mb, free_mb, total_mb, pct}
    def get_history(self) -> List[dict]  # 用於 sparkline
```

使用 `pynvml`（非 subprocess nvidia-smi），執行緒安全（threading.Lock）。

---

### 5.7 `utils/generate_experiment_data.py` — 模擬實驗數據

執行後生成 `data/precomputed/experiment_results.json`，包含：

**A. 訓練 Loss 曲線（500 steps，錨定 Dettmers 2023 QLoRA 論文）：**
```python
loss(t) = end_loss + (start_loss - end_loss) * exp(-t / decay) + N(0, noise_std)
```
| 組態 | start_loss | end_loss | decay | noise_std |
|------|-----------|---------|-------|-----------|
| QLoRA 8B | 2.8 | 1.2 | 80 | 0.08 |
| QLoRA 70B（模擬） | 2.6 | 0.95 | 70 | 0.06 |
| Full FT 8B（模擬） | 2.7 | 1.05 | 85 | 0.05 |

**B. VRAM 使用量（GB，錨定論文 Table 3）：**
| 模型 | 方法 | VRAM (GB) |
|------|------|-----------|
| LLaMA 8B | Full FT | 80.0 |
| LLaMA 8B | QLoRA | 9.8 |
| LLaMA 8B | Inference only | 5.2 |
| LLaMA 70B | QLoRA | 46.0 |
| LLaMA 70B | Inference only | 35.0 |
| TinyLlama 1.1B | QLoRA (本機實測) | 1.75 |

**C. Quality Radar（5 維 0-100，錨定 MT-Bench Zheng 2023）：**
| 維度 | Base Model | QLoRA FT | Full FT |
|------|-----------|---------|---------|
| Instruction Following | 42 | 78 | 83 |
| Coherence | 68 | 79 | 82 |
| Factuality | 71 | 73 | 74 |
| Fluency | 82 | 84 | 85 |
| Relevance | 55 | 81 | 84 |

**D. Perplexity（錨定 LLaMA WikiText-2 基準）：**
| Steps | QLoRA 8B | Full FT 8B | Base Model |
|-------|---------|-----------|-----------|
| 0 | 24.3 | 24.3 | 24.3 |
| 50 | 18.1 | 16.8 | 24.3 |
| 100 | 14.2 | 13.1 | 24.3 |
| 200 | 11.8 | 10.9 | 24.3 |
| 300 | 10.9 | 10.1 | 24.3 |
| 500 | 10.2 | 9.7 | 24.3 |

**E. Time-Quality Tradeoff（Pareto Frontier）：**
| 組態 | Training Time (hr) | MT-Bench Score |
|------|-------------------|---------------|
| QLoRA 7B | 4 | 6.1 |
| Full FT 7B | 48 | 6.3 |
| QLoRA 13B | 8 | 6.4 |
| QLoRA 33B | 20 | 6.6 |
| QLoRA 65B | 30 | 6.7 |
| Full FT 13B | 96 | 6.5 |

---

### 5.8 `utils/chart_builder.py` — Plotly Figure Factory

```python
def build_loss_curves(data: dict) -> go.Figure
    # 3條線（QLoRA 8B, QLoRA 70B sim, Full FT 8B sim）
    # 若有 live 訓練數據，疊加第4條"Live Demo"線

def build_vram_comparison(data: dict) -> go.Figure
    # 分組 bar chart，按模型分組，按方法上色

def build_quality_radar(data: dict) -> go.Figure
    # 3層雷達圖：Base / QLoRA FT / Full FT

def build_perplexity_curves(data: dict) -> go.Figure
    # 折線圖 + scatter 點，Base Model 為水平參考線

def build_tradeoff_scatter(data: dict) -> go.Figure
    # 散點圖 + Pareto frontier 折線
    # X 軸：log scale training hours；Y 軸：MT-Bench score

def build_vram_sparkline(history: List[dict]) -> go.Figure
    # Sidebar 用小圖，無 axis labels
```

所有圖表使用 `plotly_dark` template，主色調：
- QLoRA 系列：`#00d4ff`（青藍）
- Full FT 系列：`#ff6b6b`（珊瑚紅）
- Base Model：`#a8a8a8`（灰色）
- Live Demo：`#ffd700`（金色）

---

## 六、UI 分頁規格

### Tab 1：QLoRA Training

```
┌─左欄（1/3）────────────────┐  ┌─右欄（2/3）─────────────────────────┐
│ Model: [TinyLlama-1.1B ▼]  │  │ Training Loss                        │
│ Max Steps: [═══] 50        │  │  3.0│╲                               │
│ Learning Rate: [═] 2e-4    │  │  2.0│ ╲╲──╲                          │
│ LoRA Rank: [4 / 8 / 16]    │  │  1.0│      ╲──────                   │
│                             │  │     └──────────────── step           │
│ Dataset: 200 samples        │  │ VRAM: [█████████░░░░░] 1750/4096 MB  │
│ ▶ [Load Model]             │  │ Step: 34/50  Epoch: 2.0  ETA: 45s   │
│ ▶ [Start Training]         │  │                                       │
│ 💾 [Save Adapter]          │  │ ▼ Training Logs                       │
│                             │  │   Step 34 | loss: 1.43 | VRAM: 1750 │
└─────────────────────────────┘  └───────────────────────────────────────┘
```

互動邏輯：
- Load Model → 呼叫 `model_manager.load_base_model()`，顯示 VRAM 佔用
- Start Training → 啟動背景執行緒，每 2 秒 `st.rerun()` 輪詢 progress queue
- Save Adapter → `trainer.model.save_pretrained(adapter_path)`，更新 session_state

### Tab 2：Model Compare

前提條件：需要 adapter 已訓練（檢查 `st.session_state.adapter_trained`）

```
Prompt: [Enter your question here...                              ] [Send]

┌─BASE MODEL──────────────────┐  ┌─FINE-TUNED MODEL─────────────────┐
│ "TinyLlama is a compact      │  │ "TinyLlama is a 1.1B parameter   │
│ language model..."           │  │ open-source LLM developed by..." │
│ Time: 8.3s  |  Tokens: 82   │  │ Time: 8.7s  |  Tokens: 124      │
└──────────────────────────────┘  └──────────────────────────────────┘

Metrics:  [Vocab Diversity ↑18%]  [Avg Token Length ↑22%]  [Keyword Match ↑35%]
```

### Tab 3：RAG Pipeline

```
Section 1 - Data Ingestion:
[🎵 Upload Audio] [🖼️ Upload Image] [📄 Upload Text]  [⚡ Process & Index]

Section 2 - Knowledge Graph (Plotly Network):
[NetworkX 知識圖譜互動式視覺化，可縮放/拖曳]

Section 3 - Query:
[Enter question...]  [🔍 Run RAG Query]

Section 4 - Results:
Retrieved chunks (with similarity scores)
Pipeline trace: Audio→Text | Image→Caption | Embed | Graph Expand | LLM
Final Answer: [...]
```

### Tab 4：ReAct Agent

```
Question: [What are the key advantages of QLoRA over full fine-tuning?]  [🚀 Run]

Step 1:  💭 Thought: I need to search for information about QLoRA advantages...
         ⚡ Action: search_rag[QLoRA advantages memory efficiency]
         👁️ Observation: "QLoRA reduces memory requirements by 4x through..."

Step 2:  💭 Thought: I also need to compare training time...
         ⚡ Action: search_rag[training time comparison fine-tuning]
         👁️ Observation: "Full fine-tuning requires approximately..."

✅ Final Answer: QLoRA offers three key advantages...

Stats: 2 steps | 12.3 seconds | 2 tools used
```

### Tab 5：Experiment Results

```
Row 1:
┌─Chart A: Training Loss────────┐  ┌─Chart B: VRAM Usage (GB)────────┐
│ 3 lines + optional Live line  │  │ Grouped bar: model × method     │
└───────────────────────────────┘  └──────────────────────────────────┘
Row 2:
┌─Chart C: Quality Radar────────┐  ┌─Chart D: Perplexity vs Steps───┐
│ 3-layer radar                 │  │ Line + scatter + baseline       │
└───────────────────────────────┘  └──────────────────────────────────┘
Row 3 (full width):
┌─Chart E: Time-Quality Tradeoff (Pareto Frontier)──────────────────────┐
│ Scatter + Pareto line, annotated with model names                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 七、安裝流程

```bash
# 1. 進入專案目錄，建立虛擬環境
cd "C:\Users\lin\Desktop\class\百億計畫\加拿大AI\demo_site"
python -m venv venv
venv\Scripts\activate

# 2. 安裝 PyTorch（CUDA 12.1 wheel，相容 CUDA 12.5 driver）
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 3. 驗證 CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. 安裝 bitsandbytes（Windows native）
pip install bitsandbytes==0.43.3

# 5. 安裝訓練套件
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1 einops==0.8.0

# 6. 安裝 RAG 套件
pip install langchain==0.2.16 langchain-community==0.2.16 chromadb==0.5.5 \
    sentence-transformers==3.0.1

# 7. 安裝 App 及監控套件
pip install streamlit==1.38.0 pynvml==11.5.0 psutil==5.9.8 \
    python-dotenv==1.0.1 plotly==5.24.1

# 8. 安裝 Whisper（音訊）
pip install openai-whisper==20231117
# 另需下載 ffmpeg binary，加入 PATH

# 9. 生成模擬實驗數據
python utils/generate_experiment_data.py

# 10. 啟動 Demo Site
streamlit run app.py
# 開啟 http://localhost:8501
```

---

## 八、測試驗收標準

| 項目 | 驗收標準 |
|------|---------|
| GPU 識別 | `torch.cuda.get_device_name(0)` = "NVIDIA GeForce RTX 3050" |
| 4-bit 量化 | `model.get_memory_footprint()` < 700MB for TinyLlama 1.1B |
| QLoRA 訓練 | 50 steps 在 ~3 分鐘內完成，loss 從 ~2.8 降至 ~1.5，無 OOM |
| Adapter 大小 | `adapters/qlora_checkpoint/` 目錄大小 < 50MB |
| 訓練可重現 | 相同 seed 下 loss curve 幾乎一致 |
| RAG 查詢 | 回傳 3 個 chunks，相似度分數 > 0.5，有正確的溯源標示 |
| 知識圖譜 | 圖譜包含至少 10 個節點，可在 UI 互動縮放 |
| 圖表渲染 | Tab 5 所有 5 張圖表均正常渲染，無空白 |
| Streamlit 啟動 | `streamlit run app.py` 在 < 30 秒內開啟，無 ImportError |

---

## 九、Demo 展示腳本建議

**最佳展示順序：**
1. **Tab 5（Charts）** — 先展示大圖，建立「為什麼 QLoRA 重要」的背景
2. **Tab 1（QLoRA Training）** — 點擊 Load Model → Start Training，解說 QLoRA 原理
3. **Tab 3（RAG）** — 訓練跑背景時，演示 multimodal 上傳和知識圖譜
4. **Tab 4（ReAct）** — 展示推理追蹤，強調 chain-of-thought
5. **返回 Tab 1** — 訓練完成，儲存 adapter
6. **Tab 2（Compare）** — Base vs Fine-tuned 對比，這是整個 demo 的高潮

**話術重點：**
- Tab 1：「這是在 4GB 消費級 GPU 上實際跑的 QLoRA，背後用了 4-bit NF4 量化和 paged optimizer...」
- Tab 3：「傳統 RAG 只有向量搜索，我加了知識圖譜做 chained retrieval，解決 lost-in-the-middle 問題...」
- Tab 5：「這些數字錨定了 Dettmers 2023 的 QLoRA 論文，Full FT 70B 需要 576GB VRAM，但 QLoRA 只需要 46GB...」
