# CLAUDE.md — QLoRA Demo Site 開發指引

## 這個專案是什麼

這是一個 Streamlit 多分頁 ML Portfolio Demo Site，用於展示：
1. QLoRA fine-tuning pipeline（TinyLlama 1.1B，4-bit NF4，本機 RTX 3050 4GB）
2. Multimodal RAG（Whisper + BLIP + ChromaDB + NetworkX 知識圖譜）
3. ReAct reasoning agent
4. 實驗結果圖表（5 張 Plotly 圖，模擬 8B/70B QLoRA 數據）

**完整規格請參閱 `SPEC.md`**，所有開發決策均以 SPEC.md 為依據。

---

## 硬體約束（每次開發必讀）

- GPU：**RTX 3050，僅 4GB VRAM**
- `per_device_train_batch_size` 永遠為 1，不可更改
- 訓練使用 `fp16=True`（非 bf16）和 `optim="paged_adamw_8bit"`
- 模型只能選 ~1B 級別：TinyLlama-1.1B 或 LLaMA 3.2 1B
- VRAM 預算：~1750MB（安全），超過 3500MB 視為危險

---

## 目錄結構

```
demo_site/
├── app.py              # Streamlit 主入口
├── requirements.txt    # 所有 pinned 依賴
├── SPEC.md             # 詳細規格書（開發依據）
├── CLAUDE.md           # 本檔案
├── core/               # 業務邏輯（無 Streamlit 依賴）
│   ├── model_manager.py
│   ├── qlora_trainer.py
│   ├── rag_pipeline.py
│   ├── react_agent.py
│   └── vram_monitor.py
├── tabs/               # Streamlit UI 層（每分頁一檔）
│   ├── tab_qlora.py
│   ├── tab_compare.py
│   ├── tab_rag.py
│   ├── tab_agent.py
│   └── tab_charts.py
├── utils/
│   ├── generate_experiment_data.py
│   └── chart_builder.py
├── data/
│   ├── train_data/alpaca_tiny.json
│   ├── rag_docs/
│   └── precomputed/experiment_results.json
└── adapters/qlora_checkpoint/
```

---

## 開發規則

### 1. 遵守 SPEC.md
- 所有函式簽名、設定值、資料格式均以 SPEC.md 為準
- 若需修改，先更新 SPEC.md，再修改程式碼

### 2. 模組分層
- `core/` 不能 import `streamlit`（保持可測試性）
- `tabs/` 只做 UI 渲染，業務邏輯呼叫 `core/`
- `utils/` 為純函式，無副作用

### 3. Streamlit session_state 規範
- 所有跨分頁共用狀態存在 `st.session_state`
- 在 `app.py` 初始化預設值（防止 KeyError）
- 模型只允許有一個實例（`@st.cache_resource`）

### 4. VRAM 安全
- 每次載入模型後必須顯示 `get_vram_info()` 確認用量
- Tab 切換不自動載入/卸載模型（由使用者點擊按鈕控制）
- 訓練前檢查可用 VRAM > 2000MB，否則顯示警告

### 5. 錯誤處理
- 所有 torch/CUDA 操作包在 try/except 中
- OOM 錯誤顯示友善訊息 + 建議（降低 max_steps、重啟）
- 模型下載失敗顯示 HF_TOKEN 設定引導

### 6. 圖表規範
- 所有圖表使用 `plotly_dark` template
- 顏色規範：QLoRA=`#00d4ff`，Full FT=`#ff6b6b`，Base=`#a8a8a8`，Live=`#ffd700`
- 圖表必須有 title、axis labels、hover tooltips
- 數據來源標注（footnote）錨定真實論文

### 7. 模擬數據說明
- `data/precomputed/experiment_results.json` 是模擬數據
- 所有數值錨定真實研究（Dettmers 2023, MT-Bench Zheng 2023）
- 圖表中標注 "Simulated based on QLoRA paper (Dettmers et al., 2023)"

---

## 套件版本（不可隨意升級）

| 套件 | 版本 |
|------|------|
| torch | 2.3.1+cu121 |
| transformers | 4.44.2 |
| peft | 0.12.0 |
| bitsandbytes | 0.43.3 |
| trl | 0.10.1 |
| accelerate | 0.34.2 |
| streamlit | 1.38.0 |
| chromadb | 0.5.5 |
| sentence-transformers | 3.0.1 |
| langchain | 0.2.16 |
| pynvml | 11.5.0 |
| plotly | 5.24.1 |

---

## 啟動方式

```bash
# 啟動虛擬環境
venv\Scripts\activate

# 啟動 app
streamlit run app.py
# → http://localhost:8501
```

---

## 驗收標準（完成開發後自我驗測）

1. `streamlit run app.py` 成功啟動，無 ImportError
2. Tab 5 圖表全部渲染（無需 GPU）
3. Tab 1 Load Model 成功，VRAM 顯示 < 700MB
4. Tab 1 Start Training 跑 10 steps，loss 有下降，無 OOM
5. Tab 3 Process & Index 成功建立 ChromaDB collection
6. Tab 3 Query 返回有效結果
7. Tab 4 ReAct Agent 完成一個 2-step 推理
8. Tab 2 在 adapter 訓練後可正常對比
