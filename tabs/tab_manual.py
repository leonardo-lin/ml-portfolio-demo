"""
Tab 6: User Manual — 使用者操作手冊
"""

import streamlit as st


def render():
    st.header("使用者操作手冊")
    st.caption("QLoRA ML Portfolio Demo Site — 完整操作指引")

    # ── 系統需求 ──────────────────────────────────────────────────────────────
    with st.expander("一、系統需求", expanded=False):
        st.markdown("""
| 項目 | 最低需求 | 本機規格 |
|------|---------|---------|
| GPU | 4GB VRAM (NVIDIA) | RTX 3050 4GB ✓ |
| RAM | 8GB | 15GB ✓ |
| 磁碟空間 | 10GB（模型快取） | 需確認 |
| Python | 3.9–3.11 | 3.9.12 ✓ |
| CUDA Driver | 11.8+ | 12.5 ✓ |
| OS | Windows 10/11 | Windows 11 ✓ |
""")

    # ── 安裝步驟 ──────────────────────────────────────────────────────────────
    with st.expander("二、安裝步驟（首次使用）", expanded=False):
        st.markdown("**2.1 前置作業** — 確認 ffmpeg 已安裝（音訊功能需要）：")
        st.code("ffmpeg -version", language="bash")

        st.markdown("**2.2 建立虛擬環境並安裝套件**（使用 venv，避免污染 Anaconda）：")
        st.code("""cd "C:\\Users\\lin\\Desktop\\class\\百億計畫\\加拿大AI\\demo_site"

# 建立隔離環境（只需執行一次）
python -m venv venv
venv\\Scripts\\activate

# PyTorch CUDA（最重要，必須先裝，約 2GB）
pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 驗證 CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# QLoRA 套件
pip install bitsandbytes==0.43.3
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1 einops==0.8.0

# RAG 套件
pip install chromadb==0.5.5 sentence-transformers==3.0.1
pip install langchain==0.2.16 langchain-community==0.2.16

# UI 套件
pip install streamlit==1.38.0 plotly==5.24.1 pynvml==11.5.0 psutil==5.9.8 python-dotenv==1.0.1 networkx

# 音訊套件（需 ffmpeg binary）
pip install openai-whisper==20231117""", language="bash")

        st.markdown("**2.3 生成實驗數據（只需執行一次）：**")
        st.code("python utils\\generate_experiment_data.py", language="bash")

    # ── 啟動方式 ──────────────────────────────────────────────────────────────
    with st.expander("三、啟動方式", expanded=False):
        st.code("""cd "C:\\Users\\lin\\Desktop\\class\\百億計畫\\加拿大AI\\demo_site"
venv\\Scripts\\activate
streamlit run app.py""", language="bash")
        st.info("瀏覽器自動開啟，或手動前往 **http://localhost:8501**")

    # ── 各分頁說明 ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("各分頁操作說明")

    tab_info = {
        "⚡ Tab 1 — QLoRA Training（核心功能）": {
            "desc": "在 RTX 3050 上實際跑 QLoRA 微調，觀察即時 loss 下降與 VRAM 用量。",
            "steps": [
                "選擇模型：TinyLlama-1.1B-Chat（推薦）或 LLaMA 3.2 1B（需 HF Token）",
                "設定參數：Max Steps=50、Learning Rate=2e-4、LoRA Rank=8",
                "點擊【Load Model】→ 等待 30–60 秒（首次下載約 2.2GB）",
                "側邊欄顯示 VRAM ~560MB → 4-bit 量化成功",
                "點擊【Start Training】→ Loss Chart 每 2 秒更新，VRAM ~1750MB",
                "訓練完成 → 點擊【Save Adapter】→ 儲存至 adapters/qlora_checkpoint/",
            ],
            "note": "若出現 OOM：降低 Max Steps 至 20，重新 Load Model 再試",
        },
        "🔍 Tab 2 — Model Compare（效果驗證）": {
            "desc": "並排比較微調前後的模型回答差異。前提：Tab 1 已訓練並儲存 Adapter。",
            "steps": [
                "從下拉選單選擇範例問題，或自行輸入",
                "調整 Max New Tokens（100–400）",
                "點擊【Generate Comparison】",
                "左欄：Base Model 回答 / 右欄：Fine-tuned 回答",
                "底部自動計算：Response Length / Vocabulary Diversity / Avg Word Length",
            ],
            "note": "Fine-tuned 模型在 ML 問題上應產出更詳細、術語更精確的回答",
        },
        "📚 Tab 3 — RAG Pipeline（多模態知識庫）": {
            "desc": "展示多模態資料處理與混合式檢索（向量 + 知識圖譜）。無 GPU 也可使用。",
            "steps": [
                "上傳資料（任選）：音訊 → Whisper 轉文字 / 圖片 → BLIP 生成描述 / 文字 → 自動分段",
                "勾選【Also index built-in knowledge base】直接使用 QLoRA/RAG/ML 知識庫",
                "點擊【Process & Index】→ 顯示 'Index ready: XX chunks'",
                "查看互動式知識圖譜（可縮放、hover 查看節點關係）",
                "輸入問題 → 點擊【Run RAG Query】",
                "查看：相關段落（含相似度分數）/ 知識圖譜擴展上下文 / Pipeline 追蹤",
            ],
            "note": "embedding 模型跑在 CPU，不需要 GPU",
        },
        "🧠 Tab 4 — ReAct Agent（推理追蹤）": {
            "desc": "展示 AI 一步一步推理的過程（Thought → Action → Observation）。",
            "steps": [
                "無模型 = Demo Mode（預建示範追蹤） / 有模型 = Live Mode（實際推理）",
                "選擇範例問題或自行輸入",
                "調整 Max Reasoning Steps（2–8）",
                "點擊【Run Agent】",
                "藍色 = Thought / 橙色 = Action / 綠色 = Observation / 最底 = Final Answer",
            ],
            "note": "可用工具：search_rag / calculate / lookup_graph",
        },
        "🛡️ Tab 7 — Prompt Guard（Llama 4 安全分類器）": {
            "desc": "展示 Llama 4 系列的 Llama-Prompt-Guard-2-86M，偵測 prompt injection / jailbreak，並比較 Full FT vs QLoRA 的記憶體用量。",
            "steps": [
                "在 .env 設定 HF_TOKEN（Meta gated 模型必須）並重啟 app",
                "在 huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M 接受授權",
                "點擊【Load Model (fp16)】→ 下載 ~170MB，顯示 VRAM 用量",
                "點擊左側 Safe / Injection 範例，或自行輸入文字，點擊【Classify】",
                "查看 SAFE / INJECTION 標籤與信心分數",
                "點擊【Run Full Fine-Tuning】→ 30 steps，記錄 VRAM + RAM",
                "點擊【Run QLoRA】→ 同樣 30 steps，圖表疊加第二條線",
                "查看 Peak Memory 比較表與長條圖：QLoRA VRAM 應比 Full FT 少約 75%",
            ],
            "note": "QLoRA 用 4-bit NF4 量化 + LoRA (task_type=SEQ_CLS)，optimizer 改用 paged_adamw_8bit",
        },
        "📊 Tab 5 — Experiment Results（實驗圖表）": {
            "desc": "展示 QLoRA vs Full Fine-Tuning 的大規模實驗比較。不需要 GPU，隨時可用。",
            "steps": [
                "A. Training Loss Curves — QLoRA 8B / 70B / Full FT 曲線（500 steps）",
                "B. VRAM Requirements — 各模型×方法的 GPU 記憶體需求",
                "C. Quality Radar — 5 維度回答品質雷達圖（錨定 MT-Bench）",
                "D. Perplexity vs Steps — 訓練步數與困惑度折線圖",
                "E. Time-Quality Tradeoff — 訓練時間 vs 品質 Pareto 前緣",
            ],
            "note": "若已執行 Tab 1 訓練，圖表 A 自動疊加金色「Live Demo」曲線",
        },
    }

    for title, info in tab_info.items():
        with st.expander(title, expanded=False):
            st.markdown(f"**用途：** {info['desc']}")
            st.markdown("**操作步驟：**")
            for i, step in enumerate(info["steps"], 1):
                st.markdown(f"{i}. {step}")
            if info.get("note"):
                st.warning(f"注意：{info['note']}")

    # ── Demo 展示順序 ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Demo 展示建議順序")
    demo_steps = [
        ("1", "📊 圖表", "先建立「為什麼 QLoRA 重要」的背景"),
        ("2", "⚡ QLoRA", "點擊 Load Model + Start Training，邊解說 4-bit 量化原理"),
        ("3", "📚 RAG", "訓練跑背景時，展示多模態上傳和知識圖譜"),
        ("4", "🧠 ReAct", "展示推理追蹤，強調 chain-of-thought"),
        ("5", "⚡ QLoRA", "返回確認訓練完成，存 Adapter"),
        ("6", "🔍 對比", "Base vs Fine-tuned 並排 — 高潮"),
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

    # ── 常見問題 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("常見問題排解")
    qa = [
        ("OOM (Out of Memory)", "VRAM 不足", "降低 Max Steps、重啟後重試"),
        ("模型下載失敗", "需要 HuggingFace Token", "在 `.env` 填入 `HF_TOKEN=hf_xxx`"),
        ("cp950 編碼錯誤", "用系統 Python 而非 venv", "確認使用 `venv\\Scripts\\activate`"),
        ("sentence-transformers import 失敗", "numpy ABI 衝突（Anaconda 環境問題）", "必須使用 venv，不要用 Anaconda 基礎環境"),
        ("音訊無法處理", "缺少 ffmpeg binary", "安裝 ffmpeg 並加入系統 PATH"),
        ("圖表空白", "缺少實驗數據", "執行 `python utils\\generate_experiment_data.py`"),
    ]
    for problem, cause, fix in qa:
        with st.expander(f"❓ {problem}"):
            st.markdown(f"**原因：** {cause}")
            st.markdown(f"**解法：** {fix}")
