"""
QLoRA ML Portfolio Demo Site
Main Streamlit entry point.

Run: streamlit run app.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QLoRA ML Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "model_loaded": False,
    "training_active": False,
    "training_done": False,
    "training_history": [],
    "adapter_saved": False,
    "adapter_path": None,
    "rag_indexed": False,
    "qlora_trainer": None,
    "qlora_queue": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Shared singletons (cached, one per server process) ───────────────────────

@st.cache_resource
def get_model_manager():
    try:
        from core.model_manager import ModelManager
        return ModelManager()
    except ImportError:
        return None


@st.cache_resource
def get_vram_monitor():
    from core.vram_monitor import VRAMMonitor
    monitor = VRAMMonitor(poll_interval_sec=1.0, history_len=120)
    monitor.start()
    return monitor


@st.cache_resource
def get_rag_pipeline():
    try:
        from core.rag_pipeline import MultimodalRAGPipeline
        return MultimodalRAGPipeline()
    except ImportError:
        return None


model_manager = get_model_manager()
vram_monitor = get_vram_monitor()
rag_pipeline = get_rag_pipeline()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("QLoRA ML Demo")
    st.caption("RTX 3050 · 4GB VRAM · Python 3.9")
    st.divider()

    # VRAM gauge
    vram = vram_monitor.get_current()
    vram_mb = vram.get("used_mb", 0)
    total_mb = vram.get("total_mb", 4096)
    pct = vram.get("pct", 0.0)

    st.markdown("**GPU Memory**")
    st.progress(min(pct / 100, 1.0))
    st.caption(f"{vram_mb} / {total_mb} MB  ({pct:.1f}%)")

    # Sparkline
    history = vram_monitor.get_history()
    if history:
        from utils.chart_builder import build_vram_sparkline
        fig_spark = build_vram_sparkline(history, total_mb=total_mb)
        st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Hardware info
    with st.expander("Hardware Info"):
        import platform
        import psutil
        ram = psutil.virtual_memory()
        st.markdown(f"**OS:** {platform.system()} {platform.release()}")
        st.markdown(f"**CPU:** {psutil.cpu_count(logical=False)}C / {psutil.cpu_count()}T")
        st.markdown(f"**RAM:** {ram.used // (1024**3):.1f} / {ram.total // (1024**3):.1f} GB  ({ram.percent}%)")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.markdown(f"**GPU:** {gpu_name}")
                st.markdown(f"**CUDA:** {torch.version.cuda}")
            else:
                st.markdown("**GPU:** CUDA not available")
        except ImportError:
            st.markdown("**PyTorch:** Not installed")

    st.divider()

    # Model status
    if model_manager.is_loaded:
        st.success(f"Model loaded")
        footprint = model_manager.get_memory_footprint()
        if footprint > 0:
            st.caption(f"Footprint: {footprint / 1e6:.0f} MB (4-bit)")
    else:
        st.info("No model loaded")

    if st.session_state.training_done:
        st.success("Adapter trained")
    if st.session_state.get("adapter_path"):
        st.caption(f"Adapter: {st.session_state.adapter_path}")

    st.divider()
    st.caption("Built with Streamlit + HuggingFace + PEFT + ChromaDB")

# ── Main tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚡ QLoRA Training",
    "🔍 Model Compare",
    "📚 RAG Pipeline",
    "🧠 ReAct Agent",
    "📊 Experiment Results",
    "📖 Manual",
    "🛡️ Prompt Guard",
])

with tab1:
    from tabs.tab_qlora import render as render_qlora
    render_qlora(model_manager, vram_monitor)

with tab2:
    from tabs.tab_compare import render as render_compare
    render_compare(model_manager)

with tab3:
    from tabs.tab_rag import render as render_rag
    render_rag(rag_pipeline)

with tab4:
    from tabs.tab_agent import render as render_agent
    render_agent(model_manager, rag_pipeline)

with tab5:
    from tabs.tab_charts import render as render_charts
    render_charts(training_history=st.session_state.get("training_history", []))

with tab6:
    from tabs.tab_manual import render as render_manual
    render_manual()

with tab7:
    from tabs.tab_prompt_guard import render as render_prompt_guard
    render_prompt_guard()
