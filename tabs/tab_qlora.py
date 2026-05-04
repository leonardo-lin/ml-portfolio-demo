"""
Tab 1: QLoRA Training Demo
Real-time training with live loss chart and VRAM monitoring.
"""

import os
import queue
import time

import streamlit as st

try:
    import pandas as pd
    from core.model_manager import SUPPORTED_MODELS, ModelManager
    from core.qlora_trainer import QLoRATrainer, TrainingConfig
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)


def render(model_manager=None, vram_monitor=None):
    st.header("QLoRA Training Demo")
    if not _DEPS_OK:
        st.error(f"Missing dependency: `{_DEPS_ERROR}`")
        st.info("Please install: `pip install torch transformers peft trl accelerate bitsandbytes`")
        st.code("venv\\Scripts\\pip install torch==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121\nvenv\\Scripts\\pip install bitsandbytes==0.43.3 transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1", language="bash")
        return
    # Demo Mode: show explanation instead of training UI
    if st.session_state.get("demo_mode", False):
        st.info(
            "**Demo Mode is active.**\n\n"
            "The **Experiment Results** tab already shows pre-loaded training results "
            "for TinyLlama 1.1B QLoRA, including the live training curve.\n\n"
            "To run real QLoRA training, disable Demo Mode in the sidebar."
        )
        return

    st.markdown(
        "Fine-tune a 4-bit quantized language model using QLoRA on your RTX 3050 (4GB VRAM). "
        "Training runs in a background thread — the chart updates live every 2 seconds."
    )

    # ── Init session state ──────────────────────────────────────────────────
    for key, default in [
        ("qlora_trainer", None),
        ("qlora_queue", None),
        ("training_active", False),
        ("training_history", []),
        ("training_done", False),
        ("adapter_saved", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_left, col_right = st.columns([1, 2])

    # ── LEFT: Configuration ─────────────────────────────────────────────────
    with col_left:
        st.subheader("Configuration")

        model_label = st.selectbox(
            "Base Model",
            list(SUPPORTED_MODELS.keys()),
            disabled=st.session_state.training_active or model_manager.is_loaded,
        )
        model_id = SUPPORTED_MODELS[model_label]

        max_steps = st.slider("Max Steps", min_value=10, max_value=100, value=50, step=5,
                               disabled=st.session_state.training_active)
        lr_options = [1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
        lr_labels = ["1e-5", "5e-5", "1e-4", "2e-4 (default)", "3e-4", "5e-4"]
        lr_idx = st.selectbox("Learning Rate", range(len(lr_options)),
                               format_func=lambda i: lr_labels[i], index=3,
                               disabled=st.session_state.training_active)
        learning_rate = lr_options[lr_idx]

        lora_r = st.radio("LoRA Rank (r)", [4, 8, 16], index=1, horizontal=True,
                           disabled=st.session_state.training_active)

        st.divider()
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "train_data", "alpaca_tiny.json"
        )
        ds_exists = os.path.exists(dataset_path)
        if ds_exists:
            import json
            with open(dataset_path, "r", encoding="utf-8") as f:
                n_samples = len(json.load(f))
            st.info(f"Dataset: {n_samples} samples (Alpaca format)")
        else:
            st.error("Dataset not found: data/train_data/alpaca_tiny.json")

        st.divider()

        # Buttons
        if not model_manager.is_loaded:
            if st.button("Load Model", type="primary", use_container_width=True, disabled=not ds_exists):
                with st.spinner(f"Loading {model_id} in 4-bit..."):
                    try:
                        model_manager.load_base_model(model_id)
                        vram = model_manager.get_vram_info()
                        st.success(
                            f"Model loaded! VRAM: {vram['used_mb']} MB / {vram['total_mb']} MB"
                            f"  ({vram['pct']}%)\n"
                            f"Footprint: {model_manager.get_memory_footprint() / 1e6:.0f} MB"
                        )
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
        else:
            vram = model_manager.get_vram_info()
            st.success(f"Model loaded  VRAM: {vram['used_mb']} MB ({vram['pct']}%)")

            if not st.session_state.training_active and not st.session_state.training_done:
                if st.button("Start Training", type="primary", use_container_width=True):
                    _start_training(model_manager, model_id, max_steps, learning_rate, lora_r)
                    st.rerun()

            if st.session_state.training_done and not st.session_state.adapter_saved:
                if st.button("Save Adapter", type="secondary", use_container_width=True):
                    try:
                        trainer: QLoRATrainer = st.session_state.qlora_trainer
                        saved_path = trainer.save_adapter()
                        st.session_state.adapter_saved = True
                        st.session_state["adapter_path"] = saved_path
                        st.success(f"Adapter saved to {saved_path}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

            if not st.session_state.training_active:
                if st.button("Unload Model", use_container_width=True):
                    model_manager.unload()
                    st.session_state.training_active = False
                    st.session_state.training_done = False
                    st.session_state.training_history = []
                    st.session_state.adapter_saved = False
                    st.rerun()

    # ── RIGHT: Live monitoring ───────────────────────────────────────────────
    with col_right:
        st.subheader("Training Monitor")

        # VRAM gauge
        vram = model_manager.get_vram_info() if model_manager.is_loaded else {"used_mb": 0, "total_mb": 4096, "pct": 0.0}
        vram_pct = vram["pct"] / 100
        vram_color = "normal" if vram["pct"] < 70 else ("off" if vram["pct"] > 90 else "normal")
        st.metric("VRAM", f"{vram['used_mb']} / {vram['total_mb']} MB", f"{vram['pct']}%")
        st.progress(vram_pct)

        # Loss chart placeholder
        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        log_placeholder = st.empty()

        _render_chart(chart_placeholder, st.session_state.training_history)

        if st.session_state.training_active:
            _poll_queue(status_placeholder, log_placeholder)

        elif st.session_state.training_done:
            status_placeholder.success(
                f"Training complete! {len(st.session_state.training_history)} steps logged."
            )

        elif not model_manager.is_loaded:
            status_placeholder.info("Load a model to begin training.")

        _render_chart(chart_placeholder, st.session_state.training_history)
        _render_logs(log_placeholder, st.session_state.training_history)


def _start_training(model_manager, model_id, max_steps, learning_rate, lora_r):
    q: queue.Queue = queue.Queue()
    config = TrainingConfig(
        model_id=model_id,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
    )
    trainer = QLoRATrainer(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        config=config,
        progress_queue=q,
    )
    trainer.run()
    st.session_state.qlora_trainer = trainer
    st.session_state.qlora_queue = q
    st.session_state.training_active = True
    st.session_state.training_done = False
    st.session_state.training_history = []
    st.session_state.adapter_saved = False


def _poll_queue(status_placeholder, log_placeholder):
    q: queue.Queue = st.session_state.qlora_queue
    trainer: QLoRATrainer = st.session_state.qlora_trainer

    new_msgs = []
    try:
        while True:
            msg = q.get_nowait()
            new_msgs.append(msg)
    except queue.Empty:
        pass

    for msg in new_msgs:
        status = msg.get("status", "running")
        if status == "done":
            st.session_state.training_active = False
            st.session_state.training_done = True
            break
        elif status == "error":
            st.session_state.training_active = False
            st.session_state.training_done = True
            st.error(f"Training error: {msg.get('message', 'Unknown error')}")
            break
        elif status in ("running", "preparing"):
            if msg.get("step", 0) > 0:
                st.session_state.training_history.append(msg)

    if st.session_state.training_active:
        step = st.session_state.training_history[-1]["step"] if st.session_state.training_history else 0
        loss = st.session_state.training_history[-1]["loss"] if st.session_state.training_history else 0
        vram = st.session_state.training_history[-1].get("vram_mb", 0) if st.session_state.training_history else 0
        target_steps = st.session_state.qlora_trainer._config.max_steps if st.session_state.qlora_trainer else 50

        status_placeholder.info(f"Training...  Step {step}/{target_steps}  |  Loss: {loss:.4f}  |  VRAM: {vram} MB")
        time.sleep(2)
        st.rerun()


def _render_chart(placeholder, history):
    if not history:
        with placeholder.container():
            st.caption("Loss chart will appear once training starts.")
        return

    import plotly.graph_objects as go
    df_data = [h for h in history if h.get("step", 0) > 0]
    if not df_data:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h["step"] for h in df_data],
        y=[h["loss"] for h in df_data],
        mode="lines",
        name="Training Loss",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="Step",
        yaxis_title="Loss",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    with placeholder.container():
        st.plotly_chart(fig, use_container_width=True)


def _render_logs(placeholder, history):
    if not history:
        return
    last = history[-10:]
    lines = []
    for h in reversed(last):
        lines.append(
            f"Step {h.get('step','?'):>3} | "
            f"loss: {h.get('loss',0):.4f} | "
            f"VRAM: {h.get('vram_mb',0)} MB | "
            f"epoch: {h.get('epoch',0):.2f}"
        )
    with placeholder.expander("Training Logs (last 10 steps)", expanded=False):
        st.code("\n".join(lines))
