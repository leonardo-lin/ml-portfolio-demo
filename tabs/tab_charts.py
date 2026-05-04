"""
Tab 5: Experiment Results
5 Plotly charts showing QLoRA vs Full FT comparisons.
No GPU required — uses precomputed data.
"""

import os
import streamlit as st


def render(training_history: list = None):
    st.header("Experiment Results")
    st.markdown(
        "Performance analysis comparing QLoRA, Full Fine-Tuning, and baseline. "
        "Numbers are anchored to published research — see footnotes on each chart."
    )

    # Load precomputed data
    data = _load_data()
    if data is None:
        st.error(
            "Experiment data not found. Please run: `python utils/generate_experiment_data.py`"
        )
        return

    from utils.chart_builder import (
        build_loss_curves,
        build_perplexity_curves,
        build_quality_radar,
        build_tradeoff_scatter,
        build_vram_comparison,
    )

    # ── Row 1 ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A  Training Loss Curves")
        fig_loss = build_loss_curves(data, live_history=training_history)
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.subheader("B  VRAM Requirements")
        fig_vram = build_vram_comparison(data)
        st.plotly_chart(fig_vram, use_container_width=True)

    # ── Row 2 ────────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("C  Response Quality (Radar)")
        fig_radar = build_quality_radar(data)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col4:
        st.subheader("D  Perplexity vs Training Steps")
        fig_ppl = build_perplexity_curves(data)
        st.plotly_chart(fig_ppl, use_container_width=True)

    # ── Row 3 (full width) ───────────────────────────────────────────────────
    st.subheader("E  Training Time vs Quality Tradeoff (Pareto Frontier)")
    fig_tradeoff = build_tradeoff_scatter(data)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # ── Key Insights ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("VRAM Reduction", "8×", help="QLoRA 8B (9.8GB) vs Full FT 8B (80GB)")
    with c2:
        st.metric("Quality Retention", "99.3%", help="QLoRA 65B vs Full FT on Vicuna benchmark")
    with c3:
        st.metric("Live Demo (1.1B)", "~1.75 GB", help="Actual VRAM on RTX 3050 4GB")

    with st.expander("Data Sources & Methodology"):
        meta = data.get("metadata", {})
        st.markdown("**Sources:**")
        for src in meta.get("sources", []):
            st.markdown(f"- {src}")
        st.markdown(f"\n**Note:** {meta.get('note', '')}")


def _load_data():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "precomputed", "experiment_results.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
