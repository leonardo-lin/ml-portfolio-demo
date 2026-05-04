"""
Plotly figure factory for all 5 experiment charts.
All charts use plotly_dark template.

Color conventions:
  QLoRA:    #00d4ff (cyan-blue)
  Full FT:  #ff6b6b (coral red)
  Base:     #a8a8a8 (gray)
  Live:     #ffd700 (gold)
  Sim note: italic annotation on charts with simulated data
"""

import json
import os
from typing import List

import plotly.graph_objects as go
import plotly.subplots as sp

TEMPLATE = "plotly_dark"
COLOR_QLORA = "#00d4ff"
COLOR_FULLFT = "#ff6b6b"
COLOR_BASE = "#a8a8a8"
COLOR_LIVE = "#ffd700"
COLOR_QLORA_LIGHT = "#7dd3fc"

FOOTNOTE = "<i>Simulated data anchored to Dettmers et al. 2023 (QLoRA) & Zheng et al. 2023 (MT-Bench)</i>"


def _load_data(path: str = None) -> dict:
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "precomputed", "experiment_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Chart A: Training Loss Curves ────────────────────────────────────────────

def build_loss_curves(data: dict = None, live_history: List[dict] = None) -> go.Figure:
    """
    3 pre-computed lines + optional 4th 'Live Demo' line from real training.
    live_history: list of {step, loss} dicts from training callback.
    """
    if data is None:
        data = _load_data()
    d = data["loss_curves"]
    steps = d["steps"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps, y=d["qlora_8b"],
        name="QLoRA 8B", mode="lines",
        line=dict(color=COLOR_QLORA, width=2),
        hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra>QLoRA 8B</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=d["qlora_70b_sim"],
        name="QLoRA 70B (simulated)", mode="lines",
        line=dict(color=COLOR_QLORA_LIGHT, width=2, dash="dot"),
        hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra>QLoRA 70B</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=d["fullft_8b_sim"],
        name="Full FT 8B (simulated)", mode="lines",
        line=dict(color=COLOR_FULLFT, width=2, dash="dash"),
        hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra>Full FT 8B</extra>"
    ))

    if live_history and len(live_history) > 1:
        live_steps = [h["step"] for h in live_history]
        live_loss = [h["loss"] for h in live_history]
        fig.add_trace(go.Scatter(
            x=live_steps, y=live_loss,
            name="Live Demo (1.1B)", mode="lines+markers",
            line=dict(color=COLOR_LIVE, width=2),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra>Live Demo</extra>"
        ))

    fig.update_layout(
        template=TEMPLATE,
        title="Training Loss Curves",
        xaxis_title="Training Step",
        yaxis_title="Cross-Entropy Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[dict(
            text=FOOTNOTE, showarrow=False,
            xref="paper", yref="paper", x=0, y=-0.15,
            font=dict(size=10, color="#888888")
        )],
        margin=dict(b=80),
        hovermode="x unified",
    )
    return fig


# ── Chart B: VRAM Usage Comparison ───────────────────────────────────────────

def build_vram_comparison(data: dict = None) -> go.Figure:
    if data is None:
        data = _load_data()
    cats = data["vram_comparison"]["categories"]

    models = [c["model"] for c in cats]
    methods = [c["method"] for c in cats]
    vrams = [c["vram_gb"] for c in cats]
    colors = [c["color"] for c in cats]
    labels = [f"{m} ({v:.1f} GB)" for m, v in zip(methods, vrams)]

    # Group by model for grouped bar
    unique_models = list(dict.fromkeys(models))
    method_colors = {
        "Full FT": COLOR_FULLFT,
        "QLoRA": COLOR_QLORA,
        "QLoRA (Live)": COLOR_LIVE,
        "Inference": COLOR_QLORA_LIGHT,
    }

    unique_methods = list(dict.fromkeys(methods))
    fig = go.Figure()
    for method in unique_methods:
        x_vals, y_vals = [], []
        for m, mt, v in zip(models, methods, vrams):
            if mt == method:
                x_vals.append(m)
                y_vals.append(v)
        fig.add_trace(go.Bar(
            name=method,
            x=x_vals, y=y_vals,
            marker_color=method_colors.get(method, "#888"),
            text=[f"{v:.1f} GB" for v in y_vals],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1f} GB<extra>" + method + "</extra>"
        ))

    # Horizon line for RTX 3050
    fig.add_hline(y=4.0, line_dash="dot", line_color="#ff9500",
                  annotation_text="RTX 3050 (4 GB)", annotation_position="right")

    fig.update_layout(
        template=TEMPLATE,
        title="VRAM Requirements Comparison",
        xaxis_title="Model",
        yaxis_title="VRAM (GB)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            text=FOOTNOTE, showarrow=False,
            xref="paper", yref="paper", x=0, y=-0.15,
            font=dict(size=10, color="#888888")
        )],
        margin=dict(b=80),
    )
    return fig


# ── Chart C: Quality Radar ────────────────────────────────────────────────────

def build_quality_radar(data: dict = None) -> go.Figure:
    if data is None:
        data = _load_data()
    d = data["quality_radar"]
    dims = d["dimensions"]
    # Close the polygon
    dims_closed = dims + [dims[0]]

    def close(vals):
        return vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=close(d["base_model"]), theta=dims_closed,
        fill="toself", name="Base Model",
        line_color=COLOR_BASE, opacity=0.5,
        hovertemplate="%{theta}: %{r}<extra>Base Model</extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(d["qlora_finetuned"]), theta=dims_closed,
        fill="toself", name="QLoRA Fine-tuned",
        line_color=COLOR_QLORA, opacity=0.7,
        hovertemplate="%{theta}: %{r}<extra>QLoRA Fine-tuned</extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(d["full_ft"]), theta=dims_closed,
        fill="toself", name="Full FT (simulated)",
        line_color=COLOR_FULLFT, opacity=0.5,
        line_dash="dot",
        hovertemplate="%{theta}: %{r}<extra>Full FT</extra>"
    ))

    fig.update_layout(
        template=TEMPLATE,
        title="Response Quality Comparison (0–100 scale)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        annotations=[dict(
            text=FOOTNOTE, showarrow=False,
            xref="paper", yref="paper", x=0.5, y=-0.25,
            font=dict(size=10, color="#888888"), xanchor="center"
        )],
        margin=dict(b=100),
    )
    return fig


# ── Chart D: Perplexity vs Steps ─────────────────────────────────────────────

def build_perplexity_curves(data: dict = None) -> go.Figure:
    if data is None:
        data = _load_data()
    d = data["perplexity"]
    steps = d["checkpoints"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=d["base_model"],
        name="Base Model (no training)", mode="lines",
        line=dict(color=COLOR_BASE, width=2, dash="dot"),
        hovertemplate="Step %{x}<br>PPL: %{y:.1f}<extra>Base</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=d["qlora_8b"],
        name="QLoRA 8B", mode="lines+markers",
        line=dict(color=COLOR_QLORA, width=2),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="Step %{x}<br>PPL: %{y:.1f}<extra>QLoRA 8B</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=d["fullft_8b"],
        name="Full FT 8B (simulated)", mode="lines+markers",
        line=dict(color=COLOR_FULLFT, width=2, dash="dash"),
        marker=dict(size=8, symbol="square"),
        hovertemplate="Step %{x}<br>PPL: %{y:.1f}<extra>Full FT 8B</extra>"
    ))

    fig.update_layout(
        template=TEMPLATE,
        title="Perplexity vs Training Steps",
        xaxis_title="Training Steps",
        yaxis_title="Perplexity (lower is better)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            text=FOOTNOTE, showarrow=False,
            xref="paper", yref="paper", x=0, y=-0.15,
            font=dict(size=10, color="#888888")
        )],
        margin=dict(b=80),
        hovermode="x unified",
    )
    return fig


# ── Chart E: Time-Quality Tradeoff ────────────────────────────────────────────

def build_tradeoff_scatter(data: dict = None) -> go.Figure:
    if data is None:
        data = _load_data()
    points = data["tradeoff"]["points"]

    method_color = {"QLoRA": COLOR_QLORA, "Full FT": COLOR_FULLFT, "Live": COLOR_LIVE}

    fig = go.Figure()

    for method in ["Full FT", "QLoRA", "Live"]:
        pts = [p for p in points if p["method"] == method]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["time_hr"] for p in pts],
            y=[p["mt_bench"] for p in pts],
            mode="markers+text",
            name=method,
            marker=dict(color=method_color[method], size=14, symbol="circle",
                        line=dict(color="white", width=1)),
            text=[p["label"] for p in pts],
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate="%{text}<br>Time: %{x:.1f}h<br>MT-Bench: %{y:.2f}<extra>" + method + "</extra>"
        ))

    # Pareto frontier: QLoRA points sorted by time (they dominate)
    pareto = sorted([p for p in points if p["method"] == "QLoRA"], key=lambda x: x["time_hr"])
    pareto_x = [p["time_hr"] for p in pareto]
    pareto_y = [p["mt_bench"] for p in pareto]
    fig.add_trace(go.Scatter(
        x=pareto_x, y=pareto_y,
        mode="lines", name="Pareto Frontier (QLoRA)",
        line=dict(color=COLOR_QLORA, width=1, dash="dot"),
        showlegend=True,
        hoverinfo="skip"
    ))

    fig.update_layout(
        template=TEMPLATE,
        title="Training Time vs. Quality Tradeoff",
        xaxis=dict(title="Training Time (hours)", type="log"),
        yaxis=dict(title="MT-Bench Score (0–10)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            text=FOOTNOTE, showarrow=False,
            xref="paper", yref="paper", x=0, y=-0.15,
            font=dict(size=10, color="#888888")
        )],
        margin=dict(b=80),
    )
    return fig


# ── Sidebar VRAM Sparkline ────────────────────────────────────────────────────

def build_vram_sparkline(history: List[dict], total_mb: int = 4096) -> go.Figure:
    """Tiny sparkline for sidebar, no axis labels."""
    if not history:
        x, y = [0], [0]
    else:
        x = list(range(len(history)))
        y = [h.get("used_mb", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        fill="tozeroy",
        line=dict(color=COLOR_QLORA, width=1.5),
        fillcolor="rgba(0,212,255,0.15)",
        hoverinfo="skip"
    ))
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, total_mb]),
        showlegend=False,
    )
    return fig
