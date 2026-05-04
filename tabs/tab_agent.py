"""
Tab 4: ReAct Agent Demo
Step-by-step reasoning trace visualization.
"""

import streamlit as st

try:
    from core.model_manager import ModelManager
    from core.react_agent import ReActAgent, ReActTrace
    from core.rag_pipeline import MultimodalRAGPipeline
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)


def render(model_manager=None, rag_pipeline=None):
    st.header("ReAct Agent")
    if not _DEPS_OK:
        st.error(f"Missing dependency: `{_DEPS_ERROR}`")
        st.info("Please install required packages (see QLoRA Training tab).")
        return
    st.markdown(
        "Watch the AI reason step-by-step using the **ReAct** framework "
        "(Reasoning + Acting). Each step interleaves a Thought, Tool Action, and Observation."
    )

    sample_questions = [
        "What are the key advantages of QLoRA over full fine-tuning?",
        "How much VRAM does it take to train a 70B model with QLoRA?",
        "What is 9.8 multiplied by 4.7?",
        "How does the knowledge graph improve RAG retrieval quality?",
        "Compare perplexity of QLoRA vs full fine-tuning after 300 training steps.",
    ]

    q_choice = st.selectbox("Sample Questions", ["(custom)"] + sample_questions)
    question = st.text_input(
        "Your Question",
        value=q_choice if q_choice != "(custom)" else "",
        placeholder="Ask a question that requires reasoning and retrieval..."
    )

    demo_mode = not model_manager.is_loaded
    if demo_mode:
        st.info(
            "Model not loaded — running in **Demo Mode** with a pre-built trace. "
            "Load a model in the QLoRA Training tab for live inference."
        )

    max_steps = st.slider("Max Reasoning Steps", 2, 8, 5)

    if st.button("Run Agent", type="primary", disabled=not question.strip()):
        with st.spinner("Agent reasoning..."):
            trace = _run_agent(model_manager, rag_pipeline, question, max_steps, demo_mode)

        _render_trace(trace)


def _run_agent(model_manager, rag_pipeline, question, max_steps, demo_mode) -> ReActTrace:
    if demo_mode:
        return ReActAgent.demo_trace(question, rag_pipeline=rag_pipeline)

    def gen_fn(prompt: str) -> str:
        resp, _ = model_manager.generate(prompt, max_new_tokens=300, temperature=0.3, do_sample=True)
        return resp

    agent = ReActAgent(
        generate_fn=gen_fn,
        rag_pipeline=rag_pipeline,
        max_steps=max_steps,
    )
    return agent.run(question)


def _render_trace(trace: ReActTrace):
    st.divider()
    st.subheader(f"Question: {trace.question}")

    if trace.error:
        st.error(f"Agent error: {trace.error}")

    # Steps
    for i, step in enumerate(trace.steps, 1):
        st.markdown(f"**Step {i}**")
        cols = st.columns([1, 10])
        with cols[1]:
            # Thought
            st.markdown(
                f'<div style="background:#1a2744;padding:10px;border-radius:6px;'
                f'border-left:4px solid #00d4ff;margin-bottom:6px">'
                f'<b>Thought:</b> {step.thought}</div>',
                unsafe_allow_html=True,
            )
            # Action
            st.markdown(
                f'<div style="background:#2d1a00;padding:10px;border-radius:6px;'
                f'border-left:4px solid #ff9500;margin-bottom:6px">'
                f'<b>Action:</b> <code>{step.action}[{step.action_input}]</code></div>',
                unsafe_allow_html=True,
            )
            # Observation
            st.markdown(
                f'<div style="background:#1a2d1a;padding:10px;border-radius:6px;'
                f'border-left:4px solid #4caf50;margin-bottom:10px">'
                f'<b>Observation:</b> {step.observation[:500]}{"..." if len(step.observation) > 500 else ""}</div>',
                unsafe_allow_html=True,
            )

    # Final answer
    st.divider()
    st.markdown(
        f'<div style="background:#003322;padding:16px;border-radius:8px;'
        f'border:2px solid #00d4ff">'
        f'<b>Final Answer:</b><br><br>{trace.final_answer}</div>',
        unsafe_allow_html=True,
    )

    # Stats
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Reasoning Steps", trace.total_steps)
    with c2:
        st.metric("Time", f"{trace.elapsed_sec:.1f}s")
    with c3:
        tools_used = list(set(s.action for s in trace.steps))
        st.metric("Tools Used", f"{len(tools_used)}: {', '.join(tools_used)}" if tools_used else "0")
