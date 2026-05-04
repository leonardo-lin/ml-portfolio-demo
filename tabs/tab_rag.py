"""
Tab 3: RAG Pipeline Demo
Multimodal upload (audio/image/text) + indexing + query + knowledge graph visualization.
"""

import os
import tempfile

import streamlit as st

try:
    from core.rag_pipeline import MultimodalRAGPipeline
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERROR = str(_e)


def render(rag_pipeline=None):
    st.header("Multimodal RAG Pipeline")
    if not _DEPS_OK:
        st.error(f"Missing dependency: `{_DEPS_ERROR}`")
        st.info("Please install: `pip install chromadb sentence-transformers langchain langchain-community`")
        return
    st.markdown(
        "Upload audio, images, or text documents. The pipeline transcribes audio via **Whisper**, "
        "generates image captions via **BLIP**, embeds all content into **ChromaDB**, "
        "and builds a **knowledge graph** for chained retrieval."
    )

    # ── Section 1: Data Ingestion ────────────────────────────────────────────
    st.subheader("1  Data Ingestion")
    col_audio, col_image, col_text = st.columns(3)

    uploaded_audio = None
    uploaded_image = None
    uploaded_text = None

    with col_audio:
        st.markdown("**Audio (WAV/MP3)**")
        uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"],
                                           key="rag_audio", label_visibility="collapsed")
        if uploaded_audio:
            st.audio(uploaded_audio)

    with col_image:
        st.markdown("**Image (JPG/PNG)**")
        uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                                           key="rag_image", label_visibility="collapsed")
        if uploaded_image:
            st.image(uploaded_image, width=200)

    with col_text:
        st.markdown("**Text Document**")
        uploaded_text = st.file_uploader("Upload text", type=["txt", "md"],
                                          key="rag_text", label_visibility="collapsed")
        if uploaded_text:
            content = uploaded_text.read().decode("utf-8")
            st.caption(f"{len(content)} characters")

    # Default knowledge base
    st.divider()
    col_default, col_reset = st.columns([3, 1])
    with col_default:
        use_default = st.checkbox(
            "Also index built-in knowledge base (QLoRA / RAG / ML docs)",
            value=True
        )
    with col_reset:
        if st.button("Reset Index", type="secondary"):
            rag_pipeline.reset_collection()
            st.session_state["rag_indexed"] = False
            st.success("Index cleared.")

    if st.button("Process & Index", type="primary"):
        _process_and_index(rag_pipeline, uploaded_audio, uploaded_image, uploaded_text, use_default)

    # Show current index status
    count = rag_pipeline.get_collection_count()
    if count > 0:
        st.success(f"Index ready: {count} chunks stored in ChromaDB")
    else:
        st.info("No documents indexed yet. Upload files or use the built-in knowledge base.")

    # ── Section 2: Knowledge Graph ───────────────────────────────────────────
    st.divider()
    st.subheader("2  Knowledge Graph")
    fig_graph = rag_pipeline.get_graph_plotly(max_nodes=40)
    st.plotly_chart(fig_graph, use_container_width=True)

    # ── Section 3: Query ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("3  Query the Knowledge Base")

    sample_questions = [
        "What are the key advantages of QLoRA over full fine-tuning?",
        "How much VRAM does QLoRA 70B require?",
        "How does Whisper process audio for RAG pipelines?",
        "What is the difference between NF4 and INT4 quantization?",
        "How does knowledge graph expansion improve RAG retrieval?",
    ]

    q_choice = st.selectbox("Sample Questions", ["(custom)"] + sample_questions)
    question = st.text_input(
        "Your Question",
        value=q_choice if q_choice != "(custom)" else "",
        placeholder="Ask something about QLoRA, RAG, or the uploaded documents..."
    )
    top_k = st.slider("Top-K Chunks", 1, 5, 3)

    if st.button("Run RAG Query", type="primary", disabled=not question.strip() or count == 0):
        with st.spinner("Retrieving..."):
            result = rag_pipeline.query(question, top_k=top_k)

        # Retrieved chunks
        st.subheader("Retrieved Chunks")
        for i, chunk in enumerate(result.retrieved_chunks):
            with st.expander(
                f"Chunk {i+1}  |  Score: {chunk['score']:.3f}  |  Source: {chunk['source']}",
                expanded=(i == 0)
            ):
                st.markdown(chunk["text"])

        # Graph-expanded context
        if result.graph_context:
            with st.expander(f"Knowledge Graph Expansion (+{len(result.graph_context)} chunks)"):
                for gc in result.graph_context:
                    st.markdown(f"> {gc}")

        # Pipeline trace
        st.divider()
        st.subheader("Pipeline Trace")
        trace_cols = st.columns(5)
        steps = ["Upload", "Chunk/Embed", "Vector Search", "Graph Expand", "Context Built"]
        colors = ["#1e3a5f", "#1e3a5f", "#004466", "#005577", "#006688"]
        for col, step, color in zip(trace_cols, steps, colors):
            with col:
                st.markdown(
                    f'<div style="background:{color};padding:8px;border-radius:6px;'
                    f'text-align:center;font-size:12px">{step}</div>',
                    unsafe_allow_html=True
                )

        # Final context preview
        with st.expander("Final Context (sent to LLM)"):
            st.text(result.final_context[:2000] + ("..." if len(result.final_context) > 2000 else ""))

        st.info(
            "To generate an actual LLM answer, load a model in **QLoRA Training** tab "
            "and use the **ReAct Agent** tab for full pipeline."
        )


def _process_and_index(rag_pipeline, uploaded_audio, uploaded_image, uploaded_text, use_default):
    chunks_total = 0
    all_chunks = []

    with st.status("Processing documents...", expanded=True) as status:
        # Audio
        if uploaded_audio is not None:
            st.write("Transcribing audio with Whisper small...")
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded_audio.read())
                    tmp_path = tmp.name
                chunks = rag_pipeline.ingest_audio(tmp_path)
                os.unlink(tmp_path)
                all_chunks.extend(chunks)
                st.write(f"Audio: {len(chunks)} chunks from transcript")
            except Exception as e:
                st.warning(f"Audio processing failed (ffmpeg required): {e}")

        # Image
        if uploaded_image is not None:
            st.write("Generating image caption with BLIP...")
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded_image.read())
                    tmp_path = tmp.name
                caption = rag_pipeline.ingest_image(tmp_path)
                os.unlink(tmp_path)
                all_chunks.append({"text": caption, "source": uploaded_image.name})
                st.write(f"Image caption: \"{caption[:80]}...\"")
            except Exception as e:
                st.warning(f"Image captioning failed: {e}")

        # Text
        if uploaded_text is not None:
            st.write("Chunking text document...")
            content = uploaded_text.read().decode("utf-8") if not isinstance(uploaded_text, str) else uploaded_text
            chunks = rag_pipeline.ingest_text_content(content, source=uploaded_text.name)
            all_chunks.extend(chunks)
            st.write(f"Text: {len(chunks)} chunks")

        # Default knowledge base
        if use_default:
            st.write("Indexing built-in ML knowledge base...")
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(here, "data", "rag_docs", "knowledge_docs.txt")
            if os.path.exists(default_path):
                chunks = rag_pipeline.ingest_text(default_path)
                all_chunks.extend(chunks)
                st.write(f"Knowledge base: {len(chunks)} chunks")

        # Build vector index
        if all_chunks:
            st.write(f"Embedding {len(all_chunks)} chunks into ChromaDB...")
            n = rag_pipeline.build_index(all_chunks)
            chunks_total += n

            st.write("Building knowledge graph...")
            rag_pipeline.build_knowledge_graph(all_chunks)

            st.session_state["rag_indexed"] = True
            status.update(label=f"Done! {chunks_total} chunks indexed.", state="complete")
        else:
            status.update(label="Nothing to index. Upload files or enable knowledge base.", state="error")
