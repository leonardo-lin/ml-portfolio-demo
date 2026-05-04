"""
Multimodal RAG Pipeline.
Handles: audio (Whisper) -> text, image (BLIP) -> caption, text -> chunks.
Vector store: ChromaDB. Knowledge graph: NetworkX.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RAGResult:
    question: str
    retrieved_chunks: List[Dict]   # [{text, score, source, chunk_id}]
    graph_context: List[str]       # extra chunks from KG expansion
    final_context: str
    answer: str = ""
    sources: List[str] = field(default_factory=list)


class MultimodalRAGPipeline:
    CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "chroma_db")
    COLLECTION_NAME = "qlora_demo"

    def __init__(self):
        self._collection = None
        self._graph = None
        self._embedder = None
        self._whisper_model = None
        self._blip_processor = None
        self._blip_model = None
        self._indexed_chunks: List[Dict] = []  # for graph building

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.CHROMA_DIR)
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_whisper(self):
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model("small")
        return self._whisper_model

    def _get_blip(self):
        if self._blip_processor is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            model_id = "Salesforce/blip-image-captioning-base"
            self._blip_processor = BlipProcessor.from_pretrained(model_id)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(model_id)
            self._blip_model.eval()
        return self._blip_processor, self._blip_model

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_audio(self, audio_path: str) -> List[str]:
        """Transcribe audio and return sentence-level chunks."""
        model = self._get_whisper()
        result = model.transcribe(audio_path, language="en")
        text = result["text"]
        return self._chunk_text(text, source=os.path.basename(audio_path))

    def ingest_image(self, image_path: str) -> str:
        """Generate descriptive caption from image using BLIP."""
        from PIL import Image
        processor, model = self._get_blip()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def ingest_text(self, text_path: str) -> List[str]:
        """Read text file and return chunks."""
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._chunk_text(content, source=os.path.basename(text_path))

    def ingest_text_content(self, content: str, source: str = "manual") -> List[str]:
        """Ingest raw text content."""
        return self._chunk_text(content, source=source)

    def _chunk_text(self, text: str, source: str = "", chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 20:  # skip tiny fragments
                chunks.append({"text": chunk, "source": source})
            start += chunk_size - overlap
        return chunks

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Dict]) -> int:
        """Embed and store chunks in ChromaDB. Returns count added."""
        if not chunks:
            return 0
        collection = self._get_collection()
        embedder = self._get_embedder()

        texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        sources = [c.get("source", "unknown") if isinstance(c, dict) else "unknown" for c in chunks]

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        ids = [f"chunk_{collection.count()}_{i}" for i in range(len(texts))]
        metadatas = [{"source": s, "chunk_id": i} for i, s in enumerate(sources)]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self._indexed_chunks.extend(chunks)
        return len(texts)

    def build_knowledge_graph(self, chunks: List[Dict] = None):
        """Build NetworkX graph from noun-phrase co-occurrence."""
        import networkx as nx
        if chunks is None:
            chunks = self._indexed_chunks

        G = nx.Graph()
        entity_pattern = re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b')

        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            sentences = re.split(r'[.!?]', text)
            for sent in sentences:
                entities = list(set(entity_pattern.findall(sent)))
                entities = [e for e in entities if len(e) > 2 and e not in {"The", "This", "That", "These"}]
                for e in entities:
                    G.add_node(e)
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        if G.has_edge(entities[i], entities[j]):
                            G[entities[i]][entities[j]]["weight"] += 1
                        else:
                            G.add_edge(entities[i], entities[j], weight=1)

        self._graph = G
        return G

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 3) -> RAGResult:
        """Retrieve relevant chunks + graph expansion."""
        collection = self._get_collection()
        embedder = self._get_embedder()

        query_emb = embedder.encode([question]).tolist()
        n_results = min(top_k, collection.count())
        if n_results == 0:
            return RAGResult(
                question=question,
                retrieved_chunks=[],
                graph_context=[],
                final_context="No documents indexed yet.",
                answer="Please index some documents first.",
            )

        results = collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        retrieved = []
        for text, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            retrieved.append({
                "text": text,
                "score": round(1 - dist, 4),  # cosine: 1 - distance
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
            })

        # Graph expansion
        graph_context = []
        if self._graph is not None:
            graph_context = self._expand_via_graph(question, retrieved)

        # Build final context
        context_parts = [f"[Source: {r['source']}]\n{r['text']}" for r in retrieved]
        if graph_context:
            context_parts.append("[Related Context from Knowledge Graph]\n" + "\n".join(graph_context))
        final_context = "\n\n---\n\n".join(context_parts)

        return RAGResult(
            question=question,
            retrieved_chunks=retrieved,
            graph_context=graph_context,
            final_context=final_context,
            sources=list(set(r["source"] for r in retrieved)),
        )

    def _expand_via_graph(self, question: str, retrieved: List[Dict]) -> List[str]:
        """Find related entities from question and expand context via graph."""
        if self._graph is None:
            return []
        entity_pattern = re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b')
        q_entities = entity_pattern.findall(question)
        expanded = []
        for entity in q_entities:
            if self._graph.has_node(entity):
                neighbors = list(self._graph.neighbors(entity))[:3]
                for n in neighbors:
                    for chunk in self._indexed_chunks:
                        text = chunk["text"] if isinstance(chunk, dict) else chunk
                        if n.lower() in text.lower() and text not in [r["text"] for r in retrieved]:
                            expanded.append(text[:300])
                            break
        return expanded[:2]  # max 2 graph-expanded chunks

    def get_graph_plotly(self, max_nodes: int = 50):
        """Convert NetworkX graph to Plotly figure."""
        import plotly.graph_objects as go

        G = self._graph
        if G is None or len(G.nodes) == 0:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                title="Knowledge Graph (no data)",
                annotations=[dict(text="Index documents to build graph", showarrow=False,
                                   xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            return fig

        import networkx as nx
        # Limit to top-degree nodes for readability
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [n for n, _ in top_nodes]
        subgraph = G.subgraph(top_node_names)

        pos = nx.spring_layout(subgraph, seed=42, k=2.0)

        edge_x, edge_y = [], []
        for u, v in subgraph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#555"),
            hoverinfo="none",
        )

        node_x = [pos[n][0] for n in subgraph.nodes()]
        node_y = [pos[n][1] for n in subgraph.nodes()]
        node_degree = [subgraph.degree(n) for n in subgraph.nodes()]
        node_text = [f"{n} (degree: {d})" for n, d in zip(subgraph.nodes(), node_degree)]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            hoverinfo="text",
            text=list(subgraph.nodes()),
            hovertext=node_text,
            textposition="top center",
            textfont=dict(size=8, color="#cccccc"),
            marker=dict(
                showscale=True,
                colorscale="Blues",
                color=node_degree,
                size=[max(8, min(20, d * 3)) for d in node_degree],
                colorbar=dict(thickness=10, title="Degree", xanchor="left"),
                line_width=1,
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            template="plotly_dark",
            title=f"Knowledge Graph ({len(subgraph.nodes())} entities, {len(subgraph.edges())} relations)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def get_collection_count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def reset_collection(self):
        """Delete and recreate the collection."""
        import chromadb
        client = chromadb.PersistentClient(path=self.CHROMA_DIR)
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        self._indexed_chunks = []
        self._graph = None
