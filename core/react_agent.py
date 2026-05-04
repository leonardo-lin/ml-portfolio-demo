"""
ReAct (Reasoning + Acting) Agent.
Uses regex-based action parsing and tool dispatch.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class ReActTrace:
    question: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using available tools.

Available tools:
- search_rag[query]: Search the knowledge base for relevant information about ML, QLoRA, RAG, etc.
- calculate[expression]: Evaluate a safe mathematical expression (numbers and operators only)
- lookup_graph[entity]: Find concepts related to a given entity in the knowledge graph

IMPORTANT: Always follow this exact format for each step:
Thought: <your reasoning about what to do>
Action: <tool_name>[<input>]
Observation: <tool result will appear here>

After gathering enough information, end with:
Final Answer: <your complete answer>

Use at most {max_steps} steps before giving a Final Answer."""

USER_PROMPT = """Question: {question}

Begin your reasoning:
"""


class ReActAgent:
    ACTION_PATTERN = re.compile(r"Action:\s*(\w+)\[([^\]]+)\]", re.IGNORECASE)
    FINAL_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
    THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        rag_pipeline=None,
        max_steps: int = 5,
    ):
        """
        generate_fn: callable(prompt) -> str  (model generation)
        rag_pipeline: MultimodalRAGPipeline instance (optional)
        """
        self._generate = generate_fn
        self._rag = rag_pipeline
        self._max_steps = max_steps
        self._tools: Dict[str, Callable] = self._build_tools()

    def _build_tools(self) -> Dict[str, Callable]:
        tools = {}

        if self._rag is not None:
            def search_rag(query: str) -> str:
                try:
                    result = self._rag.query(query, top_k=2)
                    if result.retrieved_chunks:
                        chunks_text = "\n\n".join(
                            f"[{c['source']} | score:{c['score']}] {c['text'][:300]}"
                            for c in result.retrieved_chunks
                        )
                        return chunks_text
                    return "No relevant information found in knowledge base."
                except Exception as e:
                    return f"Search error: {e}"
            tools["search_rag"] = search_rag
        else:
            tools["search_rag"] = lambda q: "Knowledge base not available."

        def calculate(expr: str) -> str:
            allowed = re.compile(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$')
            if not allowed.match(expr.strip()):
                return "Error: Only numeric expressions allowed."
            try:
                result = eval(expr, {"__builtins__": {}}, {})  # restricted eval
                return str(result)
            except Exception as e:
                return f"Calculation error: {e}"
        tools["calculate"] = calculate

        def lookup_graph(entity: str) -> str:
            if self._rag is None or self._rag._graph is None:
                return "Knowledge graph not available."
            G = self._rag._graph
            if not G.has_node(entity):
                # Try case-insensitive search
                matches = [n for n in G.nodes() if entity.lower() in n.lower()]
                if not matches:
                    return f"Entity '{entity}' not found in knowledge graph."
                entity = matches[0]
            neighbors = list(G.neighbors(entity))[:5]
            if not neighbors:
                return f"'{entity}' has no connections in the knowledge graph."
            return f"Entities related to '{entity}': {', '.join(neighbors)}"
        tools["lookup_graph"] = lookup_graph

        return tools

    def _call_tool(self, action: str, action_input: str) -> str:
        tool_fn = self._tools.get(action.lower())
        if tool_fn is None:
            available = ", ".join(self._tools.keys())
            return f"Unknown tool '{action}'. Available: {available}"
        return tool_fn(action_input.strip())

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse model output into (thought, action, action_input, final_answer)."""
        thought_match = self.THOUGHT_PATTERN.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""

        final_match = self.FINAL_PATTERN.search(text)
        if final_match:
            return thought, None, None, final_match.group(1).strip()

        action_match = self.ACTION_PATTERN.search(text)
        if action_match:
            return thought, action_match.group(1), action_match.group(2), None

        # No action found - treat whole output as final answer
        return thought, None, None, text.strip()

    def run(self, question: str) -> ReActTrace:
        trace = ReActTrace(question=question)
        t0 = time.time()
        scratchpad = ""

        for step_num in range(self._max_steps):
            prompt = (
                SYSTEM_PROMPT.format(max_steps=self._max_steps)
                + "\n\n"
                + USER_PROMPT.format(question=question)
                + scratchpad
            )

            try:
                output = self._generate(prompt)
            except Exception as e:
                trace.error = str(e)
                trace.final_answer = f"Generation error: {e}"
                break

            thought, action, action_input, final_answer = self._parse_output(output)

            if final_answer is not None:
                trace.final_answer = final_answer
                trace.total_steps = step_num
                break

            if action is not None:
                observation = self._call_tool(action, action_input or "")
                step = ReActStep(
                    thought=thought,
                    action=action,
                    action_input=action_input or "",
                    observation=observation,
                )
                trace.steps.append(step)
                scratchpad += (
                    f"\nThought: {thought}\n"
                    f"Action: {action}[{action_input}]\n"
                    f"Observation: {observation}\n"
                )
            else:
                # No action, no final answer — treat as final
                trace.final_answer = output.strip() or "No answer generated."
                trace.total_steps = step_num
                break
        else:
            if not trace.final_answer:
                trace.final_answer = "Maximum steps reached. Please try a more specific question."
            trace.total_steps = self._max_steps

        trace.elapsed_sec = round(time.time() - t0, 2)
        if trace.total_steps == 0:
            trace.total_steps = len(trace.steps)
        return trace

    # ── Demo mode (no live model) ─────────────────────────────────────────────

    @classmethod
    def demo_trace(cls, question: str, rag_pipeline=None) -> ReActTrace:
        """Return a pre-built demo trace for showing UI without a loaded model."""
        trace = ReActTrace(question=question)

        search_fn = None
        if rag_pipeline is not None:
            try:
                result = rag_pipeline.query("QLoRA advantages memory efficiency", top_k=1)
                obs1 = result.retrieved_chunks[0]["text"][:300] if result.retrieved_chunks else \
                    "QLoRA reduces VRAM requirements by using 4-bit NF4 quantization, enabling 65B model training on 2x A100 40GB."
            except Exception:
                obs1 = "QLoRA reduces VRAM requirements by using 4-bit NF4 quantization."
        else:
            obs1 = "QLoRA reduces VRAM requirements by using 4-bit NF4 quantization, enabling 65B model training on ~46GB VRAM."

        trace.steps = [
            ReActStep(
                thought="I need to find specific information about QLoRA advantages over full fine-tuning.",
                action="search_rag",
                action_input="QLoRA advantages memory efficiency comparison",
                observation=obs1,
            ),
            ReActStep(
                thought="Now I have information about memory. Let me also check quality comparison data.",
                action="search_rag",
                action_input="QLoRA quality performance MT-Bench",
                observation="QLoRA achieves 99.3% of ChatGPT-level quality on Vicuna benchmark while requiring only 46GB VRAM for 65B models, vs 400GB+ for full fine-tuning.",
            ),
        ]
        trace.final_answer = (
            "QLoRA offers three key advantages over full fine-tuning:\n\n"
            "1. **Memory Efficiency**: Reduces VRAM requirements by ~4x using 4-bit NF4 quantization. "
            "A 65B model requires only ~46GB VRAM for training vs 400GB+ for full FT.\n\n"
            "2. **Accessibility**: Enables fine-tuning on consumer/prosumer hardware (2x A100 40GB) "
            "instead of requiring expensive data center infrastructure.\n\n"
            "3. **Minimal Quality Loss**: Achieves 99.3% of full fine-tuning quality on benchmarks, "
            "making the memory-quality tradeoff highly favorable."
        )
        trace.total_steps = len(trace.steps)
        trace.elapsed_sec = 3.2
        return trace
