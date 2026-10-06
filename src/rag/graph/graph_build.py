from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from rag.graph.nodes.generator import generator_node
from rag.graph.nodes.retriever import retriever_node
from rag.graph.state import GraphState


def build_rag_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(GraphState)

    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator_node)

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile(checkpointer=checkpointer)