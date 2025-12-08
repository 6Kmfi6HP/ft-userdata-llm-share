"""
LangGraph subgraphs for modular graph composition.

Provides:
- Analysis subgraph: 4 agents running in parallel (fan-out/fan-in)
- Debate subgraph: Bull → Bear → Judge sequential debate (for entry)
- Position debate subgraph: Position Bull → Bear → Judge (for position management)
- Position subgraph: (Legacy) Position Bull → Bear → Judge → Grounding
"""

from .analysis_graph import build_analysis_subgraph
from .debate_graph import build_debate_subgraph
from .position_debate_graph import build_position_debate_subgraph
from .position_graph import build_position_subgraph  # Legacy

__all__ = [
    "build_analysis_subgraph",
    "build_debate_subgraph",
    "build_position_debate_subgraph",
    "build_position_subgraph",  # Legacy
]
