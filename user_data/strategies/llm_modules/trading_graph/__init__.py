"""
LangGraph state machine module for LLM trading system.

This module provides:
- State schemas: TypedDict definitions for graph state
- Main graph: Primary trading decision graph
- Subgraphs: Analysis (parallel fan-out/in) and Debate (Bull→Bear→Judge)
- Nodes: Analysis agents, debate agents, and execution nodes
- Edges: Conditional routing functions
- Logging: GraphDecisionLogger for complete decision chain logging
"""

from .state import (
    Direction,
    Verdict,
    Signal,
    AgentReport,
    DebateArgument,
    JudgeVerdict,
    AnalysisState,
    DebateState,
    TradingDecisionState,
    AGENT_WEIGHTS,
)
from .main_graph import (
    build_main_graph,
    TradingGraphRunner,
    get_graph_description,
)
from .logging import GraphDecisionLogger

__all__ = [
    # Enums
    "Direction",
    "Verdict",
    # Data classes
    "Signal",
    "AgentReport",
    "DebateArgument",
    "JudgeVerdict",
    # States
    "AnalysisState",
    "DebateState",
    "TradingDecisionState",
    # Constants
    "AGENT_WEIGHTS",
    # Graph builders
    "build_main_graph",
    "TradingGraphRunner",
    "get_graph_description",
    # Logging
    "GraphDecisionLogger",
]
