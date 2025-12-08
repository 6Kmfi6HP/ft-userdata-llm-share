"""
Analysis Subgraph Builder for LangGraph.

Builds the analysis subgraph with 4 parallel analysis agents (fan-out)
followed by an aggregator node (fan-in).

Architecture:
    START
      │
      ├──► [indicator_node] ──┐
      ├──► [trend_node]     ──┼──► [aggregator_node] ──► END
      ├──► [sentiment_node] ──┤
      └──► [pattern_node]   ──┘
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from ..state import AnalysisState
from ..nodes.analysis import (
    indicator_node,
    trend_node,
    sentiment_node,
    pattern_node,
    aggregator_node,
)

logger = logging.getLogger(__name__)


def build_analysis_subgraph(
    retry_max_attempts: int = 2,
    parallel_execution: bool = True
) -> StateGraph:
    """
    Build the 4-agent parallel analysis subgraph.

    Args:
        retry_max_attempts: Max retry attempts for each node
        parallel_execution: Whether to run agents in parallel (fan-out pattern)

    Returns:
        Compiled StateGraph for analysis
    """
    logger.info("Building analysis subgraph with parallel fan-out/fan-in pattern")

    # Create graph builder with AnalysisState schema
    builder = StateGraph(AnalysisState)

    # Add analysis nodes
    builder.add_node("indicator", indicator_node)
    builder.add_node("trend", trend_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("pattern", pattern_node)
    builder.add_node("aggregator", aggregator_node)

    if parallel_execution:
        # Fan-out: START → all analysis nodes (parallel execution)
        builder.add_edge(START, "indicator")
        builder.add_edge(START, "trend")
        builder.add_edge(START, "sentiment")
        builder.add_edge(START, "pattern")

        # Fan-in: all analysis nodes → aggregator
        builder.add_edge("indicator", "aggregator")
        builder.add_edge("trend", "aggregator")
        builder.add_edge("sentiment", "aggregator")
        builder.add_edge("pattern", "aggregator")
    else:
        # Sequential execution (for debugging)
        builder.add_edge(START, "indicator")
        builder.add_edge("indicator", "trend")
        builder.add_edge("trend", "sentiment")
        builder.add_edge("sentiment", "pattern")
        builder.add_edge("pattern", "aggregator")

    # Aggregator → END
    builder.add_edge("aggregator", END)

    logger.info(
        f"Analysis subgraph built: "
        f"parallel={parallel_execution}, nodes=5 (4 agents + aggregator)"
    )

    return builder.compile()


def get_analysis_subgraph_description() -> str:
    """Get description of the analysis subgraph for documentation."""
    return """
Analysis Subgraph - 4 Agent Parallel Analysis

Nodes:
  1. indicator_node: Technical indicator analysis (RSI, MACD, ADX, Stochastic)
  2. trend_node: Trend structure analysis (EMA, support/resistance, visual)
  3. sentiment_node: Market sentiment analysis (funding, OI, Fear & Greed)
  4. pattern_node: K-line pattern recognition (candlestick, chart patterns)
  5. aggregator_node: Weighted consensus calculation

Flow:
  START → [4 agents in parallel] → aggregator → END

Agent Weights:
  - IndicatorAgent: 1.0 (baseline)
  - TrendAgent: 1.2 (highest - trend is king)
  - SentimentAgent: 0.8 (lowest - auxiliary)
  - PatternAgent: 1.1 (medium-high)

Output:
  - consensus_direction: LONG / SHORT / NEUTRAL
  - consensus_confidence: 0-100
  - key_support: Support price level
  - key_resistance: Resistance price level
  - weighted_scores: Direction score breakdown
"""
