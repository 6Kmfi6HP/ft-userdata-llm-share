"""
Position Debate Subgraph Builder for LangGraph.

Builds the position debate subgraph with:
Position Bull → Position Bear → Position Judge sequential flow.

NOTE: This is the NEW version that does NOT include grounding.
Grounding is now handled at the main graph level for unified processing.

Architecture:
    START ──► [position_bull_node] ──► [position_bear_node] ──►
              [position_judge_node] ──► END

Reference: LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md
"""

import logging

from langgraph.graph import StateGraph, START, END

from ..state import TradingDecisionState
from ..nodes.position import (
    position_bull_node,
    position_bear_node,
    position_judge_node,
)

logger = logging.getLogger(__name__)


def build_position_debate_subgraph() -> StateGraph:
    """
    Build the Position Debate subgraph (without grounding).

    Sequential execution flow:
    1. Position Bull makes case FOR holding/scaling
    2. Position Bear challenges and advocates for reducing/exiting
    3. Position Judge evaluates and renders verdict

    NOTE: Grounding is now handled at the main graph level.

    Returns:
        Compiled StateGraph for position debate
    """
    logger.info("Building position debate subgraph (without grounding)")

    # Create graph builder with TradingDecisionState schema
    builder = StateGraph(TradingDecisionState)

    # Add position debate nodes
    builder.add_node("position_bull", position_bull_node)
    builder.add_node("position_bear", position_bear_node)
    builder.add_node("position_judge", position_judge_node)

    # Sequential execution: Bull → Bear → Judge
    builder.add_edge(START, "position_bull")
    builder.add_edge("position_bull", "position_bear")
    builder.add_edge("position_bear", "position_judge")
    builder.add_edge("position_judge", END)

    logger.info("Position debate subgraph built: Bull → Bear → Judge")

    return builder.compile()


def get_position_debate_subgraph_description() -> str:
    """Get description of the position debate subgraph for documentation."""
    return """
Position Debate Subgraph - Adversarial Debate for Existing Positions

Purpose:
  Specialized debate for managing existing positions, deciding whether to:
  - HOLD: Continue holding position unchanged
  - EXIT: Close entire position
  - SCALE_IN: Add to position (+20% ~ +50%)
  - PARTIAL_EXIT: Reduce position (-30% ~ -70%)

Nodes:
  1. position_bull_node: Advocates FOR holding/scaling
     - Analyzes trend continuation evidence
     - Evaluates drawback reasonableness
     - Identifies scaling opportunities

  2. position_bear_node: Advocates FOR reducing/exiting
     - Analyzes MFE drawback severity
     - Detects stuck-in-loop patterns
     - Applies profit protection rules

  3. position_judge_node: Renders position verdict
     - Evaluates both arguments
     - Advisory profit protection rules (>50% MFE drawback + hold_count≥3)
     - Returns HOLD/EXIT/SCALE_IN/PARTIAL_EXIT + adjustment percentage

Flow:
  START → position_bull → position_bear → position_judge → END

Position Metrics Used:
  - MFE (Maximum Favorable Excursion): Peak unrealized profit
  - MAE (Maximum Adverse Excursion): Peak unrealized loss
  - Drawdown from peak: (MFE - current) / MFE
  - Hold count: Consecutive HOLD decisions
  - Time in position: Hours since entry

Token Usage:
  - 3 LLM calls: position_bull + position_bear + position_judge
"""
