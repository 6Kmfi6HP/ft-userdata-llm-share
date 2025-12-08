"""
Position Management Subgraph Builder for LangGraph.

Builds the position management subgraph with:
Position Bull → Position Bear → Position Judge → Position Grounding sequential flow.

Architecture:
    START ──► [position_bull_node] ──► [position_bear_node] ──►
              [position_judge_node] ──► [position_grounding_node] ──► END

Reference: LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md
- Position-specific adversarial debate for existing positions
- Layer 4 grounding verification for position claims
"""

import logging

from langgraph.graph import StateGraph, START, END

from ..state import TradingDecisionState
from ..nodes.position import (
    position_bull_node,
    position_bear_node,
    position_judge_node,
    position_grounding_node,
)

logger = logging.getLogger(__name__)


def build_position_subgraph() -> StateGraph:
    """
    Build the Position Management debate subgraph.

    Sequential execution flow:
    1. Position Bull makes case FOR holding/scaling
    2. Position Bear challenges and advocates for reducing/exiting
    3. Position Judge evaluates and renders verdict
    4. Position Grounding verifies claims against actual metrics

    Returns:
        Compiled StateGraph for position management
    """
    logger.info("Building position management subgraph")

    # Create graph builder with TradingDecisionState schema
    builder = StateGraph(TradingDecisionState)

    # Add position debate nodes
    builder.add_node("position_bull", position_bull_node)
    builder.add_node("position_bear", position_bear_node)
    builder.add_node("position_judge", position_judge_node)
    builder.add_node("position_grounding", position_grounding_node)

    # Sequential execution: Bull → Bear → Judge → Grounding
    builder.add_edge(START, "position_bull")
    builder.add_edge("position_bull", "position_bear")
    builder.add_edge("position_bear", "position_judge")
    builder.add_edge("position_judge", "position_grounding")
    builder.add_edge("position_grounding", END)

    logger.info("Position subgraph built: Bull → Bear → Judge → Grounding")

    return builder.compile()


def get_position_subgraph_description() -> str:
    """Get description of the position subgraph for documentation."""
    return """
Position Management Subgraph - Adversarial Debate for Existing Positions

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

  4. position_grounding_node: Layer 4 verification
     - Verifies MFE/MAE/drawdown claims
     - Verifies hold_count claims
     - Applies confidence penalty for false claims

Flow:
  START → position_bull → position_bear → position_judge → position_grounding → END

Position Metrics Used:
  - MFE (Maximum Favorable Excursion): Peak unrealized profit
  - MAE (Maximum Adverse Excursion): Peak unrealized loss
  - Drawdown from peak: (MFE - current) / MFE
  - Hold count: Consecutive HOLD decisions
  - Time in position: Hours since entry

Profit Protection Rules (Advisory):
  - MFE drawback >50% + hold_count ≥3: Suggest PARTIAL_EXIT -50%
  - MFE drawback >30% + stuck_in_loop: Suggest PARTIAL_EXIT -30%
  - MFE drawback >70%: Suggest EXIT

  Note: These rules strengthen Bear's arguments but Judge makes final decision
        based on full analysis context.

Token Usage:
  - 3 LLM calls: position_bull + position_bear + position_judge
  - 0 LLM calls: position_grounding (code-based verification)
"""
