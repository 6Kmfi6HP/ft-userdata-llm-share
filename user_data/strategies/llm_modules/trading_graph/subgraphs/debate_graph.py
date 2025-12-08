"""
Debate Subgraph Builder for LangGraph.

Builds the debate subgraph with Bull → Bear → Judge sequential flow.

Architecture:
    START ──► [bull_node] ──► [bear_node] ──► [judge_node] ──► END
"""

import logging

from langgraph.graph import StateGraph, START, END

from ..state import TradingDecisionState
from ..nodes.debate import (
    bull_node,
    bear_node,
    judge_node,
)

logger = logging.getLogger(__name__)


def build_debate_subgraph() -> StateGraph:
    """
    Build the Bull → Bear → Judge debate subgraph.

    Sequential execution flow:
    1. Bull makes case FOR the trade
    2. Bear challenges and finds flaws
    3. Judge evaluates and renders verdict

    Returns:
        Compiled StateGraph for debate
    """
    logger.info("Building debate subgraph with sequential Bull→Bear→Judge flow")

    # Create graph builder with TradingDecisionState schema
    builder = StateGraph(TradingDecisionState)

    # Add debate nodes
    builder.add_node("bull", bull_node)
    builder.add_node("bear", bear_node)
    builder.add_node("judge", judge_node)

    # Sequential execution: Bull → Bear → Judge
    builder.add_edge(START, "bull")
    builder.add_edge("bull", "bear")
    builder.add_edge("bear", "judge")
    builder.add_edge("judge", END)

    logger.info("Debate subgraph built: Bull → Bear → Judge")

    return builder.compile()


def get_debate_subgraph_description() -> str:
    """Get description of the debate subgraph for documentation."""
    return """
Debate Subgraph - Bull vs Bear Adversarial Debate

Nodes:
  1. bull_node: Opportunity Finder - makes strongest case FOR the trade
  2. bear_node: Devil's Advocate - finds every possible flaw
  3. judge_node: Impartial Arbiter - evaluates and renders final verdict

Flow:
  START → bull → bear → judge → END

Bull Agent:
  - Identifies trading opportunities based on analysis
  - Builds supporting arguments with evidence
  - Analyzes potential rewards
  - Arguments for why risks are manageable

Bear Agent:
  - Challenges Bull's arguments
  - Identifies hidden risks and contradictions
  - Analyzes failure modes
  - Provides counter-arguments with evidence

Judge Agent:
  - Evaluates both arguments (40% evidence, 30% logic, 30% risk)
  - Determines winner (Bull/Bear/Tie)
  - Renders verdict (APPROVE/REJECT/ABSTAIN)
  - Calibrates confidence: (debate_conf + original_conf) / 2

Verdict Options:
  - APPROVE: Bull wins - trade is sound, execute
  - REJECT: Bear wins - too risky, skip
  - ABSTAIN: Balanced arguments - wait for better setup
"""
