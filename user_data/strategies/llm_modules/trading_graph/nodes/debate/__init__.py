"""
Debate agent nodes for LangGraph.

Implements Bull vs Bear adversarial debate system:
- BullNode: Opportunity Finder - makes strongest case FOR the trade
- BearNode: Devil's Advocate - finds every possible flaw
- JudgeNode: Impartial arbiter - evaluates arguments and renders verdict
"""

from .bull_node import bull_node
from .bear_node import bear_node
from .judge_node import judge_node

__all__ = [
    "bull_node",
    "bear_node",
    "judge_node",
]
