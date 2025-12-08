"""
LangGraph node implementations.

Organized into:
- analysis/: 4 specialized analysis agent nodes + aggregator
- debate/: Bull, Bear, and Judge agent nodes
- position/: Position management debate nodes
- verification/: Layer 4 grounding verification node
- execution/: Validator and executor nodes
"""

from .analysis import (
    indicator_node,
    trend_node,
    sentiment_node,
    pattern_node,
    aggregator_node,
)
from .debate import (
    bull_node,
    bear_node,
    judge_node,
)
from .position import (
    position_bull_node,
    position_bear_node,
    position_judge_node,
    position_grounding_node,
)
from .verification import (
    grounding_node,
    IndicatorClaim,
    GroundingResult,
)
from .execution import (
    validator_node,
    executor_node,
)

__all__ = [
    # Analysis nodes
    "indicator_node",
    "trend_node",
    "sentiment_node",
    "pattern_node",
    "aggregator_node",
    # Debate nodes
    "bull_node",
    "bear_node",
    "judge_node",
    # Position management nodes
    "position_bull_node",
    "position_bear_node",
    "position_judge_node",
    "position_grounding_node",
    # Verification nodes
    "grounding_node",
    "IndicatorClaim",
    "GroundingResult",
    # Execution nodes
    "validator_node",
    "executor_node",
]
