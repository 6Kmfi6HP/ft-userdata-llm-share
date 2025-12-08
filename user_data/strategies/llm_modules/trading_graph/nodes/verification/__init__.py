"""
Verification Nodes for LangGraph.

Provides two verification approaches:
1. Grounding Node (legacy): Code-based comparison of claimed vs actual values
2. Reflection Node (new): LLM-based Chain-of-Verification for analysis validation

The Reflection Node is preferred as it:
- Better handles diverse LLM output formats
- Focuses on trading logic rather than numerical matching
- Provides actionable corrections in natural language
"""

from .grounding_node import (
    grounding_node,
    IndicatorClaim,
    GroundingResult,
)

from .reflection_node import (
    reflection_node,
    position_reflection_node,
    ReflectionResult,
)

__all__ = [
    # Grounding (legacy)
    "grounding_node",
    "IndicatorClaim",
    "GroundingResult",
    # Reflection (new - preferred)
    "reflection_node",
    "position_reflection_node",
    "ReflectionResult",
]
