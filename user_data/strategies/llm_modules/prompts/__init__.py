"""
Centralized prompt management for LLM trading system.

Organized into:
- analysis/: Prompts for 4 analysis agents
- debate/: Prompts for Bull, Bear, and Judge agents
"""

from .analysis import (
    INDICATOR_SYSTEM_PROMPT,
    TREND_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    PATTERN_SYSTEM_PROMPT,
)
from .debate import (
    BULL_SYSTEM_PROMPT,
    BEAR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)

__all__ = [
    # Analysis prompts
    "INDICATOR_SYSTEM_PROMPT",
    "TREND_SYSTEM_PROMPT",
    "SENTIMENT_SYSTEM_PROMPT",
    "PATTERN_SYSTEM_PROMPT",
    # Debate prompts
    "BULL_SYSTEM_PROMPT",
    "BEAR_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
]
