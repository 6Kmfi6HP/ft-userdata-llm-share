"""
Analysis agent prompts.

Provides system prompts for:
- IndicatorAgent: Technical indicator analysis
- TrendAgent: Trend structure analysis
- SentimentAgent: Market sentiment analysis
- PatternAgent: K-line pattern recognition
"""

from .indicator_prompt import INDICATOR_SYSTEM_PROMPT
from .trend_prompt import TREND_SYSTEM_PROMPT
from .sentiment_prompt import SENTIMENT_SYSTEM_PROMPT
from .pattern_prompt import PATTERN_SYSTEM_PROMPT

__all__ = [
    "INDICATOR_SYSTEM_PROMPT",
    "TREND_SYSTEM_PROMPT",
    "SENTIMENT_SYSTEM_PROMPT",
    "PATTERN_SYSTEM_PROMPT",
]
