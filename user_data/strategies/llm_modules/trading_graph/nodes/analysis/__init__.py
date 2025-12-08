"""
Analysis agent nodes for LangGraph.

Implements 4 specialized analysis agents + aggregator:
- IndicatorAgent: RSI, MACD, ADX, Stochastic analysis
- TrendAgent: EMA structure, support/resistance (supports visual)
- SentimentAgent: Funding rate, OI, Fear & Greed
- PatternAgent: K-line pattern recognition (supports visual)
- Aggregator: Weighted consensus from all agents
"""

from .indicator_node import indicator_node
from .trend_node import trend_node
from .sentiment_node import sentiment_node
from .pattern_node import pattern_node
from .aggregator_node import aggregator_node

__all__ = [
    "indicator_node",
    "trend_node",
    "sentiment_node",
    "pattern_node",
    "aggregator_node",
]
