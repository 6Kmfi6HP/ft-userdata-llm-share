"""
Sentiment Analysis Node for LangGraph.

Market sentiment analysis: funding rate, long/short ratio, OI, Fear & Greed.
"""

import logging
import re
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    AnalysisState,
    AgentReport,
    Signal,
    Direction,
    SignalStrength,
)
from ....prompts.analysis.sentiment_prompt import (
    SENTIMENT_SYSTEM_PROMPT,
    build_sentiment_prompt,
)
from ....lc_integration.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


def sentiment_node(state: AnalysisState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Market sentiment analysis node.

    Analyzes funding rate, long/short ratio, OI changes, Fear & Greed index.

    Args:
        state: Current analysis state
        config: Node configuration (contains llm_config)

    Returns:
        Dict with agent_reports and sentiment_report updates
    """
    pair = state.get("pair", "UNKNOWN")
    market_context = state.get("market_context", "")

    logger.debug(f"[SentimentAgent] Starting analysis for {pair}")

    try:
        # Get LLM config from graph config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="sentiment")

        # Build prompt
        prompt = build_sentiment_prompt(market_context)

        # Call LLM
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        parsed = _parse_sentiment_response(response_text)

        # Create report
        report = AgentReport(
            agent_name="SentimentAgent",
            analysis=response_text,
            signals=parsed["signals"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            key_levels=parsed["key_levels"]
        )

        logger.info(
            f"[SentimentAgent] {pair} analysis complete: "
            f"direction={report.direction}, confidence={report.confidence:.0f}%, "
            f"signals={len(report.signals)}"
        )

        return {
            "agent_reports": [report],
            "sentiment_report": report
        }

    except Exception as e:
        logger.error(f"[SentimentAgent] Analysis failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result with empty report."""
    error_report = AgentReport(
        agent_name="SentimentAgent",
        analysis=f"Analysis failed: {error_msg}",
        signals=[],
        confidence=0.0,
        direction=None,
        error=error_msg
    )
    return {
        "agent_reports": [error_report],
        "sentiment_report": error_report,
        "errors": [f"SentimentAgent: {error_msg}"]
    }


def _parse_sentiment_response(response: str) -> Dict[str, Any]:
    """Parse sentiment analysis response."""
    result = {
        "signals": [],
        "direction": Direction.NEUTRAL,
        "confidence": 50.0,
        "key_levels": {"support": None, "resistance": None}
    }

    if not response:
        return result

    lines = response.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section markers
        if '[信号列表]' in line:
            current_section = 'signals'
            continue
        elif '[方向判断]' in line:
            current_section = 'direction'
            continue
        elif '[置信度]' in line:
            current_section = 'confidence'
            continue
        elif '[关键价位]' in line:
            current_section = 'key_levels'
            continue
        elif '[分析摘要]' in line:
            current_section = 'summary'
            continue

        # Parse based on section
        if current_section == 'signals' and line.startswith('- '):
            signal = _parse_signal_line(line[2:])
            if signal:
                result['signals'].append(signal)

        elif current_section == 'direction':
            direction = _parse_direction(line)
            if direction:
                result['direction'] = direction

        elif current_section == 'confidence':
            confidence = _parse_confidence(line)
            if confidence is not None:
                result['confidence'] = confidence

        elif current_section == 'key_levels':
            level = _parse_key_level(line)
            if level:
                key, value = level
                result['key_levels'][key] = value

    return result


def _parse_signal_line(line: str) -> Optional[Signal]:
    """Parse a single signal line."""
    try:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            return None

        name = parts[0]
        direction = _parse_direction(parts[1]) or Direction.NEUTRAL
        strength = _parse_strength(parts[2])
        description = parts[4] if len(parts) > 4 else ""

        return Signal(
            signal_type=name,
            description=description,
            strength=strength,
            direction=direction
        )
    except Exception:
        return None


def _parse_direction(text: str) -> Optional[Direction]:
    """Parse direction from text."""
    text_lower = text.lower().strip()
    if 'long' in text_lower or '做多' in text_lower or '多' == text_lower:
        return Direction.LONG
    elif 'short' in text_lower or '做空' in text_lower or '空' == text_lower:
        return Direction.SHORT
    elif 'neutral' in text_lower or '中性' in text_lower:
        return Direction.NEUTRAL
    return None


def _parse_strength(text: str) -> SignalStrength:
    """Parse signal strength."""
    text_lower = text.lower().strip()
    if 'strong' in text_lower or '强' in text_lower:
        return SignalStrength.STRONG
    elif 'moderate' in text_lower or '中' in text_lower:
        return SignalStrength.MEDIUM
    elif 'weak' in text_lower or '弱' in text_lower:
        return SignalStrength.WEAK
    return SignalStrength.MEDIUM


def _parse_confidence(text: str) -> Optional[float]:
    """Parse confidence value."""
    try:
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            conf = float(numbers[0])
            return max(0, min(100, conf))
    except Exception:
        pass
    return None


def _parse_key_level(line: str) -> Optional[tuple]:
    """Parse key level line."""
    line_lower = line.lower()

    if '支撑' in line_lower or 'support' in line_lower:
        value = _parse_float(line)
        if value:
            return ('support', value)

    elif '阻力' in line_lower or 'resistance' in line_lower:
        value = _parse_float(line)
        if value:
            return ('resistance', value)

    return None


def _parse_float(text: str) -> Optional[float]:
    """Extract float from text."""
    if not text or 'N/A' in text.upper():
        return None
    try:
        numbers = re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))
        if numbers:
            return float(numbers[0])
    except Exception:
        pass
    return None
