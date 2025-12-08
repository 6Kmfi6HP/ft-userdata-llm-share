"""
Bull Agent Node for LangGraph.

The Opportunity Finder - makes the strongest case FOR the trade.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    DebateArgument,
    Direction,
)
from ....prompts.debate.bull_prompt import (
    BULL_SYSTEM_PROMPT,
    build_bull_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def bull_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Bull Agent node - Opportunity Finder.

    Makes the strongest possible case FOR the trade based on analysis results.

    Args:
        state: Current trading decision state with analysis results
        config: Node configuration (contains llm_config)

    Returns:
        Dict with bull_argument update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[BullAgent] Starting argument for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="bull")

        # Build analysis summary
        analysis_summary = ContextAdapter.format_all_reports(
            indicator_report=state.get("indicator_report"),
            trend_report=state.get("trend_report"),
            sentiment_report=state.get("sentiment_report"),
            pattern_report=state.get("pattern_report")
        )

        # Get consensus info
        consensus_dir = state.get("consensus_direction", Direction.NEUTRAL)
        consensus_conf = state.get("consensus_confidence", 50.0)

        # Build prompt
        prompt = build_bull_prompt(
            analysis_summary=analysis_summary,
            market_context=state.get("market_context", ""),
            pair=pair,
            current_price=state.get("current_price", 0.0),
            consensus_direction=consensus_dir.value if hasattr(consensus_dir, 'value') else str(consensus_dir),
            consensus_confidence=consensus_conf
        )

        # Call LLM
        messages = [
            {"role": "system", "content": BULL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        argument = _parse_bull_response(response_text, consensus_dir)

        logger.info(
            f"[BullAgent] {pair} argument complete: "
            f"position={argument.position.value}, confidence={argument.confidence:.0f}%"
        )

        return {
            "bull_argument": argument
        }

    except Exception as e:
        logger.error(f"[BullAgent] Argument failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_argument = DebateArgument(
        agent_role="bull",
        position=Direction.NEUTRAL,
        confidence=0.0,
        key_points=[f"Error: {error_msg}"],
        risk_factors=[],
        supporting_signals=[],
        counter_arguments=[],
        recommended_action="wait",
        reasoning=f"Bull analysis failed: {error_msg}"
    )
    return {
        "bull_argument": error_argument,
        "errors": [f"BullAgent: {error_msg}"]
    }


def _parse_bull_response(response: str, default_direction: Direction) -> DebateArgument:
    """
    Parse Bull agent response with enhanced fallback strategies.

    Args:
        response: LLM response text
        default_direction: Default direction from consensus

    Returns:
        DebateArgument dataclass
    """
    result = {
        "position": default_direction,
        "confidence": 50.0,
        "key_points": [],
        "risk_factors": [],
        "supporting_signals": [],
        "recommended_action": "wait",
        "reasoning": ""
    }

    if not response:
        return DebateArgument(agent_role="bull", **result)

    lines = response.strip().split('\n')
    current_section = None
    all_signals_lines = []  # Collect potential signal lines for fallback

    # Flexible section patterns
    section_patterns = {
        'position': ['[立场]', '**立场**', '立场:', '立场：', '[方向]', '方向:'],
        'confidence': ['[置信度]', '**置信度**', '置信度:', '置信度：', '[信心]'],
        'key_points': ['[核心论点]', '**核心论点**', '核心论点:', '[论点]', '[关键论点]', '**论点**'],
        'supporting_signals': ['[支持信号]', '**支持信号**', '支持信号:', '[信号]', '[技术信号]', '**信号**'],
        'risk_factors': ['[风险因素]', '**风险因素**', '风险因素:', '[风险]', '**风险**'],
        'action': ['[建议行动]', '**建议行动**', '建议行动:', '[行动]', '[操作建议]', '**建议**'],
        'reasoning': ['[完整论述]', '**完整论述**', '完整论述:', '[论述]', '[分析]', '**分析**'],
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section markers
        section_detected = False
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    current_section = section_name
                    section_detected = True
                    break
            if section_detected:
                break
        
        if section_detected:
            continue

        # Parse based on section
        if current_section == 'position':
            if 'long' in line.lower() or '做多' in line or '多头' in line:
                result['position'] = Direction.LONG
            elif 'short' in line.lower() or '做空' in line or '空头' in line:
                result['position'] = Direction.SHORT

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'key_points':
            # Support multiple list formats: 1. / - / • / * / numbered
            if re.match(r'^[\d\.\-•\*\+]\s*', line) or line.startswith('**'):
                point = re.sub(r'^[\d\.\-•\*\+\s]+', '', line).strip()
                point = point.strip('*').strip()
                if point and len(point) > 3:
                    result['key_points'].append(point)

        elif current_section == 'supporting_signals':
            # Support multiple list formats
            if re.match(r'^[\-•\*\+]\s*', line) or ':' in line:
                signal = re.sub(r'^[\-•\*\+\s]+', '', line).strip()
                if signal and len(signal) > 3:
                    result['supporting_signals'].append(signal)
                    all_signals_lines.append(signal)

        elif current_section == 'risk_factors':
            if re.match(r'^[\-•\*\+]\s*', line) or '风险' in line:
                risk = re.sub(r'^[\-•\*\+\s]+', '', line).strip()
                if risk and len(risk) > 3:
                    result['risk_factors'].append(risk)

        elif current_section == 'action':
            if line and len(line) > 5:
                result['recommended_action'] = line

        elif current_section == 'reasoning':
            result['reasoning'] += line + " "

    result['reasoning'] = result['reasoning'].strip()

    # ========== Fallback strategies for supporting_signals ==========
    
    # Strategy 1: Extract from key_points if supporting_signals is empty
    if not result['supporting_signals'] and result['key_points']:
        # Use first 3 key points as signals
        for point in result['key_points'][:3]:
            if len(point) > 10:
                result['supporting_signals'].append(point[:100])

    # Strategy 2: Extract from response using signal keywords
    if not result['supporting_signals']:
        signal_keywords = ['ADX', 'RSI', 'MACD', 'EMA', 'Stochastic', 'BB', '布林',
                          '资金费率', '多空比', 'OI', '持仓', '趋势', '突破', '支撑', '阻力']
        for line in lines:
            line = line.strip()
            if any(kw in line for kw in signal_keywords):
                # Clean up the line
                signal = re.sub(r'^[\d\.\-•\*\+\s\[\]]+', '', line).strip()
                if signal and len(signal) > 10 and signal not in result['supporting_signals']:
                    result['supporting_signals'].append(signal[:150])
                    if len(result['supporting_signals']) >= 5:
                        break

    # Strategy 3: Generate from direction and analysis if still empty
    if not result['supporting_signals']:
        direction = result['position'].value if hasattr(result['position'], 'value') else str(result['position'])
        conf = result['confidence']
        result['supporting_signals'].append(f"方向: {direction}, 置信度: {conf:.0f}%")

    # ========== Fallback for key_points ==========
    if not result['key_points'] and result['reasoning']:
        # Split reasoning into key points
        sentences = re.split(r'[。；;]', result['reasoning'])
        for s in sentences[:3]:
            s = s.strip()
            if len(s) > 10:
                result['key_points'].append(s[:100])

    return DebateArgument(agent_role="bull", **result)
