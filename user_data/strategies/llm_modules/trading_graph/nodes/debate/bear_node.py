"""
Bear Agent Node for LangGraph.

The Devil's Advocate - finds every possible flaw in the trade thesis.
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
from ....prompts.debate.bear_prompt import (
    BEAR_SYSTEM_PROMPT,
    build_bear_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def bear_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Bear Agent node - Devil's Advocate.

    Challenges Bull's arguments and identifies all risks and flaws.

    Args:
        state: Current trading decision state with Bull's argument
        config: Node configuration (contains llm_config)

    Returns:
        Dict with bear_argument update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[BearAgent] Starting counter-argument for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="bear")

        # Get Bull's argument
        bull_argument = state.get("bull_argument")
        if not bull_argument:
            return _create_error_result("No Bull argument to counter")

        bull_arg_text = ContextAdapter.format_debate_argument(bull_argument)

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
        prompt = build_bear_prompt(
            bull_argument=bull_arg_text,
            analysis_summary=analysis_summary,
            market_context=state.get("market_context", ""),
            pair=pair,
            current_price=state.get("current_price", 0.0),
            consensus_direction=consensus_dir.value if hasattr(consensus_dir, 'value') else str(consensus_dir),
            consensus_confidence=consensus_conf
        )

        # Call LLM
        messages = [
            {"role": "system", "content": BEAR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        argument = _parse_bear_response(response_text)

        logger.info(
            f"[BearAgent] {pair} counter-argument complete: "
            f"confidence={argument.confidence:.0f}%, risks={len(argument.risk_factors)}"
        )

        return {
            "bear_argument": argument
        }

    except Exception as e:
        logger.error(f"[BearAgent] Counter-argument failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_argument = DebateArgument(
        agent_role="bear",
        position=Direction.NEUTRAL,
        confidence=0.0,
        key_points=[],
        risk_factors=[f"Error: {error_msg}"],
        supporting_signals=[],
        counter_arguments=[],
        recommended_action="wait",
        reasoning=f"Bear analysis failed: {error_msg}"
    )
    return {
        "bear_argument": error_argument,
        "errors": [f"BearAgent: {error_msg}"]
    }


def _parse_bear_response(response: str) -> DebateArgument:
    """
    Parse Bear agent response with enhanced fallback strategies.

    Args:
        response: LLM response text

    Returns:
        DebateArgument dataclass
    """
    result = {
        "position": Direction.NEUTRAL,
        "confidence": 50.0,
        "key_points": [],
        "risk_factors": [],
        "supporting_signals": [],
        "counter_arguments": [],
        "recommended_action": "wait",
        "reasoning": ""
    }

    if not response:
        return DebateArgument(agent_role="bear", **result)

    lines = response.strip().split('\n')
    current_section = None

    # Flexible section patterns for Bear
    section_patterns = {
        'position': ['[立场]', '**立场**', '立场:', '立场：'],
        'confidence': ['[置信度]', '**置信度**', '置信度:', '置信度：'],
        'counter_arguments': ['[核心反驳]', '**核心反驳**', '核心反驳:', '[反驳]', '[反对论点]', '**反驳**'],
        'risk_factors': ['[风险因素]', '**风险因素**', '风险因素:', '[风险]', '**风险**'],
        'contradictions': ['[矛盾信号]', '**矛盾信号**', '矛盾信号:', '[矛盾]', '**矛盾**'],
        'worst_case': ['[最坏情况]', '**最坏情况**', '最坏情况:', '[极端情况]'],
        'action': ['[建议]', '**建议**', '建议:', '[行动建议]', '[操作建议]'],
        'reasoning': ['[完整论述]', '**完整论述**', '完整论述:', '[论述]', '[分析]'],
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
            if '反对' in line or 'against' in line.lower() or '拒绝' in line:
                result['position'] = Direction.SHORT  # Opposing the trade
            elif '支持' in line or 'support' in line.lower():
                result['position'] = Direction.LONG
            elif '中立' in line or 'neutral' in line.lower() or '观望' in line:
                result['position'] = Direction.NEUTRAL

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'counter_arguments':
            # Support multiple list formats
            if re.match(r'^[\d\.\-•\*\+]\s*', line) or line.startswith('**'):
                counter = re.sub(r'^[\d\.\-•\*\+\s]+', '', line).strip()
                counter = counter.strip('*').strip()
                if counter and len(counter) > 3:
                    result['counter_arguments'].append(counter)

        elif current_section == 'risk_factors':
            if re.match(r'^[\-•\*\+]\s*', line) or '风险' in line or ':' in line:
                risk = re.sub(r'^[\-•\*\+\s]+', '', line).strip()
                if risk and len(risk) > 3:
                    result['risk_factors'].append(risk)

        elif current_section == 'contradictions':
            if re.match(r'^[\-•\*\+]\s*', line) or '矛盾' in line:
                contradiction = re.sub(r'^[\-•\*\+\s]+', '', line).strip()
                if contradiction and len(contradiction) > 3:
                    result['key_points'].append(f"矛盾: {contradiction}")

        elif current_section == 'worst_case':
            if line and len(line) > 5:
                result['key_points'].append(f"最坏情况: {line}")

        elif current_section == 'action':
            if line and len(line) > 5:
                result['recommended_action'] = line

        elif current_section == 'reasoning':
            result['reasoning'] += line + " "

    result['reasoning'] = result['reasoning'].strip()

    # ========== Fallback strategies ==========
    
    # Fallback for counter_arguments: use risk_factors
    if not result['counter_arguments'] and result['risk_factors']:
        result['counter_arguments'] = result['risk_factors'][:3]

    # Fallback for risk_factors: extract from response
    if not result['risk_factors']:
        risk_keywords = ['风险', '危险', '下跌', '亏损', '止损', '回调', '回撤', '失败', '不利']
        for line in lines:
            line = line.strip()
            if any(kw in line for kw in risk_keywords):
                risk = re.sub(r'^[\d\.\-•\*\+\s\[\]]+', '', line).strip()
                if risk and len(risk) > 10 and risk not in result['risk_factors']:
                    result['risk_factors'].append(risk[:150])
                    if len(result['risk_factors']) >= 3:
                        break

    # Fallback for key_points
    if not result['key_points']:
        # Use counter_arguments as key_points
        for arg in result['counter_arguments'][:3]:
            result['key_points'].append(arg[:100])
    
    # If still no key_points, extract from reasoning
    if not result['key_points'] and result['reasoning']:
        sentences = re.split(r'[。；;]', result['reasoning'])
        for s in sentences[:3]:
            s = s.strip()
            if len(s) > 10:
                result['key_points'].append(s[:100])

    # Generate default key_point if still empty
    if not result['key_points']:
        conf = result['confidence']
        result['key_points'].append(f"Bear 反对置信度: {conf:.0f}%")

    return DebateArgument(agent_role="bear", **result)
