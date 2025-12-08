"""
Position Bear Agent Node for LangGraph.

The Profit Protector - finds every possible reason to reduce or exit the position.
Focus on MFE drawback, stuck-in-loop detection, and profit protection.
"""

import logging
import re
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    DebateArgument,
    Direction,
    PositionMetrics,
)
from ....prompts.debate.position_bear_prompt import (
    POSITION_BEAR_SYSTEM_PROMPT,
    build_position_bear_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def position_bear_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Position Bear Agent node - Profit Protector.

    Finds every possible reason to reduce or exit the position.

    Args:
        state: Current trading decision state with position metrics and bull argument
        config: Node configuration (contains llm_config)

    Returns:
        Dict with position_bear_argument update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[PositionBearAgent] Starting argument for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="position_bear")

        # Get bull argument
        bull_argument = state.get("position_bull_argument")
        if not bull_argument:
            return _create_error_result("Missing Position Bull argument")

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

        # Get position metrics
        position_metrics: PositionMetrics = state.get("position_metrics") or {}
        current_profit_pct = position_metrics.get("current_profit_pct", state.get("position_profit_pct", 0.0))
        max_profit_pct = position_metrics.get("max_profit_pct", current_profit_pct)
        max_loss_pct = position_metrics.get("max_loss_pct", 0.0)
        drawdown_from_peak_pct = position_metrics.get("drawdown_from_peak_pct", 0.0)
        hold_count = position_metrics.get("hold_count", 0)
        time_in_position_hours = position_metrics.get("time_in_position_hours", 0.0)
        position_side = position_metrics.get("position_side", state.get("position_side", "unknown"))
        entry_price = position_metrics.get("entry_price", 0.0)

        # Build prompt
        prompt = build_position_bear_prompt(
            bull_argument=bull_arg_text,
            analysis_summary=analysis_summary,
            market_context=state.get("market_context", ""),
            pair=pair,
            current_price=state.get("current_price", 0.0),
            consensus_direction=consensus_dir.value if hasattr(consensus_dir, 'value') else str(consensus_dir),
            consensus_confidence=consensus_conf,
            current_profit_pct=current_profit_pct,
            max_profit_pct=max_profit_pct,
            max_loss_pct=max_loss_pct,
            drawdown_from_peak_pct=drawdown_from_peak_pct,
            hold_count=hold_count,
            time_in_position_hours=time_in_position_hours,
            position_side=position_side,
            entry_price=entry_price
        )

        # Call LLM
        messages = [
            {"role": "system", "content": POSITION_BEAR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        argument = _parse_position_bear_response(response_text, position_side)

        logger.info(
            f"[PositionBearAgent] {pair} argument complete: "
            f"confidence={argument.confidence:.0f}%, "
            f"action={argument.recommended_action}"
        )

        return {
            "position_bear_argument": argument
        }

    except Exception as e:
        logger.error(f"[PositionBearAgent] Argument failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_argument = DebateArgument(
        agent_role="position_bear",
        position=Direction.NEUTRAL,
        confidence=0.0,
        key_points=[f"Error: {error_msg}"],
        risk_factors=[],
        supporting_signals=[],
        counter_arguments=[],
        recommended_action="hold",
        reasoning=f"Position Bear analysis failed: {error_msg}"
    )
    return {
        "position_bear_argument": error_argument,
        "errors": [f"PositionBearAgent: {error_msg}"]
    }


def _parse_position_bear_response(response: str, position_side: str) -> DebateArgument:
    """
    Parse Position Bear agent response.

    Args:
        response: LLM response text
        position_side: Current position side ("long" or "short")

    Returns:
        DebateArgument dataclass
    """
    # Determine default direction (Bear argues against current position)
    if position_side == "long":
        default_direction = Direction.SHORT  # Bear argues for exit/reduce long
    elif position_side == "short":
        default_direction = Direction.LONG  # Bear argues for exit/reduce short
    else:
        default_direction = Direction.NEUTRAL

    result = {
        "position": default_direction,
        "confidence": 50.0,
        "key_points": [],
        "risk_factors": [],
        "supporting_signals": [],
        "counter_arguments": [],
        "recommended_action": "hold",  # Default to hold if bear has no strong case
        "reasoning": ""
    }

    if not response:
        return DebateArgument(agent_role="position_bear", **result)

    lines = response.strip().split('\n')
    current_section = None
    adjustment_pct = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section markers
        if '[立场]' in line:
            current_section = 'position'
            continue
        elif '[调整比例]' in line:
            current_section = 'adjustment'
            continue
        elif '[置信度]' in line:
            current_section = 'confidence'
            continue
        elif '[核心反驳]' in line:
            current_section = 'counter_arguments'
            continue
        elif '[风险因素]' in line:
            current_section = 'risk_factors'
            continue
        elif '[MFE回撤分析]' in line:
            current_section = 'mfe_analysis'
            continue
        elif '[stuck-in-loop检测]' in line:
            current_section = 'stuck_loop'
            continue
        elif '[利润保护评估]' in line:
            current_section = 'profit_protection'
            continue
        elif '[最坏情况]' in line:
            current_section = 'worst_case'
            continue
        elif '[建议]' in line:
            current_section = 'action'
            continue
        elif '[完整论述]' in line:
            current_section = 'reasoning'
            continue

        # Parse based on section
        if current_section == 'position':
            line_upper = line.upper()
            if 'PARTIAL_EXIT' in line_upper or '减仓' in line:
                result['recommended_action'] = 'partial_exit'
            elif 'EXIT' in line_upper or '退出' in line or '平仓' in line:
                result['recommended_action'] = 'exit'
            elif '支持' in line.lower():
                # Bear supports Bull's case
                result['recommended_action'] = 'hold'

        elif current_section == 'adjustment':
            # Look for negative numbers
            numbers = re.findall(r'-?\d+', line)
            if numbers:
                adjustment_pct = int(numbers[0])
                if adjustment_pct > 0:
                    adjustment_pct = -adjustment_pct  # Ensure negative for reduction

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'counter_arguments':
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                point = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                if point:
                    result['counter_arguments'].append(point)
                    result['key_points'].append(point)  # Also add to key_points

        elif current_section == 'risk_factors':
            if line.startswith('-') or line.startswith('•'):
                risk = line.lstrip('-•').strip()
                if risk:
                    result['risk_factors'].append(risk)

        elif current_section == 'mfe_analysis':
            result['reasoning'] += f"MFE分析: {line} "

        elif current_section == 'stuck_loop':
            result['reasoning'] += f"Stuck检测: {line} "

        elif current_section == 'profit_protection':
            result['reasoning'] += f"利润保护: {line} "
            # Check if profit protection rule triggered
            if '触发' in line or '是' in line.lower():
                result['supporting_signals'].append(f"利润保护规则: {line}")

        elif current_section == 'worst_case':
            result['reasoning'] += f"最坏情况: {line} "

        elif current_section == 'action':
            line_upper = line.upper()
            if 'PARTIAL_EXIT' in line_upper or '减仓' in line:
                result['recommended_action'] = 'partial_exit'
            elif 'EXIT' in line_upper or '退出' in line or '平仓' in line:
                result['recommended_action'] = 'exit'

        elif current_section == 'reasoning':
            result['reasoning'] += line + " "

    # Finalize recommended action with adjustment
    if result['recommended_action'] == 'partial_exit' and adjustment_pct != 0:
        result['recommended_action'] = f"partial_exit {adjustment_pct}%"
    elif result['recommended_action'] == 'exit':
        result['recommended_action'] = "exit -100%"

    result['reasoning'] = result['reasoning'].strip()

    return DebateArgument(agent_role="position_bear", **result)
