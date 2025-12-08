"""
Position Bull Agent Node for LangGraph.

The Position Holder Advocate - makes the strongest case FOR holding or scaling into position.
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
from ....prompts.debate.position_bull_prompt import (
    POSITION_BULL_SYSTEM_PROMPT,
    build_position_bull_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def position_bull_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Position Bull Agent node - Position Holder Advocate.

    Makes the strongest possible case FOR holding or scaling into position.

    Args:
        state: Current trading decision state with position metrics
        config: Node configuration (contains llm_config)

    Returns:
        Dict with position_bull_argument update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[PositionBullAgent] Starting argument for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="position_bull")

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
        prompt = build_position_bull_prompt(
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
            {"role": "system", "content": POSITION_BULL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        argument = _parse_position_bull_response(response_text, position_side)

        logger.info(
            f"[PositionBullAgent] {pair} argument complete: "
            f"position={argument.position.value}, confidence={argument.confidence:.0f}%, "
            f"action={argument.recommended_action}"
        )

        return {
            "position_bull_argument": argument
        }

    except Exception as e:
        logger.error(f"[PositionBullAgent] Argument failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_argument = DebateArgument(
        agent_role="position_bull",
        position=Direction.NEUTRAL,
        confidence=0.0,
        key_points=[f"Error: {error_msg}"],
        risk_factors=[],
        supporting_signals=[],
        counter_arguments=[],
        recommended_action="hold",
        reasoning=f"Position Bull analysis failed: {error_msg}"
    )
    return {
        "position_bull_argument": error_argument,
        "errors": [f"PositionBullAgent: {error_msg}"]
    }


def _parse_position_bull_response(response: str, position_side: str) -> DebateArgument:
    """
    Parse Position Bull agent response.

    Args:
        response: LLM response text
        position_side: Current position side ("long" or "short")

    Returns:
        DebateArgument dataclass
    """
    # Determine default direction from position side
    if position_side == "long":
        default_direction = Direction.LONG
    elif position_side == "short":
        default_direction = Direction.SHORT
    else:
        default_direction = Direction.NEUTRAL

    result = {
        "position": default_direction,
        "confidence": 50.0,
        "key_points": [],
        "risk_factors": [],
        "supporting_signals": [],
        "recommended_action": "hold",
        "reasoning": ""
    }

    if not response:
        return DebateArgument(agent_role="position_bull", **result)

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
        elif '[核心论点]' in line:
            current_section = 'key_points'
            continue
        elif '[支持信号]' in line:
            current_section = 'supporting_signals'
            continue
        elif '[回撤评估]' in line:
            current_section = 'drawdown'
            continue
        elif '[风险因素]' in line:
            current_section = 'risk_factors'
            continue
        elif '[建议行动]' in line:
            current_section = 'action'
            continue
        elif '[完整论述]' in line:
            current_section = 'reasoning'
            continue

        # Parse based on section
        if current_section == 'position':
            line_upper = line.upper()
            if 'SCALE_IN' in line_upper or '加仓' in line:
                result['recommended_action'] = 'scale_in'
            elif 'HOLD' in line_upper or '持有' in line:
                result['recommended_action'] = 'hold'

        elif current_section == 'adjustment':
            numbers = re.findall(r'[+]?\d+', line)
            if numbers:
                adjustment_pct = int(numbers[0])

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'key_points':
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                point = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                if point:
                    result['key_points'].append(point)

        elif current_section == 'supporting_signals':
            if line.startswith('-') or line.startswith('•'):
                signal = line.lstrip('-•').strip()
                if signal:
                    result['supporting_signals'].append(signal)

        elif current_section == 'drawdown':
            # Store drawdown assessment as part of reasoning
            result['reasoning'] += f"回撤评估: {line} "

        elif current_section == 'risk_factors':
            if line.startswith('-') or line.startswith('•'):
                risk = line.lstrip('-•').strip()
                if risk:
                    result['risk_factors'].append(risk)

        elif current_section == 'action':
            if result['recommended_action'] == 'hold':
                # Check if action overrides position
                line_upper = line.upper()
                if 'SCALE_IN' in line_upper or '加仓' in line:
                    result['recommended_action'] = 'scale_in'

        elif current_section == 'reasoning':
            result['reasoning'] += line + " "

    # Finalize recommended action with adjustment
    if result['recommended_action'] == 'scale_in' and adjustment_pct > 0:
        result['recommended_action'] = f"scale_in +{adjustment_pct}%"

    result['reasoning'] = result['reasoning'].strip()

    return DebateArgument(agent_role="position_bull", **result)
