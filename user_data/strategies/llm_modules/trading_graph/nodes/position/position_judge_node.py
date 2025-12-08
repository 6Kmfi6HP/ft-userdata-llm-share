"""
Position Judge Agent Node for LangGraph.

The Position Arbiter - evaluates both arguments and renders position management verdict.
Outputs: HOLD, EXIT, SCALE_IN, or PARTIAL_EXIT with adjustment percentage.
"""

import logging
import re
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    PositionJudgeVerdict,
    PositionVerdict,
    Direction,
    PositionMetrics,
)
from ....prompts.debate.position_judge_prompt import (
    POSITION_JUDGE_SYSTEM_PROMPT,
    build_position_judge_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def position_judge_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Position Judge Agent node - Position Arbiter.

    Evaluates Position Bull and Bear arguments, renders final position management verdict.

    Args:
        state: Current trading decision state with both arguments
        config: Node configuration (contains llm_config)

    Returns:
        Dict with position_judge_verdict update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[PositionJudgeAgent] Starting verdict for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="position_judge")

        # Get both arguments
        bull_argument = state.get("position_bull_argument")
        bear_argument = state.get("position_bear_argument")

        if not bull_argument or not bear_argument:
            return _create_error_result("Missing Position Bull or Bear argument")

        bull_arg_text = ContextAdapter.format_debate_argument(bull_argument)
        bear_arg_text = ContextAdapter.format_debate_argument(bear_argument)

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
        prompt = build_position_judge_prompt(
            bull_argument=bull_arg_text,
            bear_argument=bear_arg_text,
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
            {"role": "system", "content": POSITION_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        verdict = _parse_position_judge_response(response_text, consensus_conf)

        # Apply confidence calibration
        calibrated_conf = (verdict.confidence + consensus_conf) / 2
        verdict.confidence = calibrated_conf

        logger.info(
            f"[PositionJudgeAgent] {pair} verdict: {verdict.verdict.value}, "
            f"confidence={verdict.confidence:.0f}%, "
            f"adjustment={verdict.adjustment_pct}%, "
            f"forced_rule={verdict.profit_protection_triggered}"
        )

        return {
            "position_judge_verdict": verdict
        }

    except Exception as e:
        logger.error(f"[PositionJudgeAgent] Verdict failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_verdict = PositionJudgeVerdict(
        verdict=PositionVerdict.HOLD,
        confidence=0.0,
        adjustment_pct=None,
        key_reasoning=f"Position Judge error: {error_msg}",
        profit_protection_triggered=False,
        forced_rule_name=None,
        bull_score=0.0,
        bear_score=0.0
    )
    return {
        "position_judge_verdict": error_verdict,
        "errors": [f"PositionJudgeAgent: {error_msg}"]
    }


def _parse_position_judge_response(response: str, original_confidence: float) -> PositionJudgeVerdict:
    """
    Parse Position Judge agent response.

    Args:
        response: LLM response text
        original_confidence: Original consensus confidence for calibration

    Returns:
        PositionJudgeVerdict dataclass
    """
    result = {
        "verdict": PositionVerdict.HOLD,
        "confidence": 50.0,
        "adjustment_pct": None,
        "key_reasoning": "",
        "profit_protection_triggered": False,
        "forced_rule_name": None,
        "bull_score": 0.0,
        "bear_score": 0.0
    }

    if not response:
        return PositionJudgeVerdict(**result)

    lines = response.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section markers
        if '[Bull 评分]' in line or '[bull 评分]' in line.lower():
            current_section = 'bull_score'
            # Try to extract score from same line
            numbers = re.findall(r'总分[:\s]*(\d+)', line)
            if numbers:
                result['bull_score'] = float(numbers[0])
            continue
        elif '[Bear 评分]' in line or '[bear 评分]' in line.lower():
            current_section = 'bear_score'
            numbers = re.findall(r'总分[:\s]*(\d+)', line)
            if numbers:
                result['bear_score'] = float(numbers[0])
            continue
        elif '[胜出方]' in line:
            current_section = 'winner'
            continue
        elif '[利润保护触发]' in line:
            current_section = 'profit_protection'
            continue
        elif '[裁决]' in line:
            current_section = 'verdict'
            continue
        elif '[调整比例]' in line:
            current_section = 'adjustment'
            continue
        elif '[置信度]' in line:
            current_section = 'confidence'
            continue
        elif '[核心理由]' in line:
            current_section = 'key_reasoning'
            continue
        elif '[风险评估]' in line:
            current_section = 'risk'
            continue
        elif '[最终建议]' in line:
            current_section = 'action'
            continue
        elif '[完整裁决理由]' in line:
            current_section = 'full_reasoning'
            continue

        # Parse based on section
        if current_section == 'bull_score':
            # Try to extract total score
            numbers = re.findall(r'总分[:\s]*(\d+)', line)
            if numbers:
                result['bull_score'] = float(numbers[0])

        elif current_section == 'bear_score':
            numbers = re.findall(r'总分[:\s]*(\d+)', line)
            if numbers:
                result['bear_score'] = float(numbers[0])

        elif current_section == 'profit_protection':
            line_lower = line.lower()
            if '是' in line_lower or 'yes' in line_lower or '触发' in line_lower:
                result['profit_protection_triggered'] = True
                # Try to extract rule name
                if '-' in line:
                    result['forced_rule_name'] = line.split('-')[-1].strip()

        elif current_section == 'verdict':
            line_upper = line.upper()
            if 'SCALE_IN' in line_upper:
                result['verdict'] = PositionVerdict.SCALE_IN
            elif 'PARTIAL_EXIT' in line_upper:
                result['verdict'] = PositionVerdict.PARTIAL_EXIT
            elif 'EXIT' in line_upper and 'PARTIAL' not in line_upper:
                result['verdict'] = PositionVerdict.EXIT
            elif 'HOLD' in line_upper:
                result['verdict'] = PositionVerdict.HOLD

        elif current_section == 'adjustment':
            numbers = re.findall(r'-?\d+', line)
            if numbers:
                adj = int(numbers[0])
                # Validate adjustment range based on verdict
                if result['verdict'] == PositionVerdict.SCALE_IN:
                    result['adjustment_pct'] = max(20, min(50, abs(adj)))
                elif result['verdict'] == PositionVerdict.PARTIAL_EXIT:
                    result['adjustment_pct'] = max(-70, min(-30, -abs(adj)))
                elif result['verdict'] == PositionVerdict.EXIT:
                    result['adjustment_pct'] = -100
                else:
                    result['adjustment_pct'] = 0

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'key_reasoning':
            result['key_reasoning'] = line

        elif current_section == 'full_reasoning':
            if result['key_reasoning']:
                result['key_reasoning'] += " " + line
            else:
                result['key_reasoning'] = line

    # Set default adjustment based on verdict if not set
    if result['adjustment_pct'] is None:
        if result['verdict'] == PositionVerdict.SCALE_IN:
            result['adjustment_pct'] = 30  # Default +30%
        elif result['verdict'] == PositionVerdict.PARTIAL_EXIT:
            result['adjustment_pct'] = -50  # Default -50%
        elif result['verdict'] == PositionVerdict.EXIT:
            result['adjustment_pct'] = -100
        else:
            result['adjustment_pct'] = 0

    return PositionJudgeVerdict(**result)
