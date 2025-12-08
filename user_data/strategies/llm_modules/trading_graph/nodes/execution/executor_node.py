"""
Trade Executor Node for LangGraph.

Converts validated decisions into executable trading actions.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from ...state import (
    TradingDecisionState,
    Direction,
)

logger = logging.getLogger(__name__)


def executor_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Trade Executor node.

    Converts validated decisions into final trading execution parameters.
    This node prepares the output format expected by the strategy.

    Args:
        state: Current trading decision state with validation results
        config: Node configuration

    Returns:
        Dict with final execution parameters
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[Executor] Executing decision for {pair}")

    try:
        # Get validation results from state
        final_action = state.get("final_action", "signal_wait")
        final_confidence = state.get("final_confidence", 0.0)
        final_leverage = state.get("final_leverage")
        final_reason = state.get("final_reason", "")

        # Get market context
        current_price = state.get("current_price", 0.0)
        key_support = state.get("key_support")
        key_resistance = state.get("key_resistance")

        # Get position adjustment percentage (for adjust_position action)
        # Try state first, then fallback to position_judge_verdict (LangGraph state merge issue workaround)
        adjustment_pct = state.get("adjustment_pct")
        if adjustment_pct is None:
            position_verdict = state.get("position_judge_verdict")
            if position_verdict and hasattr(position_verdict, 'adjustment_pct'):
                adjustment_pct = position_verdict.adjustment_pct
                logger.debug(f"[Executor] Got adjustment_pct={adjustment_pct} from position_judge_verdict (fallback)")

        # Build execution result
        execution_result = _build_execution_result(
            action=final_action,
            pair=pair,
            confidence=final_confidence,
            leverage=final_leverage,
            reason=final_reason,
            current_price=current_price,
            key_support=key_support,
            key_resistance=key_resistance,
            consensus_direction=state.get("consensus_direction"),
            judge_verdict=state.get("judge_verdict"),
            position_verdict=state.get("position_judge_verdict"),
            adjustment_pct=adjustment_pct,
            validation_warnings=state.get("validation_warnings", [])
        )

        # Log execution details (include adjustment_pct for adjust_position)
        log_msg = f"[Executor] {pair} execution prepared: action={final_action}, confidence={final_confidence:.0f}%"
        if final_action == "adjust_position":
            log_msg += f", adjustment={adjustment_pct}%"
        logger.info(log_msg)

        return {
            "execution_result": execution_result,
            "execution_completed": True,
            "execution_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[Executor] Execution failed: {e}")
        return _create_error_result(str(e))


def _build_execution_result(
    action: str,
    pair: str,
    confidence: float,
    leverage: Optional[int],
    reason: str,
    current_price: float,
    key_support: Optional[float],
    key_resistance: Optional[float],
    consensus_direction: Optional[Direction],
    judge_verdict: Any,
    position_verdict: Any = None,
    adjustment_pct: Optional[float] = None,
    validation_warnings: list = None
) -> Dict[str, Any]:
    """
    Build the execution result dictionary.

    This format matches what LLMFunctionStrategy expects from trading functions.
    """
    if validation_warnings is None:
        validation_warnings = []

    result = {
        "action": action,
        "pair": pair,
        "confidence_score": confidence,
        "reason": reason,
        "current_price": current_price,  # Include for proper default calculations
        "timestamp": datetime.now().isoformat(),
        "source": "langgraph_debate"
    }

    # Add entry-specific fields
    if action in ("signal_entry_long", "signal_entry_short"):
        result["leverage"] = leverage or 10
        result["key_support"] = key_support or current_price * 0.95
        result["key_resistance"] = key_resistance or current_price * 1.05

        # Calculate position context from consensus
        if consensus_direction:
            result["trend_direction"] = consensus_direction.value

        # Add debate context
        if judge_verdict:
            result["debate_verdict"] = judge_verdict.verdict.value if hasattr(judge_verdict.verdict, 'value') else str(judge_verdict.verdict)
            result["debate_confidence"] = judge_verdict.confidence
            result["winning_argument"] = judge_verdict.winning_argument

    # Add adjust_position-specific fields
    elif action == "adjust_position":
        result["adjustment_pct"] = adjustment_pct or 0
        result["key_support"] = key_support or current_price * 0.95
        result["key_resistance"] = key_resistance or current_price * 1.05

        # Determine adjustment type from percentage
        if adjustment_pct and adjustment_pct > 0:
            result["adjustment_type"] = "scale_in"
        elif adjustment_pct and adjustment_pct < 0:
            result["adjustment_type"] = "partial_exit"
        else:
            result["adjustment_type"] = "none"

        # Add position verdict context
        if position_verdict:
            result["position_verdict"] = position_verdict.verdict.value if hasattr(position_verdict.verdict, 'value') else str(position_verdict.verdict)
            result["position_verdict_confidence"] = position_verdict.confidence
            result["profit_protection_triggered"] = position_verdict.profit_protection_triggered

    # Add exit-specific fields
    elif action == "signal_exit":
        result["exit_reason"] = reason
        # Add position verdict if from position management
        if position_verdict:
            result["position_verdict"] = position_verdict.verdict.value if hasattr(position_verdict.verdict, 'value') else str(position_verdict.verdict)

    # Add hold-specific fields
    elif action == "signal_hold":
        result["hold_reason"] = reason
        # Add position verdict if from position management
        if position_verdict:
            result["position_verdict"] = position_verdict.verdict.value if hasattr(position_verdict.verdict, 'value') else str(position_verdict.verdict)

    # Add wait-specific fields
    elif action == "signal_wait":
        result["wait_reason"] = reason

    # Add warnings if any
    if validation_warnings:
        result["warnings"] = validation_warnings

    return result


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    return {
        "execution_result": {
            "action": "signal_wait",
            "pair": "UNKNOWN",
            "confidence_score": 0.0,
            "reason": f"Execution error: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "source": "langgraph_debate",
            "error": error_msg
        },
        "execution_completed": False,
        "execution_timestamp": datetime.now().isoformat(),
        "errors": [f"Executor: {error_msg}"]
    }


def format_execution_summary(execution_result: Dict[str, Any]) -> str:
    """
    Format execution result as human-readable summary.

    Used for logging and debugging.
    """
    action = execution_result.get("action", "unknown")
    pair = execution_result.get("pair", "UNKNOWN")
    confidence = execution_result.get("confidence_score", 0.0)
    reason = execution_result.get("reason", "")

    lines = [
        f"=== Execution Summary for {pair} ===",
        f"Action: {action}",
        f"Confidence: {confidence:.0f}%",
    ]

    if action in ("signal_entry_long", "signal_entry_short"):
        leverage = execution_result.get("leverage", 10)
        lines.append(f"Leverage: {leverage}x")
        if "debate_verdict" in execution_result:
            lines.append(f"Debate Verdict: {execution_result['debate_verdict']}")
            lines.append(f"Winning Argument: {execution_result.get('winning_argument', 'N/A')}")

    elif action == "adjust_position":
        adjustment_pct = execution_result.get("adjustment_pct", 0)
        adjustment_type = execution_result.get("adjustment_type", "unknown")
        lines.append(f"Adjustment: {adjustment_pct:+.0f}% ({adjustment_type})")
        if "position_verdict" in execution_result:
            lines.append(f"Position Verdict: {execution_result['position_verdict']}")
            if execution_result.get("profit_protection_triggered"):
                lines.append("** Profit Protection Triggered **")

    reason_display = reason[:100] + "..." if len(reason) > 100 else reason
    lines.append(f"Reason: {reason_display}")

    if execution_result.get("warnings"):
        lines.append("Warnings:")
        for warning in execution_result["warnings"]:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
