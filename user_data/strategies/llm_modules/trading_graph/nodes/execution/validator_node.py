"""
Decision Validator Node for LangGraph.

Validates Judge verdicts against risk management rules before execution.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    JudgeVerdict,
    Verdict,
    Direction,
)

logger = logging.getLogger(__name__)


# Validation thresholds
MIN_CONFIDENCE_THRESHOLD = 40.0  # Minimum confidence to proceed
MAX_LEVERAGE_CAP = 100  # Maximum allowed leverage
DEFAULT_LEVERAGE = 10  # Default leverage if not specified
MIN_LEVERAGE = 1


def validator_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Decision Validator node.

    Validates the Judge verdict against risk management rules:
    1. Confidence threshold check
    2. Leverage cap enforcement
    3. Direction consistency check
    4. Position conflict detection

    Args:
        state: Current trading decision state with judge verdict
        config: Node configuration (contains risk_config)

    Returns:
        Dict with validation results and final action
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[Validator] Validating decision for {pair}")

    try:
        # Get risk config
        risk_config = config.get("configurable", {}).get("risk_config", {}) if config else {}

        judge_verdict = state.get("judge_verdict")

        # No verdict to validate
        if not judge_verdict:
            logger.warning(f"[Validator] No judge verdict for {pair}")
            return _create_wait_result("No judge verdict available")

        # Validate based on verdict type
        validation_result = _validate_verdict(
            verdict=judge_verdict,
            consensus_direction=state.get("consensus_direction"),
            consensus_confidence=state.get("consensus_confidence", 0.0),
            has_position=state.get("has_position", False),
            position_side=state.get("position_side"),
            risk_config=risk_config
        )

        if validation_result["is_valid"]:
            logger.info(
                f"[Validator] {pair} validated: action={validation_result['final_action']}, "
                f"leverage={validation_result['final_leverage']}"
            )
        else:
            logger.info(
                f"[Validator] {pair} validation failed: {validation_result['validation_reason']}"
            )

        return validation_result

    except Exception as e:
        logger.error(f"[Validator] Validation failed: {e}")
        return _create_wait_result(str(e))


def _validate_verdict(
    verdict: JudgeVerdict,
    consensus_direction: Optional[Direction],
    consensus_confidence: float,
    has_position: bool,
    position_side: Optional[str],
    risk_config: dict
) -> Dict[str, Any]:
    """
    Validate the judge verdict.

    Args:
        verdict: Judge verdict to validate
        consensus_direction: Direction from analysis consensus
        consensus_confidence: Confidence from analysis consensus
        has_position: Whether there's an existing position
        position_side: Side of existing position ('long' or 'short')
        risk_config: Risk management configuration

    Returns:
        Validation result dictionary
    """
    result = {
        "is_valid": False,
        "final_action": "signal_wait",
        "final_confidence": 0.0,
        "final_leverage": None,
        "final_reason": "",
        "validation_warnings": [],
        "validation_reason": ""
    }

    # Get thresholds from config
    min_confidence = risk_config.get("min_confidence", MIN_CONFIDENCE_THRESHOLD)
    max_leverage = risk_config.get("max_leverage", MAX_LEVERAGE_CAP)
    default_leverage = risk_config.get("default_leverage", DEFAULT_LEVERAGE)

    # Handle different verdict types
    if verdict.verdict == Verdict.REJECT:
        # Bear wins - reject the trade
        result["is_valid"] = True
        result["final_action"] = "signal_wait"
        result["final_confidence"] = verdict.confidence
        result["final_reason"] = f"Bear wins: {verdict.key_reasoning}"
        result["validation_reason"] = "Trade rejected by debate"
        return result

    if verdict.verdict == Verdict.ABSTAIN:
        # Arguments balanced - wait for better setup
        result["is_valid"] = True
        result["final_action"] = "signal_wait"
        result["final_confidence"] = verdict.confidence
        result["final_reason"] = f"Debate inconclusive: {verdict.key_reasoning}"
        result["validation_reason"] = "Abstain verdict - waiting for clearer signal"
        return result

    # APPROVE verdict - validate before execution
    if verdict.verdict == Verdict.APPROVE:
        warnings = []

        # Check 1: Confidence threshold
        if verdict.confidence < min_confidence:
            warnings.append(
                f"Low confidence ({verdict.confidence:.0f}% < {min_confidence}%)"
            )
            result["validation_warnings"] = warnings
            result["validation_reason"] = "Confidence below threshold"
            result["final_action"] = "signal_wait"
            result["final_confidence"] = verdict.confidence
            result["final_reason"] = f"Confidence too low: {verdict.confidence:.0f}%"
            result["is_valid"] = True
            return result

        # Check 2: Determine direction from consensus
        if consensus_direction is None or consensus_direction == Direction.NEUTRAL:
            warnings.append("No clear direction from consensus")
            result["validation_warnings"] = warnings
            result["validation_reason"] = "No clear direction"
            result["final_action"] = "signal_wait"
            result["final_confidence"] = verdict.confidence
            result["final_reason"] = "Cannot determine trade direction"
            result["is_valid"] = True
            return result

        # Check 3: Position conflict detection
        if has_position and position_side:
            # Convert consensus direction to position side format
            desired_side = "long" if consensus_direction == Direction.LONG else "short"

            if position_side != desired_side:
                # Opposite direction - recommend exit
                result["is_valid"] = True
                result["final_action"] = "signal_exit"
                result["final_confidence"] = verdict.confidence
                result["final_leverage"] = None
                result["final_reason"] = (
                    f"Direction conflict: holding {position_side} but consensus is {desired_side}"
                )
                result["validation_reason"] = "Exit recommended due to direction conflict"
                return result
            else:
                # Same direction - recommend hold
                result["is_valid"] = True
                result["final_action"] = "signal_hold"
                result["final_confidence"] = verdict.confidence
                result["final_leverage"] = None
                result["final_reason"] = f"Holding {position_side} aligns with consensus"
                result["validation_reason"] = "Position aligned with consensus"
                return result

        # No position - new entry
        # Check 4: Cap leverage
        leverage = verdict.leverage if verdict.leverage else default_leverage
        if leverage > max_leverage:
            warnings.append(f"Leverage capped from {leverage}x to {max_leverage}x")
            leverage = max_leverage
        leverage = max(MIN_LEVERAGE, leverage)

        # Determine entry action
        if consensus_direction == Direction.LONG:
            result["final_action"] = "signal_entry_long"
        elif consensus_direction == Direction.SHORT:
            result["final_action"] = "signal_entry_short"
        else:
            result["final_action"] = "signal_wait"
            result["final_reason"] = "Neutral direction - no entry"
            result["validation_reason"] = "Neutral direction"
            result["is_valid"] = True
            return result

        result["is_valid"] = True
        result["final_confidence"] = verdict.confidence
        result["final_leverage"] = leverage
        result["final_reason"] = verdict.key_reasoning
        result["validation_warnings"] = warnings
        result["validation_reason"] = "Approved for execution"

    return result


def _create_wait_result(error_msg: str) -> Dict[str, Any]:
    """Create a wait result due to error."""
    return {
        "is_valid": False,
        "final_action": "signal_wait",
        "final_confidence": 0.0,
        "final_leverage": None,
        "final_reason": f"Validation error: {error_msg}",
        "validation_warnings": [],
        "validation_reason": error_msg,
        "errors": [f"Validator: {error_msg}"]
    }
