"""
Conditional Routing Functions for LangGraph.

Defines routing logic for graph edges based on state.

Key routing decisions:
- route_entry_or_position: Routes to entry debate vs position management based on has_position
- route_after_debate: Routes after entry debate to reflection
- route_after_reflection: Routes after reflection to executor or end
- route_after_position_reflection: Routes after position reflection to executor or end
"""

import logging
from typing import Literal

from ..state import (
    TradingDecisionState,
    AnalysisState,
    Verdict,
    Direction,
    PositionVerdict,
)

logger = logging.getLogger(__name__)


def route_after_analysis(
    state: TradingDecisionState
) -> Literal["debate"]:
    """
    Route after analysis subgraph completion.

    Always proceeds to debate - the adversarial debate is a core verification
    layer that should not be skipped. Even low confidence or neutral signals
    should be debated to verify if the analysis is correct or hallucinated.

    Reference: LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md - Layer 3
    "D3框架(2024)显示对抗性辩论减少30-50%的过度自信"

    Args:
        state: Current trading decision state after analysis

    Returns:
        "debate" to always proceed to adversarial debate
    """
    consensus_confidence = state.get("consensus_confidence", 0.0)
    consensus_direction = state.get("consensus_direction")

    logger.info(
        f"[Router] Proceeding to debate for verification: "
        f"direction={consensus_direction}, confidence={consensus_confidence:.0f}%"
    )
    return "debate"


def route_entry_or_position(
    state: TradingDecisionState
) -> Literal["entry_debate", "position_debate"]:
    """
    Route based on whether there's an existing position.

    This is the main branching point after analysis:
    - If no position: Go to entry debate (Bull vs Bear for new trade)
    - If has position: Go to position debate (Position Bull vs Bear for management)

    Args:
        state: Current trading decision state after analysis

    Returns:
        "entry_debate" for new trades, "position_debate" for position management
    """
    has_position = state.get("has_position", False)
    pair = state.get("pair", "UNKNOWN")

    if has_position:
        position_side = state.get("position_side", "unknown")
        position_profit = state.get("position_profit_pct", 0.0)
        logger.info(
            f"[Router] {pair}: Has {position_side} position ({position_profit:.2f}%), "
            f"routing to position debate"
        )
        return "position_debate"
    else:
        logger.info(f"[Router] {pair}: No position, routing to entry debate")
        return "entry_debate"


def route_after_debate(
    state: TradingDecisionState
) -> Literal["reflection", "end"]:
    """
    Route after debate subgraph completion.

    Check if judge reached a verdict, then proceed to reflection verification.

    Args:
        state: Current trading decision state after debate

    Returns:
        "reflection" to proceed to LLM-based verification, "end" if no verdict (error)
    """
    judge_verdict = state.get("judge_verdict")

    if not judge_verdict:
        logger.warning("[Router] No judge verdict (debate error), ending with wait")
        return "end"

    # All verdicts proceed to reflection verification
    logger.debug(f"[Router] Proceeding to reflection verification: verdict={judge_verdict.verdict}")
    return "reflection"


def route_after_reflection(
    state: TradingDecisionState
) -> Literal["executor", "end"]:
    """
    Route after reflection verification (LLM-based analysis validation).

    Reflection is supportive verification - it should rarely block trades.
    Only stops if LLM explicitly says should_proceed=False with low confidence.

    Args:
        state: Current trading decision state after reflection

    Returns:
        "executor" to proceed to Executor Agent, "end" only in extreme cases
    """
    reflection_result = state.get("reflection_result")
    should_proceed = state.get("reflection_should_proceed", True)
    
    # If no reflection result (error), proceed anyway
    if not reflection_result:
        logger.info("[Router] No reflection result (error), proceeding to executor")
        return "executor"
    
    # Get more context for logging
    direction_conf = getattr(reflection_result, 'direction_confidence', 0)
    timing_conf = getattr(reflection_result, 'timing_confidence', 0)
    critical_count = len(getattr(reflection_result, 'critical_issues', []))
    
    # Check if reflection suggests stopping (should be rare)
    if not should_proceed:
        logger.warning(
            f"[Router] Reflection suggests caution (Dir:{direction_conf:.0f}%, "
            f"Timing:{timing_conf:.0f}%, Critical:{critical_count}) - stopping"
        )
        return "end"
    
    # Check reflection result attributes
    if hasattr(reflection_result, 'should_proceed') and not reflection_result.should_proceed:
        logger.warning(
            f"[Router] Reflection flagged critical issues - stopping"
        )
        return "end"
    
    # Log reflection result and proceed
    confidence_adj = state.get("reflection_confidence_adjustment", 0)
    logger.info(
        f"[Router] Reflection verified (Dir:{direction_conf:.0f}%, "
        f"Timing:{timing_conf:.0f}%, Adj:{confidence_adj:+.0f}%) - proceeding to executor"
    )
    
    return "executor"


def route_after_position_reflection(
    state: TradingDecisionState
) -> Literal["executor", "end"]:
    """
    Route after position reflection verification.

    If reflection finds critical issues, end here.
    Otherwise proceed to Executor Agent.

    Args:
        state: Current trading decision state after position reflection

    Returns:
        "executor" to proceed to Executor Agent, "end" if issues found
    """
    reflection_result = state.get("position_reflection_result")
    should_proceed = state.get("position_reflection_should_proceed", True)

    if not reflection_result:
        logger.warning("[Router] No position reflection result, proceeding to executor")
        return "executor"

    if not should_proceed:
        logger.warning(
            f"[Router] POSITION REFLECTION SUGGESTS STOPPING - "
            f"Critical issues detected, defaulting to HOLD"
        )
        return "end"

    if hasattr(reflection_result, 'should_proceed') and not reflection_result.should_proceed:
        logger.warning("[Router] POSITION REFLECTION REJECTED - should_proceed=False")
        return "end"

    confidence_adj = state.get("position_reflection_confidence_adjustment", 0)
    if confidence_adj and confidence_adj != 0:
        logger.info(
            f"[Router] Position reflection passed with confidence adjustment: {confidence_adj:+.0f}%"
        )

    return "executor"


# ============= Legacy Grounding Routes (Deprecated) =============

def route_after_grounding(
    state: TradingDecisionState
) -> Literal["executor", "end"]:
    """
    [DEPRECATED] Route after grounding verification.
    
    Use route_after_reflection instead.
    Kept for backward compatibility.
    """
    # Delegate to reflection routing
    return route_after_reflection(state)


def route_after_position_grounding(
    state: TradingDecisionState
) -> Literal["executor", "end"]:
    """
    [DEPRECATED] Route after position grounding verification.
    
    Use route_after_position_reflection instead.
    Kept for backward compatibility.
    """
    return route_after_position_reflection(state)


# ============= Helper Functions =============

def should_skip_pattern_analysis(state: AnalysisState) -> bool:
    """
    Determine if pattern analysis should be skipped.

    Pattern analysis is expensive (vision model), skip if:
    - No chart data available
    - Trend is very clear (save tokens)

    Args:
        state: Current analysis state

    Returns:
        True if pattern analysis should be skipped
    """
    # Skip if no OHLCV data for chart generation
    if not state.get("ohlcv_data"):
        return True

    return False


def should_use_vision_analysis(state: TradingDecisionState) -> bool:
    """
    Determine if vision analysis should be used for trend/pattern agents.

    Vision analysis is more expensive but more accurate for chart patterns.

    Args:
        state: Current trading decision state

    Returns:
        True if vision analysis should be used
    """
    llm_config = state.get("llm_config", {})
    return llm_config.get("use_vision", False)


def get_skip_wait_result() -> dict:
    """
    Get default result for skipping to wait.

    Returns standard wait result when skipping debate/validation.
    """
    return {
        "final_action": "signal_wait",
        "final_confidence": 0.0,
        "final_leverage": None,
        "final_reason": "Skipped due to low confidence or neutral direction",
        "is_valid": True,
        "execution_completed": True
    }


def route_for_position_management(
    state: TradingDecisionState
) -> Literal["analyze_position", "analyze_entry"]:
    """
    Route based on whether there's an existing position.

    Position management has different analysis requirements than entry.

    Args:
        state: Current trading decision state

    Returns:
        "analyze_position" if holding, "analyze_entry" if not
    """
    has_position = state.get("has_position", False)

    if has_position:
        logger.debug("[Router] Routing to position analysis")
        return "analyze_position"

    logger.debug("[Router] Routing to entry analysis")
    return "analyze_entry"
