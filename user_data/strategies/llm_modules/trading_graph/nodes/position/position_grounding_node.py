"""
Position Grounding Verification Node for LangGraph.

Layer 4 Hallucination Prevention for position management.
Verifies MFE/MAE/drawdown/hold_count claims against actual position metrics.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    PositionMetrics,
    PositionVerdict,
)

logger = logging.getLogger(__name__)

# Tolerance for position metric verification (percentage)
POSITION_TOLERANCE_PCT = 20.0

# Confidence penalty per false position claim
CONFIDENCE_PENALTY_PER_FALSE_CLAIM = 10.0

# Hallucination threshold for rejection
HALLUCINATION_REJECTION_THRESHOLD = 50.0


@dataclass
class PositionClaim:
    """Represents a claim about position metrics made by an agent."""
    metric: str                     # e.g., "MFE", "drawdown", "hold_count"
    claimed_value: Optional[float]
    comparison: str                 # "=", ">", "<", "qualitative"
    source: str                     # "position_bull" or "position_bear"
    original_text: str
    actual_value: Optional[float] = None
    is_verified: bool = False
    is_valid: bool = True
    discrepancy_pct: float = 0.0


@dataclass
class PositionGroundingResult:
    """Result of position grounding verification with auto-correction."""
    total_claims: int = 0
    verified_claims: int = 0
    valid_claims: int = 0
    false_claims: int = 0
    hallucination_score: float = 0.0
    claims: List[PositionClaim] = field(default_factory=list)
    should_reject: bool = False
    confidence_penalty: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # NEW: Auto-correction fields
    corrected_claims: List[PositionClaim] = field(default_factory=list)
    corrected_values: Dict[str, float] = field(default_factory=dict)


def position_grounding_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Position Grounding Verification node.

    Extracts position metric claims from debate arguments and verifies against actual data.

    Args:
        state: Current trading decision state with position debate results
        config: Node configuration

    Returns:
        Dict with position_grounding_result update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[PositionGrounding] Starting verification for {pair}")

    try:
        # Get position metrics (ground truth)
        position_metrics: PositionMetrics = state.get("position_metrics") or {}
        actual_values = _build_actual_values(position_metrics, state)

        # Extract claims from both arguments
        all_claims = []

        bull_argument = state.get("position_bull_argument")
        if bull_argument:
            bull_claims = _extract_position_claims(bull_argument, "position_bull")
            all_claims.extend(bull_claims)

        bear_argument = state.get("position_bear_argument")
        if bear_argument:
            bear_claims = _extract_position_claims(bear_argument, "position_bear")
            all_claims.extend(bear_claims)

        # Verify claims
        result = _verify_position_claims(all_claims, actual_values)

        # Build corrected context for Executor Agent
        corrected_context = _build_position_corrected_context(result, actual_values, state)
        grounding_summary = _build_position_grounding_summary(result)

        # Log results
        if result.false_claims > 0:
            logger.warning(
                f"[PositionGrounding] {pair}: {result.false_claims} false claims detected "
                f"(hallucination_score={result.hallucination_score:.0f}%)"
            )
        else:
            logger.debug(
                f"[PositionGrounding] {pair}: All {result.verified_claims} claims verified"
            )

        return {
            "position_grounding_result": result,
            "grounding_verified": not result.should_reject,
            # Enhanced outputs for Executor Agent
            "position_grounding_corrected_values": result.corrected_values,
            "position_corrected_context": corrected_context,
            "position_grounding_summary": grounding_summary,
            "position_hallucination_score": result.hallucination_score,
            "position_false_claim_details": [
                {"metric": c.metric, "claimed": c.claimed_value, "actual": c.actual_value}
                for c in result.corrected_claims
            ]
        }

    except Exception as e:
        logger.error(f"[PositionGrounding] Verification failed: {e}")
        # Return permissive result on error
        return {
            "position_grounding_result": PositionGroundingResult(
                warnings=[f"Verification error: {e}"]
            ),
            "grounding_verified": True,
            "errors": [f"PositionGrounding: {e}"]
        }


def _build_actual_values(position_metrics: PositionMetrics, state: TradingDecisionState) -> Dict[str, float]:
    """Build actual values dict from position metrics and state."""
    return {
        "MFE": position_metrics.get("max_profit_pct", state.get("position_profit_pct", 0.0)),
        "MAE": position_metrics.get("max_loss_pct", 0.0),
        "current_profit": position_metrics.get("current_profit_pct", state.get("position_profit_pct", 0.0)),
        "drawdown": position_metrics.get("drawdown_from_peak_pct", 0.0),
        "hold_count": position_metrics.get("hold_count", 0),
        "time_in_position": position_metrics.get("time_in_position_hours", 0.0),
    }


def _extract_position_claims(argument, source: str) -> List[PositionClaim]:
    """
    Extract position metric claims from debate argument.

    Args:
        argument: DebateArgument dataclass
        source: "position_bull" or "position_bear"

    Returns:
        List of PositionClaim objects
    """
    claims = []

    # Combine all text sources
    text_sources = []
    text_sources.extend(argument.key_points or [])
    text_sources.extend(argument.risk_factors or [])
    text_sources.extend(argument.supporting_signals or [])
    text_sources.extend(argument.counter_arguments or [])
    if argument.reasoning:
        text_sources.append(argument.reasoning)

    for text in text_sources:
        if not text:
            continue

        # Extract MFE claims
        mfe_patterns = [
            r'MFE[:\s]*([0-9.]+)%?',
            r'最大浮盈[:\s]*([0-9.]+)%?',
            r'max_profit[:\s]*([0-9.]+)%?',
        ]
        for pattern in mfe_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(PositionClaim(
                    metric="MFE",
                    claimed_value=float(match),
                    comparison="=",
                    source=source,
                    original_text=text[:100]
                ))

        # Extract MAE claims
        mae_patterns = [
            r'MAE[:\s]*([0-9.]+)%?',
            r'最大浮亏[:\s]*([0-9.]+)%?',
            r'max_loss[:\s]*([0-9.]+)%?',
        ]
        for pattern in mae_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(PositionClaim(
                    metric="MAE",
                    claimed_value=float(match),
                    comparison="=",
                    source=source,
                    original_text=text[:100]
                ))

        # Extract drawdown claims
        drawdown_patterns = [
            r'MFE回撤[:\s]*([0-9.]+)%?',
            r'回撤[:\s]*([0-9.]+)%?',
            r'drawdown[:\s]*([0-9.]+)%?',
            r'drawback[:\s]*([0-9.]+)%?',
        ]
        for pattern in drawdown_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(PositionClaim(
                    metric="drawdown",
                    claimed_value=float(match),
                    comparison="=",
                    source=source,
                    original_text=text[:100]
                ))

        # Extract hold_count claims
        hold_patterns = [
            r'hold_count[:\s]*([0-9]+)',
            r'HOLD次数[:\s]*([0-9]+)',
            r'连续(\d+)次HOLD',
            r'连续持有(\d+)次',
        ]
        for pattern in hold_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(PositionClaim(
                    metric="hold_count",
                    claimed_value=float(match),
                    comparison="=",
                    source=source,
                    original_text=text[:100]
                ))

        # Extract comparison claims (>, <)
        comparison_patterns = [
            (r'MFE回撤\s*>\s*([0-9.]+)%?', 'drawdown', '>'),
            (r'MFE回撤\s*<\s*([0-9.]+)%?', 'drawdown', '<'),
            (r'hold_count\s*[≥>=]\s*([0-9]+)', 'hold_count', '>='),
            (r'HOLD次数\s*[≥>=]\s*([0-9]+)', 'hold_count', '>='),
        ]
        for pattern, metric, comp in comparison_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(PositionClaim(
                    metric=metric,
                    claimed_value=float(match),
                    comparison=comp,
                    source=source,
                    original_text=text[:100]
                ))

    return claims


def _verify_position_claims(claims: List[PositionClaim], actual_values: Dict[str, float]) -> PositionGroundingResult:
    """
    Verify position claims against actual values and auto-correct false claims.

    Args:
        claims: List of extracted claims
        actual_values: Dict of actual position metric values

    Returns:
        PositionGroundingResult with verification results and corrections
    """
    result = PositionGroundingResult()
    result.total_claims = len(claims)

    for claim in claims:
        # Check if we have actual value for this metric
        if claim.metric not in actual_values:
            claim.is_verified = False
            result.warnings.append(f"Cannot verify {claim.metric}: no actual value")
            continue

        actual = actual_values[claim.metric]
        claim.actual_value = actual
        claim.is_verified = True
        result.verified_claims += 1

        # Verify based on comparison type
        if claim.comparison == "=":
            is_valid, discrepancy = _verify_equals(claim.claimed_value, actual)
            claim.is_valid = is_valid
            claim.discrepancy_pct = discrepancy

        elif claim.comparison == ">":
            claim.is_valid = actual > claim.claimed_value
            claim.discrepancy_pct = 0 if claim.is_valid else abs(actual - claim.claimed_value)

        elif claim.comparison == "<":
            claim.is_valid = actual < claim.claimed_value
            claim.discrepancy_pct = 0 if claim.is_valid else abs(actual - claim.claimed_value)

        elif claim.comparison == ">=":
            claim.is_valid = actual >= claim.claimed_value
            claim.discrepancy_pct = 0 if claim.is_valid else abs(actual - claim.claimed_value)

        if claim.is_valid:
            result.valid_claims += 1
        else:
            result.false_claims += 1
            
            # AUTO-CORRECTION: Store the correct value
            result.corrected_values[claim.metric] = actual
            result.corrected_claims.append(claim)
            
            logger.info(
                f"[PositionGrounding] AUTO-CORRECTED: {claim.metric} "
                f"claimed={claim.claimed_value} -> actual={actual:.2f} "
                f"(discrepancy={claim.discrepancy_pct:.1f}%)"
            )

        result.claims.append(claim)

    # Calculate hallucination score
    if result.verified_claims > 0:
        result.hallucination_score = (result.false_claims / result.verified_claims) * 100
    else:
        result.hallucination_score = 0.0

    # Calculate confidence penalty
    result.confidence_penalty = result.false_claims * CONFIDENCE_PENALTY_PER_FALSE_CLAIM

    # Determine if should reject
    result.should_reject = result.hallucination_score >= HALLUCINATION_REJECTION_THRESHOLD

    # Log correction summary
    if result.corrected_claims:
        logger.info(
            f"[PositionGrounding] Corrected {len(result.corrected_claims)} claims: "
            f"{list(result.corrected_values.keys())}"
        )

    return result


def _verify_equals(claimed: float, actual: float) -> tuple:
    """
    Verify equality claim with tolerance.

    Args:
        claimed: Claimed value
        actual: Actual value

    Returns:
        Tuple of (is_valid, discrepancy_percentage)
    """
    if actual == 0:
        if claimed == 0:
            return True, 0.0
        else:
            return False, 100.0

    discrepancy = abs(claimed - actual) / abs(actual) * 100
    is_valid = discrepancy <= POSITION_TOLERANCE_PCT

    return is_valid, discrepancy


def _build_position_corrected_context(
    result: PositionGroundingResult,
    actual_values: Dict[str, float],
    state: TradingDecisionState
) -> str:
    """
    Build corrected context text for Position Executor Agent.
    
    Args:
        result: PositionGroundingResult with verification details
        actual_values: Dict of actual position metric values
        state: Current trading decision state
        
    Returns:
        Formatted corrected context string
    """
    lines = [
        "=== Position Grounding 验证结果 ===",
        f"幻觉检测分数: {result.hallucination_score:.0f}% "
        f"({'可接受' if result.hallucination_score < 30 else '需注意' if result.hallucination_score < 50 else '高风险'})"
    ]
    
    # List corrected claims
    if result.corrected_claims:
        lines.append("")
        lines.append("❌ 错误声明已纠正:")
        for claim in result.corrected_claims:
            source_cn = "PositionBull" if "bull" in claim.source else "PositionBear"
            lines.append(
                f"  - {source_cn}声称 \"{claim.metric}={claim.claimed_value}\" "
                f"→ 实际值: {claim.metric}={claim.actual_value:.2f}"
            )
    
    # List verified correct claims
    valid_claims = [c for c in result.claims if c.is_verified and c.is_valid]
    if valid_claims:
        lines.append("")
        lines.append("✅ 验证通过的声明:")
        for claim in valid_claims[:3]:
            lines.append(f"  - {claim.metric}: 正确")
    
    # Position metrics summary
    lines.append("")
    lines.append("📊 纠正后的持仓数据:")
    
    position_side = state.get("position_side", "unknown")
    position_profit = state.get("position_profit_pct", 0.0)
    lines.append(f"  方向: {'多仓' if position_side == 'long' else '空仓'}")
    lines.append(f"  当前盈亏: {position_profit:+.2f}%")
    
    for metric, value in sorted(actual_values.items()):
        interpretation = _get_position_metric_interpretation(metric, value)
        lines.append(f"  {metric}: {value:.2f}% {interpretation}")
    
    lines.append("")
    lines.append("⚠️ 请基于以上纠正后的持仓数据做出决策")
    
    return "\n".join(lines)


def _build_position_grounding_summary(result: PositionGroundingResult) -> str:
    """
    Build concise position grounding summary for logging.
    
    Args:
        result: PositionGroundingResult with verification details
        
    Returns:
        One-line summary string
    """
    return (
        f"持仓验证: {result.valid_claims}/{result.verified_claims} 通过 | "
        f"幻觉分: {result.hallucination_score:.0f}% | "
        f"纠正: {len(result.corrected_claims)}项 | "
        f"置信度惩罚: -{result.confidence_penalty:.0f}%"
    )


def _get_position_metric_interpretation(metric: str, value: float) -> str:
    """
    Get human-readable interpretation of position metric value.
    
    Args:
        metric: Metric name
        value: Metric value
        
    Returns:
        Interpretation string in parentheses
    """
    interpretations = {
        "MFE": lambda v: "(良好盈利)" if v > 5 else "(小幅盈利)" if v > 0 else "(无盈利)",
        "MAE": lambda v: "(高风险)" if v > 5 else "(中等风险)" if v > 2 else "(风险可控)",
        "drawdown": lambda v: "(严重回撤)" if v > 50 else "(中等回撤)" if v > 30 else "(轻微回撤)",
        "current_profit": lambda v: "(盈利中)" if v > 0 else "(亏损中)",
        "hold_count": lambda v: "(长期持有)" if v > 5 else "(正常)" if v > 0 else "(新仓位)",
        "time_in_position": lambda v: "(长时间持仓)" if v > 24 else "(正常时长)" if v > 1 else "(新仓位)",
    }
    
    if metric in interpretations:
        return interpretations[metric](value)
    return ""

