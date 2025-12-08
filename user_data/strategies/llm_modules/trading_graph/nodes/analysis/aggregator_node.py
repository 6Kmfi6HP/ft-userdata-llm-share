"""
Aggregator Node for LangGraph.

Collects and aggregates results from all analysis agents using weighted consensus.
This is the fan-in node that runs after all parallel analysis nodes complete.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    AnalysisState,
    AgentReport,
    Direction,
    AGENT_WEIGHTS,
)

logger = logging.getLogger(__name__)


def aggregator_node(state: AnalysisState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Weighted consensus aggregation node.

    Collects reports from all analysis agents and calculates:
    - Consensus direction using weighted voting
    - Consensus confidence using weighted average
    - Aggregated key support/resistance levels

    Args:
        state: Current analysis state with all agent reports
        config: Node configuration

    Returns:
        Dict with consensus_direction, consensus_confidence, key levels, weighted_scores
    """
    logger.debug("[Aggregator] Starting weighted consensus aggregation")

    # Collect all valid reports
    reports = []
    for report in [
        state.get("indicator_report"),
        state.get("trend_report"),
        state.get("sentiment_report"),
        state.get("pattern_report")
    ]:
        if report and report.is_valid():
            reports.append(report)

    if not reports:
        logger.warning("[Aggregator] No valid reports to aggregate")
        return {
            "consensus_direction": Direction.NEUTRAL,
            "consensus_confidence": 0.0,
            "key_support": None,
            "key_resistance": None,
            "weighted_scores": {}
        }

    # Calculate weighted direction scores
    direction_scores = {
        Direction.LONG: 0.0,
        Direction.SHORT: 0.0,
        Direction.NEUTRAL: 0.0
    }

    total_weight = 0.0
    weighted_confidence_sum = 0.0

    for report in reports:
        weight = AGENT_WEIGHTS.get(report.agent_name, 1.0)
        total_weight += weight

        # Add to direction score
        if report.direction:
            direction_scores[report.direction] += weight * (report.confidence / 100)

        # Add to weighted confidence
        weighted_confidence_sum += weight * report.confidence

    # Determine consensus direction (20% threshold)
    long_score = direction_scores[Direction.LONG]
    short_score = direction_scores[Direction.SHORT]

    if long_score > short_score * 1.2:
        consensus_dir = Direction.LONG
    elif short_score > long_score * 1.2:
        consensus_dir = Direction.SHORT
    else:
        consensus_dir = Direction.NEUTRAL

    # Calculate weighted confidence
    consensus_conf = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0

    # Aggregate key levels
    support_levels = []
    resistance_levels = []

    for report in reports:
        if report.key_levels:
            if report.key_levels.get("support"):
                support_levels.append(report.key_levels["support"])
            if report.key_levels.get("resistance"):
                resistance_levels.append(report.key_levels["resistance"])

    # Use median for key levels if multiple values
    key_support = _get_median(support_levels) if support_levels else None
    key_resistance = _get_median(resistance_levels) if resistance_levels else None

    logger.info(
        f"[Aggregator] Consensus: direction={consensus_dir.value}, "
        f"confidence={consensus_conf:.1f}%, "
        f"agents={len(reports)}, "
        f"scores={{LONG: {long_score:.2f}, SHORT: {short_score:.2f}}}"
    )

    if key_support or key_resistance:
        logger.info(f"[Aggregator] Key levels: support={key_support}, resistance={key_resistance}")

    return {
        "consensus_direction": consensus_dir,
        "consensus_confidence": consensus_conf,
        "key_support": key_support,
        "key_resistance": key_resistance,
        "weighted_scores": {
            "long": long_score,
            "short": short_score,
            "neutral": direction_scores[Direction.NEUTRAL]
        }
    }


def _get_median(values: List[float]) -> Optional[float]:
    """Get median value from a list."""
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def get_aggregation_summary(state: AnalysisState) -> str:
    """
    Generate a text summary of the aggregation for debate agents.

    Args:
        state: Analysis state with aggregated results

    Returns:
        Formatted summary string
    """
    lines = [
        "=== 分析智能体共识摘要 ===",
        f"共识方向: {state.get('consensus_direction', Direction.NEUTRAL).value}",
        f"共识置信度: {state.get('consensus_confidence', 0):.1f}%",
    ]

    if state.get("key_support") or state.get("key_resistance"):
        lines.append(
            f"关键价位: 支撑 {state.get('key_support', 'N/A')} / "
            f"阻力 {state.get('key_resistance', 'N/A')}"
        )

    weighted = state.get("weighted_scores", {})
    if weighted:
        lines.append(
            f"加权得分: LONG={weighted.get('long', 0):.2f}, "
            f"SHORT={weighted.get('short', 0):.2f}"
        )

    # Add individual agent summaries
    for report_key, label in [
        ("indicator_report", "指标分析"),
        ("trend_report", "趋势分析"),
        ("sentiment_report", "情绪分析"),
        ("pattern_report", "形态分析")
    ]:
        report = state.get(report_key)
        if report and report.is_valid():
            dir_str = report.direction.value if report.direction else "neutral"
            lines.append(f"  - {label}: {dir_str} ({report.confidence:.0f}%)")

    return "\n".join(lines)
