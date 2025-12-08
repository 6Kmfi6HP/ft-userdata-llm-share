"""
Grounding Verification Node for LangGraph.

Layer 4 of the 6-layer hallucination prevention architecture.
Validates that indicator claims made during debate match actual data.

Reference: LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md
"Grounding验证: 将LLM的输出与可验证的外部数据源进行对照"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.runnables import RunnableConfig

from ...state import TradingDecisionState, DebateArgument

logger = logging.getLogger(__name__)


# ============= Data Classes =============

@dataclass
class IndicatorClaim:
    """
    A claim about an indicator value made during debate.

    Example claims:
    - "RSI处于超卖区域(28)" -> indicator="RSI", claimed_value=28, comparison="="
    - "ADX>25表明强趋势" -> indicator="ADX", claimed_value=25, comparison=">"
    """
    indicator: str              # e.g., "RSI", "MACD", "ADX", "EMA"
    claimed_value: Optional[float]  # The value claimed (None if qualitative)
    comparison: str             # "=", ">", "<", ">=", "<=", "qualitative"
    source: str                 # "bull" or "bear"
    original_text: str          # Original claim text for debugging
    actual_value: Optional[float] = None  # Actual value from data
    is_verified: bool = False   # Whether claim was verified
    is_valid: bool = True       # Whether claim matches actual value
    discrepancy_pct: float = 0.0  # Percentage difference

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "claimed_value": self.claimed_value,
            "comparison": self.comparison,
            "source": self.source,
            "actual_value": self.actual_value,
            "is_verified": self.is_verified,
            "is_valid": self.is_valid,
            "discrepancy_pct": self.discrepancy_pct
        }


@dataclass
class GroundingResult:
    """
    Result of grounding verification.

    Tracks how many claims were verified and their validity.
    Now includes automatic correction of false claims.
    """
    total_claims: int = 0
    verified_claims: int = 0
    valid_claims: int = 0
    false_claims: int = 0
    hallucination_score: float = 0.0  # 0-100, higher = more hallucinations
    claims: List[IndicatorClaim] = field(default_factory=list)
    should_reject: bool = False       # If hallucination is too high
    confidence_penalty: float = 0.0   # Penalty to apply to final confidence
    warnings: List[str] = field(default_factory=list)
    # NEW: Auto-correction fields
    corrected_claims: List[IndicatorClaim] = field(default_factory=list)  # Claims that were corrected
    corrected_values: Dict[str, float] = field(default_factory=dict)  # indicator -> corrected value

    def to_dict(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "valid_claims": self.valid_claims,
            "false_claims": self.false_claims,
            "hallucination_score": self.hallucination_score,
            "should_reject": self.should_reject,
            "confidence_penalty": self.confidence_penalty,
            "warnings": self.warnings,
            "claims": [c.to_dict() for c in self.claims],
            "corrected_claims": [c.to_dict() for c in self.corrected_claims],
            "corrected_values": self.corrected_values
        }


# ============= Constants =============

# Indicator patterns for extraction
INDICATOR_PATTERNS = {
    # RSI patterns
    "RSI": [
        r"RSI[=:：]?\s*(\d+(?:\.\d+)?)",
        r"RSI(?:处于|在|为|达到|已达)?\s*(\d+(?:\.\d+)?)",
        r"RSI\s*[<>]=?\s*(\d+(?:\.\d+)?)",
        r"RSI超[买卖](?:区域)?[（\(]?(\d+(?:\.\d+)?)[）\)]?",
    ],
    # ADX patterns
    "ADX": [
        r"ADX[=:：]?\s*(\d+(?:\.\d+)?)",
        r"ADX\s*[<>]=?\s*(\d+(?:\.\d+)?)",
        r"ADX(?:仅为|低于|高于|超过)?\s*(\d+(?:\.\d+)?)",
    ],
    # MACD patterns
    "MACD": [
        r"MACD[=:：]?\s*(-?\d+(?:\.\d+)?)",
        r"MACD柱[=:：]?\s*(-?\d+(?:\.\d+)?)",
        r"MACD_hist[=:：]?\s*(-?\d+(?:\.\d+)?)",
    ],
    # Stochastic patterns
    "STOCH_K": [
        r"(?:Stoch|随机指标)?%?K[=:：]?\s*(\d+(?:\.\d+)?)",
        r"Stochastic\s*K[=:：]?\s*(\d+(?:\.\d+)?)",
    ],
    "STOCH_D": [
        r"(?:Stoch|随机指标)?%?D[=:：]?\s*(\d+(?:\.\d+)?)",
    ],
    # Price/EMA patterns
    "EMA_20": [
        r"EMA20[=:：]?\s*(\d+(?:\.\d+)?)",
        r"EMA\s*20[=:：]?\s*(\d+(?:\.\d+)?)",
    ],
    "EMA_50": [
        r"EMA50[=:：]?\s*(\d+(?:\.\d+)?)",
        r"EMA\s*50[=:：]?\s*(\d+(?:\.\d+)?)",
    ],
    # MFI patterns
    "MFI": [
        r"MFI[=:：]?\s*(\d+(?:\.\d+)?)",
        r"MFI\s*[<>]=?\s*(\d+(?:\.\d+)?)",
    ],
}

# Qualitative claim patterns (without specific values)
QUALITATIVE_PATTERNS = {
    "RSI": [
        (r"RSI超卖", "oversold", lambda v: v < 30),
        (r"RSI超买", "overbought", lambda v: v > 70),
        (r"RSI中性", "neutral", lambda v: 30 <= v <= 70),
    ],
    "ADX": [
        (r"ADX(?:显示)?强趋势", "strong_trend", lambda v: v > 25),
        (r"ADX(?:显示)?弱趋势", "weak_trend", lambda v: v < 20),
        (r"趋势强度(?:较)?(?:强|高)", "strong_trend", lambda v: v > 25),
    ],
    "MACD": [
        (r"MACD金叉", "golden_cross", lambda v: v > 0),
        (r"MACD死叉", "death_cross", lambda v: v < 0),
        (r"MACD(?:柱)?为正", "positive", lambda v: v > 0),
        (r"MACD(?:柱)?为负", "negative", lambda v: v < 0),
    ],
}

# Tolerance for value comparison (percentage)
VALUE_TOLERANCE_PCT = 15.0  # Allow 15% deviation before flagging

# Hallucination thresholds
HALLUCINATION_WARNING_THRESHOLD = 30.0  # Warn if score > 30%
HALLUCINATION_REJECT_THRESHOLD = 50.0   # Reject if score > 50%
CONFIDENCE_PENALTY_PER_FALSE_CLAIM = 5.0  # Reduce confidence by 5% per false claim


# ============= Main Node Function =============

def grounding_node(
    state: TradingDecisionState,
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Grounding Verification Node - Layer 4 Hallucination Prevention.

    Extracts indicator claims from Bull/Bear arguments and verifies
    them against actual market data.

    Args:
        state: Current trading decision state with debate arguments
        config: Node configuration

    Returns:
        Dict with grounding_result and potentially modified confidence
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[GroundingNode] Starting verification for {pair}")

    try:
        # Get debate arguments
        bull_argument = state.get("bull_argument")
        bear_argument = state.get("bear_argument")

        if not bull_argument and not bear_argument:
            logger.warning("[GroundingNode] No debate arguments to verify")
            return {"grounding_result": GroundingResult()}

        # Extract actual indicator values from market_context
        actual_values = _extract_actual_values(state)

        # Extract claims from debate arguments
        claims = []
        if bull_argument:
            claims.extend(_extract_claims_from_argument(bull_argument, "bull"))
        if bear_argument:
            claims.extend(_extract_claims_from_argument(bear_argument, "bear"))

        # Verify claims against actual values
        result = _verify_claims(claims, actual_values)

        # Calculate confidence penalty
        final_confidence = state.get("final_confidence", 50.0)
        if result.confidence_penalty > 0:
            adjusted_confidence = max(0, final_confidence - result.confidence_penalty)
            logger.info(
                f"[GroundingNode] Confidence penalty applied: "
                f"{final_confidence:.0f}% -> {adjusted_confidence:.0f}% "
                f"({result.false_claims} false claims)"
            )
        else:
            adjusted_confidence = final_confidence

        # Build corrected context for Executor Agent
        corrected_context = _build_corrected_context(result, actual_values)
        grounding_summary = _build_grounding_summary(result)

        # Log summary
        logger.info(
            f"[GroundingNode] {pair} verification: "
            f"{result.valid_claims}/{result.verified_claims} claims valid, "
            f"hallucination_score={result.hallucination_score:.0f}%"
        )

        if result.should_reject:
            logger.warning(
                f"[GroundingNode] HIGH HALLUCINATION DETECTED for {pair}! "
                f"Score={result.hallucination_score:.0f}% - recommending rejection"
            )

        return {
            "grounding_result": result,
            "final_confidence": adjusted_confidence,
            "grounding_verified": True,
            "validation_warnings": result.warnings,
            # Enhanced outputs for Executor Agent
            "grounding_corrected_values": result.corrected_values,
            "corrected_context": corrected_context,
            "grounding_summary": grounding_summary,
            "hallucination_score": result.hallucination_score,
            "total_claims": result.total_claims,
            "verified_claims": result.verified_claims,
            "false_claims": result.false_claims,
            "false_claim_details": [c.to_dict() for c in result.corrected_claims]
        }

    except Exception as e:
        logger.error(f"[GroundingNode] Verification failed: {e}")
        return {
            "grounding_result": GroundingResult(
                warnings=[f"Grounding verification error: {str(e)}"]
            ),
            "grounding_verified": False,
            "errors": [f"GroundingNode: {str(e)}"]
        }


# ============= Helper Functions =============

def _extract_actual_values(state: TradingDecisionState) -> Dict[str, float]:
    """
    Extract actual indicator values from state.

    Priority (following arXiv 2512.01123 - avoid text parsing for numerical data):
    1. verified_indicator_data (structured data from ContextBuilder)
    2. Fallback to market_context text parsing (for backward compatibility)
    """
    # PRIORITY 1: Use structured data if available (preferred method)
    verified_data = state.get("verified_indicator_data")
    if verified_data and len(verified_data) > 0:
        logger.debug(f"[GroundingNode] Using verified_indicator_data: {verified_data}")
        return verified_data.copy()
    
    # PRIORITY 2: Fallback to text parsing (backward compatibility)
    # Note: This is a valid fallback - text parsing works but is less reliable
    logger.debug(
        "[GroundingNode] No verified_indicator_data found, using text parsing fallback."
    )
    
    actual_values = {}
    market_context = state.get("market_context", "")

    # Parse RSI from market_context
    rsi_match = re.search(r"RSI[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if rsi_match:
        actual_values["RSI"] = float(rsi_match.group(1))

    # Parse ADX from market_context
    adx_match = re.search(r"ADX[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if adx_match:
        actual_values["ADX"] = float(adx_match.group(1))

    # Parse MACD from market_context
    macd_match = re.search(r"MACD柱?[=:：]\s*(-?\d+(?:\.\d+)?)", market_context)
    if macd_match:
        actual_values["MACD"] = float(macd_match.group(1))

    # Parse Stochastic %K
    stoch_k_match = re.search(r"%K[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if stoch_k_match:
        actual_values["STOCH_K"] = float(stoch_k_match.group(1))

    # Parse Stochastic %D
    stoch_d_match = re.search(r"%D[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if stoch_d_match:
        actual_values["STOCH_D"] = float(stoch_d_match.group(1))

    # Parse MFI
    mfi_match = re.search(r"MFI[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if mfi_match:
        actual_values["MFI"] = float(mfi_match.group(1))

    # Parse EMA values
    ema20_match = re.search(r"EMA\s*20[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if ema20_match:
        actual_values["EMA_20"] = float(ema20_match.group(1))

    ema50_match = re.search(r"EMA\s*50[=:：]\s*(\d+(?:\.\d+)?)", market_context)
    if ema50_match:
        actual_values["EMA_50"] = float(ema50_match.group(1))

    # Also check for indicators in a structured section
    # Pattern: "指标值: RSI=45.2, ADX=28.5, ..."
    indicator_section = re.search(
        r"(?:指标|Indicators?)[=:：]?\s*(.+?)(?:\n|$)",
        market_context,
        re.IGNORECASE
    )
    if indicator_section:
        section_text = indicator_section.group(1)
        for indicator, patterns in INDICATOR_PATTERNS.items():
            if indicator not in actual_values:
                for pattern in patterns:
                    match = re.search(pattern, section_text, re.IGNORECASE)
                    if match:
                        actual_values[indicator] = float(match.group(1))
                        break

    logger.debug(f"[GroundingNode] Extracted actual values (text parsing): {actual_values}")
    return actual_values


def _extract_claims_from_argument(
    argument: DebateArgument,
    source: str
) -> List[IndicatorClaim]:
    """
    Extract indicator claims from a debate argument.

    Looks in key_points, supporting_signals, and reasoning for claims.
    """
    claims = []

    # Collect all text to search
    texts_to_search = []

    if hasattr(argument, 'key_points') and argument.key_points:
        texts_to_search.extend(argument.key_points)

    if hasattr(argument, 'supporting_signals') and argument.supporting_signals:
        texts_to_search.extend(argument.supporting_signals)

    if hasattr(argument, 'reasoning') and argument.reasoning:
        texts_to_search.append(argument.reasoning)

    if hasattr(argument, 'risk_factors') and argument.risk_factors:
        texts_to_search.extend(argument.risk_factors)

    if hasattr(argument, 'counter_arguments') and argument.counter_arguments:
        texts_to_search.extend(argument.counter_arguments)

    # Search for quantitative claims
    for text in texts_to_search:
        if not text:
            continue

        for indicator, patterns in INDICATOR_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        claimed_value = float(match.group(1))

                        # Determine comparison type
                        context = text[max(0, match.start()-10):match.end()+10]
                        comparison = _determine_comparison(context)

                        claims.append(IndicatorClaim(
                            indicator=indicator,
                            claimed_value=claimed_value,
                            comparison=comparison,
                            source=source,
                            original_text=context.strip()
                        ))
                    except (ValueError, IndexError):
                        continue

    # Search for qualitative claims
    for text in texts_to_search:
        if not text:
            continue

        for indicator, qual_patterns in QUALITATIVE_PATTERNS.items():
            for pattern, claim_type, validator in qual_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    claims.append(IndicatorClaim(
                        indicator=indicator,
                        claimed_value=None,
                        comparison="qualitative",
                        source=source,
                        original_text=f"{indicator}:{claim_type}"
                    ))

    return claims


def _determine_comparison(context: str) -> str:
    """Determine the comparison type from context."""
    if ">=" in context or "≥" in context or "大于等于" in context:
        return ">="
    elif "<=" in context or "≤" in context or "小于等于" in context:
        return "<="
    elif ">" in context or "超过" in context or "高于" in context or "大于" in context:
        return ">"
    elif "<" in context or "低于" in context or "小于" in context:
        return "<"
    else:
        return "="


def _verify_claims(
    claims: List[IndicatorClaim],
    actual_values: Dict[str, float]
) -> GroundingResult:
    """
    Verify claims against actual values and auto-correct false claims.

    When a claim is found to be false (outside tolerance), the actual value
    is stored in corrected_values for downstream nodes to use.
    
    Multi-timeframe Support:
    - If claim mentions "ADX=36.96" but actual_values only has "ADX=21.2",
      we check if there's a matching "ADX_1h=36.96" or "ADX_4h=36.96" etc.
    - This prevents false positives when LLM references different timeframes.

    Returns GroundingResult with verification statistics and corrections.
    """
    result = GroundingResult(total_claims=len(claims))

    for claim in claims:
        # Try to find the actual value - with multi-timeframe fallback
        actual, matched_key = _find_actual_value_with_mtf_fallback(
            claim.indicator, 
            claim.claimed_value, 
            actual_values
        )
        
        if actual is None:
            # Cannot verify - indicator not found in any timeframe
            claim.is_verified = False
            result.claims.append(claim)
            continue

        claim.actual_value = actual
        claim.is_verified = True
        result.verified_claims += 1

        # Verify based on comparison type
        if claim.comparison == "qualitative":
            # Verify qualitative claims using validators
            claim.is_valid = _verify_qualitative_claim(claim, actual)
        else:
            claim.is_valid, claim.discrepancy_pct = _verify_quantitative_claim(
                claim, actual
            )

        if claim.is_valid:
            result.valid_claims += 1
        else:
            result.false_claims += 1
            
            # AUTO-CORRECTION: Store the correct value
            result.corrected_values[claim.indicator] = actual
            result.corrected_claims.append(claim)
            
            # Note: Only log if it's a significant discrepancy and not a timeframe mismatch
            if matched_key == claim.indicator:
                logger.info(
                    f"[GroundingNode] AUTO-CORRECTED: {claim.indicator} "
                    f"claimed={claim.claimed_value} -> actual={actual:.2f} "
                    f"(discrepancy={claim.discrepancy_pct:.1f}%)"
                )
            else:
                # Timeframe mismatch - this is less concerning
                logger.debug(
                    f"[GroundingNode] Timeframe mismatch: {claim.indicator}={claim.claimed_value} "
                    f"matched {matched_key}={actual:.2f} (discrepancy={claim.discrepancy_pct:.1f}%)"
                )
            
            result.warnings.append(
                f"Corrected {claim.source} claim: {claim.indicator} "
                f"claimed={claim.claimed_value} ({claim.comparison}), "
                f"corrected to actual={actual:.2f}"
            )

        result.claims.append(claim)

    # Calculate hallucination score
    if result.verified_claims > 0:
        result.hallucination_score = (
            result.false_claims / result.verified_claims
        ) * 100

    # Determine if should reject - raise threshold when total claims are low
    # With only 2 claims, 100% hallucination is misleading
    effective_threshold = HALLUCINATION_REJECT_THRESHOLD
    if result.verified_claims <= 3:
        effective_threshold = 70.0  # More lenient for small sample size
    
    result.should_reject = (
        result.hallucination_score >= effective_threshold and 
        result.false_claims >= 2  # Require at least 2 false claims to reject
    )

    # Calculate confidence penalty
    result.confidence_penalty = (
        result.false_claims * CONFIDENCE_PENALTY_PER_FALSE_CLAIM
    )

    # Add warning if hallucination is high but not rejecting
    if (result.hallucination_score >= HALLUCINATION_WARNING_THRESHOLD
            and not result.should_reject):
        result.warnings.insert(
            0,
            f"WARNING: Elevated hallucination score ({result.hallucination_score:.0f}%)"
        )
    
    # Log correction summary
    if result.corrected_claims:
        logger.info(
            f"[GroundingNode] Corrected {len(result.corrected_claims)} claims: "
            f"{list(result.corrected_values.keys())}"
        )

    return result

def _find_actual_value_with_mtf_fallback(
    indicator: str,
    claimed_value: Optional[float],
    actual_values: Dict[str, float]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Find actual value for an indicator, with multi-timeframe fallback.
    
    Strategy:
    1. First try exact match (e.g., "ADX" -> actual_values["ADX"])
    2. If not found, try MTF variants (e.g., "ADX_1h", "ADX_4h")
    3. If claimed_value is provided, prefer the MTF variant closest to claimed value
    
    Args:
        indicator: Base indicator name (e.g., "ADX")
        claimed_value: The value claimed by LLM (for MTF matching)
        actual_values: Dict of actual indicator values
        
    Returns:
        Tuple of (actual_value, matched_key) or (None, None) if not found
    """
    # Try exact match first
    if indicator in actual_values:
        return actual_values[indicator], indicator
    
    # Try multi-timeframe variants
    timeframes = ["1h", "4h", "1d"]
    mtf_candidates = []
    
    for tf in timeframes:
        mtf_key = f"{indicator}_{tf}"
        if mtf_key in actual_values:
            mtf_candidates.append((mtf_key, actual_values[mtf_key]))
    
    if not mtf_candidates:
        return None, None
    
    # If we have claimed_value, find the closest match
    if claimed_value is not None and len(mtf_candidates) > 0:
        # Find MTF value closest to claimed value
        closest_key, closest_val = min(
            mtf_candidates,
            key=lambda x: abs(x[1] - claimed_value) if x[1] else float('inf')
        )
        return closest_val, closest_key
    
    # Otherwise, return the first MTF value found
    return mtf_candidates[0][1], mtf_candidates[0][0]



def _verify_qualitative_claim(
    claim: IndicatorClaim,
    actual: float
) -> bool:
    """Verify a qualitative claim using predefined validators."""
    qual_patterns = QUALITATIVE_PATTERNS.get(claim.indicator, [])

    for pattern, claim_type, validator in qual_patterns:
        if claim_type in claim.original_text:
            return validator(actual)

    # If no specific validator found, assume valid (cannot disprove)
    return True


def _verify_quantitative_claim(
    claim: IndicatorClaim,
    actual: float
) -> Tuple[bool, float]:
    """
    Verify a quantitative claim against actual value.

    Returns (is_valid, discrepancy_percentage).
    """
    if claim.claimed_value is None:
        return True, 0.0

    claimed = claim.claimed_value

    # Calculate discrepancy
    if actual != 0:
        discrepancy_pct = abs((claimed - actual) / actual) * 100
    elif claimed != 0:
        discrepancy_pct = 100.0  # Claimed non-zero when actual is zero
    else:
        discrepancy_pct = 0.0  # Both zero

    # Verify based on comparison type
    if claim.comparison == "=":
        # Equality with tolerance
        is_valid = discrepancy_pct <= VALUE_TOLERANCE_PCT
    elif claim.comparison == ">":
        is_valid = actual > claimed or discrepancy_pct <= VALUE_TOLERANCE_PCT
    elif claim.comparison == "<":
        is_valid = actual < claimed or discrepancy_pct <= VALUE_TOLERANCE_PCT
    elif claim.comparison == ">=":
        is_valid = actual >= claimed or discrepancy_pct <= VALUE_TOLERANCE_PCT
    elif claim.comparison == "<=":
        is_valid = actual <= claimed or discrepancy_pct <= VALUE_TOLERANCE_PCT
    else:
        is_valid = True  # Unknown comparison, assume valid

    return is_valid, discrepancy_pct


def _build_corrected_context(result: GroundingResult, actual_values: Dict[str, float]) -> str:
    """
    Build corrected context text for Executor Agent.
    
    This text provides the Executor with verified/corrected indicator values
    to make decisions based on accurate data.
    
    Args:
        result: GroundingResult with verification details
        actual_values: Dict of actual indicator values
        
    Returns:
        Formatted corrected context string
    """
    lines = [
        "=== Grounding 验证结果 ===",
        f"幻觉检测分数: {result.hallucination_score:.0f}% "
        f"({'可接受' if result.hallucination_score < 30 else '需注意' if result.hallucination_score < 50 else '高风险'})"
    ]
    
    # List corrected claims
    if result.corrected_claims:
        lines.append("")
        lines.append("❌ 错误声明已纠正:")
        for claim in result.corrected_claims:
            source_cn = "Bull" if claim.source == "bull" else "Bear"
            lines.append(
                f"  - {source_cn}声称 \"{claim.indicator}={claim.claimed_value}\" "
                f"→ 实际值: {claim.indicator}={claim.actual_value:.2f}"
            )
    
    # List verified correct claims
    valid_claims = [c for c in result.claims if c.is_verified and c.is_valid]
    if valid_claims:
        lines.append("")
        lines.append("✅ 验证通过的声明:")
        for claim in valid_claims[:5]:  # Limit to 5 for brevity
            lines.append(f"  - {claim.indicator}: 正确")
    
    # List all actual indicator values
    if actual_values:
        lines.append("")
        lines.append("📊 纠正后的指标数据:")
        for indicator, value in sorted(actual_values.items()):
            interpretation = _get_indicator_interpretation(indicator, value)
            lines.append(f"  {indicator}: {value:.2f} {interpretation}")
    
    lines.append("")
    lines.append("⚠️ 请基于以上纠正后的数据做出决策")
    
    return "\n".join(lines)


def _build_grounding_summary(result: GroundingResult) -> str:
    """
    Build concise grounding summary for logging.
    
    Args:
        result: GroundingResult with verification details
        
    Returns:
        One-line summary string
    """
    return (
        f"验证: {result.valid_claims}/{result.verified_claims} 通过 | "
        f"幻觉分: {result.hallucination_score:.0f}% | "
        f"纠正: {len(result.corrected_claims)}项 | "
        f"置信度惩罚: -{result.confidence_penalty:.0f}%"
    )


def _get_indicator_interpretation(indicator: str, value: float) -> str:
    """
    Get human-readable interpretation of indicator value.
    
    Args:
        indicator: Indicator name
        value: Indicator value
        
    Returns:
        Interpretation string in parentheses
    """
    interpretations = {
        "RSI": lambda v: "(超卖)" if v < 30 else "(超买)" if v > 70 else "(中性区域)",
        "ADX": lambda v: "(弱趋势)" if v < 20 else "(中等趋势)" if v < 40 else "(强趋势)",
        "MACD": lambda v: "(看多信号)" if v > 0 else "(看空信号)" if v < 0 else "(中性)",
        "MFI": lambda v: "(超卖)" if v < 20 else "(超买)" if v > 80 else "(中性区域)",
        "STOCH_K": lambda v: "(超卖)" if v < 20 else "(超买)" if v > 80 else "(中性区域)",
        "STOCH_D": lambda v: "(超卖)" if v < 20 else "(超买)" if v > 80 else "(中性区域)",
    }
    
    if indicator in interpretations:
        return interpretations[indicator](value)
    return ""


def format_grounding_summary(result: GroundingResult) -> str:
    """
    Format grounding result as human-readable summary.

    Used for logging and debugging.
    """
    lines = [
        "=== Grounding Verification Summary ===",
        f"Total Claims: {result.total_claims}",
        f"Verified: {result.verified_claims}",
        f"Valid: {result.valid_claims}",
        f"False/Hallucinated: {result.false_claims}",
        f"Hallucination Score: {result.hallucination_score:.0f}%",
        f"Confidence Penalty: -{result.confidence_penalty:.0f}%",
    ]

    if result.should_reject:
        lines.append("*** RECOMMENDATION: REJECT DUE TO HIGH HALLUCINATION ***")

    if result.warnings:
        lines.append("\nWarnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    if result.claims:
        lines.append("\nClaims Detail:")
        for claim in result.claims:
            status = "VALID" if claim.is_valid else "FALSE"
            if not claim.is_verified:
                status = "UNVERIFIED"
            lines.append(
                f"  [{status}] {claim.source.upper()}: {claim.indicator} "
                f"claimed={claim.claimed_value} actual={claim.actual_value}"
            )

    return "\n".join(lines)

