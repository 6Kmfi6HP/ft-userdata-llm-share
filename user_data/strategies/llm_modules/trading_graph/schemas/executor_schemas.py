"""
Pydantic Schemas for Executor Agent Output Validation.

Based on research from arXiv 2511.21734 (Verification-First) and 
industry best practices for structured LLM output.

Key Design Principles:
1. LLM outputs qualitative judgments only (direction_strength, risk_level)
2. Code calculates precise numerical values (stop_loss, take_profit)
3. Pydantic validates all outputs before execution
4. Fallback to conservative defaults on validation failure
"""

from enum import Enum
from typing import Optional, List, Literal, Dict, Any
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_V2 = True
except ImportError:
    # Fallback for older pydantic
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_V2 = False

import logging

logger = logging.getLogger(__name__)


# ============= Enums =============

class DirectionStrength(str, Enum):
    """Signal direction strength - qualitative judgment from LLM."""
    STRONG = "strong"      # Multiple indicators + trend + pattern aligned
    MODERATE = "moderate"  # Partially aligned, some noise
    WEAK = "weak"          # Mixed signals, unclear direction


class RiskLevel(str, Enum):
    """Market risk level - qualitative judgment from LLM."""
    HIGH = "high"      # High volatility, near support/resistance, extreme funding
    MEDIUM = "medium"  # Normal market conditions
    LOW = "low"        # Low volatility, clear trend, stable volume


class ExecutorAction(str, Enum):
    """Valid executor actions."""
    ENTRY_LONG = "signal_entry_long"
    ENTRY_SHORT = "signal_entry_short"
    WAIT = "signal_wait"
    HOLD = "signal_hold"
    EXIT = "signal_exit"
    ADJUST = "adjust_position"


class AdjustmentType(str, Enum):
    """Position adjustment types."""
    SCALE_IN = "scale_in"
    PARTIAL_EXIT = "partial_exit"


# ============= Verification-First Output Schema =============

class VerificationResult(BaseModel):
    """Result of VF (Verification-First) check for each dimension."""
    dimension: str = Field(..., description="Verification dimension name")
    passed: bool = Field(..., description="Whether verification passed")
    reason: str = Field("", description="Reason for pass/fail")


class ExecutorVerificationOutput(BaseModel):
    """
    VF (Verification-First) verification output.
    
    The LLM first verifies the candidate answer from Judge,
    then provides its own decision.
    """
    # Verification results
    analysis_consensus_verified: bool = Field(
        True, 
        description="Whether analysis consensus is well-supported"
    )
    debate_conclusion_verified: bool = Field(
        True,
        description="Whether debate conclusion is logically consistent"
    )
    data_citation_verified: bool = Field(
        True,
        description="Whether corrected grounding data was properly used"
    )
    risk_oversight_verified: bool = Field(
        True,
        description="Whether all major risks were identified"
    )
    verification_notes: List[str] = Field(
        default_factory=list,
        description="Notes from verification process"
    )
    
    def all_passed(self) -> bool:
        """Check if all verifications passed."""
        return all([
            self.analysis_consensus_verified,
            self.debate_conclusion_verified,
            self.data_citation_verified,
            self.risk_oversight_verified
        ])
    
    def failed_dimensions(self) -> List[str]:
        """Get list of failed verification dimensions."""
        failed = []
        if not self.analysis_consensus_verified:
            failed.append("analysis_consensus")
        if not self.debate_conclusion_verified:
            failed.append("debate_conclusion")
        if not self.data_citation_verified:
            failed.append("data_citation")
        if not self.risk_oversight_verified:
            failed.append("risk_oversight")
        return failed


# ============= Qualitative Output Schema =============

class ExecutorQualitativeOutput(BaseModel):
    """
    LLM Qualitative Output Schema.
    
    Following arXiv 2512.01123: LLM provides qualitative judgments,
    code calculates precise numerical values.
    
    LLM should NOT output:
    - stop_loss_price (calculated by code)
    - take_profit_price (calculated by code)
    - exact price targets
    """
    # Core decision
    action: Literal[
        "signal_entry_long",
        "signal_entry_short",
        "signal_wait",
        "signal_hold",
        "signal_exit",
        "adjust_position"
    ] = Field(..., description="Trading action to take")
    
    confidence: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Confidence in decision (0-100)"
    )
    
    # Qualitative judgments (LLM's strength)
    direction_strength: Literal["strong", "moderate", "weak"] = Field(
        "moderate",
        description="Signal direction strength"
    )
    
    risk_level: Literal["high", "medium", "low"] = Field(
        "medium",
        description="Current market risk level"
    )
    
    # Reasoning
    reasoning: str = Field(
        "",
        description="Complete reasoning for the decision"
    )
    
    key_factors: List[str] = Field(
        default_factory=list,
        description="Key factors influencing the decision"
    )
    
    # Position adjustment (only for adjust_position)
    adjustment_pct: Optional[float] = Field(
        None,
        ge=-70,
        le=50,
        description="Adjustment percentage for position changes"
    )
    
    adjustment_type: Optional[Literal["scale_in", "partial_exit"]] = Field(
        None,
        description="Type of position adjustment"
    )

    class Config:
        extra = "ignore"  # Ignore extra fields from LLM

    if PYDANTIC_V2:
        @model_validator(mode='after')
        def validate_adjustment_requirements(self):
            """Validate adjustment fields for adjust_position action."""
            if self.action == "adjust_position":
                if self.adjustment_pct is None:
                    raise ValueError("adjust_position requires adjustment_pct")
                if self.adjustment_type is None:
                    raise ValueError("adjust_position requires adjustment_type")
            return self
    else:
        @root_validator
        def validate_adjustment_requirements(cls, values):
            """Validate adjustment fields for adjust_position action."""
            if values.get("action") == "adjust_position":
                if values.get("adjustment_pct") is None:
                    raise ValueError("adjust_position requires adjustment_pct")
                if values.get("adjustment_type") is None:
                    raise ValueError("adjust_position requires adjustment_type")
            return values


# ============= Full Executor Output Schema =============

class ExecutorOutputSchema(BaseModel):
    """
    Complete Executor Agent Output Schema with Pydantic validation.
    
    Combines:
    1. VF verification results
    2. Qualitative decision from LLM
    3. Calculated numerical values from code
    """
    # Verification (from VF phase)
    verification: Optional[ExecutorVerificationOutput] = None
    
    # Core decision
    action: Literal[
        "signal_entry_long",
        "signal_entry_short",
        "signal_wait",
        "signal_hold",
        "signal_exit",
        "adjust_position"
    ]
    
    confidence: float = Field(..., ge=0, le=100)
    leverage: Optional[int] = Field(None, ge=1, le=100)
    direction: Optional[Literal["long", "short", "neutral"]] = None
    
    # Qualitative judgments
    direction_strength: Literal["strong", "moderate", "weak"] = "moderate"
    risk_level: Literal["high", "medium", "low"] = "medium"
    
    # Risk management (calculated by code, NOT LLM)
    stop_loss_price: Optional[float] = Field(None, gt=0)
    take_profit_price: Optional[float] = Field(None, gt=0)
    stop_loss_pct: Optional[float] = Field(None, ge=0, le=100)
    take_profit_pct: Optional[float] = Field(None, ge=0, le=100)
    risk_reward_ratio: Optional[float] = Field(None, ge=0.5, le=10)
    
    # Position adjustment
    adjustment_pct: Optional[float] = Field(None, ge=-70, le=50)
    adjustment_type: Optional[Literal["scale_in", "partial_exit"]] = None
    
    # Reasoning
    reasoning: str = ""
    key_factors: List[str] = Field(default_factory=list)
    risk_assessment: str = ""
    
    # Consistency tracking
    reasoning_consistent: bool = True
    consistency_violations: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"

    if PYDANTIC_V2:
        @model_validator(mode='after')
        def validate_entry_requirements(self):
            """Validate entry actions have required risk management."""
            if self.action in ["signal_entry_long", "signal_entry_short"]:
                if not self.stop_loss_price:
                    raise ValueError(f"{self.action} requires stop_loss_price")
                if not self.take_profit_price:
                    raise ValueError(f"{self.action} requires take_profit_price")
                if self.confidence < 60:
                    raise ValueError(f"{self.action} requires confidence >= 60")
            return self
    else:
        @root_validator
        def validate_entry_requirements(cls, values):
            """Validate entry actions have required risk management."""
            action = values.get("action")
            if action in ["signal_entry_long", "signal_entry_short"]:
                if not values.get("stop_loss_price"):
                    raise ValueError(f"{action} requires stop_loss_price")
                if not values.get("take_profit_price"):
                    raise ValueError(f"{action} requires take_profit_price")
                if values.get("confidence", 0) < 60:
                    raise ValueError(f"{action} requires confidence >= 60")
            return values


# ============= Risk Management Calculator =============

@dataclass
class RiskManagementConfig:
    """Configuration for risk management calculations."""
    high_risk_sl_pct: float = 0.015      # 1.5% stop loss for high risk
    medium_risk_sl_pct: float = 0.025    # 2.5% stop loss for medium risk
    low_risk_sl_pct: float = 0.04        # 4% stop loss for low risk
    
    strong_direction_tp_mult: float = 3.0   # 3:1 RR for strong signals
    moderate_direction_tp_mult: float = 2.0  # 2:1 RR for moderate signals
    weak_direction_tp_mult: float = 1.5      # 1.5:1 RR for weak signals


def calculate_risk_management(
    qualitative: ExecutorQualitativeOutput,
    current_price: float,
    key_support: Optional[float] = None,
    key_resistance: Optional[float] = None,
    config: Optional[RiskManagementConfig] = None
) -> Dict[str, Any]:
    """
    Calculate precise risk management parameters from qualitative LLM judgments.
    
    Following arXiv 2512.01123: Let structured code handle numerical calculations,
    not LLM (which is prone to numerical hallucination).
    
    Args:
        qualitative: LLM's qualitative output
        current_price: Current market price
        key_support: Key support level
        key_resistance: Key resistance level
        config: Risk management configuration
        
    Returns:
        Dict with calculated stop_loss, take_profit, etc.
    """
    if config is None:
        config = RiskManagementConfig()
    
    # Not an entry action - no risk management needed
    if qualitative.action not in ["signal_entry_long", "signal_entry_short"]:
        return {}
    
    # Map risk level to stop loss percentage
    sl_pct_map = {
        "high": config.high_risk_sl_pct,
        "medium": config.medium_risk_sl_pct,
        "low": config.low_risk_sl_pct,
    }
    sl_pct = sl_pct_map.get(qualitative.risk_level, config.medium_risk_sl_pct)
    
    # Map direction strength to take profit multiplier
    tp_mult_map = {
        "strong": config.strong_direction_tp_mult,
        "moderate": config.moderate_direction_tp_mult,
        "weak": config.weak_direction_tp_mult,
    }
    tp_multiplier = tp_mult_map.get(qualitative.direction_strength, config.moderate_direction_tp_mult)
    
    if qualitative.action == "signal_entry_long":
        # Long position: SL below, TP above
        stop_loss = current_price * (1 - sl_pct)
        
        # Use support level if available and reasonable
        if key_support and key_support < current_price:
            # Place SL just below support (0.5% buffer)
            support_sl = key_support * 0.995
            stop_loss = max(stop_loss, support_sl)
        
        # Calculate take profit based on risk-reward ratio
        risk = current_price - stop_loss
        take_profit = current_price + (risk * tp_multiplier)
        
        # Cap at resistance if available
        if key_resistance and key_resistance > current_price:
            # Place TP just below resistance (0.5% buffer)
            take_profit = min(take_profit, key_resistance * 1.005)
            
    elif qualitative.action == "signal_entry_short":
        # Short position: SL above, TP below
        stop_loss = current_price * (1 + sl_pct)
        
        # Use resistance level if available and reasonable
        if key_resistance and key_resistance > current_price:
            # Place SL just above resistance (0.5% buffer)
            resistance_sl = key_resistance * 1.005
            stop_loss = min(stop_loss, resistance_sl)
        
        # Calculate take profit based on risk-reward ratio
        risk = stop_loss - current_price
        take_profit = current_price - (risk * tp_multiplier)
        
        # Floor at support if available
        if key_support and key_support < current_price:
            # Place TP just above support (0.5% buffer)
            take_profit = max(take_profit, key_support * 0.995)
    else:
        return {}
    
    # Calculate actual risk-reward ratio
    actual_risk = abs(current_price - stop_loss)
    actual_reward = abs(take_profit - current_price)
    actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0
    
    return {
        "stop_loss_price": round(stop_loss, 2),
        "take_profit_price": round(take_profit, 2),
        "stop_loss_pct": round(sl_pct * 100, 2),
        "take_profit_pct": round(abs(take_profit - current_price) / current_price * 100, 2),
        "risk_reward_ratio": round(actual_rr, 2),
    }


# ============= Reasoning Consistency Validator =============

def verify_reasoning_consistency(
    decision: ExecutorOutputSchema,
    state_context: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """
    Verify that Executor decision is logically consistent with prior agent conclusions.
    
    Checks for:
    1. High confidence entry with high hallucination score
    2. Entry direction contradicting debate winner
    3. Entry when Judge rejected
    4. Entry direction against strong consensus
    
    Args:
        decision: The executor's decision
        state_context: Dict containing prior agent results
        
    Returns:
        Tuple of (is_consistent, list_of_violations)
    """
    violations = []
    
    # Rule 1: High hallucination should not have high confidence entry
    hallucination_score = state_context.get("hallucination_score") or \
                          state_context.get("position_hallucination_score") or 0
    
    if hallucination_score > 50:
        if decision.action in ["signal_entry_long", "signal_entry_short"]:
            if decision.confidence > 70:
                violations.append(
                    f"High confidence ({decision.confidence}%) entry with "
                    f"high hallucination score ({hallucination_score:.0f}%)"
                )
    
    # Rule 2: Check debate winner alignment
    judge_verdict = state_context.get("judge_verdict") or \
                    state_context.get("position_judge_verdict")
    
    if judge_verdict:
        winning_arg = getattr(judge_verdict, "winning_argument", None)
        if winning_arg == "bear" and decision.action == "signal_entry_long":
            violations.append("Entry LONG when Bear wins debate")
        elif winning_arg == "bull" and decision.action == "signal_entry_short":
            violations.append("Entry SHORT when Bull wins debate")
    
    # Rule 3: Entry when Judge rejected
    if judge_verdict:
        verdict = getattr(judge_verdict, "verdict", None)
        if verdict and hasattr(verdict, "value") and verdict.value == "reject":
            if decision.action in ["signal_entry_long", "signal_entry_short"]:
                violations.append("Entry signal when Judge verdict was REJECT")
    
    # Rule 4: Direction vs consensus conflict
    consensus_direction = state_context.get("consensus_direction")
    if consensus_direction:
        dir_val = consensus_direction.value if hasattr(consensus_direction, "value") else str(consensus_direction)
        
        if dir_val == "long" and decision.action == "signal_entry_short":
            if decision.confidence > 60:
                violations.append(
                    f"Entry SHORT against LONG consensus with confidence {decision.confidence}%"
                )
        elif dir_val == "short" and decision.action == "signal_entry_long":
            if decision.confidence > 60:
                violations.append(
                    f"Entry LONG against SHORT consensus with confidence {decision.confidence}%"
                )
    
    is_consistent = len(violations) == 0
    
    if not is_consistent:
        logger.warning(f"[ReasoningConsistency] Violations detected: {violations}")
    
    return is_consistent, violations


# ============= Helper Functions =============

def create_conservative_output(
    has_position: bool,
    reason: str
) -> ExecutorOutputSchema:
    """Create a conservative fallback output for error/validation failure cases."""
    return ExecutorOutputSchema(
        action="signal_hold" if has_position else "signal_wait",
        confidence=0.0,
        direction_strength="weak",
        risk_level="high",
        reasoning=reason,
        reasoning_consistent=True,
        consistency_violations=[]
    )


def parse_qualitative_from_text(
    response_text: str,
    has_position: bool = False
) -> ExecutorQualitativeOutput:
    """
    Parse LLM response text into ExecutorQualitativeOutput.
    
    Extracts qualitative fields only; numerical risk management
    will be calculated separately by code.
    """
    import re
    
    # Default values
    action = "signal_wait" if not has_position else "signal_hold"
    confidence = 0.0
    direction_strength = "moderate"
    risk_level = "medium"
    reasoning = ""
    key_factors = []
    adjustment_pct = None
    adjustment_type = None
    
    try:
        # Parse action
        action_match = re.search(r'action:\s*(\w+)', response_text, re.IGNORECASE)
        if action_match:
            raw_action = action_match.group(1).upper()
            action_map = {
                "ENTRY_LONG": "signal_entry_long",
                "ENTRY_SHORT": "signal_entry_short",
                "LONG": "signal_entry_long",
                "SHORT": "signal_entry_short",
                "EXIT": "signal_exit",
                "HOLD": "signal_hold",
                "WAIT": "signal_wait",
                "SCALE_IN": "adjust_position",
                "PARTIAL_EXIT": "adjust_position",
                "ADJUST": "adjust_position",
            }
            action = action_map.get(raw_action, action)
        
        # Parse confidence
        conf_match = re.search(r'confidence:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Parse direction_strength
        strength_match = re.search(
            r'direction_strength:\s*(strong|moderate|weak)', 
            response_text, 
            re.IGNORECASE
        )
        if strength_match:
            direction_strength = strength_match.group(1).lower()
        
        # Parse risk_level
        risk_match = re.search(
            r'risk_level:\s*(high|medium|low)', 
            response_text, 
            re.IGNORECASE
        )
        if risk_match:
            risk_level = risk_match.group(1).lower()
        
        # Parse reasoning
        reasoning_match = re.search(
            r'\[决策理由\]\s*\n(.+?)(?=\n\[|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Parse key factors
        factors_match = re.search(
            r'\[关键因素\]\s*\n(.+?)(?=\n\[|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if factors_match:
            factors_text = factors_match.group(1)
            key_factors = [
                f.strip().lstrip('- ')
                for f in factors_text.split('\n')
                if f.strip() and f.strip() != '-'
            ]
        
        # Parse adjustment fields
        adj_match = re.search(r'adjustment_pct:\s*([+-]?\d+)', response_text, re.IGNORECASE)
        if adj_match:
            adjustment_pct = float(adj_match.group(1))
        
        adj_type_match = re.search(r'adjustment_type:\s*(scale_in|partial_exit)', response_text, re.IGNORECASE)
        if adj_type_match:
            adjustment_type = adj_type_match.group(1).lower()
            
    except Exception as e:
        logger.warning(f"[ParseQualitative] Parse error: {e}")
    
    return ExecutorQualitativeOutput(
        action=action,
        confidence=confidence,
        direction_strength=direction_strength,
        risk_level=risk_level,
        reasoning=reasoning,
        key_factors=key_factors,
        adjustment_pct=adjustment_pct,
        adjustment_type=adjustment_type
    )
