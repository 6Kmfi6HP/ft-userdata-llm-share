"""
LangGraph State Schema Definitions.

Defines TypedDict state schemas for the trading decision graph:
- AnalysisState: State for analysis subgraph (4 agents parallel)
- DebateState: State for debate subgraph (Bull→Bear→Judge)
- TradingDecisionState: Main graph state combining all stages
"""

from typing import TypedDict, List, Optional, Annotated, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
import operator


# ============= Enums =============

class Direction(str, Enum):
    """Trading direction enum."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Verdict(str, Enum):
    """Judge verdict enum."""
    APPROVE = "approve"   # Bull wins - trade is sound
    REJECT = "reject"     # Bear wins - too risky
    ABSTAIN = "abstain"   # Arguments balanced - wait


class PositionVerdict(str, Enum):
    """Position management verdict enum."""
    HOLD = "hold"               # Continue unchanged
    EXIT = "exit"               # Close entire position
    SCALE_IN = "scale_in"       # Add to position (+20%~+50%)
    PARTIAL_EXIT = "partial_exit"  # Reduce position (-30%~-70%)


class SignalStrength(str, Enum):
    """Signal strength classification."""
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


# ============= Data Classes =============

@dataclass
class Signal:
    """Individual trading signal from an agent."""
    signal_type: str          # e.g., "RSI超卖", "MACD金叉"
    description: str          # Signal description
    strength: SignalStrength  # Signal strength
    direction: Optional[Direction] = None  # Implied direction


@dataclass
class AgentReport:
    """
    Analysis agent report structure.

    Preserves the format from existing agents/base_agent.py
    """
    agent_name: str               # e.g., "IndicatorAgent", "TrendAgent"
    analysis: str                 # Full analysis text
    signals: List[Signal] = field(default_factory=list)  # Detected signals
    confidence: float = 0.0       # 0-100
    direction: Optional[Direction] = None
    key_levels: Optional[dict] = None  # {"support": float, "resistance": float}
    error: Optional[str] = None   # Error message if analysis failed

    def is_valid(self) -> bool:
        """Check if report is valid (no error and has confidence)."""
        return self.error is None and self.confidence > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "analysis": self.analysis,
            "signals": [
                {
                    "signal_type": s.signal_type,
                    "description": s.description,
                    "strength": s.strength.value,
                    "direction": s.direction.value if s.direction else None
                }
                for s in self.signals
            ],
            "confidence": self.confidence,
            "direction": self.direction.value if self.direction else None,
            "key_levels": self.key_levels,
            "error": self.error
        }


@dataclass
class DebateArgument:
    """
    Debate argument structure for Bull/Bear agents.
    """
    agent_role: str                     # "bull" or "bear"
    position: Direction                 # Advocated direction
    confidence: float                   # 0-100
    key_points: List[str] = field(default_factory=list)        # Main arguments
    risk_factors: List[str] = field(default_factory=list)      # Identified risks
    supporting_signals: List[str] = field(default_factory=list)  # Evidence
    counter_arguments: List[str] = field(default_factory=list)  # Rebuttals (Bear)
    recommended_action: str = ""        # Recommended trading action
    reasoning: str = ""                 # Full reasoning text

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_role": self.agent_role,
            "position": self.position.value,
            "confidence": self.confidence,
            "key_points": self.key_points,
            "risk_factors": self.risk_factors,
            "supporting_signals": self.supporting_signals,
            "counter_arguments": self.counter_arguments,
            "recommended_action": self.recommended_action,
            "reasoning": self.reasoning
        }


@dataclass
class JudgeVerdict:
    """
    Judge verdict structure.
    """
    verdict: Verdict                    # APPROVE, REJECT, or ABSTAIN
    confidence: float                   # 0-100 (calibrated)
    winning_argument: Optional[str] = None  # "bull" or "bear"
    key_reasoning: str = ""             # Main reasoning for verdict
    recommended_action: str = ""        # Final recommended action
    leverage: Optional[int] = None      # Recommended leverage if APPROVE
    risk_assessment: str = ""           # Overall risk assessment

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "winning_argument": self.winning_argument,
            "key_reasoning": self.key_reasoning,
            "recommended_action": self.recommended_action,
            "leverage": self.leverage,
            "risk_assessment": self.risk_assessment
        }


# ============= Position Management Data Classes =============

class PositionMetrics(TypedDict, total=False):
    """
    Position tracking metrics from PositionTracker.

    Injected into position management debate for informed decisions.
    """
    trade_id: int                       # Freqtrade trade ID
    max_profit_pct: float               # MFE (Maximum Favorable Excursion)
    max_loss_pct: float                 # MAE (Maximum Adverse Excursion)
    current_profit_pct: float           # Current unrealized P&L
    drawdown_from_peak_pct: float       # How much pulled back from MFE
    hold_count: int                     # Number of consecutive hold decisions
    hold_pattern: Dict[str, Any]        # Pattern analysis of hold behavior
    time_in_position_hours: float       # Duration of position
    entry_price: float                  # Original entry price
    current_price: float                # Current market price
    position_side: str                  # "long" or "short"
    stake_amount: float                 # Current position size


@dataclass
class PositionJudgeVerdict:
    """
    Position Judge verdict structure.

    Extends JudgeVerdict with position-specific fields for scaling decisions.
    """
    verdict: PositionVerdict            # HOLD, EXIT, SCALE_IN, or PARTIAL_EXIT
    confidence: float                   # 0-100 (calibrated)
    adjustment_pct: Optional[float] = None  # Percentage to adjust (+20 to +50 for scale_in, -30 to -70 for partial_exit)
    key_reasoning: str = ""             # Main reasoning for verdict
    profit_protection_triggered: bool = False  # Whether forced rule was applied
    forced_rule_name: Optional[str] = None  # Name of forced rule if triggered
    bull_score: float = 0.0             # Score assigned to bull argument
    bear_score: float = 0.0             # Score assigned to bear argument

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "adjustment_pct": self.adjustment_pct,
            "key_reasoning": self.key_reasoning,
            "profit_protection_triggered": self.profit_protection_triggered,
            "forced_rule_name": self.forced_rule_name,
            "bull_score": self.bull_score,
            "bear_score": self.bear_score
        }


@dataclass
class ExecutorDecision:
    """
    Executor Agent decision structure.
    
    Contains the final trading decision made by the Executor Agent.
    """
    action: str                         # Trading action (signal_entry_long, signal_entry_short, etc.)
    confidence: float                   # 0-100
    direction: Optional[Direction] = None
    leverage: Optional[int] = None
    
    # Stop loss / Take profit
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # Position adjustment
    adjustment_pct: Optional[float] = None
    adjustment_type: Optional[str] = None  # "scale_in" / "partial_exit"
    
    # Reasoning
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    model_used: Optional[str] = None
    token_usage: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "direction": self.direction.value if self.direction else None,
            "leverage": self.leverage,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "adjustment_pct": self.adjustment_pct,
            "adjustment_type": self.adjustment_type,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_assessment": self.risk_assessment,
            "confidence_breakdown": self.confidence_breakdown,
            "model_used": self.model_used,
            "token_usage": self.token_usage
        }


@dataclass
class GroundingCorrection:
    """
    Grounding correction record.
    
    Records details about a corrected indicator claim.
    """
    indicator: str                      # Indicator name (RSI, MACD, etc.)
    claimed_value: Optional[float]      # Value claimed by agent
    actual_value: float                 # Actual verified value
    source: str                         # "bull" / "bear"
    claim_text: str                     # Original claim text
    claim_type: str                     # "quantitative" / "qualitative"
    discrepancy_pct: float              # Percentage difference
    is_false: bool                      # Whether claim is false
    correction_applied: bool            # Whether correction was applied
    
    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "claimed_value": self.claimed_value,
            "actual_value": self.actual_value,
            "source": self.source,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "discrepancy_pct": self.discrepancy_pct,
            "is_false": self.is_false,
            "correction_applied": self.correction_applied
        }


# ============= Analysis Subgraph State =============

class AnalysisState(TypedDict, total=False):
    """
    State for Analysis Subgraph.

    Used by: indicator_node, trend_node, sentiment_node, pattern_node, aggregator_node
    Pattern: Parallel fan-out → fan-in aggregation
    """
    # === Input Fields (set by caller) ===
    pair: str                           # Trading pair (e.g., "BTC/USDT:USDT")
    current_price: float                # Current market price
    market_context: str                 # Full market context from ContextBuilder
    ohlcv_data: Optional[Any]           # OHLCV DataFrame for visual analysis (primary timeframe)
    ohlcv_data_htf: Optional[Any]       # OHLCV DataFrame for higher timeframe (e.g., 1h when primary is 15m)
    timeframe: str                      # Primary timeframe (e.g., "15m")
    timeframe_htf: Optional[str]        # Higher timeframe (e.g., "1h")
    llm_config: Optional[dict]          # LLM configuration (api_base, model, etc.)

    # === Agent Reports (collected in parallel) ===
    # Using Annotated with operator.add for fan-in aggregation
    agent_reports: Annotated[List[AgentReport], operator.add]

    # Individual agent reports (for direct access)
    indicator_report: Optional[AgentReport]
    trend_report: Optional[AgentReport]
    sentiment_report: Optional[AgentReport]
    pattern_report: Optional[AgentReport]

    # === Aggregated Results (set by aggregator_node) ===
    consensus_direction: Optional[Direction]
    consensus_confidence: float
    key_support: Optional[float]
    key_resistance: Optional[float]
    weighted_scores: dict               # {Direction.LONG: score, Direction.SHORT: score, ...}

    # === Error Tracking ===
    errors: Annotated[List[str], operator.add]


# ============= Debate Subgraph State =============

class DebateState(TypedDict, total=False):
    """
    State for Debate Subgraph.

    Used by: bull_node, bear_node, judge_node
    Pattern: Sequential execution (Bull → Bear → Judge)
    """
    # === Input from Analysis (formatted text) ===
    indicator_report: str               # Formatted indicator analysis
    trend_report: str                   # Formatted trend analysis
    sentiment_report: str               # Formatted sentiment analysis
    pattern_report: str                 # Formatted pattern analysis

    # Analysis summary
    consensus_direction: Direction
    consensus_confidence: float
    key_support: Optional[float]
    key_resistance: Optional[float]

    # Market context
    pair: str
    current_price: float
    market_context: str

    # === Debate Arguments ===
    bull_argument: Optional[DebateArgument]
    bear_argument: Optional[DebateArgument]
    debate_round: int                   # Current debate round (for multi-round)

    # === Judge Verdict ===
    judge_verdict: Optional[JudgeVerdict]

    # === Error Tracking ===
    errors: Annotated[List[str], operator.add]


# ============= Main Graph State =============

class TradingDecisionState(TypedDict, total=False):
    """
    Main Graph State for complete trading decision flow.

    Combines Analysis → Debate → Grounding → Executor stages.
    
    设计原则:
    - 完整记录每个阶段的输入/输出
    - 支持后续的学习系统和回测分析
    - 便于日志记录和可视化
    """
    # === Input Fields ===
    pair: str
    current_price: float
    market_context: str                 # Full market context
    ohlcv_data: Optional[Any]           # For visual analysis (primary timeframe)
    ohlcv_data_htf: Optional[Any]       # For visual analysis (higher timeframe)
    timeframe: str                      # Primary timeframe (e.g., "15m")
    timeframe_htf: Optional[str]        # Higher timeframe (e.g., "1h")
    has_position: bool                  # Whether currently in position
    position_side: Optional[str]        # "long" or "short"
    position_profit_pct: Optional[float]  # Current P&L percentage
    position_entry_price: Optional[float]  # Entry price
    position_size: Optional[float]      # Position size
    position_leverage: Optional[int]    # Position leverage
    llm_config: Optional[dict]          # LLM configuration (api_base, model, etc.)
    risk_config: Optional[dict]         # Risk configuration
    
    # === NEW: Structured Indicator Data (for Grounding) ===
    # Following arXiv 2512.01123: Avoid text parsing, use structured data directly
    # Populated from ContextBuilder indicators, used by GroundingNode
    verified_indicator_data: Optional[Dict[str, float]]  # {"RSI": 45.2, "ADX": 32.1, ...}

    # === Analysis Stage Results ===
    indicator_report: Optional[AgentReport]
    trend_report: Optional[AgentReport]
    sentiment_report: Optional[AgentReport]
    pattern_report: Optional[AgentReport]
    consensus_direction: Optional[Direction]
    consensus_confidence: float
    key_support: Optional[float]
    key_resistance: Optional[float]
    weighted_scores: Optional[dict]
    
    # Analysis metadata
    analysis_timestamp: Optional[str]
    analysis_duration_ms: Optional[float]
    analysis_token_usage: Optional[dict]

    # === Debate Stage Results ===
    bull_argument: Optional[DebateArgument]
    bear_argument: Optional[DebateArgument]
    judge_verdict: Optional[JudgeVerdict]
    
    # Position Path debate
    position_bull_argument: Optional[DebateArgument]
    position_bear_argument: Optional[DebateArgument]
    position_judge_verdict: Optional[PositionJudgeVerdict]
    
    # Debate metadata
    debate_path: Optional[str]          # "entry" or "position"
    debate_timestamp: Optional[str]
    debate_duration_ms: Optional[float]
    debate_token_usage: Optional[dict]

    # === Grounding Verification Results (Layer 4 - Enhanced) ===
    grounding_result: Optional[Any]     # GroundingResult dataclass
    grounding_verified: bool
    grounding_corrected_values: Optional[Dict[str, float]]  # Corrected indicator values
    
    # NEW: Enhanced grounding outputs
    hallucination_score: Optional[float]    # 0-100 hallucination score
    total_claims: Optional[int]
    verified_claims: Optional[int]
    false_claims: Optional[int]
    corrected_context: Optional[str]        # Text context with corrections for Executor
    grounding_summary: Optional[str]        # Concise summary for logging
    false_claim_details: Optional[List[Dict]]  # Detailed false claims

    # Position path grounding
    position_metrics: Optional[PositionMetrics]
    position_grounding_result: Optional[Any]
    position_hallucination_score: Optional[float]
    position_grounding_corrected_values: Optional[Dict[str, float]]
    position_corrected_context: Optional[str]
    position_grounding_summary: Optional[str]
    position_false_claim_details: Optional[List[Dict]]
    
    # Grounding metadata
    grounding_timestamp: Optional[str]
    grounding_duration_ms: Optional[float]

    # === Reflection Verification Results (NEW - LLM-based CoVe) ===
    # Replaces code-based Grounding with LLM reflection for better accuracy
    reflection_result: Optional[Any]        # ReflectionResult dataclass
    reflection_summary: Optional[str]       # Concise summary for logging
    reflection_reasoning: Optional[str]     # LLM's verification reasoning
    reflection_should_proceed: Optional[bool]  # Whether to continue to executor
    reflection_confidence_adjustment: Optional[float]  # Confidence penalty/boost
    reflection_suggested_direction: Optional[Direction]  # Corrected direction if any
    reflection_suggested_action: Optional[str]  # Corrected action if any
    
    # Position path reflection
    position_reflection_result: Optional[Any]
    position_reflection_summary: Optional[str]
    position_reflection_should_proceed: Optional[bool]
    position_reflection_confidence_adjustment: Optional[float]
    
    # Reflection metadata
    reflection_timestamp: Optional[str]
    reflection_duration_ms: Optional[float]

    # === Executor Agent Output (NEW - LLM-based) ===
    executor_decision: Optional[Any]    # ExecutorDecision dataclass
    
    # Final decision fields (from executor)
    final_action: Optional[str]         # "signal_entry_long", "signal_entry_short", 
                                        # "signal_exit", "signal_hold", "signal_wait",
                                        # "adjust_position"
    final_confidence: float             # 0-100
    final_direction: Optional[Direction]
    final_leverage: Optional[int]
    
    # Risk management
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]
    risk_reward_ratio: Optional[float]
    
    # Position adjustment
    adjustment_pct: Optional[float]     # +20~+50 (scale_in) or -30~-70 (partial_exit)
    adjustment_type: Optional[str]      # "scale_in" / "partial_exit"
    
    # Executor reasoning
    executor_reasoning: Optional[str]
    executor_key_factors: Optional[List[str]]
    executor_risk_assessment: Optional[str]
    executor_confidence_breakdown: Optional[Dict[str, float]]
    
    # Executor metadata
    executor_timestamp: Optional[str]
    executor_duration_ms: Optional[float]
    executor_token_usage: Optional[dict]
    executor_model: Optional[str]
    
    # === NEW: VF Verification Fields (Verification-First from arXiv 2511.21734) ===
    vf_verification_passed: Optional[bool]          # Whether all VF verifications passed
    vf_analysis_consensus_verified: Optional[bool]  # Dimension 1: Analysis consensus
    vf_debate_conclusion_verified: Optional[bool]   # Dimension 2: Debate conclusion
    vf_data_citation_verified: Optional[bool]       # Dimension 3: Grounding data used
    vf_risk_oversight_verified: Optional[bool]      # Dimension 4: Risk identification
    vf_verification_notes: Optional[List[str]]      # Notes from VF process
    
    # === NEW: Reasoning Consistency Fields (Industry best practice) ===
    reasoning_consistent: Optional[bool]            # Whether decision is logically consistent
    consistency_violations: Optional[List[str]]     # List of detected violations
    
    # === NEW: Qualitative Judgments (LLM outputs, code calculates numbers) ===
    direction_strength: Optional[str]               # "strong" / "moderate" / "weak"
    risk_level: Optional[str]                       # "high" / "medium" / "low"

    # === Legacy fields (for backward compatibility) ===
    final_reason: str
    final_key_support: Optional[float]
    final_key_resistance: Optional[float]
    is_valid: bool

    # === Execution Result ===
    execution_result: Optional[dict]    # Final result for strategy
    tool_result: Optional[dict]         # Result from trading tool execution

    # === Metadata & Tracing ===
    execution_path: Optional[str]       # "entry" or "position"
    nodes_executed: Optional[List[str]] # List of executed nodes
    
    thread_id: str                      # Unique execution ID
    created_at: str                     # Creation timestamp
    completed_at: Optional[str]         # Completion timestamp
    total_duration_ms: Optional[float]  # Total execution time
    
    # Token usage summary
    total_token_usage: Optional[dict]   # {prompt_tokens, completion_tokens, total}
    
    # Error tracking
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    
    # === Reserved fields (for learning/backtest) ===
    decision_id: Optional[str]          # Unique decision ID for reward learning
    historical_similar_decisions: Optional[List[Dict]]
    expected_outcome: Optional[str]
    backtest_mode: Optional[bool]
    backtest_timestamp: Optional[str]
    actual_outcome: Optional[Dict]      # {pnl_pct, duration_hours, exit_reason}


# ============= Agent Weights Configuration =============

# Preserved from existing orchestrator.py
AGENT_WEIGHTS = {
    "IndicatorAgent": 1.0,   # Baseline weight
    "TrendAgent": 1.2,       # Highest (trend is king)
    "SentimentAgent": 0.8,   # Lowest (auxiliary)
    "PatternAgent": 1.1,     # Medium-high
}


# ============= Helper Functions =============

def create_empty_analysis_state(
    pair: str,
    current_price: float,
    market_context: str,
    timeframe: str = "15m",
    ohlcv_data: Any = None,
    ohlcv_data_htf: Any = None,
    timeframe_htf: str = "1h"
) -> AnalysisState:
    """
    Create an empty AnalysisState with input fields populated.

    Args:
        pair: Trading pair
        current_price: Current price
        market_context: Market context string
        timeframe: Primary timeframe (e.g., "15m")
        ohlcv_data: Optional OHLCV data for primary timeframe
        ohlcv_data_htf: Optional OHLCV data for higher timeframe
        timeframe_htf: Higher timeframe (e.g., "1h")

    Returns:
        Initialized AnalysisState
    """
    return AnalysisState(
        pair=pair,
        current_price=current_price,
        market_context=market_context,
        ohlcv_data=ohlcv_data,
        ohlcv_data_htf=ohlcv_data_htf,
        timeframe=timeframe,
        timeframe_htf=timeframe_htf,
        agent_reports=[],
        indicator_report=None,
        trend_report=None,
        sentiment_report=None,
        pattern_report=None,
        consensus_direction=None,
        consensus_confidence=0.0,
        key_support=None,
        key_resistance=None,
        weighted_scores={},
        errors=[]
    )


def create_empty_trading_state(
    pair: str,
    current_price: float,
    market_context: str,
    has_position: bool = False,
    position_side: Optional[str] = None,
    position_profit_pct: Optional[float] = None,
    timeframe: str = "15m",
    ohlcv_data: Any = None,
    ohlcv_data_htf: Any = None,
    timeframe_htf: str = "1h"
) -> TradingDecisionState:
    """
    Create an empty TradingDecisionState with input fields populated.

    Args:
        pair: Trading pair
        current_price: Current price
        market_context: Market context string
        has_position: Whether in position
        position_side: Position side if in position
        position_profit_pct: Current P&L if in position
        timeframe: Primary timeframe (e.g., "15m")
        ohlcv_data: Optional OHLCV data for primary timeframe
        ohlcv_data_htf: Optional OHLCV data for higher timeframe
        timeframe_htf: Higher timeframe (e.g., "1h")

    Returns:
        Initialized TradingDecisionState
    """
    from datetime import datetime

    return TradingDecisionState(
        # Input fields
        pair=pair,
        current_price=current_price,
        market_context=market_context,
        ohlcv_data=ohlcv_data,
        ohlcv_data_htf=ohlcv_data_htf,
        timeframe=timeframe,
        timeframe_htf=timeframe_htf,
        has_position=has_position,
        position_side=position_side,
        position_profit_pct=position_profit_pct,
        
        # Analysis stage
        indicator_report=None,
        trend_report=None,
        sentiment_report=None,
        pattern_report=None,
        consensus_direction=None,
        consensus_confidence=0.0,
        key_support=None,
        key_resistance=None,
        weighted_scores=None,
        
        # Debate stage
        bull_argument=None,
        bear_argument=None,
        judge_verdict=None,
        position_bull_argument=None,
        position_bear_argument=None,
        position_judge_verdict=None,
        
        # Grounding stage
        grounding_result=None,
        grounding_verified=False,
        grounding_corrected_values=None,
        hallucination_score=None,
        corrected_context=None,
        grounding_summary=None,
        
        # Position grounding
        position_metrics=None,
        position_grounding_result=None,
        position_hallucination_score=None,
        position_grounding_corrected_values=None,
        position_corrected_context=None,
        position_grounding_summary=None,
        
        # Executor output
        executor_decision=None,
        final_action=None,
        final_confidence=0.0,
        final_direction=None,
        final_leverage=None,
        stop_loss_price=None,
        take_profit_price=None,
        adjustment_pct=None,
        adjustment_type=None,
        executor_reasoning=None,
        executor_key_factors=None,
        
        # Legacy fields
        final_reason="",
        final_key_support=None,
        final_key_resistance=None,
        is_valid=False,
        
        # Execution result
        execution_result=None,
        tool_result=None,
        
        # Metadata
        execution_path=None,
        nodes_executed=None,
        errors=[],
        warnings=[],
        thread_id="",
        created_at=datetime.now().isoformat()
    )

