"""
Graph Execution Metrics - Complete Observability System.

Based on Dynatrace / McKinsey 2025 best practices:
- Token consumption tracking
- Model behavior monitoring
- Guardrail result logging
- Non-linear flow tracing

This module provides comprehensive metrics collection for the trading graph,
enabling monitoring, debugging, and optimization.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single execution stage."""
    stage_name: str
    duration_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    
    # Stage-specific data
    stage_output_summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "success": self.success,
            "error": self.error,
            "stage_output_summary": self.stage_output_summary
        }


@dataclass
class GroundingMetrics:
    """Metrics specifically for grounding/verification stage."""
    hallucination_score: float = 0
    total_claims: int = 0
    verified_claims: int = 0
    false_claims: int = 0
    corrected_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "hallucination_score": self.hallucination_score,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "false_claims": self.false_claims,
            "corrected_indicators": self.corrected_indicators
        }


@dataclass
class VFVerificationMetrics:
    """Metrics for Verification-First verification stage."""
    all_passed: bool = True
    analysis_consensus_verified: bool = True
    debate_conclusion_verified: bool = True
    data_citation_verified: bool = True
    risk_oversight_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "analysis_consensus_verified": self.analysis_consensus_verified,
            "debate_conclusion_verified": self.debate_conclusion_verified,
            "data_citation_verified": self.data_citation_verified,
            "risk_oversight_verified": self.risk_oversight_verified,
            "verification_notes": self.verification_notes
        }


@dataclass  
class ConsistencyMetrics:
    """Metrics for reasoning consistency checks."""
    is_consistent: bool = True
    violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "violations": self.violations
        }


@dataclass
class DecisionMetrics:
    """Metrics for the final trading decision."""
    action: str = ""
    confidence: float = 0
    confidence_before_calibration: float = 0
    direction: Optional[str] = None
    direction_strength: Optional[str] = None
    risk_level: Optional[str] = None
    leverage: Optional[int] = None
    
    # Risk management (calculated by code)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "confidence_before_calibration": self.confidence_before_calibration,
            "direction": self.direction,
            "direction_strength": self.direction_strength,
            "risk_level": self.risk_level,
            "leverage": self.leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio
        }


@dataclass
class GraphExecutionMetrics:
    """Complete metrics for a full graph execution."""
    # Identification
    thread_id: str = ""
    pair: str = ""
    execution_path: str = ""  # "entry" or "position"
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: float = 0
    
    # Token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd: float = 0
    
    # Stage metrics
    stages: List[StageMetrics] = field(default_factory=list)
    
    # Grounding metrics
    grounding: Optional[GroundingMetrics] = None
    
    # VF verification metrics
    vf_verification: Optional[VFVerificationMetrics] = None
    
    # Consistency metrics
    consistency: Optional[ConsistencyMetrics] = None
    
    # Decision metrics
    decision: Optional[DecisionMetrics] = None
    
    # Outcome (for learning, filled later)
    actual_outcome: Optional[Dict[str, Any]] = None
    
    def add_stage(self, stage: StageMetrics):
        """Add a stage and update totals."""
        self.stages.append(stage)
        self.total_duration_ms += stage.duration_ms
        self.total_prompt_tokens += stage.prompt_tokens
        self.total_completion_tokens += stage.completion_tokens
    
    def calculate_cost(
        self, 
        price_per_1k_prompt: float = 0.0005, 
        price_per_1k_completion: float = 0.0015
    ):
        """Estimate API cost based on token usage."""
        self.estimated_cost_usd = (
            (self.total_prompt_tokens / 1000) * price_per_1k_prompt +
            (self.total_completion_tokens / 1000) * price_per_1k_completion
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            # Identification
            "thread_id": self.thread_id,
            "pair": self.pair,
            "execution_path": self.execution_path,
            
            # Timing
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            
            # Token usage
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            
            # Stages
            "stages": [s.to_dict() for s in self.stages],
            "stage_count": len(self.stages),
            
            # Grounding
            "grounding": self.grounding.to_dict() if self.grounding else None,
            
            # VF verification
            "vf_verification": self.vf_verification.to_dict() if self.vf_verification else None,
            
            # Consistency
            "consistency": self.consistency.to_dict() if self.consistency else None,
            
            # Decision
            "decision": self.decision.to_dict() if self.decision else None,
            
            # Outcome
            "actual_outcome": self.actual_outcome
        }
    
    def summary_line(self) -> str:
        """Get a single-line summary for logging."""
        vf_status = "VF:✓" if (self.vf_verification and self.vf_verification.all_passed) else "VF:✗"
        cons_status = "Cons:✓" if (self.consistency and self.consistency.is_consistent) else "Cons:✗"
        halluc = f"H:{self.grounding.hallucination_score:.0f}%" if self.grounding else "H:?"
        
        action = self.decision.action if self.decision else "?"
        conf = self.decision.confidence if self.decision else 0
        
        return (
            f"[{self.pair}] {action} ({conf:.0f}%) | "
            f"{vf_status} {cons_status} {halluc} | "
            f"{self.total_duration_ms:.0f}ms ${self.estimated_cost_usd:.4f}"
        )


class MetricsCollector:
    """
    Collector for graph execution metrics.
    
    Usage:
        collector = MetricsCollector()
        collector.start_execution("thread-123", "BTC/USDT", "entry")
        
        with collector.stage("analysis"):
            # ... do analysis ...
            
        with collector.stage("debate"):
            # ... do debate ...
            
        collector.record_grounding(...)
        collector.record_vf_verification(...)
        collector.record_decision(...)
        
        metrics = collector.finalize()
        logger.info(metrics.summary_line())
    """
    
    def __init__(self):
        self.current_metrics: Optional[GraphExecutionMetrics] = None
        self._stage_start_time: Optional[float] = None
    
    def start_execution(
        self, 
        thread_id: str, 
        pair: str, 
        execution_path: str
    ):
        """Start a new execution metrics collection."""
        self.current_metrics = GraphExecutionMetrics(
            thread_id=thread_id,
            pair=pair,
            execution_path=execution_path,
            start_time=datetime.now().isoformat()
        )
        logger.debug(f"[Metrics] Started collection for {pair} ({execution_path})")
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a stage."""
        start = time.time()
        stage_metrics = StageMetrics(stage_name=stage_name)
        
        try:
            yield stage_metrics
        except Exception as e:
            stage_metrics.success = False
            stage_metrics.error = str(e)
            raise
        finally:
            stage_metrics.duration_ms = (time.time() - start) * 1000
            if self.current_metrics:
                self.current_metrics.add_stage(stage_metrics)
    
    def record_stage(
        self,
        stage_name: str,
        duration_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        summary: str = ""
    ):
        """Manually record a stage's metrics."""
        stage = StageMetrics(
            stage_name=stage_name,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error=error,
            stage_output_summary=summary
        )
        if self.current_metrics:
            self.current_metrics.add_stage(stage)
    
    def record_grounding(
        self,
        hallucination_score: float,
        total_claims: int,
        verified_claims: int,
        false_claims: int,
        corrected_indicators: List[str]
    ):
        """Record grounding verification metrics."""
        if self.current_metrics:
            self.current_metrics.grounding = GroundingMetrics(
                hallucination_score=hallucination_score,
                total_claims=total_claims,
                verified_claims=verified_claims,
                false_claims=false_claims,
                corrected_indicators=corrected_indicators
            )
    
    def record_vf_verification(
        self,
        all_passed: bool,
        analysis_consensus_verified: bool = True,
        debate_conclusion_verified: bool = True,
        data_citation_verified: bool = True,
        risk_oversight_verified: bool = True,
        verification_notes: List[str] = None
    ):
        """Record VF verification metrics."""
        if self.current_metrics:
            self.current_metrics.vf_verification = VFVerificationMetrics(
                all_passed=all_passed,
                analysis_consensus_verified=analysis_consensus_verified,
                debate_conclusion_verified=debate_conclusion_verified,
                data_citation_verified=data_citation_verified,
                risk_oversight_verified=risk_oversight_verified,
                verification_notes=verification_notes or []
            )
    
    def record_consistency(
        self,
        is_consistent: bool,
        violations: List[str]
    ):
        """Record reasoning consistency metrics."""
        if self.current_metrics:
            self.current_metrics.consistency = ConsistencyMetrics(
                is_consistent=is_consistent,
                violations=violations
            )
    
    def record_decision(
        self,
        action: str,
        confidence: float,
        direction: Optional[str] = None,
        direction_strength: Optional[str] = None,
        risk_level: Optional[str] = None,
        leverage: Optional[int] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        risk_reward_ratio: Optional[float] = None,
        confidence_before_calibration: float = 0
    ):
        """Record final decision metrics."""
        if self.current_metrics:
            self.current_metrics.decision = DecisionMetrics(
                action=action,
                confidence=confidence,
                confidence_before_calibration=confidence_before_calibration,
                direction=direction,
                direction_strength=direction_strength,
                risk_level=risk_level,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                risk_reward_ratio=risk_reward_ratio
            )
    
    def record_outcome(self, outcome: Dict[str, Any]):
        """Record actual trading outcome (for learning)."""
        if self.current_metrics:
            self.current_metrics.actual_outcome = outcome
    
    def finalize(
        self,
        price_per_1k_prompt: float = 0.0005,
        price_per_1k_completion: float = 0.0015
    ) -> Optional[GraphExecutionMetrics]:
        """Finalize and return the collected metrics."""
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now().isoformat()
            self.current_metrics.calculate_cost(
                price_per_1k_prompt,
                price_per_1k_completion
            )
            
            logger.debug(
                f"[Metrics] Finalized: {self.current_metrics.summary_line()}"
            )
            
            return self.current_metrics
        return None
    
    def reset(self):
        """Reset the collector for a new execution."""
        self.current_metrics = None


# Global collector instance (can be used as singleton)
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector():
    """Reset the global metrics collector."""
    global _global_collector
    if _global_collector:
        _global_collector.reset()
    _global_collector = None
