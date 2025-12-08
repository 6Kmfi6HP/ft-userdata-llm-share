"""
Executor Agent Node for LangGraph with Verification-First (VF) Strategy.

Based on arXiv 2511.21734: "Asking LLMs to Verify First is Almost Free Lunch"
And arXiv 2512.01123: "LLM-Generated Bayesian Networks for Transparent Trading"

Key improvements:
1. VF Strategy: LLM verifies candidate answer before generating decision
2. LLM outputs qualitative judgments (direction_strength, risk_level)
3. Code calculates precise numerical values (stop_loss, take_profit)
4. Pydantic validates all outputs before execution
5. Reasoning consistency verification against prior agent conclusions
"""

import logging
import re
import time
from typing import Dict, Any, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from ...state import (
    TradingDecisionState,
    Direction,
    ExecutorDecision
)
from ...prompts.execution.executor_prompt import (
    # VF prompts (new)
    EXECUTOR_VF_SYSTEM_PROMPT,
    build_vf_executor_user_prompt,
    build_candidate_answer,
    # Original prompts (fallback)
    EXECUTOR_SYSTEM_PROMPT,
    build_executor_user_prompt,
    build_risk_rules_section,
    build_consensus_summary,
    build_debate_summary,
    build_position_info,
)
from ...schemas.executor_schemas import (
    ExecutorQualitativeOutput,
    ExecutorOutputSchema,
    RiskManagementConfig,
    calculate_risk_management,
    verify_reasoning_consistency,
    create_conservative_output,
    parse_qualitative_from_text,
)

logger = logging.getLogger(__name__)

# Default thresholds
MIN_CONFIDENCE_THRESHOLD = 60.0
HALLUCINATION_REJECT_THRESHOLD = 70.0

# Feature flags
USE_VF_STRATEGY = True  # Enable Verification-First strategy
USE_QUALITATIVE_OUTPUT = True  # LLM outputs qualitative, code calculates numbers


def executor_agent_node(
    state: TradingDecisionState,
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Executor Agent Node - Final LLM-based decision maker with VF strategy.
    
    Takes all analysis, debate, and grounding results to produce
    the final trading decision.
    
    Key improvements from research:
    1. VF (Verification-First): Verifies Judge's verdict before deciding
    2. Qualitative LLM output: LLM provides direction_strength, risk_level
    3. Numerical calculation: Code calculates stop_loss, take_profit
    4. Pydantic validation: Ensures output integrity
    5. Reasoning consistency: Checks for logical conflicts
    
    Args:
        state: Current trading decision state
        config: Node configuration
        
    Returns:
        Dict with final decision fields including VF verification results
    """
    pair = state.get("pair", "UNKNOWN")
    has_position = state.get("has_position", False)
    
    logger.info(f"[ExecutorAgent] Starting VF decision for {pair} (has_position={has_position})")
    
    start_time = time.time()
    
    try:
        # Check reflection result - if critical issues, consider returning conservative
        if has_position:
            reflection_result = state.get("position_reflection_result")
            confidence_adj = state.get("position_reflection_confidence_adjustment", 0)
        else:
            reflection_result = state.get("reflection_result")
            confidence_adj = state.get("reflection_confidence_adjustment", 0)
        
        # Log reflection context
        if reflection_result:
            logger.info(
                f"[ExecutorAgent] {pair}: Reflection - "
                f"direction_correct={getattr(reflection_result, 'direction_correct', 'N/A')}, "
                f"timing_good={getattr(reflection_result, 'entry_timing_appropriate', 'N/A')}, "
                f"confidence_adj={confidence_adj:+.0f}%"
            )
        
        # Get LLM config
        llm_config = state.get("llm_config", {})
        risk_config = state.get("risk_config", {})
        
        # Build the prompt
        if USE_VF_STRATEGY:
            result = _execute_with_vf_strategy(state, llm_config, risk_config, has_position)
        else:
            result = _execute_with_original_strategy(state, llm_config, risk_config, has_position)
        
        # Calculate execution time
        duration_ms = (time.time() - start_time) * 1000
        result["executor_duration_ms"] = duration_ms
        result["executor_timestamp"] = _get_timestamp()
        
        logger.info(
            f"[ExecutorAgent] {pair}: action={result.get('final_action')} "
            f"confidence={result.get('final_confidence', 0):.0f}% "
            f"VF_passed={result.get('vf_verification_passed', 'N/A')} "
            f"consistent={result.get('reasoning_consistent', 'N/A')} "
            f"(took {duration_ms:.0f}ms)"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[ExecutorAgent] {pair} failed: {e}")
        duration_ms = (time.time() - start_time) * 1000
        return _build_conservative_result(
            state, 
            f"Executor error: {str(e)}",
            duration_ms=duration_ms
        )


def _execute_with_vf_strategy(
    state: TradingDecisionState,
    llm_config: Dict,
    risk_config: Dict,
    has_position: bool
) -> Dict[str, Any]:
    """
    Execute with Verification-First (VF) strategy.
    
    Following arXiv 2511.21734:
    1. Present Judge's verdict as "candidate answer"
    2. LLM verifies the candidate before generating decision
    3. LLM outputs qualitative judgments
    4. Code calculates precise numbers
    5. Pydantic validates output
    6. Consistency check against prior agents
    """
    # Build VF system prompt
    system_prompt = EXECUTOR_VF_SYSTEM_PROMPT.format(
        risk_rules=build_risk_rules_section(risk_config)
    )
    
    # Build candidate answer from Judge verdict
    candidate_answer = build_candidate_answer(state)
    
    # Build VF user prompt
    user_prompt = _build_vf_user_prompt(state, candidate_answer, has_position)
    
    # Initialize LLM
    llm = _create_llm(llm_config)
    
    # Invoke LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    # Parse VF verification results
    vf_results = _parse_vf_verification(response_text)
    
    # Parse qualitative output
    qualitative = parse_qualitative_from_text(response_text, has_position)
    
    # Calculate risk management from qualitative output
    risk_params = calculate_risk_management(
        qualitative=qualitative,
        current_price=state.get("current_price", 0.0),
        key_support=state.get("key_support"),
        key_resistance=state.get("key_resistance"),
        config=_get_risk_management_config(risk_config)
    )
    
    # Build full decision with calculated values
    try:
        decision = ExecutorOutputSchema(
            verification=None,  # Could store full VF results here
            action=qualitative.action,
            confidence=qualitative.confidence,
            direction_strength=qualitative.direction_strength,
            risk_level=qualitative.risk_level,
            stop_loss_price=risk_params.get("stop_loss_price"),
            take_profit_price=risk_params.get("take_profit_price"),
            stop_loss_pct=risk_params.get("stop_loss_pct"),
            take_profit_pct=risk_params.get("take_profit_pct"),
            risk_reward_ratio=risk_params.get("risk_reward_ratio"),
            adjustment_pct=qualitative.adjustment_pct,
            adjustment_type=qualitative.adjustment_type,
            reasoning=qualitative.reasoning,
            key_factors=qualitative.key_factors,
        )
    except Exception as e:
        logger.warning(f"[ExecutorAgent] Pydantic validation failed: {e}")
        # Fallback to conservative output
        conservative = create_conservative_output(has_position, f"Validation failed: {e}")
        decision = conservative
    
    # Verify reasoning consistency
    state_context = {
        "hallucination_score": state.get("hallucination_score"),
        "position_hallucination_score": state.get("position_hallucination_score"),
        "judge_verdict": state.get("judge_verdict"),
        "position_judge_verdict": state.get("position_judge_verdict"),
        "consensus_direction": state.get("consensus_direction"),
    }
    is_consistent, violations = verify_reasoning_consistency(decision, state_context)
    
    # Update decision with consistency results
    decision.reasoning_consistent = is_consistent
    decision.consistency_violations = violations
    
    # Build result
    return _build_vf_result(decision, state, vf_results, is_consistent, violations)


def _execute_with_original_strategy(
    state: TradingDecisionState,
    llm_config: Dict,
    risk_config: Dict,
    has_position: bool
) -> Dict[str, Any]:
    """
    Execute with original strategy (fallback).
    
    LLM outputs everything including numerical values.
    """
    # Build original system prompt
    system_prompt = EXECUTOR_SYSTEM_PROMPT.format(
        risk_rules=build_risk_rules_section(risk_config)
    )
    
    user_prompt = _build_complete_user_prompt(state, has_position)
    
    # Initialize LLM
    llm = _create_llm(llm_config)
    
    # Invoke LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    # Parse response (original parser)
    decision = _parse_executor_response(response_text, state)
    
    # Build result
    return _build_result_from_decision(decision, state, 0)


def _build_vf_user_prompt(
    state: TradingDecisionState,
    candidate_answer: Dict,
    has_position: bool
) -> str:
    """Build VF user prompt from state."""
    
    # Build consensus summary
    consensus_summary = build_consensus_summary(
        consensus_direction=state.get("consensus_direction"),
        consensus_confidence=state.get("consensus_confidence", 0),
        weighted_scores=state.get("weighted_scores")
    )
    
    # Build debate summary
    if has_position:
        debate_summary = build_debate_summary(
            bull_argument=state.get("position_bull_argument"),
            bear_argument=state.get("position_bear_argument"),
            judge_verdict=state.get("position_judge_verdict")
        )
        # Use reflection summary instead of grounding
        reflection_summary = state.get("position_reflection_summary", "无验证数据")
        reflection_reasoning = state.get("position_reflection_result")
        if reflection_reasoning:
            reflection_context = getattr(reflection_reasoning, 'reflection_reasoning', '')
        else:
            reflection_context = ""
    else:
        debate_summary = build_debate_summary(
            bull_argument=state.get("bull_argument"),
            bear_argument=state.get("bear_argument"),
            judge_verdict=state.get("judge_verdict")
        )
        # Use reflection summary instead of grounding
        reflection_summary = state.get("reflection_summary", "无验证数据")
        reflection_reasoning = state.get("reflection_result")
        if reflection_reasoning:
            reflection_context = getattr(reflection_reasoning, 'reflection_reasoning', '')
        else:
            reflection_context = ""
    
    # Build position info if applicable
    position_info = None
    if has_position:
        position_metrics = state.get("position_metrics") or {}
        position_info = build_position_info(
            position_side=state.get("position_side", "long"),
            position_profit_pct=state.get("position_profit_pct", 0.0),
            entry_price=state.get("position_entry_price"),
            mfe=position_metrics.get("max_profit_pct"),
            mae=position_metrics.get("max_loss_pct"),
            drawdown=position_metrics.get("drawdown_from_peak_pct"),
            hold_count=position_metrics.get("hold_count")
        )
    
    return build_vf_executor_user_prompt(
        candidate_answer=candidate_answer,
        consensus_summary=consensus_summary,
        debate_summary=debate_summary,
        grounding_summary=reflection_summary,  # Use reflection instead of grounding
        corrected_context=reflection_context,  # Use reflection context
        current_price=state.get("current_price", 0.0),
        key_support=state.get("key_support"),
        key_resistance=state.get("key_resistance"),
        has_position=has_position,
        position_info=position_info
    )


def _parse_vf_verification(response_text: str) -> Dict[str, Any]:
    """
    Parse VF verification results from LLM response.
    
    Looks for:
    [验证结果]
    分析共识验证: PASS/FAIL (理由)
    辩论结论验证: PASS/FAIL (理由)
    数据引用验证: PASS/FAIL (理由)
    风险识别验证: PASS/FAIL (理由)
    """
    results = {
        "analysis_consensus_verified": True,
        "debate_conclusion_verified": True,
        "data_citation_verified": True,
        "risk_oversight_verified": True,
        "verification_notes": []
    }
    
    try:
        # Extract verification section
        section_match = re.search(
            r'\[验证结果\]\s*\n(.+?)(?=\n\[验证后决策\]|\n\[决策\]|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if section_match:
            section_text = section_match.group(1)
            
            # Parse each dimension
            dimensions = [
                ("分析共识验证", "analysis_consensus_verified"),
                ("辩论结论验证", "debate_conclusion_verified"),
                ("数据引用验证", "data_citation_verified"),
                ("风险识别验证", "risk_oversight_verified"),
            ]
            
            for cn_name, key in dimensions:
                pattern = rf'{cn_name}:\s*(PASS|FAIL)\s*(?:\((.+?)\))?'
                match = re.search(pattern, section_text, re.IGNORECASE)
                if match:
                    results[key] = match.group(1).upper() == "PASS"
                    if match.group(2):
                        results["verification_notes"].append(
                            f"{cn_name}: {match.group(2)}"
                        )
    
    except Exception as e:
        logger.warning(f"[ExecutorAgent] Failed to parse VF verification: {e}")
    
    # Calculate overall pass
    results["all_passed"] = all([
        results["analysis_consensus_verified"],
        results["debate_conclusion_verified"],
        results["data_citation_verified"],
        results["risk_oversight_verified"],
    ])
    
    return results


def _get_risk_management_config(risk_config: Dict) -> RiskManagementConfig:
    """Build RiskManagementConfig from risk_config dict."""
    return RiskManagementConfig(
        high_risk_sl_pct=risk_config.get("high_risk_sl_pct", 0.015),
        medium_risk_sl_pct=risk_config.get("medium_risk_sl_pct", 0.025),
        low_risk_sl_pct=risk_config.get("low_risk_sl_pct", 0.04),
        strong_direction_tp_mult=risk_config.get("strong_direction_tp_mult", 3.0),
        moderate_direction_tp_mult=risk_config.get("moderate_direction_tp_mult", 2.0),
        weak_direction_tp_mult=risk_config.get("weak_direction_tp_mult", 1.5),
    )


def _build_vf_result(
    decision: ExecutorOutputSchema,
    state: TradingDecisionState,
    vf_results: Dict,
    is_consistent: bool,
    violations: List[str]
) -> Dict[str, Any]:
    """Build state update from VF execution result."""
    
    current_price = state.get("current_price", 0.0)
    
    # Determine direction from action
    direction = None
    if decision.action == "signal_entry_long":
        direction = Direction.LONG
    elif decision.action == "signal_entry_short":
        direction = Direction.SHORT
    
    # Calculate leverage from confidence (if not set)
    leverage = None
    if decision.action in ["signal_entry_long", "signal_entry_short"]:
        leverage = _calculate_leverage(decision.confidence, state.get("risk_config", {}))
    
    return {
        # Executor decision
        "executor_decision": ExecutorDecision(
            action=decision.action,
            confidence=decision.confidence,
            direction=direction,
            leverage=leverage,
            stop_loss_price=decision.stop_loss_price,
            take_profit_price=decision.take_profit_price,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            risk_reward_ratio=decision.risk_reward_ratio,
            adjustment_pct=decision.adjustment_pct,
            adjustment_type=decision.adjustment_type,
            reasoning=decision.reasoning,
            key_factors=decision.key_factors,
            risk_assessment=decision.risk_assessment,
        ),
        
        # Final action
        "final_action": decision.action,
        "final_confidence": decision.confidence,
        "final_direction": direction,
        "final_leverage": leverage,
        
        # Risk management (calculated by code, not LLM)
        "stop_loss_price": decision.stop_loss_price,
        "take_profit_price": decision.take_profit_price,
        "stop_loss_pct": decision.stop_loss_pct,
        "take_profit_pct": decision.take_profit_pct,
        "risk_reward_ratio": decision.risk_reward_ratio,
        
        # Position adjustment
        "adjustment_pct": decision.adjustment_pct,
        "adjustment_type": decision.adjustment_type,
        
        # Qualitative judgments (NEW)
        "direction_strength": decision.direction_strength,
        "risk_level": decision.risk_level,
        
        # VF Verification results (NEW)
        "vf_verification_passed": vf_results.get("all_passed", False),
        "vf_analysis_consensus_verified": vf_results.get("analysis_consensus_verified", True),
        "vf_debate_conclusion_verified": vf_results.get("debate_conclusion_verified", True),
        "vf_data_citation_verified": vf_results.get("data_citation_verified", True),
        "vf_risk_oversight_verified": vf_results.get("risk_oversight_verified", True),
        "vf_verification_notes": vf_results.get("verification_notes", []),
        
        # Reasoning consistency (NEW)
        "reasoning_consistent": is_consistent,
        "consistency_violations": violations,
        
        # Reasoning
        "executor_reasoning": decision.reasoning,
        "executor_key_factors": decision.key_factors,
        "executor_risk_assessment": decision.risk_assessment,
        
        # Legacy compatibility
        "final_reason": decision.reasoning[:200] if decision.reasoning else "",
        "final_key_support": state.get("key_support"),
        "final_key_resistance": state.get("key_resistance"),
        "is_valid": (
            decision.confidence >= MIN_CONFIDENCE_THRESHOLD or 
            decision.action in ("signal_wait", "signal_hold")
        ) and is_consistent,
        
        # Execution result for downstream
        "execution_result": {
            "action": decision.action,
            "pair": state.get("pair"),
            "confidence_score": decision.confidence,
            "reason": decision.reasoning[:200] if decision.reasoning else "",
            "leverage": leverage,
            "current_price": current_price,
            "stop_loss": decision.stop_loss_price,
            "take_profit": decision.take_profit_price,
            "adjustment_pct": decision.adjustment_pct,
            "key_support": state.get("key_support"),
            "key_resistance": state.get("key_resistance"),
            "direction_strength": decision.direction_strength,
            "risk_level": decision.risk_level,
            "vf_passed": vf_results.get("all_passed", False),
            "reasoning_consistent": is_consistent,
            "source": "langgraph_executor_agent_vf"
        }
    }


def _calculate_leverage(confidence: float, risk_config: Dict) -> int:
    """Calculate leverage from confidence score."""
    max_leverage = risk_config.get("max_leverage", 50)
    min_leverage = risk_config.get("min_leverage", 5)
    
    # Linear scaling: 60% confidence -> min, 100% confidence -> max
    if confidence <= 60:
        return min_leverage
    elif confidence >= 100:
        return max_leverage
    else:
        ratio = (confidence - 60) / 40
        return int(min_leverage + ratio * (max_leverage - min_leverage))


def _create_llm(llm_config: Dict[str, Any]) -> ChatOpenAI:
    """Create LLM instance from config."""
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4o-mini"),
        temperature=llm_config.get("temperature", 0.1),
        openai_api_base=llm_config.get("api_base"),
        openai_api_key=llm_config.get("api_key", "not-needed"),
        model_kwargs=llm_config.get("model_kwargs", {})
    )


def _build_complete_user_prompt(state: TradingDecisionState, has_position: bool) -> str:
    """Build complete user prompt from state (original version)."""
    
    # Build consensus summary
    consensus_summary = build_consensus_summary(
        consensus_direction=state.get("consensus_direction"),
        consensus_confidence=state.get("consensus_confidence", 0),
        weighted_scores=state.get("weighted_scores")
    )
    
    # Build debate summary
    if has_position:
        debate_summary = build_debate_summary(
            bull_argument=state.get("position_bull_argument"),
            bear_argument=state.get("position_bear_argument"),
            judge_verdict=state.get("position_judge_verdict")
        )
        grounding_summary = state.get("position_grounding_summary", "无验证数据")
        corrected_context = state.get("position_corrected_context", "")
    else:
        debate_summary = build_debate_summary(
            bull_argument=state.get("bull_argument"),
            bear_argument=state.get("bear_argument"),
            judge_verdict=state.get("judge_verdict")
        )
        grounding_summary = state.get("grounding_summary", "无验证数据")
        corrected_context = state.get("corrected_context", "")
    
    # Build position info if applicable
    position_info = None
    if has_position:
        position_metrics = state.get("position_metrics") or {}
        position_info = build_position_info(
            position_side=state.get("position_side", "long"),
            position_profit_pct=state.get("position_profit_pct", 0.0),
            entry_price=state.get("position_entry_price"),
            mfe=position_metrics.get("max_profit_pct"),
            mae=position_metrics.get("max_loss_pct"),
            drawdown=position_metrics.get("drawdown_from_peak_pct"),
            hold_count=position_metrics.get("hold_count")
        )
    
    return build_executor_user_prompt(
        consensus_summary=consensus_summary,
        debate_summary=debate_summary,
        grounding_summary=grounding_summary,
        corrected_context=corrected_context,
        current_price=state.get("current_price", 0.0),
        key_support=state.get("key_support"),
        key_resistance=state.get("key_resistance"),
        has_position=has_position,
        position_info=position_info
    )


def _parse_executor_response(response_text: str, state: TradingDecisionState) -> ExecutorDecision:
    """
    Parse Executor Agent response into ExecutorDecision.
    
    Expected format:
    [决策]
    action: ENTRY_LONG
    confidence: 75
    leverage: 10
    direction: LONG
    
    [风险管理]
    stop_loss_price: 49500
    take_profit_price: 52000
    risk_reward_ratio: 2.5
    
    [决策理由]
    ...
    """
    # Default values
    action = "signal_wait"
    confidence = 0.0
    leverage = None
    direction = None
    stop_loss_price = None
    take_profit_price = None
    risk_reward_ratio = None
    adjustment_pct = None
    adjustment_type = None
    reasoning = ""
    key_factors = []
    risk_assessment = ""
    
    try:
        # Parse action
        action_match = re.search(r'action:\s*(\w+)', response_text, re.IGNORECASE)
        if action_match:
            raw_action = action_match.group(1).upper()
            action = _normalize_action(raw_action)
        
        # Parse confidence
        conf_match = re.search(r'confidence:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # Parse leverage
        leverage_match = re.search(r'leverage:\s*(\d+)', response_text, re.IGNORECASE)
        if leverage_match:
            leverage = int(leverage_match.group(1))
        
        # Parse direction
        direction_match = re.search(r'direction:\s*(LONG|SHORT|NEUTRAL)', response_text, re.IGNORECASE)
        if direction_match:
            direction = Direction(direction_match.group(1).lower())
        
        # Parse stop loss
        sl_match = re.search(r'stop_loss_price:\s*([\d.]+)', response_text, re.IGNORECASE)
        if sl_match:
            stop_loss_price = float(sl_match.group(1))
        
        # Parse take profit
        tp_match = re.search(r'take_profit_price:\s*([\d.]+)', response_text, re.IGNORECASE)
        if tp_match:
            take_profit_price = float(tp_match.group(1))
        
        # Parse risk reward
        rr_match = re.search(r'risk_reward_ratio:\s*([\d.]+)', response_text, re.IGNORECASE)
        if rr_match:
            risk_reward_ratio = float(rr_match.group(1))
        
        # Parse adjustment (for position management)
        adj_match = re.search(r'adjustment_pct:\s*([+-]?\d+)', response_text, re.IGNORECASE)
        if adj_match:
            adjustment_pct = float(adj_match.group(1))
        
        adj_type_match = re.search(r'adjustment_type:\s*(\w+)', response_text, re.IGNORECASE)
        if adj_type_match:
            adjustment_type = adj_type_match.group(1).lower()
        
        # Parse reasoning section
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
        
        # Parse risk assessment
        risk_match = re.search(
            r'\[风险评估\]\s*\n(.+?)(?=\n\[|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if risk_match:
            risk_assessment = risk_match.group(1).strip()
            
    except Exception as e:
        logger.warning(f"[ExecutorAgent] Parse error: {e}, using defaults")
    
    return ExecutorDecision(
        action=action,
        confidence=confidence,
        direction=direction,
        leverage=leverage,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_reward_ratio=risk_reward_ratio,
        adjustment_pct=adjustment_pct,
        adjustment_type=adjustment_type,
        reasoning=reasoning,
        key_factors=key_factors,
        risk_assessment=risk_assessment
    )


def _normalize_action(raw_action: str) -> str:
    """Normalize action string to expected format."""
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
    return action_map.get(raw_action, "signal_wait")


def _build_result_from_decision(
    decision: ExecutorDecision, 
    state: TradingDecisionState,
    duration_ms: float
) -> Dict[str, Any]:
    """Build state update from ExecutorDecision (original version)."""
    
    current_price = state.get("current_price", 0.0)
    
    # Calculate percentages if we have the data
    stop_loss_pct = None
    take_profit_pct = None
    
    if decision.stop_loss_price and current_price > 0:
        stop_loss_pct = abs(decision.stop_loss_price - current_price) / current_price * 100
    
    if decision.take_profit_price and current_price > 0:
        take_profit_pct = abs(decision.take_profit_price - current_price) / current_price * 100
    
    return {
        # Executor decision
        "executor_decision": decision,
        
        # Final action
        "final_action": decision.action,
        "final_confidence": decision.confidence,
        "final_direction": decision.direction,
        "final_leverage": decision.leverage,
        
        # Risk management
        "stop_loss_price": decision.stop_loss_price,
        "take_profit_price": decision.take_profit_price,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "risk_reward_ratio": decision.risk_reward_ratio,
        
        # Position adjustment
        "adjustment_pct": decision.adjustment_pct,
        "adjustment_type": decision.adjustment_type,
        
        # Reasoning
        "executor_reasoning": decision.reasoning,
        "executor_key_factors": decision.key_factors,
        "executor_risk_assessment": decision.risk_assessment,
        
        # Metadata
        "executor_duration_ms": duration_ms,
        "executor_timestamp": _get_timestamp(),
        
        # Legacy compatibility
        "final_reason": decision.reasoning[:200] if decision.reasoning else "",
        "final_key_support": state.get("key_support"),
        "final_key_resistance": state.get("key_resistance"),
        "is_valid": decision.confidence >= MIN_CONFIDENCE_THRESHOLD or decision.action in ("signal_wait", "signal_hold"),
        
        # Execution result for downstream
        "execution_result": {
            "action": decision.action,
            "pair": state.get("pair"),
            "confidence_score": decision.confidence,
            "reason": decision.reasoning[:200] if decision.reasoning else "",
            "leverage": decision.leverage,
            "current_price": current_price,
            "stop_loss": decision.stop_loss_price,
            "take_profit": decision.take_profit_price,
            "adjustment_pct": decision.adjustment_pct,
            "key_support": state.get("key_support"),
            "key_resistance": state.get("key_resistance"),
            "source": "langgraph_executor_agent"
        }
    }


def _build_conservative_result(
    state: TradingDecisionState, 
    reason: str,
    duration_ms: float = 0
) -> Dict[str, Any]:
    """Build conservative (wait/hold) result for error cases."""
    
    has_position = state.get("has_position", False)
    action = "signal_hold" if has_position else "signal_wait"
    
    return {
        "final_action": action,
        "final_confidence": 0.0,
        "final_direction": None,
        "final_leverage": None,
        "final_reason": reason,
        "final_key_support": state.get("key_support"),
        "final_key_resistance": state.get("key_resistance"),
        "is_valid": True,
        "executor_duration_ms": duration_ms,
        "executor_timestamp": _get_timestamp(),
        
        # VF fields (conservative defaults)
        "vf_verification_passed": False,
        "reasoning_consistent": True,
        "consistency_violations": [],
        
        "execution_result": {
            "action": action,
            "pair": state.get("pair"),
            "confidence_score": 0.0,
            "reason": reason,
            "source": "langgraph_executor_agent"
        }
    }


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()
