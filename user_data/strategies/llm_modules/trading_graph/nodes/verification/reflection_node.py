"""
Reflection Node for LangGraph Trading System.

Replaces the code-based Grounding Verification with LLM-based reflection.
Uses Chain-of-Verification (CoVe) pattern to validate trading analysis.

Key improvements over Grounding Node:
1. LLM understands context better than regex patterns
2. Focuses on trading logic rather than numerical matching
3. Verifies entry timing, direction correctness, and risk assessment
4. Provides actionable corrections in natural language

Reference: 
- Meta AI Chain-of-Verification (CoVe) paper
- arXiv 2511.21734 "Asking LLMs to Verify First"
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from langchain_core.runnables import RunnableConfig

from ...state import TradingDecisionState, Direction

logger = logging.getLogger(__name__)


# ============= Data Classes =============

@dataclass
class ReflectionResult:
    """Result of reflection verification."""
    
    # Overall assessment
    is_analysis_sound: bool = True
    direction_correct: bool = True
    entry_timing_appropriate: bool = True
    risk_properly_assessed: bool = True
    
    # Scores (0-100)
    overall_confidence: float = 0.0
    direction_confidence: float = 0.0
    timing_confidence: float = 0.0
    
    # Issues found
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Corrected direction if needed
    suggested_direction: Optional[Direction] = None
    suggested_action: Optional[str] = None
    
    # Reasoning
    reflection_reasoning: str = ""
    key_verification_points: List[str] = field(default_factory=list)
    
    # Whether to proceed
    should_proceed: bool = True
    confidence_adjustment: float = 0.0  # Penalty or boost to confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_analysis_sound": self.is_analysis_sound,
            "direction_correct": self.direction_correct,
            "entry_timing_appropriate": self.entry_timing_appropriate,
            "risk_properly_assessed": self.risk_properly_assessed,
            "overall_confidence": self.overall_confidence,
            "direction_confidence": self.direction_confidence,
            "timing_confidence": self.timing_confidence,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "suggested_direction": self.suggested_direction.value if self.suggested_direction else None,
            "suggested_action": self.suggested_action,
            "reflection_reasoning": self.reflection_reasoning,
            "key_verification_points": self.key_verification_points,
            "should_proceed": self.should_proceed,
            "confidence_adjustment": self.confidence_adjustment,
        }


# ============= System Prompt =============

REFLECTION_SYSTEM_PROMPT = """你是一位资深的加密货币交易分析验证专家。

你的任务是验证前序分析代理的交易建议。你的角色是提供补充视角和置信度评估，而非阻止交易。

## 重要原则

1. **支持性验证**: 你的目标是帮助完善决策，而非否定决策
2. **置信度调整**: 如有疑虑，通过降低置信度分数来表达，而非建议停止
3. **只在极端情况下建议停止**: 只有当分析存在明显致命错误时（如方向完全相反）才建议 should_proceed: FALSE
4. **问题分类要谨慎**: 
   - critical_issues: 只用于致命错误（如做多时趋势明显向下、追高杀跌等）
   - warnings: 用于需要注意但不影响交易的问题
   - 大多数问题应该是 warnings 而非 critical_issues

## 验证维度

### 1. 方向验证 (Direction Verification)
- 推荐的多空方向是否与市场趋势大致一致？（不需要完美匹配）
- 只有当方向与趋势明显相反时才标记 direction_correct: FALSE

### 2. 入场时机验证 (Entry Timing Verification)
- 当前入场点是否合理？（不需要是最佳入场点）
- 除非明显在追高/追低，否则标记 entry_timing_good: TRUE

### 3. 风险评估验证 (Risk Assessment Verification)
- 主要风险是否被识别？（不需要识别所有风险）
- 除非完全忽略主要风险，否则标记 risk_assessed: TRUE

## 输出格式

[验证结论]
direction_correct: TRUE/FALSE (默认倾向TRUE，除非明显错误)
entry_timing_good: TRUE/FALSE (默认倾向TRUE，除非明显追涨杀跌)
risk_assessed: TRUE/FALSE (默认倾向TRUE，除非完全忽略风险)
should_proceed: TRUE/FALSE (默认TRUE，只有致命错误时才FALSE)

[置信度评分]
overall_confidence: 0-100 (反映你对分析的信心程度)
direction_confidence: 0-100 (方向判断的信心)
timing_confidence: 0-100 (时机判断的信心)

[发现的问题]
critical_issues:
- (只列出致命问题，如没有则写"无")

warnings:
- (列出需要注意的问题)

[改进建议]
suggestions:
- (列出建议)

[建议修正]
suggested_direction: LONG/SHORT/NEUTRAL (只在方向明显错误时填写)
suggested_action: entry_long/entry_short/wait (只在需要完全改变决策时填写)

[验证理由]
(简要解释你的验证逻辑)"""


def build_reflection_prompt(state: TradingDecisionState, is_position: bool = False) -> str:
    """Build user prompt for reflection verification."""
    
    pair = state.get("pair", "UNKNOWN")
    current_price = state.get("current_price", 0)
    market_context = state.get("market_context", "")
    
    # Get debate results
    if is_position:
        bull_arg = state.get("position_bull_argument")
        bear_arg = state.get("position_bear_argument")
        judge = state.get("position_judge_verdict")
        position_side = state.get("position_side", "unknown")
        position_profit = state.get("position_profit_pct", 0)
        position_info = f"\n当前持仓: {position_side} 方向, 盈亏 {position_profit:.2f}%"
    else:
        bull_arg = state.get("bull_argument")
        bear_arg = state.get("bear_argument")
        judge = state.get("judge_verdict")
        position_info = ""
    
    consensus_direction = state.get("consensus_direction")
    consensus_confidence = state.get("consensus_confidence", 0)
    key_support = state.get("key_support")
    key_resistance = state.get("key_resistance")
    
    # Format bull argument
    bull_summary = "无"
    if bull_arg:
        bull_points = getattr(bull_arg, 'key_points', []) or []
        bull_summary = f"""
方向: {getattr(bull_arg, 'position', 'unknown')}
置信度: {getattr(bull_arg, 'confidence', 0):.0f}%
核心论点: {', '.join(bull_points[:3]) if bull_points else '无'}
推荐行动: {getattr(bull_arg, 'recommended_action', '')}"""
    
    # Format bear argument
    bear_summary = "无"
    if bear_arg:
        bear_points = getattr(bear_arg, 'key_points', []) or []
        risk_factors = getattr(bear_arg, 'risk_factors', []) or []
        bear_summary = f"""
方向: {getattr(bear_arg, 'position', 'unknown')}
置信度: {getattr(bear_arg, 'confidence', 0):.0f}%
核心论点: {', '.join(bear_points[:3]) if bear_points else '无'}
风险因素: {', '.join(risk_factors[:3]) if risk_factors else '无'}"""
    
    # Format judge verdict
    judge_summary = "无"
    if judge:
        judge_summary = f"""
判决: {getattr(judge, 'verdict', 'unknown')}
置信度: {getattr(judge, 'confidence', 0):.0f}%
获胜方: {getattr(judge, 'winning_argument', '无')}
推荐行动: {getattr(judge, 'recommended_action', '')}
核心理由: {getattr(judge, 'key_reasoning', '')[:200]}"""
    
    prompt = f"""请验证以下交易分析的正确性:

## 交易对信息
- 交易对: {pair}
- 当前价格: {current_price}{position_info}
- 关键支撑位: {key_support if key_support else '未识别'}
- 关键阻力位: {key_resistance if key_resistance else '未识别'}

## 分析共识
- 方向: {consensus_direction.value if consensus_direction else 'neutral'}
- 置信度: {consensus_confidence:.0f}%

## Bull Agent 论点
{bull_summary}

## Bear Agent 论点
{bear_summary}

## Judge 判决
{judge_summary}

## 市场数据摘要
{market_context[:1500] if market_context else '无市场数据'}

---

请独立验证上述分析，重点关注:
1. 方向判断是否正确？
2. 当前是入场的好时机吗？
3. 风险是否被充分识别？
4. 整体分析是否可靠？

按照指定格式输出验证结果。"""
    
    return prompt


# ============= Main Node Function =============

def reflection_node(
    state: TradingDecisionState,
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Reflection Verification Node - LLM-based analysis validation.
    
    Uses Chain-of-Verification (CoVe) to validate trading analysis.
    Focuses on direction correctness, entry timing, and risk assessment.
    
    Args:
        state: Current trading decision state with debate arguments
        config: Node configuration
        
    Returns:
        Dict with reflection_result and potentially modified confidence
    """
    start_time = time.time()
    pair = state.get("pair", "UNKNOWN")
    
    logger.info(f"[ReflectionNode] Starting reflection verification for {pair}")
    
    try:
        # Create LLM
        llm_config = state.get("llm_config", {}) or {}
        llm = _create_llm(llm_config)
        
        # Build prompt
        is_position = state.get("has_position", False) and state.get("position_judge_verdict") is not None
        user_prompt = build_reflection_prompt(state, is_position)
        
        # Call LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse response
        result = _parse_reflection_response(response_text)
        
        # Calculate execution time
        duration_ms = (time.time() - start_time) * 1000
        
        # Build summary
        summary = _build_reflection_summary(result)
        
        # Log result
        if result.should_proceed:
            logger.info(f"[ReflectionNode] {pair}: VERIFIED ✓ - {summary}")
        else:
            logger.warning(f"[ReflectionNode] {pair}: ISSUES FOUND - {summary}")
        
        # Build state update
        if is_position:
            return {
                "position_reflection_result": result,
                "position_reflection_summary": summary,
                "position_reflection_reasoning": result.reflection_reasoning,
                "reflection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reflection_duration_ms": duration_ms,
                "warnings": [f"Reflection: {w}" for w in result.warnings] if result.warnings else [],
            }
        else:
            return {
                "reflection_result": result,
                "reflection_summary": summary,
                "reflection_reasoning": result.reflection_reasoning,
                "reflection_should_proceed": result.should_proceed,
                "reflection_confidence_adjustment": result.confidence_adjustment,
                "reflection_suggested_direction": result.suggested_direction,
                "reflection_suggested_action": result.suggested_action,
                "reflection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reflection_duration_ms": duration_ms,
                "warnings": [f"Reflection: {w}" for w in result.warnings] if result.warnings else [],
            }
        
    except Exception as e:
        logger.error(f"[ReflectionNode] Error during reflection: {e}")
        duration_ms = (time.time() - start_time) * 1000
        
        # Return default (proceed anyway) on error
        default_result = ReflectionResult(
            is_analysis_sound=True,
            should_proceed=True,
            reflection_reasoning=f"Reflection failed: {str(e)}, proceeding anyway",
        )
        
        return {
            "reflection_result": default_result,
            "reflection_summary": "Reflection skipped due to error",
            "reflection_should_proceed": True,
            "reflection_duration_ms": duration_ms,
            "errors": [f"Reflection error: {str(e)}"],
        }


def position_reflection_node(
    state: TradingDecisionState,
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Position path reflection node.
    
    Validates position management analysis (hold/exit/scale decisions).
    """
    start_time = time.time()
    pair = state.get("pair", "UNKNOWN")
    
    logger.info(f"[PositionReflectionNode] Starting reflection for {pair}")
    
    try:
        llm_config = state.get("llm_config", {}) or {}
        llm = _create_llm(llm_config)
        
        user_prompt = build_reflection_prompt(state, is_position=True)
        
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result = _parse_reflection_response(response.content)
        
        duration_ms = (time.time() - start_time) * 1000
        summary = _build_reflection_summary(result)
        
        if result.should_proceed:
            logger.info(f"[PositionReflectionNode] {pair}: VERIFIED ✓ - {summary}")
        else:
            logger.warning(f"[PositionReflectionNode] {pair}: ISSUES FOUND - {summary}")
        
        return {
            "position_reflection_result": result,
            "position_reflection_summary": summary,
            "position_reflection_should_proceed": result.should_proceed,
            "position_reflection_confidence_adjustment": result.confidence_adjustment,
            "reflection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reflection_duration_ms": duration_ms,
            "warnings": [f"Position Reflection: {w}" for w in result.warnings] if result.warnings else [],
        }
        
    except Exception as e:
        logger.error(f"[PositionReflectionNode] Error: {e}")
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "position_reflection_result": ReflectionResult(should_proceed=True),
            "position_reflection_summary": "Reflection skipped",
            "position_reflection_should_proceed": True,
            "reflection_duration_ms": duration_ms,
            "errors": [f"Position reflection error: {str(e)}"],
        }


# ============= Helper Functions =============

def _create_llm(llm_config: Dict[str, Any]):
    """Create LLM instance from config."""
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        model=llm_config.get("model", "gpt-4o-mini"),
        temperature=0.3,  # Lower temperature for more consistent verification
        base_url=llm_config.get("api_base"),
        api_key=llm_config.get("api_key", "not-needed"),
        max_tokens=1500,
    )


def _parse_reflection_response(response_text: str) -> ReflectionResult:
    """Parse LLM reflection response into ReflectionResult."""
    import re
    
    result = ReflectionResult()
    
    # Parse boolean verdicts
    if re.search(r'direction_correct:\s*(TRUE|是|正确)', response_text, re.IGNORECASE):
        result.direction_correct = True
    elif re.search(r'direction_correct:\s*(FALSE|否|错误)', response_text, re.IGNORECASE):
        result.direction_correct = False
    
    if re.search(r'entry_timing_good:\s*(TRUE|是|好)', response_text, re.IGNORECASE):
        result.entry_timing_appropriate = True
    elif re.search(r'entry_timing_good:\s*(FALSE|否|差)', response_text, re.IGNORECASE):
        result.entry_timing_appropriate = False
    
    if re.search(r'risk_assessed:\s*(TRUE|是)', response_text, re.IGNORECASE):
        result.risk_properly_assessed = True
    elif re.search(r'risk_assessed:\s*(FALSE|否)', response_text, re.IGNORECASE):
        result.risk_properly_assessed = False
    
    if re.search(r'should_proceed:\s*(TRUE|是|继续)', response_text, re.IGNORECASE):
        result.should_proceed = True
    elif re.search(r'should_proceed:\s*(FALSE|否|不|停止)', response_text, re.IGNORECASE):
        result.should_proceed = False
    
    # Parse confidence scores
    overall_match = re.search(r'overall_confidence:\s*(\d+)', response_text)
    if overall_match:
        result.overall_confidence = float(overall_match.group(1))
    
    direction_match = re.search(r'direction_confidence:\s*(\d+)', response_text)
    if direction_match:
        result.direction_confidence = float(direction_match.group(1))
    
    timing_match = re.search(r'timing_confidence:\s*(\d+)', response_text)
    if timing_match:
        result.timing_confidence = float(timing_match.group(1))
    
    # Parse critical issues
    critical_section = re.search(r'critical_issues:\s*\n(.*?)(?=\n\n|\nwarnings:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if critical_section:
        issues = re.findall(r'-\s*(.+)', critical_section.group(1))
        result.critical_issues = [i.strip() for i in issues if i.strip() and i.strip() != '无']
    
    # Parse warnings
    warnings_section = re.search(r'warnings:\s*\n(.*?)(?=\n\n|\[改进建议\]|$)', response_text, re.DOTALL | re.IGNORECASE)
    if warnings_section:
        warnings = re.findall(r'-\s*(.+)', warnings_section.group(1))
        result.warnings = [w.strip() for w in warnings if w.strip() and w.strip() != '无']
    
    # Parse suggestions
    suggestions_section = re.search(r'suggestions:\s*\n(.*?)(?=\n\n|\[建议修正\]|$)', response_text, re.DOTALL | re.IGNORECASE)
    if suggestions_section:
        suggestions = re.findall(r'-\s*(.+)', suggestions_section.group(1))
        result.suggestions = [s.strip() for s in suggestions if s.strip()]
    
    # Parse suggested direction
    direction_match = re.search(r'suggested_direction:\s*(LONG|SHORT|NEUTRAL|多|空|中性)', response_text, re.IGNORECASE)
    if direction_match:
        dir_str = direction_match.group(1).upper()
        if dir_str in ['LONG', '多']:
            result.suggested_direction = Direction.LONG
        elif dir_str in ['SHORT', '空']:
            result.suggested_direction = Direction.SHORT
        else:
            result.suggested_direction = Direction.NEUTRAL
    
    # Parse suggested action
    action_match = re.search(r'suggested_action:\s*(\w+)', response_text, re.IGNORECASE)
    if action_match:
        result.suggested_action = action_match.group(1).lower()
    
    # Parse verification reasoning
    reasoning_section = re.search(r'\[验证理由\]\s*\n(.+?)(?=$)', response_text, re.DOTALL)
    if reasoning_section:
        result.reflection_reasoning = reasoning_section.group(1).strip()[:500]
    
    # Determine overall soundness (more lenient)
    # Analysis is sound if direction is correct OR confidence is high enough
    result.is_analysis_sound = (
        result.direction_correct or
        result.direction_confidence >= 70  # High confidence can override
    )
    
    # Calculate confidence adjustment (gentler penalties)
    if not result.direction_correct:
        result.confidence_adjustment -= 15  # Reduced from 20
    if not result.entry_timing_appropriate:
        result.confidence_adjustment -= 10  # Reduced from 15
    if not result.risk_properly_assessed:
        result.confidence_adjustment -= 5   # Reduced from 10
    if result.critical_issues:
        result.confidence_adjustment -= len(result.critical_issues) * 3  # Reduced from 5
    if result.warnings:
        result.confidence_adjustment -= len(result.warnings) * 1  # Reduced from 2
    
    # IMPORTANT: Only override should_proceed in extreme cases
    # Trust the LLM's judgment from the response parsing
    # Only force stop if:
    # 1. Direction is wrong AND confidence is very low
    # 2. Multiple critical issues AND low overall confidence
    if not result.direction_correct and result.direction_confidence < 50:
        result.should_proceed = False
    elif len(result.critical_issues) >= 3 and result.overall_confidence < 50:
        result.should_proceed = False
    # Otherwise, trust the LLM's should_proceed from the response
    
    return result


def _build_reflection_summary(result: ReflectionResult) -> str:
    """Build a concise summary of reflection result."""
    parts = []
    
    if result.is_analysis_sound:
        parts.append("✓ Analysis sound")
    else:
        parts.append("⚠ Issues found")
    
    parts.append(f"Dir:{result.direction_confidence:.0f}%")
    parts.append(f"Timing:{result.timing_confidence:.0f}%")
    
    if result.critical_issues:
        parts.append(f"Critical:{len(result.critical_issues)}")
    if result.warnings:
        parts.append(f"Warn:{len(result.warnings)}")
    
    if result.suggested_action:
        parts.append(f"Suggest:{result.suggested_action}")
    
    return " | ".join(parts)
