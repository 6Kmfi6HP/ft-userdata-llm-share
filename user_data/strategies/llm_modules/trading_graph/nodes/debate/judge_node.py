"""
Judge Agent Node for LangGraph.

The Impartial Arbiter - evaluates both arguments and renders final verdict.
"""

import logging
import re
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    TradingDecisionState,
    JudgeVerdict,
    Verdict,
    Direction,
)
from ....prompts.debate.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)
from ....lc_integration.llm_factory import LLMFactory
from ....lc_integration.adapters.context_adapter import ContextAdapter

logger = logging.getLogger(__name__)


def judge_node(state: TradingDecisionState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Judge Agent node - Impartial Arbiter.

    Evaluates Bull and Bear arguments, renders final verdict with calibrated confidence.

    Args:
        state: Current trading decision state with both arguments
        config: Node configuration (contains llm_config)

    Returns:
        Dict with judge_verdict update
    """
    pair = state.get("pair", "UNKNOWN")
    logger.debug(f"[JudgeAgent] Starting verdict for {pair}")

    try:
        # Get LLM config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Create LLM instance
        llm = LLMFactory.create_chat_model(llm_config, task_type="judge")

        # Get both arguments
        bull_argument = state.get("bull_argument")
        bear_argument = state.get("bear_argument")

        if not bull_argument or not bear_argument:
            return _create_error_result("Missing Bull or Bear argument")

        bull_arg_text = ContextAdapter.format_debate_argument(bull_argument)
        bear_arg_text = ContextAdapter.format_debate_argument(bear_argument)

        # Build analysis summary
        analysis_summary = ContextAdapter.format_all_reports(
            indicator_report=state.get("indicator_report"),
            trend_report=state.get("trend_report"),
            sentiment_report=state.get("sentiment_report"),
            pattern_report=state.get("pattern_report")
        )

        # Get consensus info
        consensus_dir = state.get("consensus_direction", Direction.NEUTRAL)
        consensus_conf = state.get("consensus_confidence", 50.0)

        # Build prompt
        prompt = build_judge_prompt(
            bull_argument=bull_arg_text,
            bear_argument=bear_arg_text,
            analysis_summary=analysis_summary,
            market_context=state.get("market_context", ""),
            pair=pair,
            current_price=state.get("current_price", 0.0),
            consensus_direction=consensus_dir.value if hasattr(consensus_dir, 'value') else str(consensus_dir),
            consensus_confidence=consensus_conf,
            has_position=state.get("has_position", False),
            position_side=state.get("position_side"),
            position_profit=state.get("position_profit_pct", 0.0)
        )

        # Call LLM
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response
        verdict = _parse_judge_response(response_text, consensus_conf)

        # Apply confidence calibration
        calibrated_conf = (verdict.confidence + consensus_conf) / 2
        verdict.confidence = calibrated_conf

        logger.info(
            f"[JudgeAgent] {pair} verdict: {verdict.verdict.value}, "
            f"confidence={verdict.confidence:.0f}%, winner={verdict.winning_argument}"
        )

        return {
            "judge_verdict": verdict
        }

    except Exception as e:
        logger.error(f"[JudgeAgent] Verdict failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result."""
    error_verdict = JudgeVerdict(
        verdict=Verdict.ABSTAIN,
        confidence=0.0,
        winning_argument=None,
        key_reasoning=f"Judge error: {error_msg}",
        recommended_action="signal_wait",
        leverage=None,
        risk_assessment="Unable to assess"
    )
    return {
        "judge_verdict": error_verdict,
        "errors": [f"JudgeAgent: {error_msg}"]
    }


def _parse_judge_response(response: str, original_confidence: float) -> JudgeVerdict:
    """
    Parse Judge agent response with enhanced fallback strategies.

    Args:
        response: LLM response text
        original_confidence: Original consensus confidence for calibration

    Returns:
        JudgeVerdict dataclass
    """
    result = {
        "verdict": Verdict.ABSTAIN,
        "confidence": 50.0,
        "winning_argument": None,
        "key_reasoning": "",
        "recommended_action": "signal_wait",
        "leverage": None,
        "risk_assessment": "medium"
    }

    if not response:
        return JudgeVerdict(**result)

    lines = response.strip().split('\n')
    current_section = None
    full_reasoning_lines = []  # Collect full reasoning
    key_reasoning_lines = []   # Collect key reasoning
    action_lines = []          # Collect action recommendations
    bull_score_info = ""       # Bull scoring info
    bear_score_info = ""       # Bear scoring info
    all_content_lines = []     # Collect all meaningful content for last-resort fallback

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Collect all non-empty lines for fallback
        if len(line) > 10 and not line.startswith('['):
            all_content_lines.append(line)

        # Detect section markers and handle inline content
        # Support multiple format variations: [标签], **标签**, 标签:
        section_detected = False
        inline_content = None

        # Flexible section detection with multiple patterns
        section_patterns = {
            'bull_score': ['[Bull 评分]', '[bull评分]', '**Bull 评分**', 'Bull评分:', 'Bull 评分：'],
            'bear_score': ['[Bear 评分]', '[bear评分]', '**Bear 评分**', 'Bear评分:', 'Bear 评分：'],
            'winner': ['[胜出方]', '**胜出方**', '胜出方:', '胜出方：', '[获胜方]', '获胜方:'],
            'verdict': ['[裁决]', '**裁决**', '裁决:', '裁决：', '[最终裁决]', '最终裁决:'],
            'confidence': ['[置信度]', '**置信度**', '置信度:', '置信度：', '[信心度]'],
            'leverage': ['[建议杠杆]', '**建议杠杆**', '建议杠杆:', '杠杆:', '[杠杆]'],
            'key_reasoning': ['[核心理由]', '**核心理由**', '核心理由:', '核心理由：', '[理由]', '[主要理由]', '**理由**'],
            'risk': ['[风险评估]', '**风险评估**', '风险评估:', '风险:', '[风险]', '风险等级:'],
            'action': ['[最终建议]', '**最终建议**', '最终建议:', '建议:', '[行动建议]', '**建议**'],
            'full_reasoning': ['[完整裁决理由]', '**完整裁决理由**', '完整理由:', '[完整理由]', '[详细理由]'],
        }

        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    current_section = section_name
                    section_detected = True
                    # Extract inline content after the pattern
                    idx = line.lower().find(pattern.lower())
                    if idx != -1:
                        inline_content = line[idx + len(pattern):].strip()
                        # Remove common trailing characters
                        inline_content = inline_content.lstrip(':：').strip()
                    break
            if section_detected:
                break

        # If inline content exists, use it as the line to parse
        if section_detected and inline_content:
            line = inline_content
        elif section_detected:
            continue  # Skip to next line if no inline content

        # Parse based on section
        if current_section == 'bull_score':
            if line and '总分' in line:
                bull_score_info = line
        
        elif current_section == 'bear_score':
            if line and '总分' in line:
                bear_score_info = line

        elif current_section == 'winner':
            if 'bull' in line.lower() or '多头' in line or '牛' in line:
                result['winning_argument'] = 'bull'
            elif 'bear' in line.lower() or '空头' in line or '熊' in line:
                result['winning_argument'] = 'bear'
            elif '平局' in line or '均势' in line or '无' in line:
                result['winning_argument'] = None

        elif current_section == 'verdict':
            line_upper = line.upper()
            if 'APPROVE' in line_upper or '批准' in line or '通过' in line:
                result['verdict'] = Verdict.APPROVE
            elif 'REJECT' in line_upper or '拒绝' in line or '否决' in line:
                result['verdict'] = Verdict.REJECT
            elif 'ABSTAIN' in line_upper or '弃权' in line or '观望' in line or '等待' in line:
                result['verdict'] = Verdict.ABSTAIN

        elif current_section == 'confidence':
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if numbers:
                result['confidence'] = min(100, max(0, float(numbers[0])))

        elif current_section == 'leverage':
            if 'N/A' not in line.upper() and '无' not in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    result['leverage'] = min(100, max(1, int(numbers[0])))

        elif current_section == 'key_reasoning':
            if line:  # Accumulate non-empty lines
                key_reasoning_lines.append(line)

        elif current_section == 'risk':
            line_lower = line.lower()
            if '高' in line_lower or 'high' in line_lower:
                result['risk_assessment'] = 'high'
            elif '低' in line_lower or 'low' in line_lower:
                result['risk_assessment'] = 'low'
            else:
                result['risk_assessment'] = 'medium'

        elif current_section == 'action':
            if line:
                action_lines.append(line)

        elif current_section == 'full_reasoning':
            if line:  # Accumulate non-empty lines
                full_reasoning_lines.append(line)

    # Set recommended_action from collected lines
    if action_lines:
        result['recommended_action'] = ' '.join(action_lines[:3])  # Limit to first 3 lines

    # ==================== Enhanced key_reasoning extraction ====================
    
    # Strategy 1: Use collected key_reasoning lines
    if key_reasoning_lines:
        result['key_reasoning'] = ' '.join(key_reasoning_lines)[:300]

    # Strategy 2: Use full_reasoning as fallback
    if not result['key_reasoning'] and full_reasoning_lines:
        result['key_reasoning'] = ' '.join(full_reasoning_lines)[:300]

    # Strategy 3: Build reasoning from action recommendation
    if not result['key_reasoning'] and action_lines:
        result['key_reasoning'] = f"建议: {' '.join(action_lines[:2])}"[:300]

    # Strategy 4: Build reasoning from scores comparison
    if not result['key_reasoning'] and (bull_score_info or bear_score_info):
        score_summary = []
        if bull_score_info:
            score_summary.append(f"Bull {bull_score_info}")
        if bear_score_info:
            score_summary.append(f"Bear {bear_score_info}")
        result['key_reasoning'] = "; ".join(score_summary)[:300]

    # Strategy 5: Extract from response using keyword patterns
    if not result['key_reasoning']:
        # Look for conclusion-like sentences
        conclusion_patterns = [
            r'综合[分判][析断][,，：:]([^。\n]+)',
            r'结论[：:]([^。\n]+)',
            r'因此[,，]([^。\n]+)',
            r'建议([^。\n]+)',
            r'总体来看[,，]([^。\n]+)',
            r'综上所述[,，]([^。\n]+)',
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, response)
            if match:
                result['key_reasoning'] = match.group(1).strip()[:300]
                break

    # Strategy 6: Use first meaningful content line as last resort
    if not result['key_reasoning'] and all_content_lines:
        # Filter for lines that look like reasoning (contain analysis keywords)
        reasoning_keywords = ['因为', '由于', '考虑', '鉴于', '基于', '认为', '判断', '分析', '趋势', '风险', '建议']
        for line in all_content_lines:
            if any(kw in line for kw in reasoning_keywords):
                result['key_reasoning'] = line[:300]
                break
        # If still empty, use first long content line
        if not result['key_reasoning'] and all_content_lines:
            result['key_reasoning'] = all_content_lines[0][:300]

    # Strategy 7: Generate informative default based on verdict and winner
    if not result['key_reasoning']:
        verdict_text = result['verdict'].value if hasattr(result['verdict'], 'value') else str(result['verdict'])
        winner = result['winning_argument']
        conf = result['confidence']
        risk = result['risk_assessment']
        
        if verdict_text == 'approve':
            if winner == 'bull':
                result['key_reasoning'] = f"Bull论点胜出，交易机会可行。置信度{conf:.0f}%，风险{risk}。"
            else:
                result['key_reasoning'] = f"辩论通过，批准交易。置信度{conf:.0f}%，风险评估{risk}。"
        elif verdict_text == 'reject':
            if winner == 'bear':
                result['key_reasoning'] = f"Bear论点胜出，风险过高不宜交易。置信度{conf:.0f}%，风险{risk}。"
            else:
                result['key_reasoning'] = f"辩论否决，当前不宜入场。置信度{conf:.0f}%，风险{risk}。"
        else:  # abstain
            result['key_reasoning'] = f"双方论点势均力敌，暂无明确信号。置信度{conf:.0f}%，建议观望等待更好时机。"

    return JudgeVerdict(**result)
