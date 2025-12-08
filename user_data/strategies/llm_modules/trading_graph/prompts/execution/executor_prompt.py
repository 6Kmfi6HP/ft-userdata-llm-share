"""
Verification-First (VF) Executor Prompt Templates.

Based on arXiv 2511.21734: "Asking LLMs to Verify First is Almost Free Lunch"

Key insights from the paper:
1. Verifying an answer is cognitively easier than generating the correct answer
2. VF triggers "reverse reasoning" that complements forward Chain-of-Thought
3. Even a random/trivial candidate answer provides scaffolding for better reasoning
4. VF with random answer consistently outperforms standard CoT

This module implements VF by:
1. Presenting the Judge's verdict as a "candidate answer" to verify
2. Asking the LLM to verify each dimension before generating final decision
3. LLM outputs qualitative judgments only (direction_strength, risk_level)
4. Code calculates precise numerical values (stop_loss, take_profit)
"""

from typing import Dict, Optional

# ============= VF System Prompt =============

EXECUTOR_VF_SYSTEM_PROMPT = """你是一个专业的加密货币交易执行专家。

<核心职责>
在做出最终决策前，你必须先验证前序 agents 的结论。这是 Verification-First (VF) 策略。

验证完成后，你需要提供定性判断（如方向强度、风险等级），而非精确的数值（止损/止盈价格将由系统自动计算）。
</核心职责>

<VF验证任务>
你需要验证以下4个维度:

1. 分析共识验证: 方向判断是否有足够的数据支撑？
2. 辩论结论验证: Judge 的裁决是否逻辑自洽？
3. 数据引用验证: Grounding 纠正后的数据是否被正确引用？
4. 风险识别验证: 是否存在被忽略的重大风险？

对每个维度给出 PASS 或 FAIL，并说明理由。
</VF验证任务>

<决策优先级>
1. Grounding 纠正后的数据 > Agent 的原始声明
2. 风控规则是硬性约束，不可违反
3. 当信息冲突时，以更保守的方向决策
4. 如果任何验证维度 FAIL，应降低置信度或选择保守动作
</决策优先级>

<风控规则>
{risk_rules}
</风控规则>

<定性判断说明>
direction_strength (方向强度):
- strong: 多个指标 + 趋势 + 形态一致看多/看空，信号明确
- moderate: 部分信号一致，但有一些噪音或矛盾
- weak: 信号混合，方向不明确，不适合入场

risk_level (风险等级):
- high: 高波动、临近支撑/阻力、资金费率极端、流动性差
- medium: 正常市场条件
- low: 低波动、趋势明确、成交量稳定、良好流动性
</定性判断说明>

<输出格式>
必须严格按照以下格式输出:

[验证结果]
分析共识验证: PASS/FAIL (理由)
辩论结论验证: PASS/FAIL (理由)
数据引用验证: PASS/FAIL (理由)
风险识别验证: PASS/FAIL (理由)

[验证后决策]
action: <操作类型>
confidence: <置信度 0-100>
direction: <方向 LONG/SHORT/NEUTRAL>
direction_strength: <strong/moderate/weak>
risk_level: <high/medium/low>

[调整参数] (adjust_position 时必填)
adjustment_pct: <调整百分比 +20~+50 或 -30~-70>
adjustment_type: <scale_in/partial_exit>

[决策理由]
<你的完整推理过程>

[关键因素]
- <因素1>
- <因素2>
- <因素3>
</输出格式>

<重要提示>
1. 你不需要计算止损/止盈价格，系统会根据你的 direction_strength 和 risk_level 自动计算
2. 如果任何验证维度 FAIL，应在决策理由中说明如何调整
3. 高置信度入场必须伴随 strong 或 moderate 的 direction_strength
</重要提示>
"""

# ============= VF User Prompt Builder =============

def build_vf_executor_user_prompt(
    candidate_answer: Dict,
    consensus_summary: str,
    debate_summary: str,
    grounding_summary: str,
    corrected_context: str,
    current_price: float,
    key_support: Optional[float] = None,
    key_resistance: Optional[float] = None,
    has_position: bool = False,
    position_info: Optional[str] = None
) -> str:
    """
    Build VF (Verification-First) user prompt for Executor Agent.
    
    Following arXiv 2511.21734:
    - Present Judge's verdict as "candidate answer"
    - Ask LLM to verify first, then generate decision
    
    Args:
        candidate_answer: Dict with direction, confidence, verdict from Judge
        consensus_summary: Summary of analysis consensus
        debate_summary: Summary of debate results
        grounding_summary: Summary of grounding verification
        corrected_context: Corrected data context from grounding
        current_price: Current market price
        key_support: Key support level
        key_resistance: Key resistance level
        has_position: Whether there's an existing position
        position_info: Position information string
        
    Returns:
        Formatted VF user prompt string
    """
    # Build candidate answer section
    candidate_section = f"""=== 待验证结论 (Candidate Answer) ===
来自前序 agents 的初步结论:
- 方向: {candidate_answer.get('direction', 'unknown')}
- 置信度: {candidate_answer.get('confidence', 0):.0f}%
- Judge 裁决: {candidate_answer.get('verdict', 'unknown')}
- 胜方: {candidate_answer.get('winning_argument', 'N/A')}

请先验证以上结论是否正确，然后给出你的最终决策。
"""

    # Build action options based on position status
    if has_position:
        action_options = """
[可选操作]
- HOLD: 继续持有，不做调整
- EXIT: 平仓离场
- SCALE_IN: 加仓 (需指定 adjustment_pct: +20% ~ +50%, adjustment_type: scale_in)
- PARTIAL_EXIT: 部分平仓 (需指定 adjustment_pct: -30% ~ -70%, adjustment_type: partial_exit)
"""
    else:
        action_options = """
[可选操作]
- ENTRY_LONG: 开多仓 (系统将根据你的 direction_strength 和 risk_level 计算止损止盈)
- ENTRY_SHORT: 开空仓 (系统将根据你的 direction_strength 和 risk_level 计算止损止盈)
- WAIT: 等待更好的入场机会
"""

    # Build position status section
    if has_position and position_info:
        position_section = f"已有持仓:\n{position_info}"
    else:
        position_section = "当前无持仓"

    return f"""{candidate_section}

=== 当前市场 ===
价格: {current_price}
支撑位: {key_support or 'N/A'}
阻力位: {key_resistance or 'N/A'}

=== 分析共识 (4 个 Agent 加权结果) ===
{consensus_summary}

=== 辩论结果 (Bull vs Bear) ===
{debate_summary}

=== Grounding 验证结果 (已纠正) ===
{grounding_summary}

{corrected_context}

=== 持仓状态 ===
{position_section}

{action_options}

请按照以下步骤:
1. 首先验证 "待验证结论" 中的4个维度
2. 基于验证结果，给出你的最终决策
3. 如果验证发现问题，在决策中体现调整

注意: 
- 必须基于 Grounding 纠正后的数据，而非原始声明
- 只需提供定性判断 (direction_strength, risk_level)，止损止盈由系统计算
"""


def build_candidate_answer(state_context: Dict) -> Dict:
    """
    Extract candidate answer from state context for VF prompting.
    
    Args:
        state_context: State dictionary containing Judge verdict and consensus
        
    Returns:
        Dict with direction, confidence, verdict, winning_argument
    """
    has_position = state_context.get("has_position", False)
    
    if has_position:
        judge_verdict = state_context.get("position_judge_verdict")
    else:
        judge_verdict = state_context.get("judge_verdict")
    
    consensus_direction = state_context.get("consensus_direction")
    consensus_confidence = state_context.get("consensus_confidence", 0)
    
    # Extract from judge verdict
    if judge_verdict:
        verdict = getattr(judge_verdict, "verdict", None)
        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)
        winning = getattr(judge_verdict, "winning_argument", None)
        judge_confidence = getattr(judge_verdict, "confidence", 0)
    else:
        verdict_str = "unknown"
        winning = None
        judge_confidence = 0
    
    # Extract direction
    if consensus_direction:
        direction = consensus_direction.value if hasattr(consensus_direction, "value") else str(consensus_direction)
    else:
        direction = "neutral"
    
    return {
        "direction": direction,
        "confidence": max(consensus_confidence, judge_confidence),
        "verdict": verdict_str,
        "winning_argument": winning or "N/A"
    }


# ============= Original Prompt (fallback) =============

EXECUTOR_SYSTEM_PROMPT = """你是一个专业的加密货币交易执行专家。你的职责是基于多个分析 Agent 的结果做出最终交易决策。

<核心职责>
1. 综合评估所有分析 agents 的结论
2. 基于 Grounding 纠正后的真实数据做决策
3. 严格遵守风控规则
4. 给出明确、可执行的交易指令
</核心职责>

<决策优先级>
1. Grounding 纠正后的数据 > Agent 的原始声明
2. 风控规则是硬性约束，不可违反
3. 当信息冲突时，以更保守的方向决策
</决策优先级>

<风控规则>
{risk_rules}
</风控规则>

<输出格式>
必须严格按照以下格式输出:

[决策]
action: <操作类型>
confidence: <置信度 0-100>
leverage: <杠杆倍数>
direction: <方向 LONG/SHORT/NEUTRAL>

[风险管理] (入场时必填)
stop_loss_price: <止损价格>
take_profit_price: <止盈价格>
risk_reward_ratio: <风险回报比>

[调整参数] (adjust_position 时必填)
adjustment_pct: <调整百分比>
adjustment_type: <scale_in/partial_exit>

[决策理由]
<你的完整推理过程>

[关键因素]
- <因素1>
- <因素2>
- <因素3>

[风险评估]
<对当前决策的风险评估>
</输出格式>
"""

# ============= Risk Rules Template =============

DEFAULT_RISK_RULES = """
- 单笔最大风险: 账户净值的 2%
- 最小置信度阈值: 60% (低于此值不开仓)
- 最大杠杆: 50x (根据置信度动态调整)
- 止损设置: 必须在入场价格的 1-5% 范围内
- 最小风险回报比: 1.5:1
- 同时最大持仓数: 3
"""


def build_risk_rules_section(risk_config: Optional[Dict] = None) -> str:
    """
    Build risk rules section from config.
    
    Args:
        risk_config: Risk configuration dictionary
        
    Returns:
        Formatted risk rules string
    """
    if not risk_config:
        return DEFAULT_RISK_RULES
        
    rules = []
    
    if "max_risk_per_trade" in risk_config:
        rules.append(f"- 单笔最大风险: 账户净值的 {risk_config['max_risk_per_trade']*100:.0f}%")
    
    if "min_confidence" in risk_config:
        rules.append(f"- 最小置信度阈值: {risk_config['min_confidence']}% (低于此值不开仓)")
    
    if "max_leverage" in risk_config:
        rules.append(f"- 最大杠杆: {risk_config['max_leverage']}x (根据置信度动态调整)")
    
    if "max_stop_loss_pct" in risk_config:
        rules.append(f"- 止损设置: 必须在入场价格的 1-{risk_config['max_stop_loss_pct']}% 范围内")
    
    if "min_risk_reward" in risk_config:
        rules.append(f"- 最小风险回报比: {risk_config['min_risk_reward']}:1")
    
    if "max_open_trades" in risk_config:
        rules.append(f"- 同时最大持仓数: {risk_config['max_open_trades']}")
    
    return "\n".join(rules) if rules else DEFAULT_RISK_RULES


# ============= Original User Prompt Builder (preserved) =============

def build_executor_user_prompt(
    consensus_summary: str,
    debate_summary: str,
    grounding_summary: str,
    corrected_context: str,
    current_price: float,
    key_support: Optional[float] = None,
    key_resistance: Optional[float] = None,
    has_position: bool = False,
    position_info: Optional[str] = None
) -> str:
    """
    Build the user prompt for Executor Agent (original version).
    
    Args:
        consensus_summary: Summary of analysis consensus
        debate_summary: Summary of debate results
        grounding_summary: Summary of grounding verification
        corrected_context: Corrected data context from grounding
        current_price: Current market price
        key_support: Key support level
        key_resistance: Key resistance level
        has_position: Whether there's an existing position
        position_info: Position information string (if has_position)
        
    Returns:
        Formatted user prompt string
    """
    # Build action options based on position status
    if has_position:
        action_options = """
[可选操作]
- HOLD: 继续持有，不做调整
- EXIT: 平仓离场
- SCALE_IN: 加仓 (需指定 adjustment_pct: +20% ~ +50%)
- PARTIAL_EXIT: 部分平仓 (需指定 adjustment_pct: -30% ~ -70%)
"""
    else:
        action_options = """
[可选操作]
- ENTRY_LONG: 开多仓 (需设置止损止盈)
- ENTRY_SHORT: 开空仓 (需设置止损止盈)
- WAIT: 等待更好的入场机会
"""

    # Build position status section
    if has_position and position_info:
        position_section = f"已有持仓:\n{position_info}"
    else:
        position_section = "当前无持仓"

    return f"""=== 当前市场 ===
价格: {current_price}
支撑位: {key_support or 'N/A'}
阻力位: {key_resistance or 'N/A'}

=== 分析共识 (4 个 Agent 加权结果) ===
{consensus_summary}

=== 辩论结果 (Bull vs Bear) ===
{debate_summary}

=== Grounding 验证结果 (已纠正) ===
{grounding_summary}

{corrected_context}

=== 持仓状态 ===
{position_section}

{action_options}

请基于以上信息，做出你的最终交易决策。
注意: 必须基于 Grounding 纠正后的数据，而非原始声明。
"""


def build_consensus_summary(
    consensus_direction: Optional[str],
    consensus_confidence: float,
    weighted_scores: Optional[Dict] = None
) -> str:
    """
    Build a summary of analysis consensus.
    
    Args:
        consensus_direction: Direction from analysis (long/short/neutral)
        consensus_confidence: Confidence percentage
        weighted_scores: Dict of weighted scores by direction
        
    Returns:
        Formatted consensus summary string
    """
    direction_map = {
        "long": "看多",
        "short": "看空",
        "neutral": "中性"
    }
    
    direction_cn = direction_map.get(consensus_direction, "未知")
    
    summary = f"方向: {direction_cn} | 置信度: {consensus_confidence:.0f}%"
    
    if weighted_scores:
        scores_str = " | ".join([
            f"{k}: {v:.1%}" for k, v in weighted_scores.items()
        ])
        summary += f"\n加权分数: {scores_str}"
    
    return summary


def build_debate_summary(
    bull_argument,
    bear_argument, 
    judge_verdict
) -> str:
    """
    Build a summary of the debate results.
    
    Args:
        bull_argument: Bull's debate argument
        bear_argument: Bear's debate argument
        judge_verdict: Judge's verdict
        
    Returns:
        Formatted debate summary string
    """
    lines = []
    
    # Bull summary
    if bull_argument:
        bull_confidence = getattr(bull_argument, 'confidence', 0)
        bull_action = getattr(bull_argument, 'recommended_action', 'N/A')
        lines.append(f"🐂 Bull: {bull_action} (置信度: {bull_confidence:.0f}%)")
        
        key_points = getattr(bull_argument, 'key_points', [])
        if key_points:
            lines.append(f"   要点: {'; '.join(key_points[:2])}")
    
    # Bear summary
    if bear_argument:
        bear_confidence = getattr(bear_argument, 'confidence', 0)
        bear_action = getattr(bear_argument, 'recommended_action', 'N/A')
        lines.append(f"🐻 Bear: {bear_action} (置信度: {bear_confidence:.0f}%)")
        
        risk_factors = getattr(bear_argument, 'risk_factors', [])
        if risk_factors:
            lines.append(f"   风险: {'; '.join(risk_factors[:2])}")
    
    # Judge summary
    if judge_verdict:
        verdict = getattr(judge_verdict, 'verdict', None)
        verdict_str = verdict.value if hasattr(verdict, 'value') else str(verdict)
        confidence = getattr(judge_verdict, 'confidence', 0)
        winner = getattr(judge_verdict, 'winning_argument', 'N/A')
        
        lines.append(f"⚖️ Judge: {verdict_str.upper()} (置信度: {confidence:.0f}%)")
        lines.append(f"   胜方: {winner}")
    
    return "\n".join(lines) if lines else "无辩论数据"


def build_position_info(
    position_side: str,
    position_profit_pct: float,
    entry_price: Optional[float] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
    drawdown: Optional[float] = None,
    hold_count: Optional[int] = None
) -> str:
    """
    Build position information string.
    
    Args:
        position_side: "long" or "short"
        position_profit_pct: Current P&L percentage
        entry_price: Entry price
        mfe: Maximum Favorable Excursion
        mae: Maximum Adverse Excursion
        drawdown: Drawdown from peak
        hold_count: Consecutive hold count
        
    Returns:
        Formatted position info string
    """
    side_cn = "多" if position_side == "long" else "空"
    
    lines = [
        f"方向: {side_cn}仓",
        f"当前盈亏: {position_profit_pct:+.2f}%"
    ]
    
    if entry_price:
        lines.append(f"入场价: {entry_price}")
    
    if mfe is not None:
        lines.append(f"MFE (最大浮盈): {mfe:.2f}%")
    
    if mae is not None:
        lines.append(f"MAE (最大浮亏): {mae:.2f}%")
    
    if drawdown is not None:
        lines.append(f"MFE回撤: {drawdown:.2f}%")
    
    if hold_count is not None:
        lines.append(f"连续HOLD次数: {hold_count}")
    
    return "\n".join(lines)
