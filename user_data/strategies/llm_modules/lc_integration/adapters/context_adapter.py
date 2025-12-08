"""
Context Adapter for bridging existing ContextBuilder with LangChain.

Provides utilities to:
- Convert ContextBuilder output to LangChain message format
- Format agent reports for debate input
- Build structured prompts for LangChain nodes
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class ContextAdapter:
    """
    Adapter for converting existing context formats to LangChain compatible formats.
    """

    @staticmethod
    def build_analysis_messages(
        system_prompt: str,
        market_context: str,
        additional_context: Optional[str] = None
    ) -> List[Any]:
        """
        Build LangChain messages for analysis nodes.

        Args:
            system_prompt: System role prompt (agent expertise)
            market_context: Market context from ContextBuilder
            additional_context: Any additional context to include

        Returns:
            List of LangChain messages
        """
        messages = [
            SystemMessage(content=system_prompt),
        ]

        user_content = f"请分析以下市场数据:\n\n{market_context}"

        if additional_context:
            user_content += f"\n\n{additional_context}"

        messages.append(HumanMessage(content=user_content))

        return messages

    @staticmethod
    def build_debate_messages(
        system_prompt: str,
        analysis_summary: str,
        opponent_argument: Optional[str] = None,
        debate_context: Optional[str] = None
    ) -> List[Any]:
        """
        Build LangChain messages for debate nodes.

        Args:
            system_prompt: System role prompt (Bull/Bear/Judge)
            analysis_summary: Formatted analysis from all agents
            opponent_argument: Previous argument to respond to (for Bear)
            debate_context: Additional debate context

        Returns:
            List of LangChain messages
        """
        messages = [
            SystemMessage(content=system_prompt),
        ]

        user_content = f"基于以下分析结果:\n\n{analysis_summary}"

        if opponent_argument:
            user_content += f"\n\n对方观点:\n{opponent_argument}"

        if debate_context:
            user_content += f"\n\n{debate_context}"

        messages.append(HumanMessage(content=user_content))

        return messages

    @staticmethod
    def format_agent_report(report: Any) -> str:
        """
        Format an AgentReport to readable text for debate input.

        Args:
            report: AgentReport dataclass instance

        Returns:
            Formatted string representation
        """
        if report is None:
            return "[无分析结果]"

        if hasattr(report, 'error') and report.error:
            return f"[{report.agent_name}] 分析失败: {report.error}"

        lines = [
            f"=== {report.agent_name} ===",
            f"方向: {report.direction.value if report.direction else '无'}",
            f"置信度: {report.confidence:.1f}%",
        ]

        if hasattr(report, 'signals') and report.signals:
            lines.append("信号:")
            for signal in report.signals:
                strength = signal.strength.value if hasattr(signal.strength, 'value') else signal.strength
                lines.append(f"  - {signal.signal_type}: {signal.description} ({strength})")

        if hasattr(report, 'key_levels') and report.key_levels:
            support = report.key_levels.get('support', 'N/A')
            resistance = report.key_levels.get('resistance', 'N/A')
            lines.append(f"关键价位: 支撑 {support} / 阻力 {resistance}")

        if hasattr(report, 'analysis') and report.analysis:
            # Truncate long analysis
            analysis = report.analysis[:500] + "..." if len(report.analysis) > 500 else report.analysis
            lines.append(f"分析摘要: {analysis}")

        return "\n".join(lines)

    @staticmethod
    def format_all_reports(
        indicator_report: Any = None,
        trend_report: Any = None,
        sentiment_report: Any = None,
        pattern_report: Any = None
    ) -> str:
        """
        Format all agent reports into a combined summary for debate.

        Args:
            indicator_report: IndicatorAgent report
            trend_report: TrendAgent report
            sentiment_report: SentimentAgent report
            pattern_report: PatternAgent report

        Returns:
            Combined formatted string
        """
        sections = []

        if indicator_report:
            sections.append(ContextAdapter.format_agent_report(indicator_report))

        if trend_report:
            sections.append(ContextAdapter.format_agent_report(trend_report))

        if sentiment_report:
            sections.append(ContextAdapter.format_agent_report(sentiment_report))

        if pattern_report:
            sections.append(ContextAdapter.format_agent_report(pattern_report))

        if not sections:
            return "[无可用分析结果]"

        return "\n\n".join(sections)

    @staticmethod
    def format_debate_argument(argument: Any) -> str:
        """
        Format a DebateArgument to readable text.

        Args:
            argument: DebateArgument dataclass instance

        Returns:
            Formatted string representation
        """
        if argument is None:
            return "[无辩论观点]"

        role_label = "多头倡导者 (Bull)" if argument.agent_role == "bull" else "空头魔鬼代言人 (Bear)"

        lines = [
            f"=== {role_label} ===",
            f"立场: {argument.position.value if hasattr(argument.position, 'value') else argument.position}",
            f"置信度: {argument.confidence:.1f}%",
        ]

        if argument.key_points:
            lines.append("核心观点:")
            for point in argument.key_points[:5]:  # Limit to 5 points
                lines.append(f"  • {point}")

        if argument.risk_factors:
            lines.append("风险因素:")
            for risk in argument.risk_factors[:3]:  # Limit to 3 risks
                lines.append(f"  ⚠ {risk}")

        if argument.supporting_signals:
            lines.append("支持信号:")
            for signal in argument.supporting_signals[:3]:
                lines.append(f"  ✓ {signal}")

        if argument.counter_arguments:
            lines.append("反驳论点:")
            for counter in argument.counter_arguments[:3]:
                lines.append(f"  ✗ {counter}")

        if argument.recommended_action:
            lines.append(f"建议行动: {argument.recommended_action}")

        return "\n".join(lines)

    @staticmethod
    def format_judge_verdict(verdict: Any) -> str:
        """
        Format a JudgeVerdict to readable text.

        Args:
            verdict: JudgeVerdict dataclass instance

        Returns:
            Formatted string representation
        """
        if verdict is None:
            return "[无裁决结果]"

        verdict_label = {
            "approve": "✅ 批准 (APPROVE)",
            "reject": "❌ 拒绝 (REJECT)",
            "abstain": "⏸️ 弃权 (ABSTAIN)"
        }.get(verdict.verdict.value if hasattr(verdict.verdict, 'value') else verdict.verdict, "未知")

        lines = [
            "=== 裁判裁决 ===",
            f"裁决: {verdict_label}",
            f"置信度: {verdict.confidence:.1f}%",
        ]

        if verdict.winning_argument:
            winner = "多头 (Bull)" if verdict.winning_argument == "bull" else "空头 (Bear)"
            lines.append(f"胜出方: {winner}")

        if verdict.leverage:
            lines.append(f"建议杠杆: {verdict.leverage}x")

        if verdict.key_reasoning:
            lines.append(f"核心理由: {verdict.key_reasoning}")

        if verdict.risk_assessment:
            lines.append(f"风险评估: {verdict.risk_assessment}")

        if verdict.recommended_action:
            lines.append(f"最终建议: {verdict.recommended_action}")

        return "\n".join(lines)

    @staticmethod
    def build_execution_context(
        pair: str,
        action: str,
        confidence: float,
        leverage: Optional[int] = None,
        key_support: Optional[float] = None,
        key_resistance: Optional[float] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Build execution context for trading tool invocation.

        Args:
            pair: Trading pair
            action: Action to execute
            confidence: Confidence score
            leverage: Leverage (for entry)
            key_support: Support level
            key_resistance: Resistance level
            reason: Reason for action

        Returns:
            Dict suitable for tool invocation
        """
        context = {
            "pair": pair,
            "action": action,
            "confidence_score": confidence,
            "reason": reason
        }

        if leverage is not None:
            context["leverage"] = leverage

        if key_support is not None:
            context["key_support"] = key_support

        if key_resistance is not None:
            context["key_resistance"] = key_resistance

        return context

    @staticmethod
    def extract_direction_from_action(action: str) -> Optional[str]:
        """
        Extract trading direction from action string.

        Args:
            action: Action string (e.g., "enter_long", "enter_short")

        Returns:
            Direction string ("long", "short") or None
        """
        action_lower = action.lower()

        if "long" in action_lower:
            return "long"
        elif "short" in action_lower:
            return "short"

        return None
