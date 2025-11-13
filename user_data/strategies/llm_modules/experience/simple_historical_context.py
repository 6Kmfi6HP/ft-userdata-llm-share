"""
简单的历史上下文管理器
替代 RAG 系统，使用简单的 JSONL 文件读取提供历史交易上下文
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SimpleHistoricalContext:
    """简单的历史交易上下文管理器"""

    def __init__(self, experience_config: Dict):
        """
        初始化

        Args:
            experience_config: 经验配置字典
        """
        self.decision_log_path = Path(experience_config.get(
            "decision_log_path",
            "./user_data/logs/llm_decisions.jsonl"
        ))
        self.trade_log_path = Path(experience_config.get(
            "trade_log_path",
            "./user_data/logs/trade_experience.jsonl"
        ))
        self.max_trades = experience_config.get("max_recent_trades_context", 5)
        self.max_decisions = experience_config.get("max_recent_decisions_context", 10)
        self.include_pair_specific = experience_config.get("include_pair_specific_trades", True)

        logger.info(f"初始化简单历史上下文管理器")
        logger.info(f"  交易日志: {self.trade_log_path}")
        logger.info(f"  决策日志: {self.decision_log_path}")
        logger.info(f"  最大交易数: {self.max_trades}")

    def get_recent_context(self, pair: Optional[str] = None, action_type: str = "entry") -> str:
        """
        获取最近的交易上下文

        Args:
            pair: 交易对（可选，用于筛选特定交易对的历史）
            action_type: 操作类型 (entry/exit/hold)

        Returns:
            格式化的历史上下文字符串
        """
        try:
            # 读取最近的交易
            recent_trades = self._read_recent_trades(self.max_trades)

            # 如果指定了交易对，也读取该交易对的特定历史
            pair_specific_trades = []
            if pair and self.include_pair_specific:
                pair_specific_trades = self._read_recent_trades(3, pair_filter=pair)

            # 格式化为文本
            context = self._format_context(recent_trades, pair_specific_trades, pair)

            return context

        except Exception as e:
            logger.warning(f"获取历史上下文失败: {e}")
            return ""

    def _read_recent_trades(self, limit: int, pair_filter: Optional[str] = None) -> List[Dict]:
        """
        读取最近的交易记录

        Args:
            limit: 最多读取的记录数
            pair_filter: 交易对筛选（可选）

        Returns:
            交易记录列表
        """
        if not self.trade_log_path.exists():
            logger.debug(f"交易日志文件不存在: {self.trade_log_path}")
            return []

        trades = []

        try:
            # 使用 deque 读取文件最后 N 行（效率较高）
            with open(self.trade_log_path, 'r', encoding='utf-8') as f:
                # 读取更多行以便筛选
                read_limit = limit * 3 if pair_filter else limit
                lines = deque(f, maxlen=read_limit)

            # 解析 JSON 并筛选
            for line in lines:
                if not line.strip():
                    continue

                try:
                    trade = json.loads(line)

                    # 如果指定了交易对筛选
                    if pair_filter and trade.get('pair') != pair_filter:
                        continue

                    trades.append(trade)

                    # 达到限制后停止
                    if len(trades) >= limit:
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"解析 JSON 失败: {e}")
                    continue

            return trades[-limit:]  # 返回最后 N 条

        except Exception as e:
            logger.error(f"读取交易日志失败: {e}")
            return []

    def _format_context(
        self,
        recent_trades: List[Dict],
        pair_specific_trades: List[Dict],
        current_pair: Optional[str] = None
    ) -> str:
        """
        格式化历史上下文为文本

        Args:
            recent_trades: 最近的所有交易
            pair_specific_trades: 特定交易对的交易
            current_pair: 当前交易对

        Returns:
            格式化的文本字符串
        """
        if not recent_trades and not pair_specific_trades:
            return ""

        parts = []
        parts.append("【最近交易经验】")
        parts.append("")

        # 全部交易对的历史
        if recent_trades:
            parts.append(f"所有交易对（最近{len(recent_trades)}笔）：")
            parts.append("")

            for i, trade in enumerate(recent_trades, 1):
                trade_text = self._format_single_trade(trade, i)
                parts.append(trade_text)
                parts.append("")

        # 当前交易对的特定历史
        if pair_specific_trades and current_pair:
            parts.append("=" * 50)
            parts.append(f"当前交易对 {current_pair} 的最近{len(pair_specific_trades)}笔交易：")
            parts.append("")

            for i, trade in enumerate(pair_specific_trades, 1):
                trade_text = self._format_single_trade(trade, i, detailed=True)
                parts.append(trade_text)
                parts.append("")

        # 添加思考提示
        parts.append("=" * 50)
        parts.append("思考提示：")
        parts.append("- 最近亏损的共同原因是什么？")
        parts.append("- 盈利交易的成功要素是什么？")
        parts.append("- 是否有过早平仓错失大利润的案例？")
        parts.append("- 是否有盈利回撤导致收益缩水的案例？")
        parts.append("- 当前情况与历史哪笔最相似？那次的结果如何？")

        return "\n".join(parts)

    def _format_single_trade(self, trade: Dict, index: int, detailed: bool = False) -> str:
        """
        格式化单笔交易

        Args:
            trade: 交易字典
            index: 序号
            detailed: 是否显示详细信息

        Returns:
            格式化的交易文本
        """
        pair = trade.get('pair', 'UNKNOWN')
        side = trade.get('side', 'unknown')
        profit_pct = trade.get('profit_pct', 0.0)
        duration = trade.get('duration', 'unknown')
        entry_reason = trade.get('entry_reason', '未记录')
        exit_reason = trade.get('exit_reason', '未记录')
        leverage = trade.get('leverage', 1)

        # 计算实际盈亏（考虑杠杆）
        actual_profit = profit_pct * leverage

        # 盈亏标记
        profit_marker = ""
        if actual_profit < -3:
            profit_marker = " ⚠️⚠️ 【大幅亏损】"
        elif actual_profit < 0:
            profit_marker = " ⚠️"
        elif actual_profit > 5:
            profit_marker = " ✓✓ 【大幅盈利】"
        elif actual_profit > 0:
            profit_marker = " ✓"

        # 基本信息
        lines = []
        lines.append(
            f"[{index}] {pair} - {side} "
            f"{actual_profit:+.2f}% (杠杆{leverage}x){profit_marker}"
        )

        if detailed:
            lines.append(f"    入场: {entry_reason}")
            lines.append(f"    出场: {exit_reason}")
            lines.append(f"    持仓: {duration}")
        else:
            lines.append(f"    入场: {entry_reason} | 出场: {exit_reason} | 持仓: {duration}")

        # 添加教训和警告
        lessons = trade.get('lessons', {})

        # 兼容处理字符串和字典两种格式
        if isinstance(lessons, str):
            # 字符串格式：直接显示
            if lessons:
                lines.append(f"    【经验】{lessons}")
        elif isinstance(lessons, dict):
            # 字典格式：提取 warnings 和 insights
            warnings = lessons.get('warnings', [])
            insights = lessons.get('insights', [])

            if warnings:
                for warning in warnings[:2]:  # 最多显示2条警告
                    lines.append(f"    【警告】{warning}")

            if insights and detailed:
                for insight in insights[:1]:  # 详细模式显示1条洞察
                    lines.append(f"    【洞察】{insight}")

        # 特殊标记
        if actual_profit < -2:
            lines.append("    【亏损案例】务必避免重复相同错误")

        # MFE/MAE 信息（如果有）
        if detailed:
            mfe = trade.get('max_favorable_excursion')
            mae = trade.get('max_adverse_excursion')

            if mfe is not None and mae is not None:
                lines.append(f"    最大浮盈: {mfe:.2f}% | 最大浮亏: {mae:.2f}%")

                # 如果盈利但回撤较大
                if actual_profit > 0 and mfe > 0 and (mfe - actual_profit) > 2:
                    lines.append(
                        f"    【反思】盈利从峰值回撤 {mfe - actual_profit:.2f}%，"
                        f"是否应该更早止盈？"
                    )

        return "\n".join(lines)
