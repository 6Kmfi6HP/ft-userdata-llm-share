"""
Kelly Criterion Position Calculator
基于学术论文: Risk-Constrained Kelly Gambling (Busseti, Ryu & Boyd, 2016)

Kelly公式: f* = (bp - q) / b
其中: b = 盈亏比(profit_factor), p = 胜率(win_rate), q = 1-p

实现特性:
- 半Kelly (Half-Kelly) 保守策略，降低波动性
- 最大仓位上限 (默认25%)
- 回撤调整: 当前回撤越大，建议仓位越小
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class KellyCalculator:
    """
    Kelly公式仓位计算器

    用途: 根据历史胜率和盈亏比，计算数学最优仓位比例
    设计理念: 仅提供建议，不强制执行，供LLM参考
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Kelly计算器

        Args:
            config: 配置字典，可包含:
                - max_kelly_fraction: 最大Kelly比例上限 (默认0.25)
                - use_half_kelly: 是否使用半Kelly (默认True)
                - min_sample_size: 最小样本量要求 (默认10)
                - drawdown_adjustment: 是否启用回撤调整 (默认True)
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.max_kelly_fraction = config.get("max_kelly_fraction", 0.25)
        self.use_half_kelly = config.get("use_half_kelly", True)
        self.min_sample_size = config.get("min_sample_size", 10)
        self.drawdown_adjustment = config.get("drawdown_adjustment", True)

        logger.info(f"[Kelly] 初始化完成: max_fraction={self.max_kelly_fraction}, "
                   f"half_kelly={self.use_half_kelly}, min_sample={self.min_sample_size}")

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        profit_factor: float,
        current_drawdown_pct: float = 0.0
    ) -> float:
        """
        计算Kelly最优仓位比例

        公式: f* = (b*p - q) / b
        其中: b = profit_factor, p = win_rate, q = 1-p

        Args:
            win_rate: 胜率 (0-1之间)
            profit_factor: 盈亏比 (avg_win / |avg_loss|)
            current_drawdown_pct: 当前回撤百分比 (0-100)

        Returns:
            建议仓位比例 (0 到 max_kelly_fraction 之间)
        """
        # 参数验证 (使用更保守的阈值避免极端值)
        KELLY_WIN_RATE_EPSILON = 0.001
        if win_rate <= 0.0 or win_rate >= (1.0 - KELLY_WIN_RATE_EPSILON):
            logger.debug(f"[Kelly] 胜率超出有效范围: {win_rate}, 返回0")
            return 0.0

        if profit_factor <= 0:
            logger.debug(f"[Kelly] 盈亏比异常: {profit_factor}, 返回0")
            return 0.0

        # Kelly公式核心计算
        b = profit_factor  # 盈亏比
        p = win_rate       # 胜率
        q = 1 - p          # 亏损率

        # f* = (b*p - q) / b
        kelly = (b * p - q) / b

        # 如果Kelly值为负，说明不应该下注
        if kelly <= 0:
            logger.debug(f"[Kelly] 计算值为负: {kelly:.4f}, 建议不开仓")
            return 0.0

        # 应用半Kelly (更保守，降低波动)
        if self.use_half_kelly:
            kelly *= 0.5

        # 回撤调整: 当前回撤越大，仓位越小
        if self.drawdown_adjustment and current_drawdown_pct > 5:
            # 回撤5%以上开始调整，回撤25%时仓位减半
            adjustment = max(0.5, 1 - (current_drawdown_pct - 5) / 20)
            kelly *= adjustment
            logger.debug(f"[Kelly] 回撤调整: dd={current_drawdown_pct:.1f}%, factor={adjustment:.2f}")

        # 限制最大值
        kelly = max(0.0, min(kelly, self.max_kelly_fraction))

        return kelly

    def calculate_position_size(
        self,
        available_balance: float,
        kelly_fraction: float
    ) -> float:
        """
        根据Kelly比例计算具体仓位金额

        Args:
            available_balance: 可用余额
            kelly_fraction: Kelly比例 (来自calculate_kelly_fraction)

        Returns:
            建议仓位金额 (USDT)
        """
        if kelly_fraction <= 0 or available_balance <= 0:
            return 0.0

        return available_balance * kelly_fraction

    def get_kelly_suggestion(
        self,
        pair_summary: Dict[str, Any],
        available_balance: float,
        current_drawdown_pct: float = 0.0
    ) -> Dict[str, Any]:
        """
        获取完整的Kelly建议信息

        Args:
            pair_summary: 来自historical_query.get_pair_summary()的统计数据
            available_balance: 可用余额
            current_drawdown_pct: 当前回撤百分比

        Returns:
            包含Kelly建议的完整信息字典
        """
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Kelly计算器已禁用"
            }

        total_trades = pair_summary.get("total_trades", 0)

        # 样本量检查
        if total_trades < self.min_sample_size:
            return {
                "enabled": True,
                "valid": False,
                "message": f"样本不足: {total_trades}笔 < {self.min_sample_size}笔最小要求",
                "total_trades": total_trades,
                "min_required": self.min_sample_size
            }

        win_rate = pair_summary.get("win_rate", 0)
        profit_factor = pair_summary.get("profit_factor", 0)
        avg_win_pct = pair_summary.get("avg_win_pct", 0)
        avg_loss_pct = pair_summary.get("avg_loss_pct", 0)

        # 计算Kelly比例
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=win_rate,
            profit_factor=profit_factor,
            current_drawdown_pct=current_drawdown_pct
        )

        # 计算建议仓位
        suggested_stake = self.calculate_position_size(
            available_balance=available_balance,
            kelly_fraction=kelly_fraction
        )

        return {
            "enabled": True,
            "valid": True,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "total_trades": total_trades,
            "current_drawdown_pct": current_drawdown_pct,
            "kelly_fraction": kelly_fraction,
            "kelly_fraction_pct": kelly_fraction * 100,
            "suggested_stake": suggested_stake,
            "use_half_kelly": self.use_half_kelly,
            "max_fraction": self.max_kelly_fraction
        }

    def format_for_context(self, suggestion: Dict[str, Any]) -> str:
        """
        将Kelly建议格式化为LLM上下文文本

        Args:
            suggestion: 来自get_kelly_suggestion()的结果

        Returns:
            格式化的文本，可直接加入LLM上下文
        """
        if not suggestion.get("enabled"):
            return ""

        if not suggestion.get("valid"):
            logger.debug(f"[Kelly] 无效建议: {suggestion.get('message', '数据不足')}")
            # return f"【Kelly建议】{suggestion.get('message', '数据不足')}"
            return ""

        lines = [
            "【建议仓位】",
            f"  历史胜率: {suggestion['win_rate']:.1%} ({suggestion['total_trades']}笔)",
            f"  盈亏比: {suggestion['profit_factor']:.2f} (平均盈利{suggestion['avg_win_pct']:.1f}%/平均亏损{suggestion['avg_loss_pct']:.1f}%)",
            f"  Kelly比例: {suggestion['kelly_fraction_pct']:.1f}%{'(半Kelly)' if suggestion['use_half_kelly'] else ''}",
            f"  建议仓位: {suggestion['suggested_stake']:.2f} USDT"
        ]

        if suggestion.get("current_drawdown_pct", 0) > 5:
            lines.append(f"  (已根据当前回撤{suggestion['current_drawdown_pct']:.1f}%调整)")

        return "\n".join(lines)
