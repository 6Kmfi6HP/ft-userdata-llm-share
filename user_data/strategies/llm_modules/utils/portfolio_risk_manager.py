"""
Portfolio Risk Manager
组合级风险管理模块 - 软性警告模式

基于学术论文:
- Constrained Max Drawdown Portfolio Optimization (Jaeger et al. 2024)
- Determining Optimal Stop-Loss Thresholds via Bayesian Analysis (Yang & Zhong 2016)

设计理念:
- 追踪组合回撤和连续亏损
- 生成警告信息供LLM参考
- 不硬性禁止开仓，保持LLM决策自主权
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    组合风险管理器（软性警告模式）

    功能:
    1. 追踪组合级别回撤
    2. 追踪连续亏损次数
    3. 生成风险等级和警告信息
    4. 不强制阻断交易，仅提供信息
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化组合风险管理器

        Args:
            config: 配置字典，可包含:
                - enabled: 是否启用 (默认True)
                - warning_drawdown: 警告级别回撤阈值 (默认0.05 = 5%)
                - critical_drawdown: 严重级别回撤阈值 (默认0.10 = 10%)
                - consecutive_loss_warning: 连续亏损警告阈值 (默认3)
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.warning_drawdown = config.get("warning_drawdown", 0.05)
        self.critical_drawdown = config.get("critical_drawdown", 0.10)
        self.consecutive_loss_warning = config.get("consecutive_loss_warning", 3)

        # 内部状态
        self._peak_balance: Optional[float] = None
        self._current_balance: Optional[float] = None
        self._current_drawdown: float = 0.0
        self._consecutive_losses: int = 0
        self._recent_trades: List[Dict[str, Any]] = []  # 最近10笔交易
        self._last_update_time: Optional[datetime] = None

        logger.info(f"[PortfolioRisk] 初始化完成: warning_dd={self.warning_drawdown*100:.0f}%, "
                   f"critical_dd={self.critical_drawdown*100:.0f}%, loss_warning={self.consecutive_loss_warning}")

    def update_balance(self, current_balance: float) -> None:
        """
        更新当前余额，自动追踪峰值和回撤

        Args:
            current_balance: 当前账户余额
        """
        if not self.enabled:
            return

        self._current_balance = current_balance
        self._last_update_time = datetime.now()

        # 更新峰值
        if self._peak_balance is None or current_balance > self._peak_balance:
            self._peak_balance = current_balance
            self._current_drawdown = 0.0
        else:
            # 计算回撤
            self._current_drawdown = (self._peak_balance - current_balance) / self._peak_balance

    def record_trade_result(self, profit_pct: float, trade_info: Optional[Dict[str, Any]] = None) -> None:
        """
        记录交易结果，更新连续亏损计数

        Args:
            profit_pct: 交易盈亏百分比 (正数=盈利，负数=亏损)
            trade_info: 可选的交易详情
        """
        if not self.enabled:
            return

        # 更新连续亏损计数
        if profit_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # 记录最近交易
        trade_record = {
            "profit_pct": profit_pct,
            "timestamp": datetime.now().isoformat(),
            **(trade_info or {})
        }
        self._recent_trades.append(trade_record)

        # 只保留最近10笔
        if len(self._recent_trades) > 10:
            self._recent_trades = self._recent_trades[-10:]

        logger.debug(f"[PortfolioRisk] 记录交易: profit={profit_pct:.2f}%, "
                    f"consecutive_losses={self._consecutive_losses}")

    def get_risk_level(self) -> str:
        """
        获取当前风险等级

        Returns:
            "critical" | "warning" | "normal"
        """
        if not self.enabled:
            return "normal"

        # 回撤检查
        if self._current_drawdown >= self.critical_drawdown:
            return "critical"

        if self._current_drawdown >= self.warning_drawdown:
            return "warning"

        # 连续亏损检查
        if self._consecutive_losses >= self.consecutive_loss_warning:
            return "warning"

        return "normal"

    def get_warning_message(self, risk_level: str) -> str:
        """
        根据风险等级生成警告消息

        Args:
            risk_level: 风险等级

        Returns:
            警告消息文本
        """
        if risk_level == "critical":
            messages = []
            if self._current_drawdown >= self.critical_drawdown:
                messages.append(f"组合回撤已达{self._current_drawdown*100:.1f}%，建议降低仓位或暂停开仓")
            if self._consecutive_losses >= self.consecutive_loss_warning:
                messages.append(f"连续{self._consecutive_losses}笔亏损，建议谨慎评估市场环境")
            return "；".join(messages) if messages else ""

        elif risk_level == "warning":
            messages = []
            if self._current_drawdown >= self.warning_drawdown:
                messages.append(f"组合回撤{self._current_drawdown*100:.1f}%，注意风险控制")
            if self._consecutive_losses >= self.consecutive_loss_warning:
                messages.append(f"已连续{self._consecutive_losses}笔亏损，建议审慎")
            return "；".join(messages) if messages else ""

        return ""

    def get_risk_status(self) -> Dict[str, Any]:
        """
        获取完整的风险状态信息

        Returns:
            包含所有风险指标的字典
        """
        if not self.enabled:
            return {
                "enabled": False,
                "risk_level": "normal",
                "message": "组合风险管理已禁用"
            }

        risk_level = self.get_risk_level()
        warning_message = self.get_warning_message(risk_level)

        return {
            "enabled": True,
            "risk_level": risk_level,
            "drawdown_pct": self._current_drawdown * 100,
            "peak_balance": self._peak_balance,
            "current_balance": self._current_balance,
            "consecutive_losses": self._consecutive_losses,
            "warning_message": warning_message,
            "thresholds": {
                "warning_drawdown": self.warning_drawdown * 100,
                "critical_drawdown": self.critical_drawdown * 100,
                "consecutive_loss_warning": self.consecutive_loss_warning
            },
            "last_update": self._last_update_time.isoformat() if self._last_update_time else None
        }

    def format_for_context(self) -> str:
        """
        将风险状态格式化为LLM上下文文本

        Returns:
            格式化的文本，仅在有警告时返回非空字符串
        """
        if not self.enabled:
            return ""
        
        # 未初始化时返回空 (避免 _peak_balance 为 None 导致格式化错误)
        if self._peak_balance is None:
            return ""

        risk_status = self.get_risk_status()
        risk_level = risk_status["risk_level"]

        # 正常状态不显示
        if risk_level == "normal":
            return ""

        # 警告或严重状态显示
        icon = "🔴" if risk_level == "critical" else "⚠️"
        level_text = "严重" if risk_level == "critical" else "注意"

        lines = [
            f"【{icon} 组合风险{level_text}】",
            f"  当前回撤: {risk_status['drawdown_pct']:.1f}%",
            f"  连续亏损: {risk_status['consecutive_losses']}笔"
        ]

        if risk_status["warning_message"]:
            lines.append(f"  建议: {risk_status['warning_message']}")

        return "\n".join(lines)

    def should_reduce_position(self) -> Tuple[bool, float]:
        """
        判断是否建议减仓，以及建议的减仓比例

        Returns:
            (是否建议减仓, 建议保留比例)
            例如: (True, 0.5) 表示建议减仓至50%
        """
        if not self.enabled:
            return False, 1.0

        risk_level = self.get_risk_level()

        if risk_level == "critical":
            # 严重风险: 建议减仓至50%
            return True, 0.5

        if risk_level == "warning":
            # 警告风险: 建议减仓至70%
            return True, 0.7

        return False, 1.0

    def reset_peak(self) -> None:
        """
        重置峰值（通常在策略重启或手动干预后调用）
        """
        self._peak_balance = self._current_balance
        self._current_drawdown = 0.0
        logger.info("[PortfolioRisk] 峰值已重置")

    def reset_consecutive_losses(self) -> None:
        """
        重置连续亏损计数（通常在策略调整后调用）
        """
        self._consecutive_losses = 0
        logger.info("[PortfolioRisk] 连续亏损计数已重置")
