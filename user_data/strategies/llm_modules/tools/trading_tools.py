"""
交易控制工具模块（简化版）
提供LLM可调用的6个核心交易操作
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingTools:
    """交易控制工具集（简化版）"""

    def __init__(self, strategy_instance):
        """
        初始化交易工具

        Args:
            strategy_instance: freqtrade策略实例
        """
        self.strategy = strategy_instance
        self._signal_cache = {}  # 缓存本周期的信号

    def get_tools_schema(self) -> list[Dict[str, Any]]:
        """获取所有交易工具的OpenAI函数schema"""
        return [
            {
                "name": "signal_entry_long",
                "description": "开多仓 - 做多开仓并指定杠杆",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对，例如 BTC/USDT:USDT"
                        },
                        "leverage": {
                            "type": "number",
                            "description": "杠杆倍数 (1-100)"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)，表示你对这个决策的信心程度。>80高信心，60-80中等，<60低信心"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "trend_strength": {
                            "type": "string",
                            "description": "趋势强度评估: '强势' | '中等' | '弱势'"
                        },
                        "stake_amount": {
                            "type": "number",
                            "description": "本次计划投入的USDT金额（留空则使用默认仓位）"
                        },
                        "reason": {
                            "type": "string",
                            "description": "开仓理由 - 说明为什么做多，包括技术面、趋势判断等"
                        }
                    },
                    "required": ["pair", "leverage", "confidence_score", "key_support", "key_resistance", "rsi_value", "trend_strength", "reason"]
                }
            },
            {
                "name": "signal_entry_short",
                "description": "开空仓 - 做空开仓并指定杠杆",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "leverage": {
                            "type": "number",
                            "description": "杠杆倍数 (1-100)"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)，表示你对这个决策的信心程度。>80高信心，60-80中等，<60低信心"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "trend_strength": {
                            "type": "string",
                            "description": "趋势强度评估: '强势' | '中等' | '弱势'"
                        },
                        "stake_amount": {
                            "type": "number",
                            "description": "本次计划投入的USDT金额（留空则使用默认仓位）"
                        },
                        "reason": {
                            "type": "string",
                            "description": "开仓理由"
                        }
                    },
                    "required": ["pair", "leverage", "confidence_score", "key_support", "key_resistance", "rsi_value", "trend_strength", "reason"]
                }
            },
            {
                "name": "signal_exit",
                "description": "平仓 - 平掉当前持仓，并对本次交易进行自我评价",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "trade_score": {
                            "type": "number",
                            "description": "【重要】对本次交易质量的自我评分 (0-100)。综合考虑：入场时机、持仓管理、盈亏结果、风险控制。评分标准：90+优秀，70-90良好，50-70及格，<50差"
                        },
                        "reason": {
                            "type": "string",
                            "description": "平仓理由 - 说明为什么平仓，以及对本次交易的反思和教训"
                        }
                    },
                    "required": ["pair", "confidence_score", "rsi_value", "trade_score", "reason"]
                }
            },
            {
                "name": "adjust_position",
                "description": "加仓/减仓 - 调整现有持仓大小。正数=加仓(趋势加强),负数=减仓(部分止盈/风险降低)。推荐: 盈利10%+时可用-30~-70减仓锁定部分利润",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "adjustment_pct": {
                            "type": "number",
                            "description": "调整百分比 (正数=加仓, 负数=减仓)。例: 50=加仓50%, -30=减仓30%(部分止盈), -50=减仓50%(大幅止盈)"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格(做多)/阻力位价格(做空) - 用于后续追踪"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格(做多)/支撑位价格(做空) - 用于后续追踪"
                        },
                        "reason": {
                            "type": "string",
                            "description": "调整理由 - 说明为什么加仓或减仓,包括技术面信号和盈利保护考虑"
                        }
                    },
                    "required": ["pair", "adjustment_pct", "confidence_score", "key_support", "key_resistance", "reason"]
                }
            },
            {
                "name": "signal_hold",
                "description": "保持 - 持仓不动，维持当前仓位（用于已有仓位时）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100) - 表示继续持有的信心"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "保持理由 - 说明为什么继续持有"
                        }
                    },
                    "required": ["pair", "confidence_score", "rsi_value", "reason"]
                }
            },
            {
                "name": "signal_wait",
                "description": "等待 - 空仓观望，不进行任何操作（用于无仓位时）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100) - 表示不开仓的信心（信心低说明可能有机会但不确定）"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "等待理由 - 说明为什么不开仓"
                        }
                    },
                    "required": ["pair", "confidence_score", "rsi_value", "reason"]
                }
            }
        ]

    def signal_entry_long(
        self,
        pair: str,
        leverage: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        rsi_value: float,
        trend_strength: str,
        reason: str,
        stake_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        发出做多信号

        Args:
            pair: 交易对
            leverage: 杠杆倍数
            confidence_score: 决策置信度 (1-100)
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            rsi_value: RSI数值
            trend_strength: 趋势强度
            reason: 开仓理由
            stake_amount: 投入金额

        Returns:
            执行结果
        """
        try:
            # 验证参数
            if leverage < 1 or leverage > 100:
                return {"success": False, "message": "杠杆必须在1-100之间"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            if stake_amount is not None and stake_amount <= 0:
                return {"success": False, "message": "投入金额必须大于0"}

            # 缓存信号
            self._signal_cache[pair] = {
                "action": "enter_long",
                "leverage": leverage,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason,
                "stake_amount": stake_amount
            }

            # 设置杠杆到策略缓存
            if not hasattr(self.strategy, '_leverage_cache'):
                self.strategy._leverage_cache = {}

            self.strategy._leverage_cache[pair] = leverage

            stake_msg = f" | 投入: {stake_amount:.2f} USDT" if stake_amount else ""
            logger.info(f"[做多信号] {pair} | 置信度: {confidence_score} | 杠杆: {leverage}x{stake_msg}")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance} | RSI: {rsi_value} | 趋势强度: {trend_strength}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"做多信号已发出 - 置信度{confidence_score}，杠杆{leverage}x",
                "pair": pair,
                "leverage": leverage,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason,
                "stake_amount": stake_amount
            }

        except Exception as e:
            logger.error(f"发出做多信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_entry_short(
        self,
        pair: str,
        leverage: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        rsi_value: float,
        trend_strength: str,
        reason: str,
        stake_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        发出做空信号（市价单）

        Args:
            pair: 交易对
            leverage: 杠杆倍数
            confidence_score: 决策置信度
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            rsi_value: RSI数值
            trend_strength: 趋势强度
            reason: 开仓理由
            stake_amount: 投入金额

        Returns:
            执行结果
        """
        try:
            # 验证参数
            if leverage < 1 or leverage > 100:
                return {"success": False, "message": "杠杆必须在1-100之间"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            if stake_amount is not None and stake_amount <= 0:
                return {"success": False, "message": "投入金额必须大于0"}

            # 缓存信号
            self._signal_cache[pair] = {
                "action": "enter_short",
                "leverage": leverage,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason,
                "stake_amount": stake_amount
            }

            # 设置杠杆
            if not hasattr(self.strategy, '_leverage_cache'):
                self.strategy._leverage_cache = {}

            self.strategy._leverage_cache[pair] = leverage

            stake_msg = f" | 投入: {stake_amount:.2f} USDT" if stake_amount else ""
            logger.info(f"[做空信号] {pair} | 置信度: {confidence_score} | 杠杆: {leverage}x{stake_msg}")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance} | RSI: {rsi_value} | 趋势强度: {trend_strength}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"做空信号已发出 - 置信度{confidence_score}，杠杆{leverage}x",
                "pair": pair,
                "leverage": leverage,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason,
                "stake_amount": stake_amount
            }

        except Exception as e:
            logger.error(f"发出做空信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_exit(
        self,
        pair: str,
        confidence_score: float,
        rsi_value: float,
        trade_score: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        发出平仓信号（市价单），并记录模型自我评分

        Args:
            pair: 交易对
            confidence_score: 决策置信度
            rsi_value: RSI数值
            trade_score: 模型对本次交易的自我评分 (0-100)
            reason: 平仓理由（包含反思和教训）

        Returns:
            执行结果
        """
        try:
            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "exit",
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "trade_score": trade_score,
                "reason": reason
            }

            logger.info(f"[平仓信号] {pair} | 置信度: {confidence_score} | 自我评分: {trade_score}/100")
            logger.info(f"  RSI: {rsi_value}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"平仓信号已发出 - 置信度{confidence_score}，自我评分{trade_score}",
                "pair": pair,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "trade_score": trade_score,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"发出平仓信号失败: {e}")
            return {"success": False, "message": str(e)}

    def adjust_position(
        self,
        pair: str,
        adjustment_pct: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        调整仓位（加仓/减仓，市价单）

        Args:
            pair: 交易对
            adjustment_pct: 调整百分比 (正数加仓，负数减仓)
            confidence_score: 决策置信度
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            reason: 调整理由

        Returns:
            执行结果
        """
        try:
            if adjustment_pct == 0:
                return {"success": False, "message": "调整幅度不能为0"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            # 缓存调整信号
            if not hasattr(self.strategy, '_position_adjustment_cache'):
                self.strategy._position_adjustment_cache = {}

            self.strategy._position_adjustment_cache[pair] = {
                "adjustment_pct": adjustment_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "reason": reason
            }

            action = "加仓" if adjustment_pct > 0 else "减仓"
            logger.info(f"[{action}] {pair} | 置信度: {confidence_score} | 幅度: {abs(adjustment_pct):.1f}%")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"{action} {abs(adjustment_pct):.1f}% - 置信度{confidence_score}",
                "pair": pair,
                "adjustment_pct": adjustment_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"调整仓位失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_hold(
        self,
        pair: str,
        confidence_score: float,
        rsi_value: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        保持持仓不动

        Args:
            pair: 交易对
            confidence_score: 决策置信度
            rsi_value: RSI数值
            reason: 保持理由

        Returns:
            执行结果
        """
        try:
            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "hold",
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

            logger.debug(f"[保持] {pair} | 置信度: {confidence_score} | RSI: {rsi_value}")
            logger.debug(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"持仓保持不变 - 置信度{confidence_score}",
                "pair": pair,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"保持信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_wait(
        self,
        pair: str,
        confidence_score: float,
        rsi_value: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        空仓等待观望

        Args:
            pair: 交易对
            confidence_score: 决策置信度
            rsi_value: RSI数值
            reason: 等待理由

        Returns:
            执行结果
        """
        try:
            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "wait",
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

            logger.debug(f"[等待] {pair} | 置信度: {confidence_score} | RSI: {rsi_value}")
            logger.debug(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"空仓等待 - 置信度{confidence_score}",
                "pair": pair,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"等待信号失败: {e}")
            return {"success": False, "message": str(e)}

    def get_signal(self, pair: str) -> Optional[Dict[str, Any]]:
        """获取缓存的信号"""
        return self._signal_cache.get(pair)

    def clear_signal_for_pair(self, pair: str):
        """
        清除指定交易对的信号缓存（线程安全）

        🔧 修复C4: 使用按交易对清除，避免多交易对环境下的竞态条件

        Args:
            pair: 交易对名称（如 "BTC/USDT:USDT"）
        """
        if pair in self._signal_cache:
            del self._signal_cache[pair]
            logger.debug(f"已清除 {pair} 的信号缓存")

    def clear_signals(self):
        """
        清空所有信号缓存

        ⚠️ DEPRECATED: 在多交易对环境下可能导致竞态条件
        请使用 clear_signal_for_pair(pair) 代替
        """
        self._signal_cache.clear()
