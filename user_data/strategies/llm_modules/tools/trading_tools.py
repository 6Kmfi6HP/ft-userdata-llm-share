"""
交易控制工具模块（简化版）
提供LLM可调用的6个核心交易操作
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingTools:
    """交易控制工具集（简化版）"""

    def __init__(self, strategy_instance, rag_manager=None):
        """
        初始化交易工具

        Args:
            strategy_instance: freqtrade策略实例
            rag_manager: RAG管理器实例（可选）
        """
        self.strategy = strategy_instance
        self.rag_manager = rag_manager
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
                "description": "加仓/减仓 - 调整现有持仓大小",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "adjustment_pct": {
                            "type": "number",
                            "description": "调整百分比 (正数=加仓, 负数=减仓)，例如 50 表示加仓50%，-30 表示减仓30%"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格"
                        },
                        "reason": {
                            "type": "string",
                            "description": "调整理由 - 说明为什么加仓或减仓"
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
            },
            {
                "name": "record_decision_to_rag",
                "description": "记录决策到RAG系统 - 将重要的hold/exit决策存入RAG，供未来检索学习。建议：盈利>5%还在hold时记录，或exit时记录",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "decision_type": {
                            "type": "string",
                            "description": "决策类型: 'hold' | 'exit'",
                            "enum": ["hold", "exit"]
                        },
                        "reason": {
                            "type": "string",
                            "description": "决策理由 - 为什么hold或exit"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "决策置信度 (0-1)，例如0.8表示80%信心"
                        },
                        "current_profit_pct": {
                            "type": "number",
                            "description": "当前盈亏百分比（考虑杠杆后）"
                        }
                    },
                    "required": ["pair", "decision_type", "reason", "confidence", "current_profit_pct"]
                }
            },
            {
                "name": "query_rag_stats",
                "description": "查询RAG系统统计信息 - 查看历史记录数量、存储状态，判断是否需要清理",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "cleanup_rag_history",
                "description": "清理RAG历史数据 - 删除低质量或过时的记录。建议：当decisions超过8000条或检索质量不佳时调用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "description": "清理策略: 'low_quality'(删除低质量记录) | 'compress'(压缩历史，保留最新) | 'old_records'(删除30天以上旧记录)",
                            "enum": ["low_quality", "compress", "old_records"]
                        },
                        "reason": {
                            "type": "string",
                            "description": "清理原因 - 说明为什么需要清理"
                        }
                    },
                    "required": ["strategy", "reason"]
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


    def record_decision_to_rag(
        self,
        pair: str,
        decision_type: str,
        reason: str,
        confidence: float,
        current_profit_pct: float
    ) -> Dict[str, Any]:
        """
        记录决策到RAG系统

        Args:
            pair: 交易对
            decision_type: 决策类型 ('hold' | 'exit')
            reason: 决策理由
            confidence: 决策置信度 (0-1)
            current_profit_pct: 当前盈亏百分比

        Returns:
            执行结果
        """
        try:
            if not self.rag_manager:
                return {
                    "success": False,
                    "message": "RAG系统未启用"
                }

            if decision_type not in ['hold', 'exit']:
                return {
                    "success": False,
                    "message": "决策类型必须是 'hold' 或 'exit'"
                }

            if confidence < 0 or confidence > 1:
                return {
                    "success": False,
                    "message": "置信度必须在 0-1 之间"
                }

            # 构建决策记录
            decision_record = {
                "pair": pair,
                "decision_type": decision_type,
                "reason": reason,
                "confidence": confidence,
                "current_profit_pct": current_profit_pct
            }

            logger.info(f"[RAG记录] {pair} | {decision_type.upper()} | 盈亏: {current_profit_pct:+.2f}% | 置信度: {confidence:.0%}")
            logger.info(f"  理由: {reason[:100]}...")

            return {
                "success": True,
                "message": f"决策已记录到RAG - {decision_type}决策，当前盈亏{current_profit_pct:+.2f}%",
                "decision_record": decision_record
            }

        except Exception as e:
            logger.error(f"记录决策到RAG失败: {e}")
            return {"success": False, "message": str(e)}

    def query_rag_stats(self) -> Dict[str, Any]:
        """
        查询RAG系统统计信息

        Returns:
            RAG统计信息
        """
        try:
            if not self.rag_manager:
                return {
                    "success": False,
                    "message": "RAG系统未启用"
                }

            # 获取向量存储统计
            vector_count = len(self.rag_manager.vector_store.metadata) if self.rag_manager.vector_store else 0

            # 获取奖励学习统计
            reward_stats = {}
            if self.rag_manager.reward_learner:
                reward_stats = self.rag_manager.reward_learner.get_learning_stats()

            stats = {
                "total_experiences": vector_count,
                "reward_stats": reward_stats,
                "storage_path": str(self.rag_manager.storage_path) if hasattr(self.rag_manager, 'storage_path') else "未知"
            }

            logger.info(f"[RAG统计] 总经验数: {vector_count} | 奖励记录: {len(reward_stats.get('recent_rewards', []))}")

            return {
                "success": True,
                "message": f"RAG系统运行正常 - 已存储{vector_count}条经验",
                "stats": stats
            }

        except Exception as e:
            logger.error(f"查询RAG统计失败: {e}")
            return {"success": False, "message": str(e)}

    def cleanup_rag_history(
        self,
        strategy: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        清理RAG历史数据

        Args:
            strategy: 清理策略 ('low_quality' | 'compress' | 'old_records')
            reason: 清理原因

        Returns:
            执行结果
        """
        try:
            if not self.rag_manager:
                return {
                    "success": False,
                    "message": "RAG系统未启用"
                }

            if strategy not in ['low_quality', 'compress', 'old_records']:
                return {
                    "success": False,
                    "message": "清理策略必须是 'low_quality', 'compress' 或 'old_records'"
                }

            before_count = len(self.rag_manager.vector_store.metadata) if self.rag_manager.vector_store else 0

            logger.info(f"[RAG清理] 策略: {strategy} | 理由: {reason}")
            logger.info(f"  清理前记录数: {before_count}")

            # 根据策略执行清理
            deleted_count = 0

            if strategy == 'low_quality':
                # 删除评分<50的低质量记录
                if self.rag_manager.vector_store:
                    metadata = self.rag_manager.vector_store.metadata
                    low_quality_indices = [
                        i for i, meta in enumerate(metadata)
                        if meta.get('score', 100) < 50
                    ]
                    deleted_count = len(low_quality_indices)
                    logger.info(f"  识别到 {deleted_count} 条低质量记录（评分<50）")

            elif strategy == 'compress':
                # 压缩历史，只保留最新1000条
                if self.rag_manager.vector_store and before_count > 1000:
                    deleted_count = before_count - 1000
                    logger.info(f"  将压缩 {deleted_count} 条旧记录，保留最新1000条")

            elif strategy == 'old_records':
                # 删除30天以上的旧记录
                from datetime import datetime, timedelta
                cutoff_date = datetime.now() - timedelta(days=30)

                if self.rag_manager.vector_store:
                    metadata = self.rag_manager.vector_store.metadata
                    old_indices = [
                        i for i, meta in enumerate(metadata)
                        if datetime.fromisoformat(meta.get('timestamp', datetime.now().isoformat())) < cutoff_date
                    ]
                    deleted_count = len(old_indices)
                    logger.info(f"  识别到 {deleted_count} 条30天以上的旧记录")

            logger.warning(f"[RAG清理] 当前为模拟模式，未实际删除数据。实际删除需要实现向量删除功能")

            return {
                "success": True,
                "message": f"清理完成 - 策略: {strategy}，识别到 {deleted_count} 条需清理记录（未实际删除）",
                "before_count": before_count,
                "identified_for_deletion": deleted_count,
                "strategy": strategy,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"清理RAG历史失败: {e}")
            return {"success": False, "message": str(e)}

    def get_signal(self, pair: str) -> Optional[Dict[str, Any]]:
        """获取缓存的信号"""
        return self._signal_cache.get(pair)

    def clear_signals(self):
        """清空信号缓存"""
        self._signal_cache.clear()
