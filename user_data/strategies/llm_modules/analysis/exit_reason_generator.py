"""
退出原因生成器 - 为自动退出生成与 LLM 主动退出一致的原因结构

为 Layer 1/2/4 自动止盈止损生成详细的退出原因，确保所有退出场景
都有统一的数据结构：{exit_reason, trade_score, confidence_score}
"""

import logging
from typing import Dict, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class ExitReasonGenerator:
    """为自动退出生成与 LLM 主动退出一致的原因结构"""

    def __init__(self, llm_client, config: dict):
        """
        初始化退出原因生成器

        Args:
            llm_client: LLM 客户端实例
            config: 配置字典
        """
        self.llm_client = llm_client
        self.config = config.get('exit_reason_generation', {})
        self.enabled = self.config.get('enabled', True)
        self.timeout = self.config.get('timeout', 5)
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 500)
        self.fallback_on_error = self.config.get('fallback_on_error', True)

    def generate_exit_reason(
        self,
        pair: str,
        exit_layer: str,
        exit_metadata: Dict,
        current_dataframe: pd.DataFrame
    ) -> Dict:
        """
        调用 LLM 生成退出原因

        Args:
            pair: 交易对
            exit_layer: 退出层 ("layer1" | "layer2" | "layer4")
            exit_metadata: 触发时的技术参数
            current_dataframe: 当前市场数据

        Returns:
            {
                "action": "exit",
                "reason": str,  # 详细退出原因
                "trade_score": float,  # 0-100
                "confidence_score": float  # 0-100
            }
        """
        if not self.enabled:
            return self._fallback_reason(exit_layer, exit_metadata)

        try:
            # 构建 prompt
            prompt = self._build_prompt(pair, exit_layer, exit_metadata, current_dataframe)

            # 调用 LLM
            logger.info(f"Calling LLM for exit reason generation: {pair} ({exit_layer})")

            content = self.llm_client.simple_call(
                messages=[
                    {
                        "role": "system",
                        "content": "你是交易分析专家，评估退出决策的合理性。用简洁的语言分析退出原因，并给出评分。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            # 检查响应是否为空
            if content is None:
                raise ValueError("LLM returned None response")

            # 解析响应
            result = self._parse_response(content, exit_layer, exit_metadata)

            logger.info(
                f"Exit reason generated for {pair}: "
                f"score={result['trade_score']}, confidence={result['confidence_score']}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to generate exit reason for {pair}: {e}", exc_info=True)

            if self.fallback_on_error:
                logger.warning(f"Using fallback reason for {pair}")
                return self._fallback_reason(exit_layer, exit_metadata)
            else:
                raise

    def _build_prompt(
        self,
        pair: str,
        exit_layer: str,
        exit_metadata: Dict,
        dataframe: pd.DataFrame
    ) -> str:
        """构建 LLM 分析 prompt"""

        # 提取最新的市场指标
        latest = dataframe.iloc[-1]
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        ema_20 = latest.get('ema_20', latest['close'])
        ema_50 = latest.get('ema_50', latest['close'])
        ema_200 = latest.get('ema_200', latest['close'])
        close = latest['close']

        # 判断趋势
        if close > ema_20 > ema_50 > ema_200:
            trend = "强上涨趋势"
        elif close > ema_200:
            trend = "上涨趋势"
        elif close < ema_20 < ema_50 < ema_200:
            trend = "强下跌趋势"
        elif close < ema_200:
            trend = "下跌趋势"
        else:
            trend = "震荡整理"

        if exit_layer == "layer2":
            return self._build_layer2_prompt(pair, exit_metadata, rsi, macd, trend)
        elif exit_layer == "layer1":
            return self._build_layer1_prompt(pair, exit_metadata, rsi, macd, trend)
        elif exit_layer == "layer4":
            return self._build_layer4_prompt(pair, exit_metadata, rsi, macd, trend)
        else:
            raise ValueError(f"Unknown exit layer: {exit_layer}")

    def _build_layer2_prompt(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 2 (ATR 追踪止损) 分析 prompt"""

        trigger_profit = exit_metadata.get('trigger_profit', 0) * 100
        profit_zone = exit_metadata.get('profit_zone', 'unknown')
        atr_multiplier = exit_metadata.get('atr_multiplier', 1.0)

        return f"""
交易对: {pair}
ATR 追踪止损触发:
- ROI: {trigger_profit:.1f}%
- 盈利区间: {profit_zone}
- ATR倍数: {atr_multiplier}x

当前市场状态:
- RSI: {rsi:.1f}
- MACD: {macd:.3f}
- 趋势: {trend}

请评估:
1. 这次止盈是否合理？如果不退出，预计市场走势如何？
2. 给出 trade_score (0-100，越高表示决策越好) 和 confidence_score (0-100，越高表示越确定)

请用以下格式输出（不要额外的文字）:
reason: <详细原因，包含市场状态分析和预测>
trade_score: <数字>
confidence_score: <数字>
"""

    def _build_layer1_prompt(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 1 (交易所硬止损 -10%) 分析 prompt"""

        return f"""
交易对: {pair}
交易所硬止损 -10% 触发

当前市场状态:
- RSI: {rsi:.1f}
- MACD: {macd:.3f}
- 趋势: {trend}

请评估:
1. 为何入场后直接触发止损？入场时机是否有问题？
2. 如果不止损继续持有，是否会继续下跌？

请用以下格式输出（不要额外的文字）:
reason: <失败原因分析>
trade_score: <数字 (0-50，因为是亏损退出)>
confidence_score: <数字>
"""

    def _build_layer4_prompt(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 4 (极端止盈保护) 分析 prompt"""

        trigger_profit = exit_metadata.get('trigger_profit', 0) * 100
        rsi_value = exit_metadata.get('rsi_value', rsi)

        return f"""
交易对: {pair}
极端止盈保护触发:
- ROI: {trigger_profit:.1f}%
- RSI: {rsi_value:.1f}

当前市场状态:
- MACD: {macd:.3f}
- 趋势: {trend}

请评估:
1. 在 {trigger_profit:.1f}% ROI 时退出是否正确？
2. RSI {rsi_value:.1f} 的极端值是否预示反转？

请用以下格式输出（不要额外的文字）:
reason: <止盈原因和市场预测>
trade_score: <数字 (70-100，因为是高盈利退出)>
confidence_score: <数字>
"""

    def _parse_response(
        self,
        content: str,
        exit_layer: str,
        exit_metadata: Dict
    ) -> Dict:
        """解析 LLM 响应为标准格式

        Args:
            content: LLM 返回的字符串内容
            exit_layer: 退出层标识
            exit_metadata: 退出元数据

        Returns:
            解析后的退出原因字典
        """

        try:
            # 直接处理字符串内容
            content = content.strip()

            # 解析三行格式
            lines = content.split('\n')
            reason = None
            trade_score = None
            confidence_score = None

            for line in lines:
                line = line.strip()
                if line.startswith('reason:'):
                    reason = line.replace('reason:', '').strip()
                elif line.startswith('trade_score:'):
                    trade_score = float(line.replace('trade_score:', '').strip())
                elif line.startswith('confidence_score:'):
                    confidence_score = float(line.replace('confidence_score:', '').strip())

            # 验证必要字段
            if reason is None or trade_score is None or confidence_score is None:
                raise ValueError("Missing required fields in LLM response")

            # 限制分数范围
            trade_score = max(0, min(100, trade_score))
            confidence_score = max(0, min(100, confidence_score))

            return {
                "action": "exit",
                "reason": reason,
                "trade_score": trade_score,
                "confidence_score": confidence_score
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response content: {content[:500] if content else 'No content'}")

            # 降级处理
            return self._fallback_reason(exit_layer, exit_metadata)

    def _fallback_reason(self, exit_layer: str, exit_metadata: Dict) -> Dict:
        """生成降级的退出原因（不调用 LLM）"""

        if exit_layer == "layer2":
            trigger_profit = exit_metadata.get('trigger_profit', 0) * 100
            profit_zone = exit_metadata.get('profit_zone', 'unknown')
            atr_multiplier = exit_metadata.get('atr_multiplier', 1.0)

            reason = (
                f"Layer 2 ATR 追踪止损触发: {profit_zone} 盈利区间，"
                f"ROI {trigger_profit:.1f}%，使用 {atr_multiplier}x ATR 追踪"
            )

            # 根据盈利区间给分
            if trigger_profit >= 15:
                trade_score = 85
            elif trigger_profit >= 6:
                trade_score = 75
            elif trigger_profit >= 2:
                trade_score = 65
            else:
                trade_score = 55

            confidence_score = 60

        elif exit_layer == "layer1":
            reason = "Layer 1 交易所硬止损 -10% 触发，入场后趋势逆转"
            trade_score = 30
            confidence_score = 70

        elif exit_layer == "layer4":
            trigger_profit = exit_metadata.get('trigger_profit', 0) * 100
            rsi_value = exit_metadata.get('rsi_value', 0)

            reason = (
                f"Layer 4 极端止盈保护触发: ROI {trigger_profit:.1f}%，"
                f"RSI {rsi_value:.1f} 极端值，保护暴利"
            )
            trade_score = 90
            confidence_score = 75

        else:
            reason = f"Unknown exit layer: {exit_layer}"
            trade_score = 50
            confidence_score = 50

        return {
            "action": "exit",
            "reason": reason,
            "trade_score": trade_score,
            "confidence_score": confidence_score
        }
