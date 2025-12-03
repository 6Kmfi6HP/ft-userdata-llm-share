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

    # 优化后的 System Prompt - 提供详细分析框架
    SYSTEM_PROMPT = """你是专业的加密货币交易分析师，负责评估自动止盈止损决策的合理性。

你的任务：
1. **市场状态分析**：根据提供的多时间框架技术指标，判断当前趋势（上涨/下跌/震荡）、动量强度、关键支撑/阻力位
2. **退出时机评估**：分析此时退出是否最优，考虑趋势延续性、动量变化、盈利回撤风险
3. **未来预测**：如果不退出，预测接下来1-3根K线的可能走势及概率
4. **经验提取**：总结此次交易可复用的关键教训

评分标准：
- trade_score: 0-100，评估这笔交易的整体质量
  - 90-100: 完美执行，入场和退出时机都很好
  - 70-89: 良好交易，有小的改进空间
  - 50-69: 一般交易，入场或退出有明显问题
  - 30-49: 较差交易，判断失误
  - 0-29: 失败交易，严重错误

- confidence_score: 0-100，对你分析的确信度
  - 80+: 信号明确，判断有高度把握
  - 60-79: 有一定把握，但存在不确定因素
  - 40-59: 信号混合，判断困难
  - <40: 市场噪音大，难以判断"""

    def __init__(self, llm_client, config: dict, context_builder=None):
        """
        初始化退出原因生成器

        Args:
            llm_client: LLM 客户端实例
            config: 配置字典
            context_builder: ContextBuilder 实例（可选，用于生成详细市场上下文）
        """
        self.llm_client = llm_client
        self.config = config.get('exit_reason_generation', {})
        self.context_builder = context_builder
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
                        "content": self.SYSTEM_PROMPT
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

        # 如果有 context_builder，使用详细市场上下文
        if self.context_builder:
            # 构建包含 exit_layer 的元数据
            full_exit_metadata = {
                'exit_layer': exit_layer,
                **exit_metadata
            }

            # 调用 build_exit_context 获取详细市场数据
            market_context = self.context_builder.build_exit_context(
                dataframe=dataframe,
                metadata={'pair': pair},
                trade=exit_metadata.get('trade'),
                exit_metadata=full_exit_metadata
            )

            # 根据 exit_layer 获取分析重点
            analysis_focus = self._get_layer_analysis_focus(exit_layer, exit_metadata)

            return f"""{market_context}

{analysis_focus}

请按照以下格式输出分析结果（不要额外的文字）:
reason: <详细原因，包含市场状态分析、退出时机评估、未来预测>
trade_score: <数字 0-100>
confidence_score: <数字 0-100>
lesson: <供下次交易参考的关键教训，1-2句话>
"""

        # 降级：没有 context_builder 时使用简化版本
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
            return self._build_layer2_prompt_simple(pair, exit_metadata, rsi, macd, trend)
        elif exit_layer == "layer1":
            return self._build_layer1_prompt_simple(pair, exit_metadata, rsi, macd, trend)
        elif exit_layer == "layer4":
            return self._build_layer4_prompt_simple(pair, exit_metadata, rsi, macd, trend)
        else:
            raise ValueError(f"Unknown exit layer: {exit_layer}")

    def _get_layer_analysis_focus(self, exit_layer: str, exit_metadata: Dict) -> str:
        """根据退出层获取分析重点"""

        trigger_profit = exit_metadata.get('trigger_profit', 0) * 100

        if exit_layer == "layer2":
            profit_zone = exit_metadata.get('profit_zone', 'unknown')
            return f"""## 分析重点 (Layer 2 ATR追踪止损)

当前盈利 {trigger_profit:.1f}% 触发了 {profit_zone} 区间的ATR追踪止损。

请重点分析：
1. **盈利回撤评估**：从最高点回撤了多少？ATR追踪是否过于敏感？
2. **趋势延续性**：根据指标历史，趋势是否还有延续空间？
3. **退出时机**：这是最优退出点吗？如果继续持有会怎样？"""

        elif exit_layer == "layer1":
            return f"""## 分析重点 (Layer 1 交易所硬止损)

⚠️ 触发了 -10% 硬止损，这是一笔亏损交易。

请重点分析：
1. **入场失误**：入场时的信号是否有效？哪些指标被误读了？
2. **趋势判断**：是逆势入场还是趋势反转太快？
3. **止损合理性**：如果不止损继续持有，后续走势会怎样？
4. **避免重复**：如何在未来避免类似错误？"""

        elif exit_layer == "layer4":
            rsi_value = exit_metadata.get('rsi_value', 0)
            adx_value = exit_metadata.get('adx_value', 0)
            return f"""## 分析重点 (Layer 4 极端止盈保护)

🎯 触发了极端止盈保护，ROI {trigger_profit:.1f}%，RSI {rsi_value:.1f}，ADX {adx_value:.1f}。

请重点分析：
1. **趋势疲竭信号**：RSI/ADX 是否显示趋势即将结束？
2. **止盈时机**：这是接近顶部/底部吗？还有上升/下跌空间吗？
3. **二次机会**：如果趋势延续，是否有回调后再次入场的机会？"""

        else:
            return f"## 分析重点 (未知层 {exit_layer})\n\n请分析退出决策的合理性。"

    def _build_layer2_prompt_simple(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 2 (ATR 追踪止损) 分析 prompt（简化版，无 context_builder 时使用）"""

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

    def _build_layer1_prompt_simple(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 1 (交易所硬止损 -10%) 分析 prompt（简化版，无 context_builder 时使用）"""

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

    def _build_layer4_prompt_simple(
        self,
        pair: str,
        exit_metadata: Dict,
        rsi: float,
        macd: float,
        trend: str
    ) -> str:
        """构建 Layer 4 (极端止盈保护) 分析 prompt（简化版，无 context_builder 时使用）"""

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
            解析后的退出原因字典，包含:
            - action: "exit"
            - reason: 详细退出原因
            - trade_score: 0-100
            - confidence_score: 0-100
            - lesson: 可选，供下次交易参考的教训
        """

        try:
            # 直接处理字符串内容
            content = content.strip()

            # 解析多行格式
            lines = content.split('\n')
            reason = None
            trade_score = None
            confidence_score = None
            lesson = None

            for line in lines:
                line = line.strip()
                if line.startswith('reason:'):
                    reason = line.replace('reason:', '').strip()
                elif line.startswith('trade_score:'):
                    try:
                        trade_score = float(line.replace('trade_score:', '').strip())
                    except ValueError:
                        pass
                elif line.startswith('confidence_score:'):
                    try:
                        confidence_score = float(line.replace('confidence_score:', '').strip())
                    except ValueError:
                        pass
                elif line.startswith('lesson:'):
                    lesson = line.replace('lesson:', '').strip()

            # 验证必要字段
            if reason is None or trade_score is None or confidence_score is None:
                raise ValueError("Missing required fields in LLM response")

            # 限制分数范围
            trade_score = max(0, min(100, trade_score))
            confidence_score = max(0, min(100, confidence_score))

            result = {
                "action": "exit",
                "reason": reason,
                "trade_score": trade_score,
                "confidence_score": confidence_score
            }

            # 添加可选的 lesson 字段
            if lesson:
                result["lesson"] = lesson

            return result

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
