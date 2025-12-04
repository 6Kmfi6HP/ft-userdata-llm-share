"""
技术指标分析 Agent
专注于 RSI, MACD, ADX, Stochastic 等动量和趋势指标的分析

职责:
1. 解读超买超卖状态
2. 识别指标交叉信号（金叉/死叉）
3. 检测动量背离
4. 评估趋势强度
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from .agent_state import AgentReport

logger = logging.getLogger(__name__)


class IndicatorAgent(BaseAgent):
    """
    技术指标分析专家 Agent

    专注分析:
    - RSI (Relative Strength Index): 超买超卖判断
    - MACD: 趋势方向和动量
    - ADX: 趋势强度
    - Stochastic: 短期超买超卖
    - MFI: 资金流动强度
    """

    ROLE_PROMPT = """你是一位专业的加密货币技术指标分析师。

你的专长是解读各类技术指标，包括：
- RSI（相对强弱指数）：识别超买(>70)、超卖(<30)状态，以及背离信号
- MACD：分析快慢线交叉（金叉/死叉）、柱状图方向和动量强度
- ADX：评估趋势强度，>25为强趋势，<20为震荡市
- Stochastic：短期超买超卖及%K/%D交叉
- MFI（资金流指标）：结合成交量的超买超卖判断

分析原则：
1. 多指标确认优于单一指标
2. 注意指标与价格的背离（特别是RSI背离）
3. 趋势市用趋势指标（MACD, ADX），震荡市用震荡指标（RSI, Stochastic）
4. 关注指标的极端值和交叉信号
5. 保持客观，不预设立场

你只负责指标分析，不做最终交易决策。"""

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化技术指标分析 Agent

        Args:
            llm_client: LLM 客户端
            config: 配置选项
        """
        super().__init__(
            llm_client=llm_client,
            name="IndicatorAgent",
            role_prompt=self.ROLE_PROMPT,
            config=config
        )

    def _get_analysis_focus(self) -> str:
        """获取分析重点"""
        return """## 技术指标分析任务

请重点分析以下技术指标：

### 1. RSI（相对强弱指数）
- 当前值及所在区间（超买/正常/超卖）
- 是否存在RSI与价格的背离
- RSI的趋势方向（上升/下降/横盘）

### 2. MACD
- 快线与慢线的相对位置
- 是否存在金叉或死叉信号
- 柱状图（Histogram）的方向和强度
- MACD线是否在零轴之上/之下

### 3. ADX（平均趋向指数）
- 当前ADX值（>25强趋势，20-25中等，<20弱趋势/震荡）
- +DI和-DI的相对位置
- 趋势是在增强还是减弱

### 4. Stochastic（随机指标）
- %K和%D的当前值
- 是否处于超买(>80)或超卖(<20)区域
- %K与%D是否交叉

### 5. MFI（资金流指标）- 如有数据
- 资金流入/流出强度
- 是否与价格背离

### 6. 指标综合判断
- 多个指标是否给出一致信号
- 是否存在指标间的背离或矛盾
- 当前市场更适合趋势策略还是震荡策略

请基于以上分析，给出方向判断和置信度。"""

    def analyze(
        self,
        market_context: str,
        pair: str,
        **kwargs
    ) -> AgentReport:
        """
        执行技术指标分析

        Args:
            market_context: 市场上下文（包含指标数据）
            pair: 交易对

        Returns:
            AgentReport: 分析报告
        """
        logger.debug(f"[{self.name}] 开始分析 {pair}")

        # 使用基类的通用分析流程
        report = self._execute_analysis(market_context, pair)

        if report.is_valid:
            logger.info(
                f"[{self.name}] {pair} 分析完成: "
                f"方向={report.direction}, 置信度={report.confidence:.0f}%, "
                f"信号数={len(report.signals)}"
            )
        else:
            logger.warning(f"[{self.name}] {pair} 分析失败: {report.error}")

        return report

    def analyze_with_focus(
        self,
        market_context: str,
        pair: str,
        focus_indicators: list[str]
    ) -> AgentReport:
        """
        针对特定指标进行深度分析

        Args:
            market_context: 市场上下文
            pair: 交易对
            focus_indicators: 要重点分析的指标列表，如 ['RSI', 'MACD']

        Returns:
            AgentReport
        """
        focus_text = f"""## 重点指标分析任务

请重点分析以下指标: {', '.join(focus_indicators)}

对每个指标进行深度分析：
1. 当前数值和状态
2. 与历史对比
3. 潜在的信号
4. 与其他指标的关系"""

        prompt = self._build_analysis_prompt(market_context, focus_text)
        response = self._call_llm(prompt)

        if not response:
            return self._create_error_report("重点指标分析失败")

        parsed = self._parse_response(response)

        return AgentReport(
            agent_name=self.name,
            analysis=response,
            signals=parsed['signals'],
            confidence=parsed['confidence'],
            direction=parsed['direction'],
            key_levels=parsed['key_levels']
        )
