"""
市场情绪分析 Agent
专注于资金费率、多空比、OI变化、恐惧贪婪指数等情绪指标的分析

职责:
1. 解读市场情绪指标
2. 识别极端情绪状态
3. 分析资金流向
4. 评估市场拥挤程度
"""

import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from .agent_state import AgentReport

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    市场情绪分析专家 Agent

    专注分析:
    - Funding Rate（资金费率）
    - Long/Short Ratio（多空比）
    - Open Interest（持仓量变化）
    - Fear & Greed Index（恐惧贪婪指数）
    - 大户/散户行为
    """

    ROLE_PROMPT = """你是一位专业的加密货币市场情绪分析师。

你的专长是解读市场情绪和资金流向指标：
- 资金费率：正费率表示多头拥挤，负费率表示空头拥挤
- 多空比：>1多头占优，<1空头占优，极端值暗示反转
- 持仓量(OI)：OI增加+价格上涨=新多头入场；OI减少+价格下跌=多头平仓
- 恐惧贪婪指数：<25极度恐惧（潜在买点），>75极度贪婪（潜在卖点）
- 清算数据：大量清算可能导致短期反向波动

分析原则：
1. 极端情绪往往是反向信号（"别人恐惧时贪婪"）
2. 情绪指标是滞后的，需要与技术面结合
3. 关注情绪的边际变化，而不仅是绝对值
4. 资金费率极端时，注意轧空/轧多风险
5. 保持客观，情绪分析是辅助而非决定因素

你只负责情绪分析，不做最终交易决策。"""

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化市场情绪分析 Agent

        Args:
            llm_client: LLM 客户端
            config: 配置选项
        """
        super().__init__(
            llm_client=llm_client,
            name="SentimentAgent",
            role_prompt=self.ROLE_PROMPT,
            config=config
        )

    def _get_analysis_focus(self) -> str:
        """获取分析重点"""
        return """## 市场情绪分析任务

请重点分析以下情绪指标：

### 1. 资金费率 (Funding Rate)
- 当前费率水平（正/负）
- 费率的历史对比（是否处于极端）
- 费率趋势（上升/下降/稳定）
- 对持仓成本的影响
- 潜在的轧空/轧多风险

### 2. 多空比 (Long/Short Ratio)
- 当前多空比数值
- 是否处于极端区域（>2.0 或 <0.5）
- 多空比的变化趋势
- 散户vs大户的多空分布（如有数据）

### 3. 持仓量变化 (Open Interest)
- OI的变化方向和幅度
- OI变化与价格变化的关系：
  - 价涨+OI增 = 新多入场（趋势延续）
  - 价涨+OI减 = 空头平仓（可能反转）
  - 价跌+OI增 = 新空入场（趋势延续）
  - 价跌+OI减 = 多头平仓（可能反转）
- OI的多周期趋势

### 4. 恐惧与贪婪指数 (Fear & Greed Index)
- 当前指数值和分类（极度恐惧/恐惧/中性/贪婪/极度贪婪）
- 历史30天的变化趋势
- 是否处于极端区域（<25 或 >75）
- 情绪转折点判断

### 5. 微观结构数据（如有）
- 盘口买卖压力
- 大单交易方向
- 清算数据（多头清算vs空头清算）

### 6. 情绪综合判断
- 多个情绪指标是否一致
- 当前情绪是否处于极端
- 是否存在情绪与价格的背离
- 情绪面是否支持技术面判断

请基于以上分析，给出方向判断和置信度。注意：极端情绪往往是反向信号。"""

    def analyze(
        self,
        market_context: str,
        pair: str,
        **kwargs
    ) -> AgentReport:
        """
        执行市场情绪分析

        Args:
            market_context: 市场上下文（包含情绪数据）
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
                f"信号={report.get_signal_summary()}"
            )
        else:
            logger.warning(f"[{self.name}] {pair} 分析失败: {report.error}")

        return report

    def analyze_extreme_sentiment(
        self,
        market_context: str,
        pair: str
    ) -> Dict[str, Any]:
        """
        专门分析极端情绪状态

        Args:
            market_context: 市场上下文
            pair: 交易对

        Returns:
            极端情绪分析结果
        """
        focus_text = """## 极端情绪状态分析

请专门识别以下极端情绪信号：

1. 资金费率极端
   - 是否 > 0.1% 或 < -0.1%（8小时费率）
   - 极端费率持续时间

2. 多空比极端
   - 是否 > 2.5（极度多头拥挤）
   - 是否 < 0.4（极度空头拥挤）

3. 恐惧贪婪极端
   - 是否 < 20（极度恐惧）
   - 是否 > 80（极度贪婪）

4. 清算风暴
   - 是否出现大量单边清算
   - 清算后的潜在反向机会

5. 反转信号评估
   - 当前极端情绪是否暗示潜在反转
   - 反转的可能时机和条件

输出格式：
[极端状态]
是/否

[极端类型]
多头拥挤 / 空头拥挤 / 极度恐惧 / 极度贪婪 / 无

[反转风险]
高/中/低

[建议]
简要建议"""

        prompt = self._build_analysis_prompt(market_context, focus_text)
        response = self._call_llm(prompt)

        if not response:
            return {"error": "极端情绪分析失败"}

        # 解析结果
        result = {
            "is_extreme": False,
            "extreme_type": None,
            "reversal_risk": "low",
            "suggestion": "",
            "raw_analysis": response
        }

        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()

            if '[极端状态]' in line or '[extreme]' in line_lower:
                result['is_extreme'] = '是' in line or 'yes' in line_lower

            elif '[极端类型]' in line or '[type]' in line_lower:
                if '多头拥挤' in line or 'long crowd' in line_lower:
                    result['extreme_type'] = 'long_crowded'
                elif '空头拥挤' in line or 'short crowd' in line_lower:
                    result['extreme_type'] = 'short_crowded'
                elif '恐惧' in line or 'fear' in line_lower:
                    result['extreme_type'] = 'extreme_fear'
                elif '贪婪' in line or 'greed' in line_lower:
                    result['extreme_type'] = 'extreme_greed'

            elif '[反转风险]' in line or '[reversal]' in line_lower:
                if '高' in line or 'high' in line_lower:
                    result['reversal_risk'] = 'high'
                elif '中' in line or 'medium' in line_lower:
                    result['reversal_risk'] = 'medium'

            elif '[建议]' in line or '[suggestion]' in line_lower:
                # 获取下一行作为建议
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    result['suggestion'] = lines[idx + 1].strip()

        return result

    def get_contrarian_signal(
        self,
        market_context: str,
        pair: str
    ) -> Optional[str]:
        """
        获取反向情绪信号

        当情绪极端时，返回与主流情绪相反的交易方向

        Args:
            market_context: 市场上下文
            pair: 交易对

        Returns:
            'long', 'short' 或 None（无明显反向信号）
        """
        extreme_analysis = self.analyze_extreme_sentiment(market_context, pair)

        if not extreme_analysis.get('is_extreme'):
            return None

        extreme_type = extreme_analysis.get('extreme_type')
        reversal_risk = extreme_analysis.get('reversal_risk')

        # 只有在反转风险较高时才给出反向信号
        if reversal_risk not in ('high', 'medium'):
            return None

        # 反向逻辑
        contrarian_map = {
            'long_crowded': 'short',      # 多头拥挤 -> 考虑做空
            'short_crowded': 'long',      # 空头拥挤 -> 考虑做多
            'extreme_fear': 'long',       # 极度恐惧 -> 考虑做多
            'extreme_greed': 'short',     # 极度贪婪 -> 考虑做空
        }

        return contrarian_map.get(extreme_type)
