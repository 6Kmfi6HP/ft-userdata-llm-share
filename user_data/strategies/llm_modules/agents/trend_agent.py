"""
趋势结构分析 Agent
专注于价格结构、趋势方向、支撑阻力位的分析

职责:
1. 识别趋势方向和阶段
2. 判断支撑和阻力位
3. 分析价格结构（高低点）
4. 评估趋势的健康程度

支持模式:
- 文本分析: 基于市场上下文数据
- 视觉分析: 基于带趋势线的K线图（梯度下降优化）

依赖:
- LLMClient.vision_call(): 视觉分析调用
- ChartGenerator: 趋势线图生成（梯度下降优化）
"""

import logging
import time
from typing import Dict, Any, Optional
import pandas as pd

from .base_agent import BaseAgent
from .agent_state import AgentReport, Signal, Direction, SignalStrength

logger = logging.getLogger(__name__)


class TrendAgent(BaseAgent):
    """
    趋势结构分析专家 Agent

    专注分析:
    - EMA 均线结构（多头/空头排列）
    - 价格高低点结构
    - 支撑和阻力位（梯度下降优化趋势线）
    - 趋势通道和轨道
    - 关键价格区域

    支持:
    - 文本分析模式
    - 视觉分析模式（基于趋势线图）
    """

    ROLE_PROMPT = """你是一位专业的加密货币趋势分析师。

你的专长是分析价格结构和趋势，包括：
- EMA均线系统：分析EMA20/50/200的排列和距离
- 价格结构：识别更高高点(HH)、更高低点(HL)、更低高点(LH)、更低低点(LL)
- 支撑阻力：识别关键价格区域和转折点
- 趋势阶段：判断趋势的初期、中期、末期或转折期
- 突破确认：判断价格突破的有效性
- 趋势线分析：识别上升/下降趋势线、通道

分析原则：
1. 趋势是你的朋友，顺势而为
2. 更高时间框架的趋势优先级更高
3. 支撑阻力位需要多次验证才更可靠
4. 关注趋势的动量和结构变化
5. 保持客观，识别趋势但不预测转折点
6. 趋势线突破需要成交量确认

你只负责趋势分析，不做最终交易决策。"""

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化趋势分析 Agent

        Args:
            llm_client: LLM 客户端（需支持 vision_call 用于视觉分析）
            config: 配置选项
        """
        super().__init__(
            llm_client=llm_client,
            name="TrendAgent",
            role_prompt=self.ROLE_PROMPT,
            config=config
        )

        # 图表生成器（延迟初始化）
        self._chart_generator = None

        # 配置
        self.num_candles = self.config.get("num_candles", 50)
        self.vision_timeout = self.config.get("vision_timeout", 45)
        self.prefer_vision = self.config.get("prefer_vision", True)  # 优先使用视觉分析

    @property
    def chart_generator(self):
        """延迟加载图表生成器"""
        if self._chart_generator is None:
            try:
                from ..utils.chart_generator import ChartGenerator
                self._chart_generator = ChartGenerator({
                    "num_candles": self.num_candles
                })
            except ImportError as e:
                logger.warning(f"无法加载 ChartGenerator: {e}")
        return self._chart_generator

    def _get_analysis_focus(self) -> str:
        """获取分析重点（文本分析模式）"""
        return """## 趋势结构分析任务

请重点分析以下方面：

### 1. EMA均线结构
- EMA20/50/200的相对位置
- 是否形成多头排列（价格>EMA20>EMA50>EMA200）或空头排列
- 价格与各均线的距离（用ATR衡量）
- 均线的斜率和方向

### 2. 价格结构分析
- 识别最近的重要高点和低点
- 判断是否形成更高高点(HH)/更高低点(HL)（上升趋势）
- 或更低高点(LH)/更低低点(LL)（下降趋势）
- 当前价格在结构中的位置

### 3. 支撑与阻力
- 识别关键支撑位（多次反弹的价格区域）
- 识别关键阻力位（多次受阻的价格区域）
- 评估当前价格距离关键位置的距离
- 这些关键位是否被测试或突破

### 4. 趋势阶段判断
- 初期：刚形成，动量强
- 中期：稳定运行，可能有回调
- 末期：动量减弱，可能反转
- 转折期：趋势正在改变
- 震荡：无明显趋势

### 5. 多时间框架分析（如有数据）
- 更高时间框架的趋势方向
- 是否与当前时间框架一致
- 时间框架间的支撑阻力对齐

### 6. 突破分析
- 是否存在突破关键位的迹象
- 突破的有效性判断（成交量确认、回测确认）

请基于以上分析，给出方向判断、关键价位和置信度。"""

    def _get_vision_analysis_focus(self) -> str:
        """获取视觉分析重点"""
        return """## 趋势线视觉分析任务

请仔细观察这张带趋势线的K线图，分析以下方面：

### 1. 趋势线分析
- **支撑趋势线（绿色）**: 斜率如何？是否有效支撑价格？
- **阻力趋势线（红色）**: 斜率如何？是否有效压制价格？
- **通道识别**: 是否形成上升/下降/横盘通道？
- **趋势线角度**: 陡峭（强势）还是平缓（弱势）？

### 2. 价格与趋势线关系
- 当前价格距离支撑线多远？
- 当前价格距离阻力线多远？
- 价格是否正在测试趋势线？
- 是否有突破趋势线的迹象？

### 3. 趋势方向判断
- 支撑线和阻力线是否同向（平行通道）？
- 是否收敛（三角形）或发散？
- 主趋势方向是什么？

### 4. 关键价位识别
- 从图中识别最近的支撑价位
- 从图中识别最近的阻力价位
- 趋势线与当前价格的交汇点

### 5. 均线系统（如图中显示）
- 均线排列顺序
- 价格与均线的位置关系

### 6. 输出格式

[信号列表]
- 信号名称 | 方向(long/short/neutral) | 强度(strong/moderate/weak) | 数值(如有) | 描述

[方向判断]
long / short / neutral

[置信度]
0-100 之间的整数

[关键价位]
支撑: 从图中识别的支撑价格
阻力: 从图中识别的阻力价格

[趋势线状态]
支撑线斜率: 上升/下降/平坦
阻力线斜率: 上升/下降/平坦
通道类型: 上升通道/下降通道/收敛三角/横盘

[分析摘要]
50字以内的简要分析总结"""

    def analyze(
        self,
        market_context: str,
        pair: str,
        ohlcv_data: Optional[pd.DataFrame] = None,
        image_base64: Optional[str] = None,
        **kwargs
    ) -> AgentReport:
        """
        执行趋势结构分析

        支持两种模式:
        1. 视觉分析（优先）: 使用带趋势线的K线图
        2. 文本分析: 基于市场上下文数据

        Args:
            market_context: 市场上下文
            pair: 交易对
            ohlcv_data: OHLCV 数据 DataFrame（可选，用于生成趋势线图）
            image_base64: 预生成的趋势线图 base64（可选）
            **kwargs: 额外参数
                - timeframe: 时间框架

        Returns:
            AgentReport: 分析报告
        """
        logger.debug(f"[{self.name}] 开始分析 {pair}")
        start_time = time.time()

        # 决定使用哪种分析模式
        use_vision = False
        if self.prefer_vision and hasattr(self.llm_client, 'vision_call'):
            if image_base64 or (ohlcv_data is not None and self.chart_generator):
                use_vision = True

        if use_vision:
            report = self._execute_vision_analysis(
                market_context, pair, ohlcv_data, image_base64, **kwargs
            )
        else:
            # 使用基类的文本分析流程
            report = self._execute_analysis(market_context, pair)

        # 计算执行时间
        report.execution_time_ms = (time.time() - start_time) * 1000

        if report.is_valid:
            # 记录关键价位
            levels_str = ""
            if report.key_levels:
                support = report.key_levels.get('support')
                resistance = report.key_levels.get('resistance')
                if support or resistance:
                    levels_str = f", 支撑={support}, 阻力={resistance}"

            mode = "📸视觉" if use_vision else "📝文本"
            logger.info(
                f"[{self.name}] {pair} {mode}分析完成: "
                f"方向={report.direction}, 置信度={report.confidence:.0f}%"
                f"{levels_str}"
            )
        else:
            logger.warning(f"[{self.name}] {pair} 分析失败: {report.error}")

        return report

    def _execute_vision_analysis(
        self,
        market_context: str,
        pair: str,
        ohlcv_data: Optional[pd.DataFrame],
        image_base64: Optional[str],
        **kwargs
    ) -> AgentReport:
        """
        执行视觉分析（使用趋势线图）

        Args:
            market_context: 市场上下文
            pair: 交易对
            ohlcv_data: OHLCV 数据
            image_base64: 预生成的图片 base64
            **kwargs: 额外参数

        Returns:
            AgentReport
        """
        timeframe = kwargs.get("timeframe", "")
        trendline_info = {}

        # 获取或生成趋势线图
        if image_base64:
            chart_image = image_base64
            image_description = "用户提供的趋势线图"
        elif ohlcv_data is not None and self.chart_generator:
            # 生成带趋势线的K线图（使用梯度下降优化）
            chart_result = self.chart_generator.generate_trend_image(
                ohlcv_data,
                pair=pair,
                timeframe=timeframe,
                num_candles=self.num_candles,
                use_gradient_descent=True  # 使用梯度下降优化
            )

            if not chart_result.get("success"):
                logger.warning(f"[{self.name}] 趋势线图生成失败: {chart_result.get('error')}，降级为文本分析")
                return self._execute_analysis(market_context, pair)

            chart_image = chart_result["image_base64"]
            image_description = chart_result.get("image_description", "趋势线图")

            # 保存趋势线信息
            trendline_info = {
                "support_trendline": chart_result.get("support_trendline"),
                "resistance_trendline": chart_result.get("resistance_trendline"),
                "support_levels": chart_result.get("support_levels", []),
                "resistance_levels": chart_result.get("resistance_levels", [])
            }

            logger.debug(f"[{self.name}] 趋势线图已生成: {image_description}")
        else:
            # 无图片可用，降级为文本分析
            logger.warning(f"[{self.name}] 无可用图片数据，降级为文本分析")
            return self._execute_analysis(market_context, pair)

        # 构建视觉分析提示词
        analysis_prompt = self._build_vision_prompt(market_context, pair, trendline_info)

        # 调用视觉 LLM
        try:
            response = self.llm_client.vision_call(
                text_prompt=analysis_prompt,
                image_base64=chart_image,
                system_prompt=self.role_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.vision_timeout
            )

            if not response:
                logger.warning(f"[{self.name}] 视觉 LLM 调用失败，降级为文本分析")
                return self._execute_analysis(market_context, pair)

            # 解析响应
            parsed = self._parse_trend_response(response, trendline_info)

            return AgentReport(
                agent_name=self.name,
                analysis=f"[视觉分析]\n{response}",
                signals=parsed['signals'],
                confidence=parsed['confidence'],
                direction=parsed['direction'],
                key_levels=parsed['key_levels']
            )

        except Exception as e:
            logger.error(f"[{self.name}] 视觉分析异常: {e}")
            return self._execute_analysis(market_context, pair)

    def _build_vision_prompt(
        self,
        market_context: str,
        pair: str,
        trendline_info: Dict[str, Any]
    ) -> str:
        """
        构建视觉分析提示词

        Args:
            market_context: 市场上下文
            pair: 交易对
            trendline_info: 趋势线信息

        Returns:
            完整的分析提示词
        """
        vision_focus = self._get_vision_analysis_focus()

        # 添加趋势线参数信息（如果有）
        trendline_context = ""
        if trendline_info:
            support_tl = trendline_info.get("support_trendline")
            resist_tl = trendline_info.get("resistance_trendline")

            if support_tl or resist_tl:
                trendline_context = "\n## 趋势线参数（算法计算结果）\n"
                if support_tl:
                    trendline_context += f"- 支撑线: 斜率={support_tl.get('slope', 0):.6f}, 起点价格={support_tl.get('start_price', 'N/A')}, 终点价格={support_tl.get('end_price', 'N/A')}\n"
                if resist_tl:
                    trendline_context += f"- 阻力线: 斜率={resist_tl.get('slope', 0):.6f}, 起点价格={resist_tl.get('start_price', 'N/A')}, 终点价格={resist_tl.get('end_price', 'N/A')}\n"

        return f"""# {pair} 趋势线视觉分析

{vision_focus}
{trendline_context}

# 补充市场信息（供参考）

{market_context}

请基于趋势线图进行视觉分析，判断趋势方向和关键价位。"""

    def _parse_trend_response(
        self,
        response: str,
        trendline_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解析趋势分析响应

        Args:
            response: LLM 响应文本
            trendline_info: 趋势线信息（用于补充关键价位）

        Returns:
            解析后的字典
        """
        # 使用基类的解析方法
        result = self._parse_response(response)

        # 如果没有解析到关键价位，使用算法计算的值
        if trendline_info:
            if not result['key_levels'].get('support') and trendline_info.get('support_levels'):
                result['key_levels']['support'] = trendline_info['support_levels'][0]
            if not result['key_levels'].get('resistance') and trendline_info.get('resistance_levels'):
                result['key_levels']['resistance'] = trendline_info['resistance_levels'][0]

            # 也可以使用趋势线的终点价格作为参考
            support_tl = trendline_info.get('support_trendline')
            resist_tl = trendline_info.get('resistance_trendline')

            if not result['key_levels'].get('support') and support_tl:
                result['key_levels']['support'] = support_tl.get('end_price')
            if not result['key_levels'].get('resistance') and resist_tl:
                result['key_levels']['resistance'] = resist_tl.get('end_price')

        # 解析趋势线状态
        lines = response.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '[趋势线状态]' in line:
                current_section = 'trendline_status'
                continue

            if current_section == 'trendline_status':
                if '上升' in line.lower():
                    # 添加上升趋势信号
                    if '支撑' in line:
                        result['signals'].append(Signal(
                            name="支撑趋势线上升",
                            direction=Direction.LONG,
                            strength=SignalStrength.MODERATE,
                            description="支撑线斜率为正，表明买盘持续"
                        ))
                    elif '阻力' in line:
                        result['signals'].append(Signal(
                            name="阻力趋势线上升",
                            direction=Direction.LONG,
                            strength=SignalStrength.WEAK,
                            description="阻力线斜率为正"
                        ))
                elif '下降' in line.lower():
                    if '支撑' in line:
                        result['signals'].append(Signal(
                            name="支撑趋势线下降",
                            direction=Direction.SHORT,
                            strength=SignalStrength.WEAK,
                            description="支撑线斜率为负"
                        ))
                    elif '阻力' in line:
                        result['signals'].append(Signal(
                            name="阻力趋势线下降",
                            direction=Direction.SHORT,
                            strength=SignalStrength.MODERATE,
                            description="阻力线斜率为负，表明卖压持续"
                        ))

                if '通道类型' in line or '收敛' in line or '横盘' in line:
                    current_section = None

        return result

    def identify_key_levels(
        self,
        market_context: str,
        pair: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        专门识别关键价位

        Args:
            market_context: 市场上下文
            pair: 交易对
            current_price: 当前价格

        Returns:
            包含关键价位的字典
        """
        focus_text = f"""## 关键价位识别任务

当前价格: {current_price}

请识别以下关键价位：

1. 最近的强支撑位（至少2次反弹确认）
2. 最近的强阻力位（至少2次受阻确认）
3. 心理关口（整数位）
4. 止损建议位
5. 目标位建议

输出格式：
[关键价位]
强支撑: 价格
弱支撑: 价格
强阻力: 价格
弱阻力: 价格
止损建议: 价格 (距当前价 X%)
目标位: 价格 (距当前价 X%)"""

        prompt = self._build_analysis_prompt(market_context, focus_text)
        response = self._call_llm(prompt)

        if not response:
            return {"error": "关键价位识别失败"}

        # 简单解析
        result = {
            "strong_support": None,
            "weak_support": None,
            "strong_resistance": None,
            "weak_resistance": None,
            "suggested_stop": None,
            "suggested_target": None,
            "raw_analysis": response
        }

        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            value = self._parse_float(line)

            if '强支撑' in line_lower or 'strong support' in line_lower:
                result['strong_support'] = value
            elif '弱支撑' in line_lower or 'weak support' in line_lower:
                result['weak_support'] = value
            elif '强阻力' in line_lower or 'strong resistance' in line_lower:
                result['strong_resistance'] = value
            elif '弱阻力' in line_lower or 'weak resistance' in line_lower:
                result['weak_resistance'] = value
            elif '止损' in line_lower or 'stop' in line_lower:
                result['suggested_stop'] = value
            elif '目标' in line_lower or 'target' in line_lower:
                result['suggested_target'] = value

        return result
