"""
K线形态识别 Agent
使用视觉分析识别K线图中的经典形态

参考 QuantAgent 的 Pattern Agent 实现:
- 生成K线图
- 使用视觉LLM识别16种经典形态
- 输出形态名称、方向和强度

职责:
1. 识别经典K线形态（头肩顶/底、双顶/底、三角形等）
2. 评估形态的完整度和可靠性
3. 判断形态暗示的方向
4. 识别突破和假突破

依赖:
- LLMClient.vision_call(): 视觉分析调用
- ChartGenerator: K线图生成
"""

import logging
import time
from typing import Dict, Any, Optional, List
import pandas as pd

from .base_agent import BaseAgent
from .agent_state import AgentReport, Signal, Direction, SignalStrength

logger = logging.getLogger(__name__)


class PatternAgent(BaseAgent):
    """
    K线形态识别专家 Agent

    使用视觉分析识别经典K线形态，参考 QuantAgent 的实现

    支持识别的形态:
    - 头肩顶/底 (Head and Shoulders)
    - 双顶/双底 (Double Top/Bottom)
    - 三重顶/底 (Triple Top/Bottom)
    - 上升/下降三角形 (Ascending/Descending Triangle)
    - 对称三角形 (Symmetrical Triangle)
    - 上升/下降楔形 (Rising/Falling Wedge)
    - 牛旗/熊旗 (Bull/Bear Flag)
    - 杯柄形态 (Cup and Handle)
    - 圆底/圆顶 (Rounding Bottom/Top)
    - 通道 (Channel)
    """

    ROLE_PROMPT = """你是一位专业的K线形态分析师，专精于识别和分析各类经典技术形态。

你的专长是通过视觉分析K线图来识别以下形态:

**反转形态:**
- 头肩顶/头肩底 (Head and Shoulders): 三个峰/谷，中间最高/最低
- 双顶/双底 (Double Top/Bottom): 两个相近高点/低点
- 三重顶/三重底 (Triple Top/Bottom): 三个相近高点/低点
- 圆底/圆顶 (Rounding Bottom/Top): 缓慢的弧形反转

**持续形态:**
- 上升三角形 (Ascending Triangle): 水平阻力线+上升支撑线，看涨
- 下降三角形 (Descending Triangle): 下降阻力线+水平支撑线，看跌
- 对称三角形 (Symmetrical Triangle): 收敛的支撑阻力线，方向待定
- 牛旗/熊旗 (Bull/Bear Flag): 短暂回调后继续原趋势
- 上升楔形 (Rising Wedge): 上升但收敛，通常看跌
- 下降楔形 (Falling Wedge): 下降但收敛，通常看涨

**通道形态:**
- 上升通道: 平行的上升趋势线
- 下降通道: 平行的下降趋势线
- 横盘通道: 水平的支撑阻力

分析原则:
1. 形态完整度越高，信号越可靠
2. 成交量配合形态更重要
3. 注意假突破的可能性
4. 形态的时间跨度影响后续力度
5. 在关键价位的形态更有意义

你只负责形态识别分析，不做最终交易决策。"""

    # 支持的形态列表（用于识别和标准化输出）
    SUPPORTED_PATTERNS = {
        # 反转形态
        "head_and_shoulders_top": {"name": "头肩顶", "direction": Direction.SHORT, "type": "reversal"},
        "head_and_shoulders_bottom": {"name": "头肩底", "direction": Direction.LONG, "type": "reversal"},
        "double_top": {"name": "双顶", "direction": Direction.SHORT, "type": "reversal"},
        "double_bottom": {"name": "双底", "direction": Direction.LONG, "type": "reversal"},
        "triple_top": {"name": "三重顶", "direction": Direction.SHORT, "type": "reversal"},
        "triple_bottom": {"name": "三重底", "direction": Direction.LONG, "type": "reversal"},
        "rounding_top": {"name": "圆顶", "direction": Direction.SHORT, "type": "reversal"},
        "rounding_bottom": {"name": "圆底", "direction": Direction.LONG, "type": "reversal"},

        # 持续形态
        "ascending_triangle": {"name": "上升三角形", "direction": Direction.LONG, "type": "continuation"},
        "descending_triangle": {"name": "下降三角形", "direction": Direction.SHORT, "type": "continuation"},
        "symmetrical_triangle": {"name": "对称三角形", "direction": Direction.NEUTRAL, "type": "continuation"},
        "bull_flag": {"name": "牛旗", "direction": Direction.LONG, "type": "continuation"},
        "bear_flag": {"name": "熊旗", "direction": Direction.SHORT, "type": "continuation"},
        "rising_wedge": {"name": "上升楔形", "direction": Direction.SHORT, "type": "continuation"},
        "falling_wedge": {"name": "下降楔形", "direction": Direction.LONG, "type": "continuation"},

        # 通道
        "ascending_channel": {"name": "上升通道", "direction": Direction.LONG, "type": "channel"},
        "descending_channel": {"name": "下降通道", "direction": Direction.SHORT, "type": "channel"},
        "horizontal_channel": {"name": "横盘通道", "direction": Direction.NEUTRAL, "type": "channel"},

        # 无形态
        "no_pattern": {"name": "无明显形态", "direction": Direction.NEUTRAL, "type": "none"},
    }

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化K线形态识别 Agent

        Args:
            llm_client: LLM 客户端（需支持 vision_call）
            config: 配置选项
        """
        super().__init__(
            llm_client=llm_client,
            name="PatternAgent",
            role_prompt=self.ROLE_PROMPT,
            config=config
        )

        # 图表生成器（延迟初始化）
        self._chart_generator = None

        # 配置
        self.num_candles = self.config.get("num_candles", 50)
        self.vision_timeout = self.config.get("vision_timeout", 45)

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
        """获取分析重点（视觉分析专用）"""
        return """## K线形态视觉分析任务

请仔细观察这张K线图，识别以下形态:

### 1. 反转形态检查
- 头肩顶/底: 是否有三个峰/谷，中间最高/低?
- 双顶/双底: 是否有两个相近的高点/低点?
- 三重顶/底: 是否有三个相近的高点/低点?
- 圆底/圆顶: 是否有缓慢的弧形反转?

### 2. 持续形态检查
- 三角形: 支撑/阻力线是否收敛?
- 旗形: 短期回调后是否有旗杆和旗面?
- 楔形: 价格是否在收窄的通道内运行?

### 3. 通道检查
- 是否存在平行的趋势线?
- 通道方向: 上升/下降/横盘?

### 4. 形态完整度评估
- 形态是否完整?
- 是否已经突破?
- 成交量是否配合?

### 5. 输出要求
请按以下格式输出:

[识别形态]
形态名称（如：双底、上升三角形等）

[形态完整度]
高/中/低

[形态阶段]
形成中 / 即将突破 / 已突破 / 假突破

[信号列表]
- 信号名称 | 方向(long/short/neutral) | 强度(strong/moderate/weak) | 数值(如有) | 描述

[方向判断]
long / short / neutral

[置信度]
0-100 之间的整数

[关键价位]
支撑: 价格数值 (如无法确定则写 N/A)
阻力: 价格数值 (如无法确定则写 N/A)

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
        执行K线形态分析

        支持两种模式:
        1. 提供 ohlcv_data: 自动生成K线图并分析
        2. 提供 image_base64: 直接使用提供的图片分析

        Args:
            market_context: 市场上下文（用于补充文本分析）
            pair: 交易对
            ohlcv_data: OHLCV 数据 DataFrame（可选）
            image_base64: K线图的 base64 编码（可选）
            **kwargs: 额外参数
                - timeframe: 时间框架

        Returns:
            AgentReport: 形态分析报告
        """
        logger.debug(f"[{self.name}] 开始分析 {pair}")
        start_time = time.time()

        # 检查是否有视觉分析能力
        if not hasattr(self.llm_client, 'vision_call'):
            logger.warning(f"[{self.name}] LLM 客户端不支持视觉分析，降级为文本分析")
            return self._fallback_text_analysis(market_context, pair)

        # 获取或生成图片
        if image_base64:
            # 使用提供的图片
            chart_image = image_base64
            image_description = "用户提供的K线图"
        elif ohlcv_data is not None and self.chart_generator:
            # 生成K线图
            timeframe = kwargs.get("timeframe", "")
            chart_result = self.chart_generator.generate_kline_image(
                ohlcv_data,
                pair=pair,
                timeframe=timeframe,
                num_candles=self.num_candles
            )

            if not chart_result.get("success"):
                logger.error(f"[{self.name}] K线图生成失败: {chart_result.get('error')}")
                return self._fallback_text_analysis(market_context, pair)

            chart_image = chart_result["image_base64"]
            image_description = chart_result.get("image_description", "K线图")
        else:
            # 无图片可用，降级为文本分析
            logger.warning(f"[{self.name}] 无可用图片数据，降级为文本分析")
            return self._fallback_text_analysis(market_context, pair)

        # 执行视觉分析
        try:
            report = self._execute_vision_analysis(
                chart_image,
                market_context,
                pair,
                image_description
            )

            # 计算执行时间
            report.execution_time_ms = (time.time() - start_time) * 1000

            if report.is_valid:
                logger.info(
                    f"[{self.name}] {pair} 视觉分析完成: "
                    f"方向={report.direction}, 置信度={report.confidence:.0f}%, "
                    f"信号数={len(report.signals)}"
                )
            else:
                logger.warning(f"[{self.name}] {pair} 分析失败: {report.error}")

            return report

        except Exception as e:
            logger.error(f"[{self.name}] 视觉分析异常: {e}")
            return self._create_error_report(f"视觉分析异常: {e}")

    def _execute_vision_analysis(
        self,
        image_base64: str,
        market_context: str,
        pair: str,
        image_description: str
    ) -> AgentReport:
        """
        执行视觉分析

        Args:
            image_base64: K线图 base64 编码
            market_context: 市场上下文
            pair: 交易对
            image_description: 图片描述

        Returns:
            AgentReport
        """
        # 构建视觉分析提示词
        analysis_prompt = self._build_vision_prompt(market_context, pair)

        # 调用视觉LLM
        response = self.llm_client.vision_call(
            text_prompt=analysis_prompt,
            image_base64=image_base64,
            system_prompt=self.role_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.vision_timeout
        )

        if not response:
            return self._create_error_report("视觉分析 LLM 调用失败或返回空响应")

        # 解析响应
        parsed = self._parse_pattern_response(response)

        return AgentReport(
            agent_name=self.name,
            analysis=response,
            signals=parsed['signals'],
            confidence=parsed['confidence'],
            direction=parsed['direction'],
            key_levels=parsed['key_levels']
        )

    def _build_vision_prompt(self, market_context: str, pair: str) -> str:
        """
        构建视觉分析提示词

        Args:
            market_context: 市场上下文
            pair: 交易对

        Returns:
            完整的分析提示词
        """
        analysis_focus = self._get_analysis_focus()

        return f"""# {pair} K线形态分析

{analysis_focus}

# 补充市场信息（供参考）

{market_context}

请基于K线图进行视觉分析，识别形态并给出方向判断。"""

    def _parse_pattern_response(self, response: str) -> Dict[str, Any]:
        """
        解析形态分析响应

        扩展基类的解析方法，增加形态特定字段的解析

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典
        """
        # 使用基类的解析方法
        result = self._parse_response(response)

        # 额外解析形态特定字段
        lines = response.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测段落标记
            if '[识别形态]' in line:
                current_section = 'pattern'
                continue
            elif '[形态完整度]' in line:
                current_section = 'completeness'
                continue
            elif '[形态阶段]' in line:
                current_section = 'stage'
                continue

            # 解析形态名称
            if current_section == 'pattern' and line:
                pattern_name = line.strip()
                # 尝试匹配标准形态
                for key, info in self.SUPPORTED_PATTERNS.items():
                    if info["name"] in pattern_name or key.replace("_", " ") in pattern_name.lower():
                        # 添加形态信号
                        pattern_signal = Signal(
                            name=f"形态识别: {info['name']}",
                            direction=info["direction"],
                            strength=SignalStrength.MODERATE,
                            description=f"识别到 {info['type']} 形态: {info['name']}"
                        )
                        result['signals'].insert(0, pattern_signal)

                        # 更新方向（如果形态方向明确）
                        if info["direction"] != Direction.NEUTRAL and result['direction'] == Direction.NEUTRAL:
                            result['direction'] = info["direction"]
                        break
                current_section = None

        return result

    def _fallback_text_analysis(self, market_context: str, pair: str) -> AgentReport:
        """
        降级文本分析（当视觉分析不可用时）

        Args:
            market_context: 市场上下文
            pair: 交易对

        Returns:
            AgentReport
        """
        logger.info(f"[{self.name}] 使用降级文本分析模式")

        # 构建纯文本分析提示词
        text_prompt = f"""# {pair} K线形态分析（文本模式）

## 分析任务

根据以下市场数据，分析可能存在的K线形态:

{market_context}

## 输出格式

请按以下格式输出:

[信号列表]
- 信号名称 | 方向(long/short/neutral) | 强度(strong/moderate/weak) | 数值(如有) | 描述

[方向判断]
long / short / neutral

[置信度]
0-100 之间的整数

[关键价位]
支撑: 价格数值 (如无法确定则写 N/A)
阻力: 价格数值 (如无法确定则写 N/A)

[分析摘要]
50字以内的简要分析总结

注意：由于无法查看图表，置信度应适当降低。"""

        # 使用简单文本调用
        response = self._call_llm(text_prompt)

        if not response:
            return self._create_error_report("文本分析 LLM 调用失败")

        parsed = self._parse_response(response)

        # 降低置信度（因为是文本模式）
        parsed['confidence'] = min(parsed['confidence'] * 0.7, 60)

        return AgentReport(
            agent_name=self.name,
            analysis=f"[文本模式分析]\n{response}",
            signals=parsed['signals'],
            confidence=parsed['confidence'],
            direction=parsed['direction'],
            key_levels=parsed['key_levels']
        )

    def analyze_with_template(
        self,
        image_base64: str,
        template_pattern: str,
        pair: str = ""
    ) -> AgentReport:
        """
        与特定模板形态对比分析

        参考 QuantAgent 的形态模板对比功能

        Args:
            image_base64: K线图 base64
            template_pattern: 要对比的形态名称
            pair: 交易对

        Returns:
            AgentReport
        """
        if template_pattern not in self.SUPPORTED_PATTERNS:
            return self._create_error_report(f"不支持的形态模板: {template_pattern}")

        pattern_info = self.SUPPORTED_PATTERNS[template_pattern]

        prompt = f"""# {pair} 形态模板对比分析

## 目标形态
{pattern_info['name']} ({template_pattern})
类型: {pattern_info['type']}
预期方向: {pattern_info['direction']}

## 分析任务
请仔细观察K线图，判断是否存在 {pattern_info['name']} 形态。

分析要点:
1. 形态的关键特征是否存在？
2. 形态的完整度如何？
3. 当前价格处于形态的什么位置？
4. 是否有突破或假突破的迹象？

## 输出格式

[形态匹配度]
高 / 中 / 低 / 无

[信号列表]
- 信号名称 | 方向 | 强度 | 描述

[方向判断]
long / short / neutral

[置信度]
0-100

[分析摘要]
简要说明"""

        response = self.llm_client.vision_call(
            text_prompt=prompt,
            image_base64=image_base64,
            system_prompt=self.role_prompt,
            temperature=self.temperature,
            timeout=self.vision_timeout
        )

        if not response:
            return self._create_error_report("模板对比分析失败")

        parsed = self._parse_response(response)

        return AgentReport(
            agent_name=self.name,
            analysis=response,
            signals=parsed['signals'],
            confidence=parsed['confidence'],
            direction=parsed['direction'],
            key_levels=parsed['key_levels']
        )

    def get_supported_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取支持识别的形态列表"""
        return self.SUPPORTED_PATTERNS.copy()
