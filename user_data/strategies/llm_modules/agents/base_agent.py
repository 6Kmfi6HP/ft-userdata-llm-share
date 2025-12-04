"""
Agent 基类模块
定义所有专业 Agent 的通用接口和辅助方法

设计原则:
1. 抽象接口 - 子类必须实现 analyze() 方法
2. 统一提示词构建 - 提供标准化的提示词模板
3. 响应解析 - 提供通用的 LLM 响应解析逻辑
4. 错误处理 - 统一的异常处理和降级策略
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from .agent_state import AgentReport, Signal, Direction, SignalStrength

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    专业分析 Agent 基类

    所有专业 Agent（Indicator, Trend, Sentiment）都继承此类
    """

    def __init__(
        self,
        llm_client,
        name: str,
        role_prompt: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 Agent

        Args:
            llm_client: LLM 客户端实例（支持 simple_call 方法）
            name: Agent 名称标识
            role_prompt: 角色定义提示词
            config: Agent 配置选项
        """
        self.llm_client = llm_client
        self.name = name
        self.role_prompt = role_prompt
        self.config = config or {}

        # 配置选项
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 800)
        self.timeout = self.config.get("timeout", 30)

    @abstractmethod
    def analyze(
        self,
        market_context: str,
        pair: str,
        **kwargs
    ) -> AgentReport:
        """
        执行分析并返回报告

        子类必须实现此方法

        Args:
            market_context: 市场上下文（来自 ContextBuilder）
            pair: 交易对
            **kwargs: 额外参数

        Returns:
            AgentReport: 分析报告
        """
        pass

    @abstractmethod
    def _get_analysis_focus(self) -> str:
        """
        获取分析重点说明

        子类必须实现，返回该 Agent 专注分析的内容描述
        """
        pass

    def _build_analysis_prompt(
        self,
        market_context: str,
        specific_focus: Optional[str] = None
    ) -> str:
        """
        构建分析提示词

        Args:
            market_context: 市场上下文
            specific_focus: 特定分析重点（可选，否则使用默认）

        Returns:
            完整的分析提示词
        """
        focus = specific_focus or self._get_analysis_focus()

        return f"""# 分析任务

{focus}

# 市场数据

{market_context}

# 输出格式要求

请严格按照以下格式输出分析结果：

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

# 输出示例

[信号列表]
- RSI超卖反弹 | long | moderate | 28.5 | RSI从超卖区回升，动量转正
- MACD金叉 | long | strong | N/A | MACD快线上穿慢线，柱状图转正
- 成交量萎缩 | neutral | weak | 0.6x | 成交量低于均值，观望为主

[方向判断]
long

[置信度]
72

[关键价位]
支撑: 42500.00
阻力: 44200.00

[分析摘要]
RSI超卖反弹配合MACD金叉，短期看多，但成交量不足需警惕假突破"""

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        调用 LLM 获取响应

        Args:
            prompt: 提示词

        Returns:
            LLM 响应文本，失败返回 None
        """
        try:
            messages = [
                {"role": "system", "content": self.role_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.llm_client.simple_call(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            return response

        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}")
            return None

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应

        Args:
            response: LLM 原始响应文本

        Returns:
            解析后的字典，包含 signals, direction, confidence, key_levels, summary
        """
        result = {
            'signals': [],
            'direction': Direction.NEUTRAL,
            'confidence': 50.0,
            'key_levels': {'support': None, 'resistance': None},
            'summary': ''
        }

        if not response:
            return result

        lines = response.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测段落标记
            if '[信号列表]' in line:
                current_section = 'signals'
                continue
            elif '[方向判断]' in line:
                current_section = 'direction'
                continue
            elif '[置信度]' in line:
                current_section = 'confidence'
                continue
            elif '[关键价位]' in line:
                current_section = 'key_levels'
                continue
            elif '[分析摘要]' in line:
                current_section = 'summary'
                continue

            # 根据当前段落解析内容
            if current_section == 'signals' and line.startswith('- '):
                signal = self._parse_signal_line(line[2:])
                if signal:
                    result['signals'].append(signal)

            elif current_section == 'direction':
                direction = self._parse_direction(line)
                if direction:
                    result['direction'] = direction

            elif current_section == 'confidence':
                confidence = self._parse_confidence(line)
                if confidence is not None:
                    result['confidence'] = confidence

            elif current_section == 'key_levels':
                level = self._parse_key_level(line)
                if level:
                    key, value = level
                    result['key_levels'][key] = value

            elif current_section == 'summary':
                result['summary'] = line

        return result

    def _parse_signal_line(self, line: str) -> Optional[Signal]:
        """
        解析单个信号行

        格式: 信号名称 | 方向 | 强度 | 数值 | 描述
        """
        try:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                return None

            name = parts[0]
            direction = self._parse_direction(parts[1]) or Direction.NEUTRAL
            strength = self._parse_strength(parts[2])
            value = self._parse_float(parts[3]) if len(parts) > 3 else None
            description = parts[4] if len(parts) > 4 else ""

            return Signal(
                name=name,
                direction=direction,
                strength=strength,
                value=value,
                description=description
            )

        except Exception as e:
            logger.debug(f"[{self.name}] 解析信号行失败: {line}, 错误: {e}")
            return None

    def _parse_direction(self, text: str) -> Optional[Direction]:
        """解析方向"""
        text_lower = text.lower().strip()
        if 'long' in text_lower or '做多' in text_lower or '多' == text_lower:
            return Direction.LONG
        elif 'short' in text_lower or '做空' in text_lower or '空' == text_lower:
            return Direction.SHORT
        elif 'neutral' in text_lower or '中性' in text_lower or '观望' in text_lower:
            return Direction.NEUTRAL
        return None

    def _parse_strength(self, text: str) -> SignalStrength:
        """解析信号强度"""
        text_lower = text.lower().strip()
        if 'strong' in text_lower or '强' in text_lower:
            return SignalStrength.STRONG
        elif 'moderate' in text_lower or '中' in text_lower:
            return SignalStrength.MODERATE
        elif 'weak' in text_lower or '弱' in text_lower:
            return SignalStrength.WEAK
        return SignalStrength.NONE

    def _parse_confidence(self, text: str) -> Optional[float]:
        """解析置信度"""
        try:
            # 提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                conf = float(numbers[0])
                return max(0, min(100, conf))  # 限制在 0-100
        except:
            pass
        return None

    def _parse_key_level(self, line: str) -> Optional[Tuple[str, float]]:
        """解析关键价位行"""
        line_lower = line.lower()

        if '支撑' in line_lower or 'support' in line_lower:
            value = self._parse_float(line)
            if value:
                return ('support', value)

        elif '阻力' in line_lower or 'resistance' in line_lower:
            value = self._parse_float(line)
            if value:
                return ('resistance', value)

        return None

    def _parse_float(self, text: str) -> Optional[float]:
        """从文本中提取浮点数"""
        if not text or 'N/A' in text.upper():
            return None
        try:
            numbers = re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))
            if numbers:
                return float(numbers[0])
        except:
            pass
        return None

    def _create_error_report(self, error_msg: str) -> AgentReport:
        """
        创建错误报告

        Args:
            error_msg: 错误信息

        Returns:
            包含错误信息的 AgentReport
        """
        return AgentReport(
            agent_name=self.name,
            analysis=f"分析失败: {error_msg}",
            signals=[],
            confidence=0.0,
            direction=None,
            error=error_msg
        )

    def _execute_analysis(
        self,
        market_context: str,
        pair: str
    ) -> AgentReport:
        """
        执行分析的通用流程

        子类可以覆盖此方法进行定制，或直接使用

        Args:
            market_context: 市场上下文
            pair: 交易对

        Returns:
            AgentReport
        """
        start_time = time.time()

        # 构建提示词
        prompt = self._build_analysis_prompt(market_context)

        # 调用 LLM
        response = self._call_llm(prompt)

        if not response:
            return self._create_error_report("LLM 调用失败或返回空响应")

        # 解析响应
        parsed = self._parse_response(response)

        # 计算执行时间
        execution_time_ms = (time.time() - start_time) * 1000

        # 构建报告
        return AgentReport(
            agent_name=self.name,
            analysis=response,
            signals=parsed['signals'],
            confidence=parsed['confidence'],
            direction=parsed['direction'],
            key_levels=parsed['key_levels'],
            execution_time_ms=execution_time_ms
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取 Agent 统计信息"""
        return {
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
