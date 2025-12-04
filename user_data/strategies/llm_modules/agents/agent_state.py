"""
Agent 状态管理模块
参考 QuantAgent 的 TypedDict 模式，定义多 Agent 系统的共享状态

设计原则:
1. 不可变状态更新 - 每个 Agent 返回新的状态字典
2. 类型安全 - 使用 TypedDict 和 dataclass 确保类型正确
3. 可追溯 - 记录每个 Agent 的分析时间戳和序列
"""

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    """交易方向枚举"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

    def __str__(self) -> str:
        return self.value


class SignalStrength(str, Enum):
    """信号强度枚举"""
    STRONG = "strong"      # 强信号，多个指标确认
    MODERATE = "moderate"  # 中等信号，部分确认
    WEAK = "weak"          # 弱信号，单一指标
    NONE = "none"          # 无明显信号

    def __str__(self) -> str:
        return self.value


@dataclass
class Signal:
    """单个交易信号"""
    name: str                           # 信号名称，如 "RSI超卖"
    direction: Direction                # 信号方向
    strength: SignalStrength            # 信号强度
    description: str = ""               # 详细描述
    value: Optional[float] = None       # 相关数值，如 RSI=28

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": str(self.direction),
            "strength": str(self.strength),
            "description": self.description,
            "value": self.value
        }


@dataclass
class AgentReport:
    """
    单个 Agent 的分析报告

    每个专业 Agent 完成分析后返回此结构
    """
    agent_name: str                     # Agent 名称标识
    analysis: str                       # 完整分析文本
    signals: List[Signal]               # 检测到的信号列表
    confidence: float                   # 置信度 0-100
    direction: Optional[Direction] = None  # 建议方向
    key_levels: Optional[Dict[str, float]] = None  # 关键价位（支撑/阻力）
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0      # 执行耗时（毫秒）
    error: Optional[str] = None         # 错误信息（如有）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "agent_name": self.agent_name,
            "analysis": self.analysis,
            "signals": [s.to_dict() for s in self.signals],
            "confidence": self.confidence,
            "direction": str(self.direction) if self.direction else None,
            "key_levels": self.key_levels,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error
        }

    def get_signal_summary(self) -> str:
        """获取信号摘要（用于日志）"""
        if not self.signals:
            return "无信号"

        signal_strs = [f"{s.name}({s.direction})" for s in self.signals[:3]]
        suffix = f"...+{len(self.signals)-3}" if len(self.signals) > 3 else ""
        return ", ".join(signal_strs) + suffix

    @property
    def is_valid(self) -> bool:
        """检查报告是否有效"""
        return self.error is None and self.confidence > 0


class AgentState(TypedDict, total=False):
    """
    多 Agent 系统的共享状态

    使用 TypedDict 确保类型安全，total=False 允许部分字段可选
    状态在各 Agent 间传递，每个 Agent 添加自己的分析结果

    流程:
    1. Orchestrator 初始化状态（pair, market_context）
    2. IndicatorAgent 添加 indicator_report
    3. TrendAgent 添加 trend_report
    4. SentimentAgent 添加 sentiment_report
    5. PatternAgent 添加 pattern_report（视觉分析）
    6. Orchestrator 聚合结果（consensus_*）
    """

    # ===== 输入数据 =====
    pair: str                           # 交易对，如 "BTC/USDT:USDT"
    current_price: float                # 当前价格
    market_context: str                 # 来自 ContextBuilder 的市场上下文

    # ===== 各 Agent 的分析报告 =====
    indicator_report: Optional[AgentReport]   # 技术指标分析
    trend_report: Optional[AgentReport]       # 趋势结构分析
    sentiment_report: Optional[AgentReport]   # 市场情绪分析
    pattern_report: Optional[AgentReport]     # K线形态分析（视觉分析）

    # ===== 聚合结果 =====
    consensus_direction: Optional[str]  # 共识方向: 'long', 'short', 'neutral', 'wait'
    consensus_confidence: float         # 共识置信度 (0-100)
    consensus_signals: List[Signal]     # 聚合后的关键信号
    combined_analysis: str              # 合并后的分析文本（注入到决策提示词）

    # ===== 关键价位 =====
    key_support: Optional[float]        # 关键支撑位
    key_resistance: Optional[float]     # 关键阻力位

    # ===== 元数据 =====
    execution_time_ms: float            # 总执行时间（毫秒）
    agent_sequence: List[str]           # Agent 执行序列
    created_at: str                     # 状态创建时间 (ISO format)


def create_initial_state(
    pair: str,
    current_price: float,
    market_context: str
) -> AgentState:
    """
    创建初始 Agent 状态

    Args:
        pair: 交易对
        current_price: 当前价格
        market_context: 市场上下文

    Returns:
        初始化的 AgentState
    """
    return AgentState(
        pair=pair,
        current_price=current_price,
        market_context=market_context,
        indicator_report=None,
        trend_report=None,
        sentiment_report=None,
        pattern_report=None,  # K线形态分析（视觉分析）
        consensus_direction=None,
        consensus_confidence=0.0,
        consensus_signals=[],
        combined_analysis="",
        key_support=None,
        key_resistance=None,
        execution_time_ms=0.0,
        agent_sequence=[],
        created_at=datetime.utcnow().isoformat()
    )


def merge_state(base: AgentState, updates: Dict[str, Any]) -> AgentState:
    """
    合并状态更新（不可变更新模式）

    Args:
        base: 基础状态
        updates: 要更新的字段

    Returns:
        新的状态字典
    """
    new_state = dict(base)
    new_state.update(updates)
    return AgentState(**new_state)
