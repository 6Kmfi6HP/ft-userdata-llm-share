"""
多 Agent 分析系统
参考 QuantAgent 架构，实现专业化分工的 LLM Agent 协作

模块结构:
- agent_state: Agent 状态和报告数据结构
- base_agent: Agent 基类，定义通用接口
- indicator_agent: 技术指标分析专家
- trend_agent: 趋势结构分析专家
- sentiment_agent: 市场情绪分析专家
- pattern_agent: K线形态识别专家（视觉分析）
- orchestrator: Agent 编排器，协调多 Agent 执行
"""

from .agent_state import AgentState, AgentReport
from .base_agent import BaseAgent
from .indicator_agent import IndicatorAgent
from .trend_agent import TrendAgent
from .sentiment_agent import SentimentAgent
from .pattern_agent import PatternAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    'AgentState',
    'AgentReport',
    'BaseAgent',
    'IndicatorAgent',
    'TrendAgent',
    'SentimentAgent',
    'PatternAgent',
    'AgentOrchestrator',
]
