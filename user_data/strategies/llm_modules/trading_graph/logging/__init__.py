"""
LangGraph 决策日志模块

记录 LangGraph 交易决策系统的完整决策链，包括：
- 分析阶段（4个 Agent 的分析结果）
- 辩论阶段（Bull/Bear 论点 + Judge 裁决）
- Grounding 验证（幻觉检测结果）
- 最终执行决策
- 完整可观测性指标
"""

from .graph_logger import GraphDecisionLogger
from .graph_metrics import (
    # Metric classes
    StageMetrics,
    GroundingMetrics,
    VFVerificationMetrics,
    ConsistencyMetrics,
    DecisionMetrics,
    GraphExecutionMetrics,
    
    # Collector
    MetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
)

__all__ = [
    # Logger
    "GraphDecisionLogger",
    
    # Metric classes
    "StageMetrics",
    "GroundingMetrics",
    "VFVerificationMetrics",
    "ConsistencyMetrics",
    "DecisionMetrics",
    "GraphExecutionMetrics",
    
    # Collector
    "MetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
]

