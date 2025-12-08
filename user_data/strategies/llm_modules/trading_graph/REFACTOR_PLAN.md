# 📋 LangGraph Agents 决策流程重构方案

> 创建时间: 2025-12-05
> 状态: ✅ 已实施 (2025-12-05)

---

## 🎯 设计目标

1. **统一 Entry Path 和 Position Path** - 两条路径的步骤数量和结构一致
2. **Grounding Node 增强** - 检测幻觉的同时**纠正错误数据**
3. **新增 Executor Agent (LLM)** - 替代现有的纯代码 `executor_node`/`validator_node`，基于前面所有 agents 的结果进行最终决策
4. **丰富 State 数据结构** - 为后续开发(学习系统、回测分析、可视化)奠定基础

---

## 🔄 架构对比

### 当前架构 vs 新架构

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                              当前架构                                          │
├───────────────────────────────────┬───────────────────────────────────────────┤
│        ENTRY PATH (7 步)          │        POSITION PATH (6 步)               │
├───────────────────────────────────┼───────────────────────────────────────────┤
│ 1. Analysis (4x LLM 并行)         │ 1. Analysis (4x LLM 并行)                  │
│ 2. Aggregator (代码)              │ 2. Aggregator (代码)                       │
│ 3. Bull/Bear/Judge (3x LLM 顺序)  │ 3. PosBull/PosBear/PosJudge (3x LLM 顺序) │
│ 4. Grounding (代码) - 只检测       │ 4. PosGrounding (代码) - 只检测            │
│ 5. Validator (代码)               │ 5. PosValidator (代码)                     │
│ 6. Executor (代码)                │ 6. Executor (代码)                         │
│                                   │                                           │
│ 问题: 路径不一致, 纯代码决策       │ 问题: 缺少独立 Grounding 步骤              │
└───────────────────────────────────┴───────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              新架构 (统一)                                     │
├───────────────────────────────────┬───────────────────────────────────────────┤
│        ENTRY PATH (5 步)          │        POSITION PATH (5 步)               │
├───────────────────────────────────┼───────────────────────────────────────────┤
│ 1. Analysis (4x LLM 并行)         │ 1. Analysis (4x LLM 并行)                  │
│ 2. Aggregator (代码)              │ 2. Aggregator (代码)                       │
│ 3. Bull/Bear/Judge (3x LLM 顺序)  │ 3. PosBull/PosBear/PosJudge (3x LLM 顺序) │
│ 4. Grounding (代码) - 检测+纠正   │ 4. PosGrounding (代码) - 检测+纠正         │
│ 5. ExecutorAgent (🤖 LLM)        │ 5. ExecutorAgent (🤖 LLM)                 │
│                                   │                                           │
│ ✅ 统一结构, LLM 最终决策         │ ✅ 统一结构, LLM 最终决策                  │
└───────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 📊 新架构流程图

```
                                    START
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       STAGE 1: Analysis Subgraph    │
                    │           (并行 Fan-out)             │
                    │                                     │
                    │   🤖 indicator_node  🤖 trend_node   │
                    │   🤖 sentiment_node  🤖 pattern_node │
                    │            ↓  (Fan-in)  ↓           │
                    │       ⚙️ aggregator_node             │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    route_entry_or_position 路由      │
                    │         (基于 has_position)         │
                    └───────────┬─────────────┬───────────┘
                                │             │
            has_position=False  │             │  has_position=True
                                ▼             ▼
            ┌─────────────────────┐   ┌──────────────────────┐
            │     ENTRY PATH      │   │    POSITION PATH     │
            └─────────────────────┘   └──────────────────────┘
                     │                         │
                     ▼                         ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐
│    STAGE 2: Debate Subgraph    │   │ STAGE 2: Position Debate       │
│                                │   │                                │
│    🤖 bull_node                │   │    🤖 position_bull_node       │
│         ↓                      │   │         ↓                      │
│    🤖 bear_node                │   │    🤖 position_bear_node       │
│         ↓                      │   │         ↓                      │
│    🤖 judge_node               │   │    🤖 position_judge_node      │
└────────────────┬───────────────┘   └────────────────┬───────────────┘
                 │                                    │
                 ▼                                    ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐
│  STAGE 3: Enhanced Grounding   │   │  STAGE 3: Position Grounding   │
│     ⚙️ grounding_node           │   │  ⚙️ position_grounding_node    │
│                                │   │                                │
│  • 提取指标声明                 │   │  • 提取 MFE/MAE 声明            │
│  • 对比实际数据                 │   │  • 对比实际持仓数据             │
│  • ❌ 标记错误声明              │   │  • ❌ 标记错误声明              │
│  • ✅ 注入正确值到 state        │   │  • ✅ 注入正确值到 state        │
│  • 计算幻觉分数                 │   │  • 计算幻觉分数                 │
│                                │   │                                │
│  新增输出:                      │   │  新增输出:                      │
│  - corrected_context (text)    │   │  - corrected_context (text)    │
│  - grounding_summary (text)    │   │  - grounding_summary (text)    │
└────────────────┬───────────────┘   └────────────────┬───────────────┘
                 │                                    │
                 │ (幻觉分>70%直接END)                 │ (幻觉分>70%直接END)
                 ▼                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                  STAGE 4: Executor Agent (🤖 LLM)                  │
│                                                                    │
│  输入:                                                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ • Analysis 共识: direction + confidence                       │ │
│  │ • Debate 结果: Bull论点 + Bear论点 + Judge裁决                 │ │
│  │ • Grounding 结果:                                             │ │
│  │   - 幻觉分数                                                   │ │
│  │   - 错误声明列表                                               │ │
│  │   - 纠正后的数据上下文 (corrected_context)                     │ │
│  │ • 风控配置: 最大杠杆、最小置信度、持仓限制等                    │ │
│  │ • 当前市场价格、支撑/阻力位                                    │ │
│  │ • (Position Path) 持仓信息: MFE/MAE/drawdown/hold_count       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  职责:                                                             │
│  1. 综合评估所有 agents 的分析结果                                  │
│  2. 考虑 Grounding 纠正后的真实数据                                 │
│  3. 应用风控规则 (内化到 prompt 中)                                 │
│  4. 做出最终交易决策                                               │
│                                                                    │
│  输出:                                                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ • action: signal_entry_long/short/exit/hold/wait/adjust      │ │
│  │ • confidence: 0-100                                          │ │
│  │ • leverage: 1-100 (自动根据风控计算)                           │ │
│  │ • stop_loss: 价格                                             │ │
│  │ • take_profit: 价格                                           │ │
│  │ • adjustment_pct: (用于 adjust_position)                      │ │
│  │ • reasoning: 完整决策推理                                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                    END
```

---

## 📈 LLM 调用汇总

| 阶段          | 节点                                     |  LLM  | 说明                       |
| ------------- | ---------------------------------------- | :---: | -------------------------- |
| Analysis      | indicator_node                           |   🤖   | 技术指标分析               |
| Analysis      | trend_node                               |   🤖   | 趋势结构分析 (可用 Vision) |
| Analysis      | sentiment_node                           |   🤖   | 市场情绪分析               |
| Analysis      | pattern_node                             |   🤖   | K线形态识别 (可用 Vision)  |
| Analysis      | aggregator_node                          |   ⚙️   | 加权共识计算               |
| Debate        | bull_node / position_bull_node           |   🤖   | 看多方论证                 |
| Debate        | bear_node / position_bear_node           |   🤖   | 看空方论证                 |
| Debate        | judge_node / position_judge_node         |   🤖   | 仲裁员裁决                 |
| Grounding     | grounding_node / position_grounding_node |   ⚙️   | 幻觉检测+纠正              |
| **Execution** | **executor_agent**                       | **🤖** | **最终交易决策 (NEW)**     |

**总计: 8 次 LLM 调用** (4 分析 + 3 辩论 + 1 执行)

---

## 📁 文件修改清单

### 1. 新增文件

| 文件路径                               | 说明                               |
| -------------------------------------- | ---------------------------------- |
| `prompts/execution/__init__.py`        | 模块初始化                         |
| `prompts/execution/executor_prompt.py` | Executor Agent 的 prompt 模板      |
| `nodes/execution/executor_agent.py`    | 新的 LLM-based Executor Agent 节点 |

### 2. 修改文件

| 文件路径                                    | 修改内容                                                                  |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| `main_graph.py`                             | 移除 `validator_node`/`position_validator_node`，统一 Entry/Position 路径 |
| `nodes/verification/grounding_node.py`      | 增强：检测 + 纠正，输出 `corrected_context`                               |
| `nodes/position/position_grounding_node.py` | 增强：检测 + 纠正，输出 `corrected_context`                               |
| `state.py`                                  | 新增丰富的 State 字段                                                     |
| `edges/routing.py`                          | 简化路由逻辑                                                              |
| `subgraphs/position_graph.py`               | 移除内部 grounding (移到主图统一处理)                                     |

### 3. 删除/废弃文件

| 文件路径                                            | 原因                          |
| --------------------------------------------------- | ----------------------------- |
| `nodes/execution/validator_node.py`                 | 风控逻辑内化到 Executor Agent |
| `main_graph.py` 中的 `position_validator_node` 函数 | 同上                          |

---

## 🗂️ State 数据结构设计 (丰富版)

### 核心设计原则

1. **完整性**: 保存每个阶段的完整输出，方便回溯
2. **可追溯**: 记录时间戳、执行路径、错误信息
3. **可扩展**: 预留字段用于学习系统、回测分析
4. **序列化友好**: 所有字段都可以 JSON 序列化

### 新增 State 字段

```python
class TradingDecisionState(TypedDict, total=False):
    """
    Main Graph State - 丰富版
    
    设计原则:
    - 完整记录每个阶段的输入/输出
    - 支持后续的学习系统和回测分析
    - 便于日志记录和可视化
    """
    
    # ==================== 输入字段 ====================
    
    # 基础信息
    pair: str                               # 交易对，如 "BTC/USDT:USDT"
    current_price: float                    # 当前价格
    timeframe: str                          # 主时间框架，如 "15m"
    timeframe_htf: Optional[str]            # 高时间框架，如 "1h"
    
    # 市场数据
    market_context: str                     # 完整市场上下文 (ContextBuilder 输出)
    ohlcv_data: Optional[Any]               # 主时间框架 OHLCV 数据
    ohlcv_data_htf: Optional[Any]           # 高时间框架 OHLCV 数据
    
    # 持仓状态
    has_position: bool                      # 是否有持仓
    position_side: Optional[str]            # 持仓方向 "long" / "short"
    position_profit_pct: Optional[float]    # 当前盈亏百分比
    position_entry_price: Optional[float]   # 入场价格
    position_size: Optional[float]          # 持仓数量
    position_leverage: Optional[int]        # 持仓杠杆
    
    # 配置
    llm_config: Optional[dict]              # LLM 配置
    risk_config: Optional[dict]             # 风控配置
    
    # ==================== Analysis 阶段输出 ====================
    
    # 个体 Agent 报告
    indicator_report: Optional[AgentReport]
    trend_report: Optional[AgentReport]
    sentiment_report: Optional[AgentReport]
    pattern_report: Optional[AgentReport]
    
    # 聚合结果
    consensus_direction: Optional[Direction]
    consensus_confidence: float
    key_support: Optional[float]
    key_resistance: Optional[float]
    weighted_scores: Optional[dict]         # {"long": 0.8, "short": 0.2, "neutral": 0.0}
    
    # 分析元数据
    analysis_timestamp: Optional[str]       # 分析完成时间
    analysis_duration_ms: Optional[float]   # 分析耗时 (毫秒)
    analysis_token_usage: Optional[dict]    # Token 使用统计
    
    # ==================== Debate 阶段输出 ====================
    
    # Entry Path 辩论
    bull_argument: Optional[DebateArgument]
    bear_argument: Optional[DebateArgument]
    judge_verdict: Optional[JudgeVerdict]
    
    # Position Path 辩论
    position_bull_argument: Optional[DebateArgument]
    position_bear_argument: Optional[DebateArgument]
    position_judge_verdict: Optional[PositionJudgeVerdict]
    
    # 辩论元数据
    debate_path: Optional[str]              # "entry" 或 "position"
    debate_timestamp: Optional[str]
    debate_duration_ms: Optional[float]
    debate_token_usage: Optional[dict]
    debate_round: Optional[int]             # 辩论轮次 (预留多轮辩论)
    
    # ==================== Grounding 阶段输出 (增强) ====================
    
    # 核心结果
    grounding_result: Optional[Any]         # GroundingResult dataclass
    grounding_verified: bool                # 是否通过验证
    
    # 幻觉检测详情
    hallucination_score: Optional[float]    # 幻觉分数 0-100
    total_claims: Optional[int]             # 总声明数
    verified_claims: Optional[int]          # 已验证声明数
    false_claims: Optional[int]             # 错误声明数
    
    # 纠正后的数据 (NEW - 核心增强)
    corrected_values: Optional[Dict[str, float]]  # 纠正后的指标值
    corrected_context: Optional[str]              # 纠正后的上下文文本 (给 Executor)
    grounding_summary: Optional[str]              # 简洁的验证摘要
    
    # 错误声明详情 (用于分析和学习)
    false_claim_details: Optional[List[Dict]]     # 错误声明详情列表
    # 格式: [{"claim": "RSI=28", "actual": 45.2, "source": "bull", "type": "quantitative"}]
    
    # Position Path Grounding (独立字段)
    position_grounding_result: Optional[Any]
    position_hallucination_score: Optional[float]
    position_corrected_values: Optional[Dict[str, float]]
    position_corrected_context: Optional[str]
    position_grounding_summary: Optional[str]
    position_false_claim_details: Optional[List[Dict]]
    
    # Grounding 元数据
    grounding_timestamp: Optional[str]
    grounding_duration_ms: Optional[float]
    
    # ==================== Executor Agent 输出 (NEW) ====================
    
    # 最终决策
    final_action: Optional[str]             # "signal_entry_long", "signal_entry_short", 
                                            # "signal_exit", "signal_hold", "signal_wait",
                                            # "adjust_position"
    final_confidence: float                 # 最终置信度 0-100
    final_direction: Optional[Direction]    # 最终方向
    
    # 交易参数
    final_leverage: Optional[int]           # 杠杆倍数
    stop_loss_price: Optional[float]        # 止损价格
    take_profit_price: Optional[float]      # 止盈价格
    stop_loss_pct: Optional[float]          # 止损百分比
    take_profit_pct: Optional[float]        # 止盈百分比
    risk_reward_ratio: Optional[float]      # 风险回报比
    
    # 持仓调整 (Position Path)
    adjustment_pct: Optional[float]         # 调整百分比 (+20~+50 / -30~-70)
    adjustment_type: Optional[str]          # "scale_in" / "partial_exit"
    
    # 决策推理 (NEW)
    executor_reasoning: Optional[str]       # Executor Agent 的完整推理过程
    executor_key_factors: Optional[List[str]]  # 关键决策因素列表
    executor_risk_assessment: Optional[str]    # 风险评估
    executor_confidence_breakdown: Optional[Dict[str, float]]  # 置信度分解
    # 格式: {"analysis": 70, "debate": 80, "grounding": 90}
    
    # Executor 元数据
    executor_timestamp: Optional[str]
    executor_duration_ms: Optional[float]
    executor_token_usage: Optional[dict]
    executor_model: Optional[str]           # 使用的模型名称
    
    # ==================== 执行结果 ====================
    
    execution_result: Optional[dict]        # 最终执行结果 (传给策略)
    is_valid: bool                          # 决策是否有效
    
    # ==================== 元数据与追溯 ====================
    
    # 执行路径追溯
    execution_path: Optional[str]           # "entry" 或 "position"
    nodes_executed: Optional[List[str]]     # 执行的节点列表
    # 例: ["analysis", "entry_debate", "entry_grounding", "executor"]
    
    # 时间追溯
    thread_id: str                          # 唯一执行 ID
    created_at: str                         # 创建时间
    completed_at: Optional[str]             # 完成时间
    total_duration_ms: Optional[float]      # 总耗时
    
    # Token 使用汇总
    total_token_usage: Optional[dict]       # 总 Token 使用
    # 格式: {"prompt_tokens": 5000, "completion_tokens": 2000, "total": 7000}
    
    # 错误追溯
    errors: Annotated[List[str], operator.add]  # 错误列表
    warnings: Annotated[List[str], operator.add]  # 警告列表
    
    # ==================== 预留字段 (学习系统/回测) ====================
    
    # 学习系统字段
    decision_id: Optional[str]              # 决策唯一 ID (用于关联后续 reward)
    historical_similar_decisions: Optional[List[Dict]]  # 相似历史决策
    expected_outcome: Optional[str]         # 预期结果
    
    # 回测分析字段
    backtest_mode: Optional[bool]           # 是否回测模式
    backtest_timestamp: Optional[str]       # 回测时间点
    actual_outcome: Optional[Dict]          # 实际结果 (回测后填充)
    # 格式: {"pnl_pct": 2.5, "duration_hours": 4, "exit_reason": "take_profit"}
    
    # 可视化字段
    chart_image_path: Optional[str]         # 图表图片路径
    chart_annotations: Optional[List[Dict]] # 图表标注
    # 格式: [{"type": "entry", "price": 50000, "timestamp": "..."}]
```

### 新增数据类

```python
@dataclass
class ExecutorDecision:
    """
    Executor Agent 决策结构
    """
    action: str                             # 交易动作
    confidence: float                       # 置信度 0-100
    direction: Optional[Direction]          # 方向
    leverage: Optional[int]                 # 杠杆
    
    # 止损止盈
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # 持仓调整
    adjustment_pct: Optional[float] = None
    adjustment_type: Optional[str] = None
    
    # 推理
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    model_used: Optional[str] = None
    token_usage: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """转为字典"""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "direction": self.direction.value if self.direction else None,
            "leverage": self.leverage,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "adjustment_pct": self.adjustment_pct,
            "adjustment_type": self.adjustment_type,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "risk_assessment": self.risk_assessment,
            "confidence_breakdown": self.confidence_breakdown,
            "model_used": self.model_used,
            "token_usage": self.token_usage
        }


@dataclass
class GroundingCorrection:
    """
    Grounding 纠正记录
    """
    indicator: str                          # 指标名称
    claimed_value: Optional[float]          # 声称的值
    actual_value: float                     # 实际值
    source: str                             # 来源 "bull" / "bear"
    claim_text: str                         # 原始声明文本
    claim_type: str                         # "quantitative" / "qualitative"
    discrepancy_pct: float                  # 偏差百分比
    is_false: bool                          # 是否为错误声明
    correction_applied: bool                # 是否已纠正
    
    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "claimed_value": self.claimed_value,
            "actual_value": self.actual_value,
            "source": self.source,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "discrepancy_pct": self.discrepancy_pct,
            "is_false": self.is_false,
            "correction_applied": self.correction_applied
        }


@dataclass 
class ExecutionMetrics:
    """
    执行指标 (用于监控和优化)
    """
    total_duration_ms: float
    analysis_duration_ms: float
    debate_duration_ms: float
    grounding_duration_ms: float
    executor_duration_ms: float
    
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    
    llm_calls_count: int
    successful_llm_calls: int
    failed_llm_calls: int
    
    nodes_executed: List[str]
    execution_path: str  # "entry" / "position"
    
    def to_dict(self) -> dict:
        return {
            "total_duration_ms": self.total_duration_ms,
            "analysis_duration_ms": self.analysis_duration_ms,
            "debate_duration_ms": self.debate_duration_ms,
            "grounding_duration_ms": self.grounding_duration_ms,
            "executor_duration_ms": self.executor_duration_ms,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls_count": self.llm_calls_count,
            "successful_llm_calls": self.successful_llm_calls,
            "failed_llm_calls": self.failed_llm_calls,
            "nodes_executed": self.nodes_executed,
            "execution_path": self.execution_path
        }
```

---

## 🔧 Grounding Node 增强设计

### corrected_context 输出格式

```
=== Grounding 验证结果 ===
幻觉检测分数: 25% (可接受)

❌ 错误声明已纠正:
  - Bull声称 "RSI=28 (超卖)" → 实际值: RSI=45 (中性)
  - Bear声称 "ADX>50 (强趋势)" → 实际值: ADX=32 (中等趋势)

✅ 验证通过的声明:
  - MACD金叉: 正确
  - EMA200上方: 正确

📊 纠正后的指标数据:
  RSI: 45.2 (中性区域, 30-70之间)
  MACD: 0.0025 (正值, 看多信号)
  ADX: 32.1 (中等趋势强度)
  资金费率: 0.01% (中性)
  OI变化: +2.3% (多头增持)

⚠️ 请基于以上纠正后的数据做出决策
```

### grounding_summary 格式

```
验证: 8/10 通过 | 幻觉分: 25% | 纠正: 2项 | 置信度惩罚: -10%
```

---

## 📝 Executor Agent Prompt 设计

### System Prompt

```python
EXECUTOR_SYSTEM_PROMPT = """
你是一个专业的加密货币交易执行专家。你的职责是基于多个分析 Agent 的结果做出最终交易决策。

<核心职责>
1. 综合评估所有分析 agents 的结论
2. 基于 Grounding 纠正后的真实数据做决策
3. 严格遵守风控规则
4. 给出明确、可执行的交易指令
</核心职责>

<决策优先级>
1. Grounding 纠正后的数据 > Agent 的原始声明
2. 风控规则是硬性约束，不可违反
3. 当信息冲突时，以更保守的方向决策
</决策优先级>

<风控规则>
{risk_rules}
</风控规则>

<输出格式>
必须严格按照以下格式输出:

[决策]
action: <操作类型>
confidence: <置信度 0-100>
leverage: <杠杆倍数>
direction: <方向 LONG/SHORT/NEUTRAL>

[风险管理] (入场时必填)
stop_loss_price: <止损价格>
take_profit_price: <止盈价格>
risk_reward_ratio: <风险回报比>

[调整参数] (adjust_position 时必填)
adjustment_pct: <调整百分比>
adjustment_type: <scale_in/partial_exit>

[决策理由]
<你的完整推理过程>

[关键因素]
- <因素1>
- <因素2>
- <因素3>

[风险评估]
<对当前决策的风险评估>
</输出格式>
"""
```

### User Prompt Builder

```python
def build_executor_prompt(
    consensus_summary: str,
    debate_summary: str,
    grounding_summary: str,
    corrected_context: str,
    risk_config: Dict,
    has_position: bool,
    position_info: Optional[str],
    current_price: float,
    key_support: Optional[float],
    key_resistance: Optional[float]
) -> str:
    """构建 Executor Agent 的 prompt"""
    
    if has_position:
        action_options = """
[可选操作]
- HOLD: 继续持有，不做调整
- EXIT: 平仓离场  
- SCALE_IN: 加仓 (需指定 adjustment_pct: +20% ~ +50%)
- PARTIAL_EXIT: 部分平仓 (需指定 adjustment_pct: -30% ~ -70%)
"""
    else:
        action_options = """
[可选操作]
- ENTRY_LONG: 开多仓 (需设置止损止盈)
- ENTRY_SHORT: 开空仓 (需设置止损止盈)
- WAIT: 等待更好的入场机会
"""
    
    return f"""
=== 当前市场 ===
价格: {current_price}
支撑位: {key_support or 'N/A'}
阻力位: {key_resistance or 'N/A'}

=== 分析共识 (4 个 Agent 加权结果) ===
{consensus_summary}

=== 辩论结果 (Bull vs Bear) ===
{debate_summary}

=== Grounding 验证结果 (已纠正) ===
{grounding_summary}

{corrected_context}

=== 持仓状态 ===
{'已有持仓:\n' + position_info if has_position else '当前无持仓'}

{action_options}

请基于以上信息，做出你的最终交易决策。
注意: 必须基于 Grounding 纠正后的数据，而非原始声明。
"""
```

---

## 📊 新的 Main Graph 结构

```python
def build_main_graph():
    """构建新的统一主图"""
    builder = StateGraph(TradingDecisionState)
    
    # === Stage 1: Analysis ===
    analysis_graph = build_analysis_subgraph()
    builder.add_node("analysis", analysis_graph)
    
    # === Stage 2: Debate ===
    debate_graph = build_debate_subgraph()
    position_debate_graph = build_position_debate_subgraph()  # 不含 grounding
    
    builder.add_node("entry_debate", debate_graph)
    builder.add_node("position_debate", position_debate_graph)
    
    # === Stage 3: Grounding (统一) ===
    builder.add_node("entry_grounding", grounding_node)
    builder.add_node("position_grounding", position_grounding_node)
    
    # === Stage 4: Executor Agent (NEW - LLM) ===
    builder.add_node("executor", executor_agent_node)
    
    # === Edges ===
    builder.add_edge(START, "analysis")
    
    # Analysis → Routing
    builder.add_conditional_edges(
        "analysis",
        route_entry_or_position,
        {
            "entry_debate": "entry_debate",
            "position_debate": "position_debate"
        }
    )
    
    # === Entry Path ===
    builder.add_edge("entry_debate", "entry_grounding")
    builder.add_conditional_edges(
        "entry_grounding",
        route_after_grounding,
        {
            "executor": "executor",
            "end": END
        }
    )
    
    # === Position Path === 
    builder.add_edge("position_debate", "position_grounding")
    builder.add_conditional_edges(
        "position_grounding",
        route_after_grounding,
        {
            "executor": "executor", 
            "end": END
        }
    )
    
    # === Executor → END ===
    builder.add_edge("executor", END)
    
    return builder.compile()
```

---

## 🚀 实施计划

### Phase 1: 基础设施 (Day 1) ✅

- [x] 创建 `prompts/execution/__init__.py`
- [x] 创建 `prompts/execution/executor_prompt.py`
- [x] 更新 `state.py` 添加新字段和数据类

### Phase 2: Grounding 增强 (Day 1-2) ✅

- [x] 修改 `nodes/verification/grounding_node.py`
  - [x] 添加 `_build_corrected_context()` 函数
  - [x] 添加 `_build_grounding_summary()` 函数
  - [x] 更新返回值包含新字段
- [x] 修改 `nodes/position/position_grounding_node.py`
  - [x] 同样的增强

### Phase 3: Executor Agent (Day 2) ✅

- [x] 创建 `nodes/execution/executor_agent.py`
  - [x] 实现 `executor_agent_node()` 函数
  - [x] 实现响应解析器
- [ ] 编写单元测试

### Phase 4: Main Graph 重构 (Day 3) ✅

- [x] 修改 `main_graph.py`
  - [x] 移除 `validator_node` 相关代码
  - [x] 移除 `position_validator_node` 函数
  - [x] 统一 Entry/Position 路径
  - [x] 添加新的 executor 节点
- [x] 修改 `edges/routing.py`
  - [x] 简化路由逻辑
  - [x] 更新路由到 executor
- [x] 创建 `subgraphs/position_debate_graph.py`
  - [x] 不含 grounding 的 position debate 子图

### Phase 5: 清理与测试 (Day 3-4)

- [ ] 删除 `nodes/execution/validator_node.py` (保留为 Legacy)
- [x] 更新 `nodes/execution/__init__.py`
- [ ] 运行集成测试
- [ ] 修复发现的问题

### Phase 6: 文档与优化 (Day 4)

- [ ] 更新 README 文档
- [ ] 调优 Executor Agent prompt
- [ ] 性能测试

---

## ✅ 验收标准

1. **功能验收**
   - [ ] Entry Path 和 Position Path 步骤数一致 (5 步)
   - [ ] Grounding 输出包含 `corrected_context`
   - [ ] Executor Agent 能正确解析并执行
   - [ ] 风控规则在 Executor Agent 中生效

2. **性能验收**
   - [ ] 总 LLM 调用次数: 8 次
   - [ ] 总执行时间增加不超过 20%

3. **质量验收**
   - [ ] 所有现有测试通过
   - [ ] 新增测试覆盖关键路径
   - [ ] 日志记录完整

---

## 📌 注意事项

1. **向后兼容**: `execution_result` 格式保持兼容，策略层无需修改
2. **错误处理**: Executor Agent 失败时回退到保守决策 (WAIT/HOLD)
3. **Token 控制**: Executor prompt 控制在 2000 tokens 以内
4. **幻觉阈值**: >70% 幻觉分直接 END，不进入 Executor
