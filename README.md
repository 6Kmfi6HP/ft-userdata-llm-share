# Freqtrade LLM Function Calling Strategy

> 基于大语言模型（LLM）Function Calling 和 RAG 技术的智能加密货币交易策略

[![Freqtrade](https://img.shields.io/badge/freqtrade-stable-blue)](https://www.freqtrade.io/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [RAG学习系统](#rag学习系统)
- [使用指南](#使用指南)
- [故障排除](#故障排除)
- [技术细节](#技术细节)
- [更新日志](#更新日志)

---

## 🎯 项目简介

这是一个创新的加密货币自动化交易策略，将 **Freqtrade** 交易框架与 **大语言模型（LLM）** 深度整合，通过 OpenAI Function Calling 和 RAG（检索增强生成）技术实现智能交易决策。

### 为什么选择 LLM 策略？

- **🧠 智能决策**: LLM 可以理解复杂的市场情况，提供类人的交易判断
- **📚 经验学习**: RAG 系统从历史交易中学习，不断优化决策质量
- **🎯 精准控制**: Function Calling 提供 9 个核心交易函数，实现精细化交易管理
- **📊 多维度分析**: 综合技术指标、账户状态、持仓情况、市场情绪进行全局决策
- **🔄 自我评价**: 模型对每笔交易打分并反思，持续优化决策能力

---

## ✨ 核心特性

### 1. **OpenAI Function Calling 完整交易控制**

LLM 通过 **9 个核心函数**完全控制交易流程：

#### 交易控制函数（6个）

| 函数名称 | 功能描述 | 主要参数 |
|---------|---------|---------|
| `signal_entry_long` | 开多仓 | 杠杆、置信度、关键位、投入金额 |
| `signal_entry_short` | 开空仓 | 杠杆、置信度、关键位、投入金额 |
| `signal_exit` | 平仓 + 自我评价 | 置信度、RSI、**trade_score**(0-100) |
| `adjust_position` | 加仓/减仓 | 调整百分比、关键位、理由 |
| `signal_hold` | 保持持仓 | 置信度、理由 |
| `signal_wait` | 空仓观望 | 置信度、理由 |

#### RAG学习函数（3个）

| 函数名称 | 功能描述 | 用途 |
|---------|---------|------|
| `record_decision_to_rag` | 记录决策到RAG | 盈利>5%时记录持仓决策，供未来学习 |
| `query_rag_stats` | 查询RAG统计 | 查看历史记录数量、存储状态 |
| `cleanup_rag_history` | 清理RAG历史 | 删除低质量或过时记录 |

### 2. **RAG 完整学习系统**

#### 核心组件

```
RAG Learning System
├── 向量语义检索
│   ├── FAISS 向量存储（毫秒级检索）
│   ├── text-embedding-bge-m3 嵌入模型
│   └── 相似历史交易检索
│
├── 模型自我评价
│   ├── 平仓时对交易打分（0-100）
│   ├── 反思入场时机、持仓管理、风险控制
│   └── 评分自动记录到RAG系统
│
├── 交易评估器（TradeEvaluator）
│   ├── 盈利评分（profit_score）
│   ├── 风险管理评分（risk_score）
│   ├── 时机把握评分（timing_score）
│   ├── 资金效率评分（efficiency_score）
│   └── 综合评级（S/A/B/C/D/F）
│
└── 奖励学习系统（RewardLearning）
    ├── 基于评分构建奖励函数
    ├── 自动识别成功/失败模式
    ├── 生成学习指导和警告
    └── 追踪累计奖励趋势
```

#### RAG工作流程

```
平仓时:
  ├─ 1. 模型调用 signal_exit，提供 trade_score(0-100)
  ├─ 2. TradeEvaluator 分析交易质量
  │      ├─ 盈利评分（是否达到目标）
  │      ├─ 风险评分（止损执行、杠杆使用）
  │      ├─ 时机评分（入场/出场时机）
  │      └─ 效率评分（收益/时间比）
  ├─ 3. 生成交易总结
  │      ├─ 优点：做对了什么
  │      ├─ 缺点：哪里可以改进
  │      └─ 教训：未来如何避免
  ├─ 4. 向量化并存储到FAISS
  │      └─ 使用 text-embedding-bge-m3 生成向量
  ├─ 5. 奖励学习
  │      ├─ 计算奖励值（reward）
  │      ├─ 记录奖励历史
  │      └─ 更新学习曲线
  └─ 6. 下次决策时
         └─ 检索相似历史 → 提供给LLM参考
```

### 3. **增强决策模块**

| 模块名称 | 功能描述 |
|---------|---------|
| **PositionTracker** | 持仓追踪：实时记录MFE/MAE、决策历史 |
| **MarketStateComparator** | 市场对比：对比开仓时和当前的市场变化 |
| **DecisionChecker** | 决策检查：验证开仓信号是否符合规则 |
| **TradeReviewer** | 交易复盘：生成详细的交易报告和教训 |

### 4. **期货交易完整支持**

- ✅ 多空双向交易（做多/做空）
- ✅ 动态杠杆（1-100x，由 LLM 决定）
- ✅ 灵活投入（可指定具体USDT金额）
- ✅ 仓位调整（加仓/减仓）
- ✅ 多重风控（止损、最大回撤、仓位限制）

### 5. **多时间框架技术分析**

支持 4 个时间框架同时分析：
- **30分钟**（主时间框架）
- **1小时**
- **4小时**
- **日线**

技术指标包括：
- 趋势：EMA(20/50/200)
- 动量：RSI、MACD
- 波动：布林带、ATR
- 强度：ADX、MFI、OBV
- 结构：价格形态、支撑阻力

### 6. **完整的日志和监控**

```
日志系统
├── freqtrade.log              # 主日志：策略运行、交易执行
├── llm_decisions.jsonl        # LLM决策日志：每次决策的上下文和结果
├── trade_experience.jsonl     # 交易经验日志：完整的交易记录
└── rag/                       # RAG存储
    ├── vector_store/          # FAISS向量索引
    ├── metadata.json          # 交易元数据
    └── rewards.jsonl          # 奖励学习记录
```

---

## 🏗️ 系统架构

```
ft-userdata-llm/
├── docker-compose.yml          # Docker 编排配置
├── Dockerfile.custom           # 自定义镜像（numpy + faiss-cpu）
├── manage.sh                   # 一键管理脚本
├── README.md                   # 本文档
├── README_RAG.md              # RAG系统详细文档
│
└── user_data/
    ├── config.json             # 核心配置文件
    │
    ├── strategies/
    │   ├── LLMFunctionStrategy.py    # 主策略文件
    │   │
    │   └── llm_modules/              # LLM 模块
    │       │
    │       ├── llm/                  # LLM 核心
    │       │   ├── llm_client.py         # OpenAI API 封装
    │       │   └── function_executor.py  # Function Calling 执行器
    │       │
    │       ├── learning/             # 学习系统 ⭐
    │       │   ├── embedding_service.py  # 嵌入服务（bge-m3）
    │       │   ├── vector_store.py       # FAISS 向量存储
    │       │   ├── rag_manager.py        # RAG 管理器
    │       │   ├── trade_evaluator.py    # 交易评估器
    │       │   └── reward_learning.py    # 奖励学习
    │       │
    │       ├── tools/                # 交易工具
    │       │   └── trading_tools.py      # 9个核心交易函数
    │       │
    │       ├── experience/           # 经验系统
    │       │   ├── experience_manager.py # 经验管理
    │       │   ├── trade_logger.py       # 交易日志
    │       │   ├── trade_reviewer.py     # 交易复盘
    │       │   ├── position_tracker.py   # 持仓追踪
    │       │   ├── market_comparator.py  # 市场对比
    │       │   ├── decision_checker.py   # 决策检查
    │       │   └── simple_historical_context.py
    │       │
    │       └── utils/                # 工具类
    │           ├── config_loader.py      # 配置加载
    │           ├── context_builder.py    # 上下文构建
    │           ├── indicator_calculator.py
    │           └── market_sentiment.py   # 市场情绪
    │
    ├── data/
    │   └── rag/                  # RAG 数据存储
    │       ├── vector_store/     # FAISS 向量索引
    │       └── metadata/         # 交易元数据
    │
    └── logs/
        ├── freqtrade.log         # 主日志
        ├── llm_decisions.jsonl   # LLM 决策日志
        └── trade_experience.jsonl # 交易经验日志
```

---

## 🚀 快速开始

### 前置要求

- **Docker** 和 **Docker Compose** 已安装
- **LLM API**: 支持 OpenAI Function Calling 的 API
- **Embedding API**: 支持 text-embedding-bge-m3 的 API
- **币安账户**（或其他支持的交易所）

### 1. 配置 LLM 和 Embedding API

编辑 `user_data/config.json`：

```json
{
  "llm_config": {
    "api_base": "http://host.docker.internal:3120",
    "api_key": "sk-your-api-key",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 60
  },

  "experience_config": {
    "enable_rag": true,
    "rag_config": {
      "embedding": {
        "model_name": "text-embedding-bge-m3",
        "api_url": "http://host.docker.internal:3120",
        "api_key": "sk-your-api-key",
        "api_type": "openai",
        "dimension": 1024
      },
      "vector_store": {
        "index_type": "flat",
        "storage_path": "./user_data/rag/vector_store"
      },
      "similarity_threshold": 0.7,
      "top_k": 5
    }
  }
}
```

**支持的 API 类型**:
- `openai`: OpenAI 兼容 API（推荐）
- `ollama`: Ollama 本地部署
- `local`: sentence-transformers 本地模型

### 2. 配置交易所

```json
{
  "exchange": {
    "name": "binance",
    "key": "your-api-key",
    "secret": "your-api-secret",
    "ccxt_config": {
      "enableRateLimit": true,
      "options": {
        "defaultType": "future"
      }
    }
  }
}
```

### 3. 启动策略

```bash
# 赋予执行权限
chmod +x manage.sh

# 快速启动（推荐）
./manage.sh start

# 完整部署（首次启动或更新后）
./manage.sh deploy
```

**manage.sh 功能**:
```
1) 快速启动 (直接启动 + 查看日志) ⚡
2) 快速重启 (重启容器 + 查看日志)
3) 完整部署 (构建镜像 + 启动 + 查看日志)
4) 只查看日志
5) 清理所有数据
6) 停止服务
```

### 4. 监控运行

- **日志**: `./manage.sh logs` 或 `docker logs -f freqtrade-llm`
- **Web UI**: http://localhost:8086
  - 用户名: `freqtrader`
  - 密码: 见配置文件
- **API**: http://localhost:8086/api/v1/

---

## ⚙️ 配置说明

### 核心配置项

#### 1. **LLM 配置**

```json
"llm_config": {
    "api_base": "http://host.docker.internal:3120",
    "api_key": "sk-xxx",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "temperature": 0.7,                // 创造性 (0.0-1.0)
    "max_tokens": 2000,                // 最大输出长度
    "timeout": 60,                     // 请求超时时间
    "retry_times": 2                   // 失败重试次数
}
```

**temperature 建议**:
- `0.0-0.3`: 保守稳健，严格遵循技术指标
- `0.4-0.7`: 平衡模式（推荐），结合技术和直觉
- `0.8-1.0`: 激进创新，可能产生意外决策

**推荐模型**:
- **qwen/qwen3-30b-a3b-thinking-2507**: 深度思考能力强（推荐）
- **gpt-4-turbo**: OpenAI 官方，稳定可靠
- **deepseek-coder**: 成本低，速度快

#### 2. **RAG 配置**

```json
"experience_config": {
    "enable_rag": true,
    "log_decisions": true,
    "log_trades": true,

    "rag_config": {
        "embedding": {
            "model_name": "text-embedding-bge-m3",
            "api_url": "http://host.docker.internal:3120",
            "api_key": "sk-xxx",
            "api_type": "openai",
            "dimension": 1024,
            "batch_size": 8
        },
        "vector_store": {
            "index_type": "flat",          // flat|hnsw|ivf
            "storage_path": "./user_data/rag/vector_store"
        },
        "similarity_threshold": 0.7,       // 相似度阈值
        "top_k": 5,                        // 检索数量
        "enable_reward_learning": true,    // 启用奖励学习
        "min_trades_for_learning": 10      // 最少交易数
    }
}
```

**FAISS 索引类型**:
- `flat`: 精确搜索，适合 <10k 记录
- `hnsw`: HNSW 近似搜索，适合 10k-1M 记录
- `ivf`: IVF 索引，适合 >1M 记录

#### 3. **交易配置**

```json
{
    "max_open_trades": 5,              // 最大持仓数
    "stake_currency": "USDT",
    "stake_amount": "unlimited",       // unlimited = 动态分配
    "tradable_balance_ratio": 0.99,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "dry_run": true,                   // 模拟交易
    "dry_run_wallet": 1340
}
```

#### 4. **风险管理**

```json
"risk_management": {
    "max_leverage": 100,               // LLM 最大可用杠杆
    "default_leverage": 10,
    "max_position_pct": 50,            // 单仓位最大占用
    "max_open_trades": 5,
    "allow_model_freedom": true,       // 允许 LLM 自由决策
    "emergency_stop_loss": -0.15       // 账户紧急止损
}
```

#### 5. **上下文配置**

```json
"context_config": {
    "max_context_tokens": 6000,
    "include_multi_timeframe_data": true,
    "indicator_history_points": 20,
    "multi_timeframe_history": {
        "1h": {"candles": 200, "fields": [...]},
        "4h": {"candles": 180, "fields": [...]},
        "1d": {"candles": 150, "fields": [...]}
    }
}
```

---

## 📚 RAG学习系统

### 模型自我评价

平仓时，模型需要对自己的交易表现打分：

```python
signal_exit(
    pair="BTC/USDT:USDT",
    confidence_score=85,
    rsi_value=72,
    trade_score=78,  # 🌟 自我评分 0-100
    reason="""
    平仓理由：目标位已达，RSI超买

    自我反思：
    ✓ 优点：入场时机准确，在支撑位附近开仓
    ✓ 优点：持仓过程中耐心等待，没有过早平仓
    ✗ 缺点：可以在75000附近部分获利了结
    ✗ 缺点：持仓时间略长，资金效率不够高

    教训：下次在盈利超过8%时可考虑部分止盈
    """
)
```

### 交易评估器

TradeEvaluator 会对每笔交易进行多维度评分：

| 维度 | 权重 | 评分标准 |
|-----|------|---------|
| **盈利评分** | 30% | 是否达到盈利目标，亏损控制 |
| **风险评分** | 25% | 止损执行、杠杆合理性、回撤控制 |
| **时机评分** | 25% | 入场位置、出场时机、是否接近峰值 |
| **效率评分** | 20% | 收益/时间比，资金利用率 |

**最终评级**:
- **S级** (90-100): 完美交易
- **A级** (80-89): 优秀交易
- **B级** (70-79): 良好交易
- **C级** (60-69): 及格交易
- **D级** (50-59): 需改进
- **F级** (<50): 失败交易

### 奖励学习

基于交易评分构建奖励函数：

```python
奖励计算公式：
reward = (profit_pct / 100) * (score / 100) * leverage_factor

示例：
- 盈利 +8%, 评分 85, 杠杆 10x
  → reward = 0.08 * 0.85 * 1.0 = +0.068

- 亏损 -5%, 评分 60, 杠杆 15x
  → reward = -0.05 * 0.60 * 1.2 = -0.036
```

### RAG 检索示例

决策时，系统会检索相似历史：

```
🔍 检索到 3 条相似历史交易：

[1] BTC/USDT 做多 (相似度: 0.89) | 评分: 82/100
    入场: 支撑位反弹，RSI 45
    持仓: 3.5小时
    盈利: +12.3%
    教训: 支撑位开仓成功率高，耐心持有是关键

[2] ETH/USDT 做多 (相似度: 0.83) | 评分: 75/100
    入场: EMA20突破
    持仓: 2.1小时
    盈利: +6.8%
    教训: 突破后应等待回踩确认

[3] SOL/USDT 做多 (相似度: 0.78) | 评分: 45/100
    入场: 假突破
    持仓: 1.8小时
    亏损: -4.2%
    教训: 量能不足的突破容易失败
```

详细说明请查看 [README_RAG.md](README_RAG.md)

---

## 📖 使用指南

### 交易函数详解

#### 1. 开多仓 - signal_entry_long

```python
signal_entry_long(
    pair="BTC/USDT:USDT",
    leverage=10,                  # 杠杆倍数
    confidence_score=85,          # 置信度 1-100
    key_support=94000.0,          # 关键支撑位
    key_resistance=96000.0,       # 关键阻力位
    rsi_value=45,                 # 当前RSI
    trend_strength="强势",        # 趋势强度
    stake_amount=500.0,           # 🌟 投入500 USDT（可选）
    reason="价格突破EMA20，RSI超卖反弹，日线趋势向上"
)
```

#### 2. 平仓 + 自我评价 - signal_exit

```python
signal_exit(
    pair="BTC/USDT:USDT",
    confidence_score=90,
    rsi_value=78,
    trade_score=85,  # 🌟 自我评分 0-100
    reason="""
    平仓理由：达到目标利润，RSI超买

    自我反思：
    ✓ 入场时机好，在支撑位开仓
    ✓ 持仓耐心，没有过早平仓
    ✗ 可以在中途部分止盈

    教训：盈利>8%时考虑分批止盈
    """
)
```

#### 3. 记录决策到RAG - record_decision_to_rag

```python
# 在盈利>5%且继续持有时记录
record_decision_to_rag(
    pair="BTC/USDT:USDT",
    decision_type="hold",         # hold | exit
    reason="趋势仍然强劲，ADX高位，继续持有",
    confidence=0.85,
    current_profit_pct=7.5
)
```

#### 4. 查询RAG统计 - query_rag_stats

```python
query_rag_stats()  # 返回当前RAG系统状态

# 返回示例:
{
    "total_experiences": 156,
    "reward_stats": {
        "total_trades": 156,
        "avg_score": 73.5,
        "cumulative_reward": 12.34
    }
}
```

#### 5. 清理RAG历史 - cleanup_rag_history

```python
cleanup_rag_history(
    strategy="low_quality",       # low_quality | compress | old_records
    reason="删除评分<50的低质量记录"
)
```

### 完整决策流程示例

```
1. 新K线到来（30分钟周期）
   ↓
2. 构建市场上下文
   - 当前价格: 95,123 USDT
   - RSI: 45 (主), 52 (1h), 58 (4h), 62 (1d)
   - MACD: 转正
   - EMA20: 突破
   - 账户余额: 1,200 USDT
   - 持仓数: 2/5
   ↓
3. RAG检索相似历史
   - 找到3条相似的支撑位反弹案例
   - 成功率: 2/3
   - 平均盈利: +8.5%
   ↓
4. LLM分析决策
   思考过程:
   "价格在EMA20获得支撑并反弹
    RSI在超卖区域
    MACD即将金叉
    类似历史案例成功率高
    → 决定开多仓"
   ↓
5. 调用 signal_entry_long
   - 投入: 400 USDT
   - 杠杆: 10x
   - 置信度: 82
   ↓
6. 执行交易
   - 策略接收信号
   - 验证参数
   - 发送订单
   ↓
7. 记录决策
   - 保存到 llm_decisions.jsonl
   - 记录上下文和推理过程
```

---

## 🔧 故障排除

### 常见问题

#### 1. "工具实例缺少方法: record_decision_to_rag"

**原因**: RAG管理器未正确初始化或注册

**解决**:
```bash
# 检查配置文件
grep "enable_rag" user_data/config.json

# 重启容器
./manage.sh restart

# 查看日志确认
docker logs freqtrade-llm | grep "RAG"
```

应该看到：
```
✓ RAG 学习系统已启用
✓ RAG 工具函数已注册
```

#### 2. "OpenAI 嵌入失败: Expecting value"

**原因**: Embedding API路径或返回格式错误

**解决**:
```bash
# 测试API
curl http://host.docker.internal:3120/v1/embeddings \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-bge-m3",
    "input": ["test"]
  }'
```

修复后系统会自动尝试多个路径：
- `/v1/embeddings`
- `/embeddings`

#### 3. "can't subtract offset-naive and offset-aware datetimes"

**状态**: ✅ 已修复

**说明**: datetime时区兼容性问题已在最新版本中解决

#### 4. embedding返回随机向量

**状态**: ✅ 已修复

**说明**: 现在API失败时返回零向量而非随机向量，避免污染向量库

#### 5. LLM决策超时

```json
"llm_config": {
    "timeout": 120,               // 增加到120秒
    "max_tokens": 1500,           // 减少输出长度
    "temperature": 0.5            // 降低创造性
}
```

#### 6. FAISS索引创建失败

```bash
# 检查依赖
docker exec freqtrade-llm pip list | grep faiss

# 应该看到
faiss-cpu    1.7.0

# 如果没有，重新构建
./manage.sh deploy
```

#### 7. 清理所有数据重新开始

```bash
./manage.sh clean
```

**警告**: 这会删除：
- FAISS 向量数据库
- RAG 元数据
- 交易数据库
- 所有日志文件

---

## 🔬 技术细节

### RAG向量检索流程

```python
# 1. 交易完成时
trade_summary = """
    BTC/USDT 做多
    入场: 支撑位反弹 (95000)
    持仓: 3.5小时
    盈利: +12.3%
    评分: 85/100
    教训: 支撑位开仓成功率高
"""

# 2. 使用 bge-m3 生成向量
vector = embedding_service.embed(trade_summary)
# → [1024维向量]

# 3. 存储到 FAISS
vector_store.add(
    vector=vector,
    metadata={
        "pair": "BTC/USDT:USDT",
        "score": 85,
        "profit_pct": 12.3,
        "timestamp": "2025-01-15T10:30:00"
    }
)

# 4. 下次决策时检索
current_context = "BTC价格在EMA20支撑位反弹"
query_vector = embedding_service.embed(current_context)

similar_trades = vector_store.search(
    query_vector=query_vector,
    top_k=5,
    min_similarity=0.7
)

# 5. 返回相似历史
[
    {"similarity": 0.89, "metadata": {...}},
    {"similarity": 0.83, "metadata": {...}},
    ...
]
```

### 评分系统详解

```python
TradeEvaluator 评分逻辑:

1. 盈利评分 (30%)
   - profit > 10%: 满分 30
   - profit 5-10%: 20-30分
   - profit 0-5%: 10-20分
   - loss < -5%: 0-10分

2. 风险评分 (25%)
   - 止损执行: ±10分
   - 杠杆合理性: ±10分
   - 回撤控制: ±5分

3. 时机评分 (25%)
   - 入场位置 (MAE): ±12分
   - 出场时机 (接近MFE): ±13分

4. 效率评分 (20%)
   - profit/hour > 1%: 满分 20
   - profit/hour 0.5-1%: 15分
   - profit/hour 0.1-0.5%: 10分
   - profit/hour < 0.1%: 0-10分

最终得分 = 盈利 + 风险 + 时机 + 效率
```

### 奖励函数

```python
def calculate_reward(trade):
    """
    奖励 = 盈利% * (评分/100) * 杠杆系数

    杠杆系数:
    - 1-5x: 1.0
    - 6-10x: 1.1
    - 11-20x: 1.2
    - >20x: 1.3
    """
    base_reward = trade.profit_pct / 100
    score_factor = trade.score / 100
    leverage_factor = calculate_leverage_factor(trade.leverage)

    reward = base_reward * score_factor * leverage_factor

    return reward

# 示例
trade_1 = {
    "profit_pct": 12.5,
    "score": 85,
    "leverage": 10
}
reward_1 = 0.125 * 0.85 * 1.1 = 0.117  # 优秀交易

trade_2 = {
    "profit_pct": -5.0,
    "score": 55,
    "leverage": 15
}
reward_2 = -0.05 * 0.55 * 1.2 = -0.033  # 失败交易
```

### 上下文Token管理

```json
{
  "context_config": {
    "max_context_tokens": 6000,
    "allocation": {
      "system_prompt": 500,           // 系统指令
      "current_market": 800,          // 当前市场数据
      "account_info": 200,            // 账户余额、持仓
      "technical_indicators": 1000,   // 技术指标历史
      "rag_similar_trades": 1500,     // RAG检索结果
      "multi_timeframe": 2000         // 多时间框架
    }
  }
}
```

---

## 📊 性能监控

### Web UI 监控

访问 http://localhost:8086：

| 页面 | 功能 |
|-----|------|
| Dashboard | 实时持仓、收益曲线、账户余额 |
| Trades | 交易历史、盈亏统计、持仓分析 |
| Performance | 回撤分析、夏普比率、胜率 |
| Logs | 策略日志、错误日志 |

### 日志查看

```bash
# 主日志（包含LLM决策推理）
tail -f user_data/logs/freqtrade.log

# LLM决策详细日志
tail -f user_data/logs/llm_decisions.jsonl | jq .

# 交易经验日志
tail -f user_data/logs/trade_experience.jsonl | jq .
```

### RAG统计查看

```bash
# 在策略中查看
docker exec freqtrade-llm python3 << 'EOF'
from strategies.llm_modules.learning.rag_manager import RAGManager
from strategies.llm_modules.utils.config_loader import load_config

config = load_config("user_data/config.json")
rag_config = config["experience_config"]["rag_config"]
rag = RAGManager(rag_config)

print(f"总交易数: {len(rag.vector_store.metadata)}")
print(f"平均评分: {rag.reward_learner.get_learning_stats()['avg_score']:.1f}")
print(f"累计奖励: {rag.reward_learner.get_learning_stats()['cumulative_reward']:.2f}")
EOF
```

---

## 📈 更新日志

### v2.0.0 (2025-01-15)

#### 新增功能
- ✅ **RAG学习系统**
  - FAISS向量存储（替换ChromaDB）
  - text-embedding-bge-m3嵌入模型
  - 模型自我评价机制（trade_score）
  - 交易评估器（4维度评分）
  - 奖励学习系统

- ✅ **3个RAG函数**
  - `record_decision_to_rag`: 记录决策到RAG
  - `query_rag_stats`: 查询RAG统计
  - `cleanup_rag_history`: 清理RAG历史

- ✅ **增强模块**
  - PositionTracker: 持仓追踪
  - MarketStateComparator: 市场对比
  - DecisionChecker: 决策检查
  - TradeReviewer: 交易复盘

#### Bug修复
- ✅ 修复 datetime 时区兼容性问题
- ✅ 修复 embedding API 路径兼容性
- ✅ 修复 trade_evaluator 评分计算错误
- ✅ 修复 embedding 降级方案（零向量而非随机）
- ✅ 修复 dataframe 空检查

#### 优化改进
- ⚡ 降低向量检索延迟（FAISS）
- 📝 完善日志系统
- 🛡️ 增强错误处理和重试机制
- 📊 改进上下文构建和Token管理

---

## ⚠️ 风险提示

1. **加密货币交易存在高风险**，可能导致本金损失
2. **LLM决策不保证盈利**，需要持续监控和优化
3. **建议先使用模拟交易**（`dry_run: true`）充分测试
4. **合理设置止损和仓位**，单笔风险控制在5%以内
5. **定期检查日志和RAG质量**，确保系统健康运行
6. **高杠杆有爆仓风险**，建议杠杆≤10x
7. **RAG系统需要积累**，前20笔交易效果可能不明显

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

特别感谢：
- Freqtrade 社区
- OpenAI Function Calling
- FAISS 向量检索库
- BGE 嵌入模型

---

## 📞 支持与资源

### 文档
- [Freqtrade 官方文档](https://www.freqtrade.io/)
- [RAG系统详细说明](README_RAG.md)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

### 社区
- GitHub Issues: 报告问题和建议
- Freqtrade Discord: 交流策略
- Freqtrade Telegram: 实时讨论

### 学习资源
- [加密货币交易基础](https://academy.binance.com/)
- [RAG技术原理](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [向量数据库入门](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)

---

**祝交易顺利！🚀**

> "The best time to plant a tree was 20 years ago. The second best time is now."
> — 最好的交易时机是20年前开始学习，次好的时机是现在开始优化。
