# Freqtrade LLM Function Calling Strategy

> 基于大语言模型（LLM）Function Calling 与自我反思学习体系的智能加密货币交易策略

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
- [经验学习系统](#经验学习系统)
- [使用指南](#使用指南)
- [故障排除](#故障排除)
- [技术细节](#技术细节)
- [更新日志](#更新日志)
- [风险提示](#风险提示)
- [许可证](#许可证)
- [贡献](#贡献)
- [支持与资源](#支持与资源)

---

## 🎯 项目简介

这是一个基于 **Freqtrade** 的自动化交易示例策略。策略使用 **OpenAI Function Calling** 接入大语言模型，在 30m 主时间框架下对市场进行全局分析：

- LLM 通过 6 个交易函数直接下达开仓/加仓/平仓等指令
- 真实交易日志（JSONL）驱动历史回放、模式分析与自我反思
- 轻量级经验系统替代了早期的 RAG / 向量检索依赖，部署更简单
- 所有状态、日志和数据库都通过 `manage.sh` 一键管理

适合想要验证「LLM + 交易框架」可行性的研究者与工程师。

---

## ✨ 核心特性

### 1. OpenAI Function Calling 控盘

LLM 通过 6 个核心函数完成交易生命周期：

| 函数名称 | 功能描述 | 关键字段 |
|----------|----------|----------|
| `signal_entry_long` | 做多开仓，指定杠杆/投入金额 | `pair`, `leverage`, `stake_amount`, `confidence_score` |
| `signal_entry_short` | 做空开仓 | `pair`, `leverage`, `trend_strength`, `reason` |
| `signal_exit` | 市价平仓 + 自评打分 | `trade_score`, `reason`, `confidence_score` |
| `adjust_position` | 加仓/减仓 | `adjustment_pct`, `key_support`, `key_resistance` |
| `signal_hold` | 有仓位时保持不动 | `confidence_score`, `rsi_value`, `reason` |
| `signal_wait` | 无仓位时观望 | `confidence_score`, `reason` |

所有函数都带有严格的参数校验，确保 LLM 只能在允许的风险边界内操作。

### 2. 经验日志 + 自我反思

早期的 RAG / 向量存储已经废弃。现在的学习闭环完全基于结构化日志：

```
LLM 决策 → 写入 llm_decisions.jsonl
平仓后 → trade_experience.jsonl + 模型自评（trade_score）
历史查询 → JSONL 扫描 + 模式分析
自我反思 → SelfReflectionEngine 输出教训/总结
奖励学习 → reward_learning.json（可选）记录分值
```

优点：部署轻量、不依赖额外的嵌入服务，也不会出现向量索引损坏的问题。

### 3. 决策增强模块

| 模块 | 作用 |
|------|------|
| `PositionTracker` | 追踪持仓 MFE/MAE、信号历史 |
| `MarketStateComparator` | 对比开仓时与当前市场状态 |
| `DecisionQualityChecker` | 快速回放最近 50 次决策质量 |
| `TradeReviewer` | 生成平仓后的复盘摘要 |
| `HistoricalQueryEngine` | 直接从 JSONL 查询最近交易/统计 |
| `PatternAnalyzer` | 统计常见成功/失败模式 |
| `SelfReflectionEngine` | 根据入场/出场表现输出教训 |

### 4. 期货交易完整支持

- 多空双向、隔离保证金、动态杠杆 (1-100x)
- 自定义投入金额或 `stake_amount = "unlimited"`
- 多次加仓/减仓（受 `max_entry_position_adjustment` 限制）
- `DecisionChecker` 对 LLM 请求做二级风控

### 5. 多时间框架技术分析

- 主时间框架：30m
- 辅助：1h / 4h / 1d（可配置）
- 指标：EMA、RSI、MACD、ATR、ADX、MFI、OBV 等
- `ContextBuilder` 自动压缩上下文，控制 token 使用

### 6. 日志与可观测性

```
user_data/logs/
├── freqtrade.log            # Freqtrade 运行日志
├── llm_decisions.jsonl      # 每次 LLM 决策完整上下文
├── trade_experience.jsonl   # 平仓后总结、收益、教训
└── reward_learning.json     # 奖励学习记录（可选）
```

所有日志都可通过 `manage.sh` 或 `tail -f` 直接查看。

---

## 🏗️ 系统架构

```
ft-userdata-llm/
├── docker-compose.yml          # Docker 编排
├── Dockerfile.custom           # 追加 Python 依赖
├── manage.sh                   # 一键管理脚本
├── README.md                   # 当前文档
└── user_data/
    ├── config.json             # 策略配置
    ├── logs/                   # 日志目录
    │   ├── freqtrade.log
    │   ├── llm_decisions.jsonl
    │   ├── trade_experience.jsonl
    │   └── reward_learning.json
    ├── strategies/
    │   ├── LLMFunctionStrategy.py
    │   └── llm_modules/
    │       ├── llm/                # LLM 客户端 & 工具执行器
    │       ├── tools/              # Function Calling 定义
    │       ├── experience/         # 日志 & 复盘模块
    │       ├── learning/           # HistoricalQuery / Pattern / Reflection
    │       ├── utils/              # Config/Context/Decision 工具
    │       └── indicators/         # 技术指标扩展
    └── tradesv3.sqlite*        # 交易数据库
```

---

## ⚡ 快速开始

1. **准备环境**
   - macOS / Linux / WSL 均可
   - Docker Desktop 已启动
   - 已克隆仓库并进入 `ft-userdata-llm`

2. **赋权 & 一键启动**

```bash
chmod +x manage.sh
./manage.sh start        # 已启用容器会直接看日志
```

3. **完整部署（重新构建镜像）**

```bash
./manage.sh deploy
```

4. **`manage.sh` 菜单功能**

| 序号 | 功能 | 说明 |
|------|------|------|
| 1 | 快速启动 | 检查容器 → 启动 → 跟随日志 |
| 2 | 快速重启 | `docker compose restart` + 日志 |
| 3 | 完整部署 | 检查镜像 → 构建 → 启动 |
| 4 | 查看容器日志 | 等价 `docker logs -f freqtrade-llm` |
| 5 | 查看 LLM 决策日志 | `tail -f user_data/logs/llm_decisions.jsonl` |
| 6 | 查看交易经验日志 | `tail -f user_data/logs/trade_experience.jsonl` |
| 7 | 清理日志和数据库 | 删除 JSONL / sqlite / reward 数据 |
| 8 | 检查镜像版本 | 比对 `freqtrade:stable` 摘要 |
| 9 | 停止服务 | `docker-compose down` |

命令行速查：`./manage.sh decisions`, `./manage.sh trades`, `./manage.sh version` 等。

---

## ⚙️ 配置说明

### 1. LLM 配置

```json
"llm_config": {
    "api_base": "http://host.docker.internal:3120",
    "api_key": "sk-xxx",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "temperature": 0.6,
    "max_tokens": 2000,
    "timeout": 60,
    "retry_times": 2
}
```

- 推荐模型：`qwen3-30b-a3b-thinking`、`gpt-4.1-mini`、`deepseek-coder`（保证 Function Calling 能力即可）
- OpenAI 兼容 API：保持 `/v1/chat/completions` 接口即可

### 2. 交易所配置

本策略支持 **Binance** 和 **Hyperliquid** 交易所。详细配置模板请参考 [CONFIG_TEMPLATE.md](CONFIG_TEMPLATE.md)。

#### Binance (默认)

```json
"stake_currency": "USDT",
"exchange": {
    "name": "binance",
    "key": "your-api-key",
    "secret": "your-api-secret",
    "ccxt_config": {
        "enableRateLimit": true,
        "options": { "defaultType": "future" }
    },
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 200,
        "timeout": 30000
    }
}
```

#### Hyperliquid

```json
"stake_currency": "USDC",
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "0x<your-wallet-address-40-hex-chars>",
    "privateKey": "0x<your-api-wallet-private-key-64-hex-chars>",
    "ccxt_config": {
        "enableRateLimit": true
    }
}
```

| 特性 | Binance | Hyperliquid |
|------|---------|-------------|
| 结算货币 | USDT | USDC |
| 认证方式 | API key + secret | 钱包地址 + 私钥 |
| 交易对格式 | `BTC/USDT:USDT` | `BTC/USDC:USDC` |
| 历史K线 | 无限制 | 最多5000根 |

### 3. 风险管理

```json
"risk_management": {
    "max_leverage": 100,
    "default_leverage": 10,
    "max_position_pct": 50,
    "max_open_trades": 4,
    "allow_model_freedom": true,
    "emergency_stop_loss": -0.15
}
```

### 4. 经验系统配置

```json
"experience_config": {
    "log_decisions": true,
    "log_trades": true,
    "decision_log_path": "./user_data/logs/llm_decisions.jsonl",
    "trade_log_path": "./user_data/logs/trade_experience.jsonl",
    "max_recent_trades_context": 5,
    "max_recent_decisions_context": 10,
    "include_pair_specific_trades": true
}
```

> 日志路径可自定义，但仍建议放在 `user_data/logs/` 以便 `manage.sh` 统一处理。

### 5. 上下文配置（节选）

```json
"context_config": {
    "max_context_tokens": 6000,
    "indicator_history_points": 80,
    "raw_kline_history_points": 80,
    "include_multi_timeframe_data": false,
    "multi_timeframe_history": {}
}
```

根据大模型价格和速度，自行调整 token 配额。

---

## 📚 经验学习系统

### 模型自我评价

LLM 在调用 `signal_exit` 时必须提交 `trade_score` 和自我反思：

```python
signal_exit(
    pair="BTC/USDT:USDT",
    confidence_score=85,
    rsi_value=72,
    trade_score=78,
    reason="""
    平仓理由：RSI 超买 + 达到预期阻力位
    自我反思：入场时机准确，但中途没有分批止盈，导致利润回吐。
    教训：盈利>8% 时优先锁仓。
    """
)
```

### 日志 & 查询

- `TradeLogger` 将决策与交易数据写入 JSONL
- `HistoricalQueryEngine` 定期重载文件，支持：
  - 最近 N 笔交易
  - 某交易对过去 30 天统计
  - 胜率 / 亏损率拆解
- `PatternAnalyzer` 会统计常见成功信号、失败原因、时间段表现

### 自我反思 + 奖励学习

- `SelfReflectionEngine` 根据盈利、持仓时长、MFE/MAE 输出总结/教训
- `RewardLearningSystem` 记录 `reward_learning.json`，用于观察累计奖励趋势（可关闭）
- 所有文本会再次作为上下文提供给下一次 LLM 决策

### 经验系统工作流

```
1. 决策阶段
   ├─ LLM 参考市场 + 历史摘要
   └─ 决策写入 llm_decisions.jsonl
2. 持仓阶段
   ├─ PositionTracker 更新 MFE/MAE
   └─ MarketComparator 跟踪市场变化
3. 平仓阶段
   ├─ LLM 调用 signal_exit 并打分
   ├─ TradeLogger 写入 trade_experience.jsonl
   ├─ SelfReflectionEngine 输出教训
   └─ RewardLearningSystem 更新奖励趋势
4. 下一次决策
   └─ HistoricalQuery + PatternAnalyzer 提供统计提示
```

---

## 🧭 使用指南

1. **启动/重启**：`./manage.sh start` 或 `./manage.sh restart`
2. **查看日志**：
   - 容器：`./manage.sh logs`
   - 决策：`./manage.sh decisions`
   - 经验：`./manage.sh trades`
3. **导出日志**：`cp user_data/logs/*.jsonl ~/backup/`
4. **数据库**：`user_data/tradesv3.sqlite` 可直接用 `sqlite3` 查看
5. **监控**：
   - Web UI: http://localhost:8086（用户名 `freqtrader`）
   - API: http://localhost:8086/api/v1/
6. **清理测试数据**：`./manage.sh clean`（会删除 JSONL + sqlite，请谨慎）

---

## 🛠️ 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Strategy analysis took ...` | LLM 请求耗时过长 | 减少 pair、降低上下文数据量、调低 `max_context_tokens` |
| `trade_experience.jsonl not found` | 首次运行文件未创建 | 触发一次平仓或手动 `touch user_data/logs/trade_experience.jsonl` |
| `工具实例缺少方法 signal_*` | `TradingTools` 未正确注册 | 重启策略或检查 `LLMFunctionStrategy` 是否报错 |
| LLM 请求超时 | API 不可达或延迟大 | 提高 `timeout`、减少 `max_tokens`、检查网络代理 |
| 无法查看日志 | 文件未生成或权限问题 | 使用 `manage.sh decisions` 自动创建并 tail |
| Docker 无法启动 | 端口或磁盘权限问题 | `docker-compose down -v` 后重新启动，确认 8086 端口空闲 |

如需更多调试，可进入容器：`docker exec -it freqtrade-llm bash`。

---

## 🔬 技术细节

- **ContextBuilder**：组合行情、账户、形态、历史经验，拆分开仓/持仓两套系统提示词。
- **FunctionExecutor**：接收 LLM 返回的 JSON，匹配到 `TradingTools`，并把结果回写给策略。
- **PositionTracker**：保存最近持仓的 MFE/MAE、杠杆、信号，供反思使用。
- **DecisionQualityChecker**：统计最近 50 次决策的胜率、平均置信度，避免连亏。
- **SelfReflectionEngine**：
  - 入场分析：方向、时机、关键词
  - 出场分析：盈利保留、止损及时性、持仓时长
  - 产出 `lessons` 与 `summary`
- **RewardLearningSystem**：
  - `reward = profit_pct/100 * score/100 * leverage_factor`
  - 用 deque 维护 1000 条历史，支持滚动平均

---

## 📈 更新日志

### v2.1.0 (2025-02-01)

- ✂️ **移除 RAG / 向量检索依赖**，改为纯 JSONL 经验系统
- 🧠 **新增轻量学习闭环**：HistoricalQuery、PatternAnalyzer、SelfReflectionEngine
- 🗂️ **日志结构统一**：决策/交易/奖励均放在 `user_data/logs`
- 🛠️ **manage.sh 升级**：新增决策/经验日志查看、镜像指纹比较、精准清理
- 📄 **README 重写**：同步最新功能与使用方式

### v2.0.0 (2025-01-15, 已废弃)

- 曾引入 FAISS + RAG 管理器（现已移除）
- 保留历史记录以供参考

---

## ⚠️ 风险提示

1. 加密货币期货波动剧烈，务必在 `dry_run` 下验证策略
2. LLM 决策具有随机性，不保证盈利，需人工监控
3. 建议单笔风险 ≤ 5%，杠杆建议 ≤ 10x
4. 定期备份 `user_data/logs/` 与 `tradesv3.sqlite`
5. 若长期离线，请停止容器并撤掉 API 权限
6. 自定义模型/提示词后务必重新回测或纸面验证
7. 任何自动化策略都可能因所依赖服务故障而失效

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue / PR：
- 修复 Bug 或补充文档
- 分享更优的提示词与风控策略
- 优化日志结构或分析脚本

---

## 📞 支持与资源

- [Freqtrade 官方文档](https://www.freqtrade.io/)
- [OpenAI Function Calling 指南](https://platform.openai.com/docs/guides/function-calling)
- [CCXT 交易所配置参考](https://docs.ccxt.com/)
- [Docker 官方手册](https://docs.docker.com/)

社区 & 学习：
- Freqtrade Discord / Telegram
- Binance Academy（期货基础）
- Open-source AI 交易社区

---

**祝交易顺利！🚀**
