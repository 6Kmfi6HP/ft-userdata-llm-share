# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Freqtrade-based cryptocurrency trading bot** that uses **Large Language Models (LLMs)** with OpenAI Function Calling to make autonomous trading decisions on Binance Futures markets. The LLM controls position entry, exit, and management through six trading functions, with integrated self-learning from structured trade logs.

**Key Architecture**: Modular LLM trading system replacing traditional rule-based strategies with adaptive AI decision-making, featuring four-layer stop-loss protection and JSONL-based experience learning with few-shot Chain-of-Thought prompting.

## Development Commands

### Container Management (via manage.sh)
```bash
./manage.sh start           # Start bot (or view logs if running)
./manage.sh restart         # Quick restart without rebuild
./manage.sh deploy          # Full rebuild and deploy
./manage.sh stop            # Stop container
```

### Log Monitoring
```bash
./manage.sh logs            # Container logs
./manage.sh decisions       # LLM decision log (llm_decisions.jsonl)
./manage.sh trades          # Trade experience log (trade_experience.jsonl)
```

### Data Management
```bash
./manage.sh clean           # Delete logs and database (WARNING: destructive)
./manage.sh version         # Check Docker image version
```

### Testing
```bash
# Run all tests
docker compose run --rm freqtrade-llm pytest user_data/tests

# Run single test file
docker compose run --rm freqtrade-llm pytest user_data/tests/test_critical_fixes.py -v

# Run specific test function
docker compose run --rm freqtrade-llm pytest user_data/tests/test_file.py::test_function_name -v

# Syntax check (inside container or with docker exec)
python -m py_compile user_data/strategies/LLMFunctionStrategy.py
```

### Backtesting
```bash
freqtrade backtesting \
  --strategy LLMFunctionStrategy \
  --timerange 20241120-20241219 \
  --timeframe 30m \
  --max-open-trades 5 \
  --stake-amount 500
```

### Direct Docker Access
```bash
docker-compose up -d        # Start in background
docker logs -f freqtrade-llm  # Stream container logs
docker exec -it freqtrade-llm bash  # SSH into container
```

## Architecture & Code Structure

### Core Trading Flow
```
Market Data (30m candles)
    ↓
ContextBuilder → Builds market context + technical indicators
    ↓
PromptBuilder → Generates system prompts (entry/position management)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Multi-Agent Pre-Analysis (Optional, configurable)              │
│   ├─ IndicatorAgent → RSI, MACD, ADX analysis                  │
│   ├─ TrendAgent → EMA structure, support/resistance            │
│   └─ SentimentAgent → Funding rate, OI, Fear & Greed           │
│   ↓                                                            │
│ AgentOrchestrator → Weighted consensus aggregation             │
└─────────────────────────────────────────────────────────────────┘
    ↓
ConsensusClient → Dual-role verification (Opportunity Finder + Risk Assessor)
    ↓
LLMClient → Calls LLM API with function definitions
    ↓
Function Calling Loop (max 5 iterations):
    ├─ LLM analyzes and calls trading function
    ├─ FunctionExecutor validates and executes
    ├─ TradingTools performs action (entry/exit/hold/adjust)
    └─ Loop continues until LLM finishes
    ↓
TradeLogger → Writes decision to llm_decisions.jsonl
    ↓
Execute on Binance Futures
```

### Module Responsibilities

**`LLMFunctionStrategy.py`** (Main entry point)
- Extends Freqtrade's IStrategy with hooks: `populate_indicators()`, `populate_entry_trend()`, `custom_exit()`, `custom_stoploss()`, `leverage_callback()`
- Implements four-layer stop-loss protection (Layer 2: `custom_stoploss()`, Layer 4: `custom_exit()`)

**`llm_modules/llm/`** - LLM client (`llm_client.py`) and function call executor (`function_executor.py`)

**`llm_modules/tools/trading_tools.py`** - 6 core trading functions: `signal_entry_long/short`, `signal_exit`, `adjust_position`, `signal_hold`, `signal_wait`

**`llm_modules/context/`** - `prompt_builder.py` (4-step CoT prompts), `few_shot_examples.py` (6 CoT examples)

**`llm_modules/learning/`** - JSONL-based self-learning: `historical_query.py`, `pattern_analyzer.py`, `self_reflection.py`, `trade_evaluator.py`

**`llm_modules/utils/`** - Key modules: `context_builder.py` (market context), `stoploss_calculator.py` (dynamic stop-loss), `decision_checker.py` (risk validation), `position_tracker.py` (MFE/MAE tracking)

**`llm_modules/indicators/indicator_calculator.py`** - Technical indicators: EMA, RSI, MACD, Stochastic (%K/%D), Williams %R, ATR, ADX (+DI/-DI), MFI, OBV, ROC, Amihud

**`llm_modules/agents/`** - Multi-agent pre-analysis system (see below)

### Multi-Agent Pre-Analysis System

Optional system providing specialized market analysis before main LLM decision. Inspired by QuantAgent's multi-agent architecture.

**Architecture**:
```
AgentOrchestrator
    ├─ IndicatorAgent (weight: 1.0)
    │   └─ RSI, MACD, ADX (+DI/-DI), Stochastic (%K/%D), Williams %R, MFI analysis
    ├─ TrendAgent (weight: 1.2)
    │   └─ EMA structure, price structure, support/resistance
    └─ SentimentAgent (weight: 0.8)
        └─ Funding rate, long/short ratio, OI, Fear & Greed
            ↓
    Weighted Consensus Aggregation → Injected into ConsensusClient
```

**Module Files**:
- `agents/agent_state.py`: State management (`AgentState`, `AgentReport`, `Signal`, `Direction`)
- `agents/base_agent.py`: Abstract base class with standardized prompt/response handling
- `agents/indicator_agent.py`: Technical indicator analysis specialist
- `agents/trend_agent.py`: Trend structure and price level analysis
- `agents/sentiment_agent.py`: Market sentiment and positioning analysis
- `agents/orchestrator.py`: Coordinates agents, aggregates results via weighted voting

**Consensus Calculation**:
- Direction votes weighted by agent confidence × agent weight
- Default weights: Trend (1.2) > Indicator (1.0) > Sentiment (0.8)
- Confidence = average of agreeing agents' confidence levels
- Key levels aggregated from all agent reports

**Agent Report Format** (standardized output):
```
[信号列表]
- signal_type: description (strength: strong/medium/weak)

[方向判断]
long / short / neutral

[置信度]
0-100

[关键价位]
支撑: price
阻力: price

[分析摘要]
Brief analysis summary
```

**Integration with ConsensusClient**:
- Agents run before dual-role verification
- Analysis injected as system context for both roles
- Disabled by default (`multi_agent_enabled: false`)

### Four-Layer Stop Loss & Take Profit System

**Layer 1: Exchange Hard Stop** (config.json)
- Fixed -10% stop on mark price (protects against liquidation)
- Executed by exchange (fastest execution)
- Config: `stoploss: -0.10`, `stoploss_on_exchange: true`, `stoploss_price_type: "mark"`

**Layer 2: ATR Dynamic Trailing Stop** (`custom_stoploss()` + `stoploss_calculator.py`)
- Profit-based tracking with smooth transitions ("让利润奔跑" strategy):
  - **>15%: 2.0×ATR (min 3.0%)** - Relaxed stops to let profits run
  - 6-15%: 1.0×ATR (min 2.0%)
  - 2-6%: 1.5×ATR (min 1.5%)
  - <2%: Use hard stop-loss (-10%)
- **Special handling for extreme profits (>80%):**
  - Widens to 4.0×ATR (min 8%) to defer to Layer 4's trend analysis
  - Layer 4 uses ADX + MACD to detect trend weakening
- Time decay: Tightens 20% if held >2h without 6% profit
- Trend adaptation: Relaxes 20% when ADX >25

**Layer 3: LLM Contextual Stop** (`context_builder.py`)
- Stop-loss levels shown to LLM in market context
- LLM can preemptively exit based on strategic analysis
- Position management prompts include profit protection rules

**Layer 4: Extreme Take-Profit Protection** (`custom_exit()`)
- Triggers before LLM can react to extreme market moves:
  - **ROI >80% + trend weakening** (ADX <20 or ADX <25 with MACD hist <0) → Immediate exit
  - **ROI >100% unconditional** → Immediate exit
- Purpose: Protect exceptional profits from sudden reversals

**Execution order**: `custom_exit` (Layer 4) → `custom_stoploss` (Layer 2) → exchange stop

### Experience Learning System (JSONL-based)

**Flow**:
```
1. Decision Phase
   ├─ LLM analyzes market + historical patterns
   └─ Write to llm_decisions.jsonl
2. Position Tracking
   ├─ PositionTracker updates MFE/MAE
   └─ MarketComparator monitors state changes
3. Exit Phase
   ├─ LLM calls signal_exit with trade_score + reason
   ├─ TradeLogger writes to trade_experience.jsonl
   ├─ SelfReflectionEngine generates lessons
   └─ RewardLearningSystem updates reward_learning.json
4. Next Decision
   └─ HistoricalQueryEngine + PatternAnalyzer provide context
```

**Key Files**:
- `user_data/logs/llm_decisions.jsonl`: Every LLM decision with full context
- `user_data/logs/trade_experience.jsonl`: Closed trade summaries with lessons
- `user_data/logs/reward_learning.json`: Cumulative reward tracking (optional)

### Prompt Engineering Framework

Based on Google Prompt Engineering Whitepaper, the system enforces structured Chain-of-Thought reasoning:

**Mandatory 4-Step CoT Framework** (`prompt_builder.py`):
1. **Market State Identification**: Classify as Trend/Range/Reversal (EMA structure, ADX, momentum)
2. **Strategy Selection**: Choose from 7 predefined strategies (trend following, range trading, reversal)
3. **Entry Condition Verification**: Require ≥2 independent confirmations
4. **Risk-Reward Assessment**: Validate R:R ≥ 2:1 before entry

**Few-Shot Examples** (`few_shot_examples.py`):
- 6 detailed examples covering all major scenarios
- Integrated via `get_format_examples_entry()` and `get_format_examples_position()`

**Position Management Prompts**:
- Noise filtering: Ignore <30% ATR fluctuations unless sustained 3+ candles
- MFE drawback >50% → Reduce position 50%; >30% → Reduce 30%
- Batch take-profit: ≥5% profit → Reduce 40%; ≥8% → Reduce 50%

**Temperature Setting**: 0.0 for deterministic reasoning (reduces hallucination in function calling)

## Configuration (config.json)

### Critical Settings

**LLM Config**:
```json
"llm_config": {
    "api_base": "http://192.168.8.225:3899",
    "model": "gemini-flash-lite-latest",
    "temperature": 0.0,
    "max_tokens": 65536,
    "timeout": 60
}
```
- Supports any OpenAI-compatible API
- Current model: Gemini Flash Lite (changed from qwen3-30b-a3b-thinking for cost optimization)
- Temperature set to 0.0 for deterministic CoT reasoning
- Max tokens increased to 65536 to accommodate detailed reasoning traces
- Recommended models: gemini-2.5-flash, qwen3-30b-a3b-thinking, gpt-4.1-mini, deepseek-coder

**Risk Management**:
```json
"risk_management": {
    "max_leverage": 100,
    "default_leverage": 10,
    "max_position_pct": 50,
    "max_open_trades": 4
}
```

**Trading Mode**:
```json
"trading_mode": "futures",
"margin_mode": "isolated",
"dry_run": true,
"dry_run_wallet": 10000
```

**Experience System**:
```json
"experience_config": {
    "log_decisions": true,
    "max_recent_trades_context": 5,
    "include_pair_specific_trades": true
}
```

**Custom Stop-Loss Config** (`custom_stoploss_config`):
- Defines Layer 2 dynamic stop-loss: profit thresholds, ATR multipliers, min distances
- Time decay and trend adaptation parameters (see CONFIG_TEMPLATE.md for full schema)

**Custom Exit Config** (`custom_exit_config`):
- Defines Layer 4 extreme take-profit triggers (ROI thresholds, RSI conditions)

**Multi-Agent Config** (within `consensus_config`):
```json
"consensus_config": {
    "enabled": true,
    "multi_agent_enabled": true,
    "agent_config": {
        "parallel_execution": true,
        "agent_weights": {
            "IndicatorAgent": 1.0,
            "TrendAgent": 1.2,
            "SentimentAgent": 0.8
        },
        "enabled_agents": ["indicator", "trend", "sentiment"]
    }
}
```
- `multi_agent_enabled`: Toggle multi-agent pre-analysis (default: false)
- `parallel_execution`: Run agents concurrently (default: true)
- `agent_weights`: Relative importance for consensus calculation
- `enabled_agents`: Which agents to activate (all three by default)

## Working with This Codebase

### When Modifying Trading Logic

1. **Function Calling Changes**: Edit `llm_modules/tools/trading_tools.py`
   - Modify function signatures carefully - LLM must match schema
   - Update both function definition AND schema in `get_trading_functions()`

2. **Prompt Engineering**: Edit `llm_modules/context/prompt_builder.py`
   - Separate prompts for entry (`_build_entry_prompt`) vs position management (`_build_position_prompt`)
   - Balance detail vs token usage (context limit: `max_context_tokens` in config)

3. **Risk Controls**:
   - Hard limits: `llm_modules/utils/decision_checker.py`
   - Dynamic stop-loss: `LLMFunctionStrategy.py` custom_stoploss() method
   - Exchange stop: config.json `stoploss` field

4. **Testing Changes**:
   - Always backtest after modifications: See BACKTEST_GUIDE.md
   - Check dry_run logs before live trading
   - Monitor `llm_decisions.jsonl` for unexpected behavior

### When Adding Technical Indicators

1. Add calculation to `llm_modules/indicators/indicator_calculator.py`
2. Update `LLMFunctionStrategy.populate_indicators()` to call it
3. Modify `context_builder.py` to include in market context
4. Update prompt in `prompt_builder.py` to explain indicator usage

### When Debugging LLM Decisions

1. Check `user_data/logs/llm_decisions.jsonl` for full LLM request/response
2. Each entry contains: market context, function calls, validation results, execution outcomes
3. Useful `jq` commands:
```bash
# Parse all decisions
jq -s '.' user_data/logs/llm_decisions.jsonl | less

# Count decisions by action type
jq -r '.action' user_data/logs/llm_decisions.jsonl | sort | uniq -c

# Find losing trades
jq 'select(.profit_pct < 0)' user_data/logs/trade_experience.jsonl

# Recent 5 trades summary
tail -5 user_data/logs/trade_experience.jsonl | jq '{pair, profit_pct, side}'
```

### When Adjusting Learning System

1. **Historical Queries**: `llm_modules/learning/historical_query.py`
   - Modify `query_recent_trades()` for different lookback periods
   - Adjust `get_pair_statistics()` for different metrics

2. **Pattern Analysis**: `llm_modules/learning/pattern_analyzer.py`
   - Edit `analyze_patterns()` to extract different patterns
   - Modify thresholds for success/failure classification

3. **Self-Reflection**: `llm_modules/learning/self_reflection.py`
   - Customize reflection prompts for different insights
   - Adjust trade evaluation criteria

### When Working with Multi-Agent System

1. **Adding a New Agent**:
   - Create new file in `llm_modules/agents/` inheriting from `BaseAgent`
   - Implement `_get_analysis_focus()` to return agent-specific analysis prompt
   - Register in `orchestrator.py` (`_create_default_agents()` method)
   - Add to `__init__.py` exports
   - Update config schema in `consensus_client.py`

2. **Modifying Agent Prompts**:
   - Each agent has `ROLE_PROMPT` class variable defining expertise
   - `_get_analysis_focus()` returns task-specific analysis instructions
   - Output format is standardized in `base_agent.py` (`_build_analysis_prompt()`)

3. **Adjusting Consensus Weights**:
   - Edit `agent_weights` in config.json under `consensus_config.agent_config`
   - Higher weight = more influence on final direction
   - Default: Trend (1.2) > Indicator (1.0) > Sentiment (0.8)

4. **Debugging Agent Analysis**:
   - Use `consensus_client.get_last_agent_state()` for last analysis state
   - Check individual agent reports in returned state
   - Enable debug logging: `logging.getLogger('llm_modules.agents').setLevel(logging.DEBUG)`
   - Agent analysis is logged when enabled

5. **Performance Optimization**:
   - Set `parallel_execution: true` for concurrent agent calls (default)
   - Disable unused agents via `enabled_agents` config
   - Reduce agent prompt complexity if timeout issues occur

### Critical Implementation Patterns

**Race Condition Prevention**:
- Always clear signals per-pair using `clear_signal_for_pair(pair)`, never globally
- Use atomic `pop()` operations for shared state (e.g., `_position_adjustment_cache`)
- Validate `remaining_stake >= min_stake` before applying position reductions

**Stop-Loss Calculation**:
- Use epsilon-based comparison for float thresholds (`abs(a - b) < 1e-10`)
- Handle ATR = 0 with fallback to min_distance
- Short positions: stop direction is `1 + distance` (not `1 - distance`)
- Test both long and short positions when modifying stop-loss logic

## Important Constraints

- **Security**: Never commit real API keys; use environment variables
- **Performance**: LLM requests block analysis - keep timeout ≤60s; reduce `max_context_tokens` if slow
- **Trading Safety**: Always test in `dry_run` first; do not disable Layer 1/2 stop-loss; max leverage 10x recommended
- **Data**: JSONL logs grow indefinitely - implement rotation for long-term; backup before `./manage.sh clean`

## Common Issues

**"Strategy analysis took too long"**
- Reduce `max_context_tokens` in config.json
- Lower `indicator_history_points`
- Decrease number of trading pairs
- Increase LLM `timeout`

**LLM makes invalid function calls**
- Check schema in `trading_tools.py` matches function signature
- Review `llm_decisions.jsonl` for prompt issues
- Verify model supports function calling properly

**Stop-loss not triggering**
- Layer 1 (exchange): Check `stoploss_on_exchange: true` and `stoploss: -0.10` in config
- Layer 2 (ATR dynamic): Verify ATR indicator is calculated in populate_indicators and `use_custom_stoploss: true`
- Layer 3 (LLM): Check context_builder includes stop-loss info in market context
- Layer 4 (take-profit): Verify `custom_exit_config` is present and custom_exit() method is active

**JSONL logs not found**
- Files created on first write - trigger a trade cycle
- Check paths in config.json `experience_config`
- Use `./manage.sh decisions` to auto-create and tail

## Technology Stack

- **Freqtrade 2.x**: Trading framework (Docker: freqtradeorg/freqtrade:stable)
- **Python 3.11+**: Primary language
- **CCXT**: Exchange API abstraction
- **Pandas + TA-Lib**: Technical analysis
- **OpenAI-compatible API**: LLM integration
- **SQLite**: Trade database
- **Docker + Docker Compose**: Containerization

## Coding Style

- Python 3.11; follow PEP 8 with 4-space indents
- snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants
- Log in JSONL format when touching `llm_decisions` or `trade_experience`; preserve existing keys and schema
- Type hints encouraged; keep docstrings concise and action-oriented
- Commit messages: `type: short description` (`feat`, `fix`, `chore`, `refactor`); imperative mood

## Related Documentation

- `README.md`: Comprehensive project documentation (Chinese)
- `AGENTS.md`: Repository guidelines, testing, and PR conventions
- `CONFIG_TEMPLATE.md`: Configuration reference
