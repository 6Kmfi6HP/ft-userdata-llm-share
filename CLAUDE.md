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

### Core Trading Flow (LangGraph Architecture)
```
Market Data (30m candles)
    ↓
ContextBuilder → Builds market context + technical indicators
    ↓
PromptBuilder → Generates system prompts (entry/position management)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LangGraphClient (Bull vs Bear Debate System)                    │
│                                                                 │
│   Stage 1: Analysis Subgraph (Parallel)                         │
│   ├─ IndicatorAgent → RSI, MACD, ADX, Stochastic                │
│   ├─ TrendAgent → EMA structure, support/resistance (vision)    │
│   ├─ SentimentAgent → Funding rate, OI, Fear & Greed            │
│   ├─ PatternAgent → K-line pattern recognition (vision)         │
│   └─ Aggregator → Weighted consensus                            │
│                         ↓                                       │
│   Stage 2: Debate Subgraph (Sequential) - Layer 3 Verification  │
│   ├─ BullAgent → Makes strongest case FOR the trade             │
│   ├─ BearAgent → Finds every possible flaw                      │
│   └─ JudgeAgent → Evaluates arguments, renders verdict          │
│                         ↓                                       │
│   Stage 3: Grounding Verification - Layer 4 Verification        │
│   └─ Compares LLM claims against actual market data             │
│                         ↓                                       │
│   Stage 4: Execution                                            │
│   ├─ Validator → Risk management checks                         │
│   └─ Executor → Prepares trading action                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
TradeLogger → Writes decision to llm_decisions.jsonl
    ↓
Execute on Binance Futures
```

### Module Responsibilities

**`LLMFunctionStrategy.py`** (Main entry point)
- Extends Freqtrade's IStrategy with hooks: `populate_indicators()`, `populate_entry_trend()`, `custom_exit()`, `custom_stoploss()`, `leverage_callback()`
- Implements four-layer stop-loss protection (Layer 2: `custom_stoploss()`, Layer 4: `custom_exit()`)

**`llm_modules/llm/`** - LLM clients and execution:
- `llm_client.py`: Direct LLM client with function calling
- `langgraph_client.py`: LangGraph-based Bull vs Bear debate client (primary)
- `consensus_client.py`: Legacy dual-role consensus (deprecated)
- `function_executor.py`: Function call executor

**`llm_modules/trading_graph/`** - LangGraph trading decision system (see below)

**`llm_modules/lc_integration/`** - LangChain integration:
- `llm_factory.py`: Creates LangChain chat models from config
- `tools/`: LangChain tool schemas (entry, exit, position tools)
- `adapters/`: Context adapters for LangGraph state

**`llm_modules/prompts/`** - Modular prompt templates:
- `analysis/`: Agent analysis prompts (indicator, sentiment, pattern, trend)
- `debate/`: Debate prompts (bull, bear, judge for entry and position)

**`llm_modules/tools/trading_tools.py`** - 6 core trading functions: `signal_entry_long/short`, `signal_exit`, `adjust_position`, `signal_hold`, `signal_wait`

**`llm_modules/context/`** - `prompt_builder.py` (4-step CoT prompts), `few_shot_examples.py` (6 CoT examples)

**`llm_modules/learning/`** - JSONL-based self-learning: `historical_query.py`, `pattern_analyzer.py`, `self_reflection.py`, `trade_evaluator.py`

**`llm_modules/utils/`** - Key modules: `context_builder.py` (market context), `stoploss_calculator.py` (dynamic stop-loss), `decision_checker.py` (risk validation), `position_tracker.py` (MFE/MAE tracking), `chart_generator.py` (K-line charts for vision)

**`llm_modules/indicators/indicator_calculator.py`** - Technical indicators: EMA, RSI, MACD, Stochastic (%K/%D), Williams %R, ATR, ADX (+DI/-DI), MFI, OBV, ROC, Amihud

### LangGraph Trading System (`trading_graph/`)

LangGraph-based multi-stage trading decision system with 6-layer hallucination prevention. Reference: `.agent/LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md`

**6-Layer Hallucination Prevention Architecture**:

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Input Grounding | Market data from ContextBuilder (real OHLC, indicators) |
| 2 | Parallel Analysis | 4 specialized agents analyze independently |
| 3 | Adversarial Debate | Bull vs Bear challenge each other's reasoning |
| 4 | Grounding Verification | Compares LLM claims against actual data |
| 5 | Validation | Risk management and confidence checks |
| 6 | Execution | Final action with calibrated confidence |

**Graph Flow**:
```
START
  │
  ▼
[analysis_subgraph] ──► 4 agents parallel (fan-out/fan-in)
  │
  ▼
[route_entry_or_position]
  │
  ├─── has_position=False ───► ENTRY PATH
  │    │
  │    ▼
  │  [debate_subgraph] ──► Bull → Bear → Judge (Layer 3)
  │    │
  │    ▼
  │  [grounding_node] ──► Layer 4 verification
  │    │
  │    ├── Hallucination > 50% ──► END (signal_wait)
  │    └── Hallucination < 50% ──► Continue
  │    │
  │    ▼
  │  [validator_node] → [executor_node] → END
  │
  └─── has_position=True ────► POSITION PATH
       │
       ▼
     [position_subgraph] ──► PosBull → PosBear → PosJudge → PosGrounding
       │
       ▼
     [position_validator] → [executor_node] → END
```

**Module Files**:
- `trading_graph/state.py`: TypedDict state schemas (`TradingDecisionState`, `AnalysisState`, `DebateState`, enums)
- `trading_graph/main_graph.py`: Main StateGraph builder with `TradingGraphRunner` class
- `trading_graph/edges/routing.py`: Conditional edge routing functions
- `trading_graph/subgraphs/analysis_graph.py`: Parallel analysis fan-out/fan-in
- `trading_graph/subgraphs/debate_graph.py`: Bull → Bear → Judge sequential
- `trading_graph/subgraphs/position_graph.py`: Position management debate
- `trading_graph/nodes/analysis/`: Analysis agent nodes (indicator, trend, sentiment, pattern, aggregator)
- `trading_graph/nodes/debate/`: Debate nodes (bull, bear, judge)
- `trading_graph/nodes/position/`: Position management nodes
- `trading_graph/nodes/verification/`: Grounding verification node
- `trading_graph/nodes/execution/`: Validator and executor nodes

**Agent Weights** (in `state.py`):
```python
AGENT_WEIGHTS = {
    "IndicatorAgent": 1.0,   # Baseline
    "TrendAgent": 1.2,       # Highest (trend is king)
    "SentimentAgent": 0.8,   # Lowest (auxiliary)
    "PatternAgent": 1.1,     # Medium-high
}
```

**Verdict Enums**:
- `Verdict`: APPROVE (Bull wins) / REJECT (Bear wins) / ABSTAIN (tie → wait)
- `PositionVerdict`: HOLD / EXIT / SCALE_IN (+20%~+50%) / PARTIAL_EXIT (-30%~-70%)

**Grounding Verification Details**:
- Extracts quantitative claims (e.g., "RSI=28", "ADX>25")
- Extracts qualitative claims (e.g., "RSI超卖", "MACD金叉")
- Tolerance: 15% deviation allowed before flagging
- Hallucination thresholds: 30% warning, 50% rejection
- Confidence penalty: -5% per false claim

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

**LangGraph Debate Config** (within `llm_config`):
```json
"llm_config": {
    "debate_config": {
        "enabled": true,
        "min_debate_quality": 60,
        "confidence_calibration": true,
        "fallback_on_error": true
    }
}
```
- `enabled`: Toggle Bull vs Bear debate system (default: true)
- `min_debate_quality`: Minimum quality threshold for debate arguments
- `confidence_calibration`: Apply confidence adjustments based on debate outcome
- `fallback_on_error`: Fall back to signal_wait if LangGraph execution fails

**Consensus Config** (for confidence threshold):
```json
"consensus_config": {
    "enabled": true,
    "confidence_threshold": 80
}
```
- `confidence_threshold`: Minimum confidence required for entry signals (post-validation)

**Vision Analysis Config** (within `llm_config`):
```json
"llm_config": {
    "use_vision": true,
    "multi_timeframe_vision": true,
    "vision_model": "gpt-4o"
}
```
- `use_vision`: Enable K-line chart visual analysis (default: false)
- `multi_timeframe_vision`: Send dual timeframe charts (15m + 1h) for better analysis (default: true when vision enabled)
- `vision_model`: Vision-capable model (GPT-4V/GPT-4o, Gemini Pro Vision, etc.)

**Multi-Timeframe Vision Analysis**:
When `use_vision: true` and `multi_timeframe_vision: true`, PatternAgent and TrendAgent will:
1. Generate two K-line charts: primary timeframe (15m) + higher timeframe (1h)
2. Send both images to Vision LLM in a single request
3. Analyze short-term patterns/trends AND long-term context together
4. Report mode tag: `[Vision-MTF:15m+1h]`

Coverage with default 50 candles:
| Timeframe | Coverage |
|-----------|----------|
| 15m | 12.5 hours (short-term patterns, entry timing) |
| 1h | 50 hours (main trend, key support/resistance) |

**Requirements**:
- Vision-capable LLM (GPT-4V, GPT-4o, Gemini Pro Vision)
- OHLCV data for both timeframes in State (`ohlcv_data`, `ohlcv_data_htf`)
- Dependencies: `mplfinance`, `matplotlib`, `pandas`, `numpy`

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

### When Working with LangGraph Trading System

1. **Adding a New Analysis Agent**:
   - Create node in `trading_graph/nodes/analysis/` (e.g., `new_agent_node.py`)
   - Add prompt template in `prompts/analysis/` (e.g., `new_agent_prompt.py`)
   - Register in `subgraphs/analysis_graph.py` parallel fan-out
   - Update `AGENT_WEIGHTS` in `state.py`
   - Add to aggregator node's agent list

2. **Modifying Debate Flow**:
   - Edit prompts in `prompts/debate/` (bull_prompt, bear_prompt, judge_prompt)
   - For position management: edit `position_bull_prompt.py`, `position_bear_prompt.py`, etc.
   - Modify node logic in `trading_graph/nodes/debate/`
   - Adjust routing logic in `edges/routing.py`

3. **Adding Grounding Checks**:
   - Extend `nodes/verification/grounding_node.py`
   - Add claim extraction patterns for new indicators
   - Adjust confidence penalty calculations
   - Update hallucination threshold if needed

4. **Position Management Path**:
   - Position debate uses separate subgraph (`subgraphs/position_graph.py`)
   - Has its own Bull/Bear/Judge nodes in `nodes/position/`
   - Uses `PositionVerdict` enum: HOLD, EXIT, SCALE_IN, PARTIAL_EXIT
   - Adjustment percentages: +20%~+50% (scale_in), -30%~-70% (partial_exit)

5. **Debugging LangGraph**:
   - Use `TradingGraphRunner(debug=True)` to enable verbose logging
   - Access last state: `langgraph_client._last_state` or `langgraph_client.get_last_agent_state()`
   - Enable debug logging: `logging.getLogger('llm_modules.trading_graph').setLevel(logging.DEBUG)`
   - Check `llm_decisions.jsonl` for full decision trace with debate results

6. **Performance Optimization**:
   - Analysis agents run in parallel by default (fan-out/fan-in pattern)
   - Debate is sequential by design (Bull needs to go before Bear)
   - Reduce agent prompt complexity if timeout issues occur
   - Vision mode adds latency; disable with `use_vision: false` if not needed

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

**LangGraph execution fails**
- Check LangChain/LangGraph dependencies: `pip install langgraph langchain-openai`
- Verify `llm_factory.py` can create chat models with your API config
- DataFrame serialization issues: checkpointer is disabled by default for this reason
- Check for import errors in `trading_graph/` modules

**"No clear signal provided" (未提供明确信号) with LangGraph**
- Ensure `trading_tools` instance is passed to `LangGraphClient` constructor
- Check `_store_signal_in_cache()` is called successfully after graph execution
- Verify signal storage isn't cleared before `LLMFunctionStrategy.get_signal()` reads it
- Enable debug logging to trace signal flow

**Grounding verification rejects valid trades**
- Check hallucination threshold in `grounding_node.py` (default: 50%)
- Review claim extraction patterns - may need adjustment for your indicators
- Tolerance is 15% by default; increase if indicators have natural variance
- Confidence penalty (-5% per false claim) may be too aggressive

## Technology Stack

- **Freqtrade 2.x**: Trading framework (Docker: freqtradeorg/freqtrade:stable)
- **Python 3.11+**: Primary language
- **LangGraph + LangChain**: Multi-agent orchestration and LLM abstraction
- **CCXT**: Exchange API abstraction
- **Pandas + TA-Lib**: Technical analysis
- **mplfinance**: K-line chart generation for vision analysis
- **OpenAI-compatible API**: LLM integration (supports any OpenAI-compatible endpoint)
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
- `.agent/LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md`: 6-layer hallucination prevention research and design
