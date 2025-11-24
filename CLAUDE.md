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
- Extends Freqtrade's IStrategy
- Hooks: `populate_indicators()`, `populate_entry_trend()`, `custom_exit()`, `custom_stoploss()`, `leverage_callback()`
- Orchestrates all modules and manages trading lifecycle
- Implements four-layer stop-loss protection (Layer 2: lines 1100-1230, Layer 4: lines 862-947)

**`llm_modules/llm/`** - LLM Integration
- `llm_client.py`: OpenAI-compatible API client with retry logic
- `function_executor.py`: Parses LLM function calls and executes trading actions

**`llm_modules/tools/`** - Trading Functions
- `trading_tools.py`: Defines 6 core functions:
  - `signal_entry_long/short`: Open positions with leverage
  - `signal_exit`: Close position with self-assessment
  - `adjust_position`: Add/reduce position size
  - `signal_hold`: Keep current position
  - `signal_wait`: Observe market without action

**`llm_modules/context/`** - Market Analysis & Prompt Engineering
- `prompt_builder.py`: Generates system prompts with mandatory 4-step CoT framework (2025-01-23 overhaul)
- `few_shot_examples.py`: Six detailed CoT examples based on Google Prompt Engineering Whitepaper (546 lines)
- `market_analyzer.py`: Analyzes current market state
- `data_formatter.py`: Formats data for LLM consumption

**`llm_modules/experience/`** - Trade Logging
- `trade_logger.py`: Writes decisions to JSONL files
- `experience_manager.py`: Aggregates trade experience
- `trade_reviewer.py`: Generates post-trade reviews with LLM self-assessment

**`llm_modules/learning/`** - Self-Learning System
- `historical_query.py`: Query trades from JSONL logs
- `pattern_analyzer.py`: Extract success/failure patterns
- `self_reflection.py`: Generate lessons from trade outcomes
- `trade_evaluator.py`: Score trade quality
- `reward_learning.py`: Track cumulative rewards
- **DEPRECATED** (files exist but NOT USED): `embedding_service.py`, `rag_manager.py`, `vector_store.py` - RAG system replaced by JSONL-based learning in v2.0. Config option `enable_rag` is ignored.

**`llm_modules/utils/`** - Core Utilities
- `config_loader.py`: Load config.json
- `context_builder.py`: Build complete market context for LLM
- `position_tracker.py`: Track MFE/MAE and position signals
- `market_comparator.py`: Compare current vs entry market state
- `decision_checker.py`: Validate LLM decisions against risk limits
- `stoploss_calculator.py`: Dynamic stop-loss calculations with smooth transitions, time decay, and trend adaptation (200 lines)

**`llm_modules/indicators/`** - Technical Analysis
- `indicator_calculator.py`: Calculate EMA, RSI, MACD, ATR, ADX, MFI, OBV

### Four-Layer Stop Loss & Take Profit System

**Layer 1: Exchange Hard Stop** (config.json)
- Fixed -10% stop on mark price (protects against liquidation)
- Executed by exchange (fastest execution)
- Config: `stoploss: -0.10`, `stoploss_on_exchange: true`, `stoploss_price_type: "mark"`

**Layer 2: ATR Dynamic Trailing Stop** (LLMFunctionStrategy.py:1402-1689)
- Profit-based tracking with smooth transitions ("让利润奔跑" strategy):
  - **>15%: 2.0×ATR (min 3.0%)** ✅ Relaxed stops to let profits run
  - 6-15%: 1.0×ATR (min 2.0%)
  - 2-6%: 1.5×ATR (min 1.5%)
  - <2%: Use hard stop-loss (-10%)
- **Special handling for extreme profits (>80%):**
  - Widens to 4.0×ATR (min 8%) to defer to Layer 4's trend analysis
  - This creates a coordinated handoff: Layer 2 → Layer 4 for intelligent exits
  - Layer 4 uses ADX + MACD to detect trend weakening, avoiding premature mechanical stops
- Time decay: Tightens 20% if held >2h without 6% profit
- Trend adaptation: Relaxes 20% when ADX >25
- Helper module: `stoploss_calculator.py` with smooth interpolation logic

**Why Relaxed Stops at High Profits?**
1. Aligns with Freqtrade best practice: "let profits run"
2. Prevents ATR stop from cutting winning trends prematurely
3. Coordinates with Layer 4 for smarter, context-aware exits
4. Execution order: `custom_exit` (Layer 4) → `custom_stoploss` (Layer 2) → exchange stop

**Layer 3: LLM Contextual Stop** (context_builder.py:350-398)
- Stop-loss levels shown to LLM in market context
- LLM can preemptively exit based on strategic analysis
- Position management prompts include profit protection rules

**Layer 4: Extreme Take-Profit Protection** (LLMFunctionStrategy.py:1045-1157)
- Triggers before LLM can react to extreme market moves:
  - **ROI >80% + trend weakening** (ADX <20 or ADX <25 with MACD hist <0) → Immediate exit
  - **ROI >100% unconditional** → Immediate exit
- Purpose: Protect exceptional profits from sudden reversals
- **Note:** Changed from RSI to trend strength (ADX + MACD) as RSI can stay extreme during strong trends

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

### Prompt Engineering Framework (2025-01-23 Overhaul)

Based on Google Prompt Engineering Whitepaper, the system now enforces structured Chain-of-Thought reasoning:

**Mandatory 4-Step CoT Framework** (prompt_builder.py:95-130):
1. **Market State Identification**: Classify as Trend/Range/Reversal based on:
   - EMA structure (20/50/200 alignment)
   - Price action relative to EMAs
   - ADX strength and momentum divergence

2. **Strategy Selection**: Choose from 7 predefined strategies:
   - **Trend Following** (3 variants): Pullback entry, breakout continuation, strong continuation
   - **Range Trading** (2 variants): Support bounce, resistance fade
   - **Reversal Trading** (2 variants): New trend start, failed breakout

3. **Entry Condition Verification**: Require ≥2 independent confirmations:
   - Technical signals (RSI, MACD, volume)
   - Price structure (support/resistance, candlestick patterns)
   - Momentum indicators (ADX, trend strength)

4. **Risk-Reward Assessment**: Validate R:R ≥ 2:1 before entry
   - Calculate stop-loss distance (Layer 1 + Layer 2)
   - Identify profit targets (key resistance/support levels)
   - Verify position sizing within risk limits

**Few-Shot Examples** (few_shot_examples.py):
- 6 detailed examples covering all major scenarios
- Each example demonstrates complete CoT reasoning
- Integrated into prompts via `get_entry_examples_for_prompt(max_examples=2)`

**Position Management Prompts** (prompt_builder.py:275-318):
- Noise filtering: Ignore <30% ATR fluctuations unless sustained 3+ candles
- Profit protection rules:
  - MFE drawback >50% → Reduce position 50%
  - MFE drawback >30% → Reduce position 30%
- Batch take-profit:
  - ≥5% profit → Reduce 40%
  - ≥8% profit → Reduce 50%
- Anchor checks: EMA structure, key levels, momentum alignment

**Temperature Setting**: Changed to 0.0 (from 0.6) for deterministic reasoning
- Ensures consistent CoT structure
- Reduces hallucination in function calling
- Trade-off: Less creativity in unusual market conditions

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
    "include_pair_specific_trades": true,
    "enable_rag": true  // IGNORED - RAG system deprecated
}
```

**Custom Stop-Loss Config** (NEW):
```json
"custom_stoploss_config": {
    "profit_thresholds": [0.0, 0.02, 0.06, 0.15],
    "atr_multipliers": [2.0, 1.5, 1.0, 0.8],
    "min_distances": [0.005, 0.015, 0.010, 0.005],
    "time_decay_hours": 2,
    "time_decay_factor": 0.8,
    "trend_strength_threshold": 25,
    "trend_strength_factor": 1.2
}
```
- Defines Layer 2 dynamic stop-loss behavior
- Smooth transitions between profit levels
- Time decay and trend adaptation parameters

**Custom Exit Config** (NEW):
```json
"custom_exit_config": {
    "extreme_profit_threshold": 0.50,
    "extreme_rsi_threshold": 90,
    "exceptional_profit_threshold": 0.70
}
```
- Defines Layer 4 extreme take-profit triggers
- Protects exceptional profits from sudden reversals

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
2. Each entry contains:
   - Market context sent to LLM
   - Function calls made
   - Validation results
   - Execution outcomes
3. Use `jq` for parsing: `jq -s '.' llm_decisions.jsonl | less`

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

### Critical Bug Fixes (fix/race-conditions-and-stoploss-bugs branch)

**Race Condition Fixes**:
1. **C4: Signal Cache Contamination** (LLMFunctionStrategy.py:619, 858)
   - Issue: `clear_signals()` cleared signals for ALL pairs, causing cross-pair interference
   - Fix: Changed to `clear_signal_for_pair(pair)` for atomic per-pair clearing
   - Impact: Prevents one pair's decision from affecting another pair's signal

2. **C6: Position Adjustment Cache Race** (LLMFunctionStrategy.py:1048)
   - Issue: Multiple threads could read `_position_adjustment_cache[pair]` simultaneously
   - Fix: Atomic `pop()` operation instead of read-then-delete
   - Impact: Ensures exactly-once position adjustment execution

3. **M5: Remaining Stake Validation** (LLMFunctionStrategy.py:1072-1084)
   - Issue: Position reduction could leave remaining stake below `min_stake`, creating orphan positions
   - Fix: Validate `remaining_stake >= min_stake` before applying reduction
   - Impact: Prevents unclosable micro-positions

**Stop-Loss Calculation Fixes**:
1. **H2/H4: Floating Point Comparison** (stoploss_calculator.py)
   - Issue: Direct float equality checks caused missed threshold transitions
   - Fix: Use epsilon-based comparison (`abs(a - b) < 1e-10`)
   - Impact: Reliable profit threshold detection

2. **M6-M12: Edge Cases** (LLMFunctionStrategy.py, stoploss_calculator.py)
   - ATR = 0 handling (fallback to min_distance)
   - Negative current_profit handling
   - Short position stop-loss direction (1 + distance vs 1 - distance)
   - Smooth transition interpolation at profit boundaries

**When Modifying Code**:
- Always clear signals per-pair, never globally
- Use atomic operations for shared state (pop/get with defaults)
- Validate stake amounts after any position size changes
- Test both long and short positions in stop-loss calculations

## Important Constraints

### Security
- Never commit real API keys - use environment variables
- Review function parameter bounds in decision_checker.py before deployment
- Validate LLM outputs before execution (already implemented in FunctionExecutor)

### Performance
- LLM requests block strategy analysis - keep timeout reasonable (60s default)
- Reduce `max_context_tokens` if strategy analysis too slow
- Monitor `indicator_history_points` - more data = more tokens

### Trading Safety
- **ALWAYS** test in dry_run mode first
- Four-layer stop-loss system is critical - do not disable Layer 1 or Layer 2 without replacement
- LLM decisions are deterministic with temperature=0.0 but still require monitoring
- Recommended max leverage: 10x (config allows 20x, but safe_max_leverage caps based on stop-loss)
- Single position risk should not exceed 5% of wallet
- Layer 4 take-profit protection is optional but recommended for extreme volatility pairs

### Data Management
- JSONL logs grow indefinitely - implement rotation if running long-term
- `tradesv3.sqlite` contains Freqtrade trade history
- Backup logs before running `./manage.sh clean`

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

## Recent Changes (Branch: fix/race-conditions-and-stoploss-bugs)

**Current Branch**: fix/race-conditions-and-stoploss-bugs (5 commits ahead of main)

**Major Changes** (+1478 lines, -200 lines across 9 files):

1. **LLMFunctionStrategy.py** (+415 lines)
   - Implemented four-layer stop-loss system (Layer 2: dynamic ATR trailing, Layer 4: extreme take-profit)
   - Fixed race conditions (C4: per-pair signal clearing, C6: atomic cache operations, M5: stake validation)
   - Added leverage_callback() for safe max leverage calculation

2. **prompt_builder.py** (+695 lines)
   - Complete overhaul based on Google Prompt Engineering Whitepaper (2025-01-23)
   - Mandatory 4-step CoT framework with 7 predefined trading strategies
   - Integrated few-shot examples from few_shot_examples.py
   - Position management prompts with profit protection rules

3. **few_shot_examples.py** (NEW, 546 lines)
   - Six detailed CoT examples covering trend/range/reversal scenarios
   - Demonstrates complete reasoning process for entry and exit decisions

4. **stoploss_calculator.py** (NEW, 200 lines)
   - Dynamic stop-loss calculations with smooth transitions
   - Time decay and trend adaptation logic
   - Handles edge cases (ATR=0, short positions, floating point comparisons)

5. **llm_client.py** (+100 lines)
   - Enhanced retry logic when LLM doesn't call functions
   - Support for thinking models (<think>...</think> extraction)
   - Improved error handling

6. **trading_tools.py** (+116 lines)
   - Refined parameter validation
   - Atomic position adjustment caching

7. **context_builder.py** (+101 lines)
   - Dynamic stop-loss level display in market context
   - Enhanced multi-timeframe data formatting

**Configuration Changes**:
- Model: `qwen3-30b-a3b-thinking` → `gemini-flash-lite-latest`
- Temperature: `0.6` → `0.0`
- Max tokens: `2500` → `65536`
- Stop-loss: `-6%` → `-10%`
- Added: `custom_stoploss_config`, `custom_exit_config`

**New Files in Project Root**:
- `AGENTS.md`, `OPTIMIZATION_SUMMARY.md`, `PARAMETERS_ANALYSIS.md`
- `LEARNING_SYSTEM_ARCHITECTURE.md`, `JSONL_LOGGING_ANALYSIS.md`
- Various analysis reports from development process

**Commit History** (most recent first):
```
511d5b9 fix: 修复竞态条件与止损计算bug，优化风险管理系统
1da35c3 feat: prioritize profit protection in position adjustment prompts
7f51517 feat: refactor prompt builder for enhanced funds management
c7e2403 feat: enhance trading bot modules for better risk management
62d7b6e fix: remove remaining text truncations in lessons and trade reviews
```

## Technology Stack

- **Freqtrade 2.x**: Trading framework (Docker: freqtradeorg/freqtrade:stable)
- **Python 3.11+**: Primary language
- **CCXT**: Exchange API abstraction
- **Pandas + TA-Lib**: Technical analysis
- **OpenAI-compatible API**: LLM integration
- **SQLite**: Trade database
- **Docker + Docker Compose**: Containerization

## Related Documentation

- `README.md`: Comprehensive project documentation (Chinese)
- `BACKTEST_GUIDE.md`: Backtesting scenarios and verification
- `IMPLEMENTATION_SUMMARY.md`: Stop-loss implementation details
- `TAKE_PROFIT_IMPLEMENTATION.md`: Position protection strategies
- `CONFIG_TEMPLATE.md`: Configuration reference
