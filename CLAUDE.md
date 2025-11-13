# CLAUDE.md - Freqtrade LLM Strategy Project

## Project Overview

Freqtrade LLM Function Calling Strategy is an advanced cryptocurrency trading bot integrating:
- Freqtrade trading framework
- OpenAI-compatible LLM for intelligent decisions
- RAG system with FAISS vector search
- Automated trade evaluation and learning

### Core Purpose
Automate crypto futures trading with LLM intelligence through OpenAI Function Calling (9 trading functions).

### Key Innovation
LLM calls trading functions, rates trades 0-100, learns from historical patterns via FAISS vector search.

---

## Directory Structure

ft-userdata-llm-share-v2/
├── docker-compose.yml              # Docker orchestration
├── Dockerfile.custom               # Custom image (faiss-cpu, numpy)
├── manage.sh                       # One-click management
├── README.md                       # Full documentation
│
└── user_data/
    ├── config.json                 # Main configuration
    ├── strategies/
    │   ├── LLMFunctionStrategy.py   # Main strategy class
    │   └── llm_modules/
    │       ├── llm/                # OpenAI API integration
    │       ├── learning/           # RAG, FAISS, evaluation
    │       ├── tools/              # Trading functions
    │       ├── experience/         # Logging
    │       ├── context/            # Prompt building
    │       ├── indicators/         # Technical analysis
    │       └── utils/              # Configuration
    ├── data/
    │   ├── binance/                # OHLC data
    │   └── rag/                    # FAISS vectors
    └── logs/
        ├── freqtrade.log
        ├── llm_decisions.jsonl
        └── trade_experience.jsonl

---

## Technology Stack

Core: Python 3.11+, Freqtrade, OpenAI-compatible LLM
Vector DB: FAISS (text-embedding-bge-m3, 1024-dim)
Database: SQLite

Dependencies:
- numpy>=1.20.0
- faiss-cpu>=1.7.0
- requests>=2.31.0
- pandas, talib-binary

External Services:
- LLM API (OpenAI-compatible)
- Embedding API (bge-m3)
- Binance Futures

---

## Key Configuration Files

config.json: LLM config, risk management, experience config, context config
docker-compose.yml: Service definition, ports, volumes
Dockerfile.custom: Base freqtrade, adds faiss-cpu, numpy, chromadb

---

## Architecture Overview

Trading Loop (per 30-min candle):
1. Load market data
2. Calculate indicators
3. Build LLM context
4. Query RAG for similar trades
5. LLM calls trading function
6. Execute trade
7. Log to llm_decisions.jsonl
8. Track position
9. On close: evaluate, score, store in FAISS

RAG Learning:
- LLM scores trade (0-100)
- TradeEvaluator scores on 4 dimensions
- Embed and store in FAISS
- Next decision: retrieve similar trades

Indicators: EMA(20/50/200), RSI, MACD, Bollinger Bands, ATR, ADX, MFI, OBV

---

## Core Modules

Strategy: LLMFunctionStrategy.py (main entry point)
LLM: llm_client.py (API calls), function_executor.py (9 functions)
RAG: rag_manager.py, embedding_service.py, vector_store.py, trade_evaluator.py, reward_learning.py
Experience: trade_logger.py, experience_manager.py, position_tracker.py
Context: context_builder.py, indicator_calculator.py
Utils: config_loader.py, decision_checker.py, market_sentiment.py

---

## Build & Deployment

Docker (Recommended):
- ./manage.sh start (quick start)
- ./manage.sh deploy (full deployment)
- ./manage.sh logs (view logs)
- ./manage.sh restart
- ./manage.sh clean (DESTRUCTIVE)

REST API: http://localhost:8086/api/v1/
Auth: Basic Auth from config.json

---

## Configuration Best Practices

LLM Models:
- claude-haiku-4-5-20251001 (recommended)
- qwen/qwen3-30b-a3b-thinking (deep reasoning)
- gpt-4-turbo (stable)

Temperature: 0.0-0.3 (conservative), 0.4-0.7 (balanced), 0.8-1.0 (creative)

Risk Management:
- Max Leverage: 10-20x
- Position Size: 1-10% per trade
- Max Trades: 5-10
- Emergency Stop: -15%

RAG:
- Index: flat (<10k), hnsw (10k-1M), ivf (>1M)
- Similarity: 0.7 threshold
- Top K: 3-5
- Min Trades: 10-20

---

## Trade Scoring

Score 0-100: S(90-100) A(80-89) B(70-79) C(60-69) D(50-59) F(<50)

Dimensions:
- Profit (30%): Achieved target?
- Risk (25%): Stop loss, leverage, drawdown?
- Timing (25%): Entry/exit quality?
- Efficiency (20%): Profit per time?

Reward = (profit% / 100) * (score / 100) * leverage_factor

---

## Common Issues

1. "tool instance lacks method": Check enable_rag=true, restart
2. "Expecting value" from API: Test endpoint, check format
3. LLM timeouts: Increase timeout, reduce max_tokens, lower temperature
4. FAISS fails: Install faiss-cpu or rebuild
5. Database locked: Clean and restart

---

## Security & Risk

SECURITY:
- API keys in config.json - secure this file
- Never commit with real keys
- Use env variables in production

TRADING RISKS:
- Volatile market, 100% loss possible
- Leverage amplifies losses
- Not guaranteed profitable
- Liquidation possible

---

## Resources

Freqtrade: https://www.freqtrade.io/
OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
FAISS: https://github.com/facebookresearch/faiss
BGE: https://github.com/FlagOpen/FlagEmbedding

---

## Metadata

Version: 2.0.0
Date: January 15, 2025
Status: Stable
License: MIT
Python: 3.11+

Last Updated: November 13, 2025
Maintained By: Claude Code
