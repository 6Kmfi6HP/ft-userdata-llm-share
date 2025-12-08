"""
Main Graph Builder for LangGraph Trading System.

Combines Analysis, Debate, Position Management, and Reflection subgraphs
into a complete trading decision flow.

NEW Architecture (with Reflection replacing Grounding):

    START
      │
      ▼
    [analysis_subgraph] ──► parallel fan-out/fan-in of 4 agents
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
      │  [entry_reflection] ──► LLM-based CoVe verification
      │    │
      │    ├── Critical issues found ──► END (signal_wait)
      │    └── Verified ──► Continue
      │    │
      │    ▼
      │  [executor_agent] ──► 🤖 LLM final decision ──► END
      │
      └─── has_position=True ────► POSITION PATH
           │
           ▼
         [position_debate] ──► PosBull → PosBear → PosJudge
           │
           ▼
         [position_reflection] ──► LLM-based CoVe verification
           │
           ├── Critical issues found ──► END (signal_hold)
           └── Verified ──► Continue
           │
           ▼
         [executor_agent] ──► 🤖 LLM final decision ──► END

Reference: LLM_TRADING_HALLUCINATION_SOLUTION_REPORT.md
- Layer 3: Adversarial Debate (Bull vs Bear)
- Layer 4: Reflection Verification (LLM-based Chain-of-Verification)
- Executor Agent: LLM-based final decision
"""

import logging
import re
import time
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TradingDecisionState
from .subgraphs.analysis_graph import build_analysis_subgraph
from .subgraphs.debate_graph import build_debate_subgraph
from .subgraphs.position_debate_graph import build_position_debate_subgraph
from .nodes.verification import reflection_node, position_reflection_node
from .nodes.execution.executor_agent import executor_agent_node
from .edges.routing import (
    route_entry_or_position,
    route_after_debate,
    route_after_reflection,
    route_after_position_reflection,
)
from .logging import GraphDecisionLogger

logger = logging.getLogger(__name__)


def build_main_graph(
    checkpointer: Optional[MemorySaver] = None,
    debug: bool = False
) -> StateGraph:
    """
    Build the main trading decision graph.

    This graph combines:
    1. Analysis subgraph (4 agents parallel)
    2. Entry/Position routing based on has_position
    3. Entry path: Debate → Reflection → Executor Agent
    4. Position path: Position Debate → Position Reflection → Executor Agent

    Args:
        checkpointer: Optional checkpointer for state persistence
        debug: Enable debug logging

    Returns:
        Compiled StateGraph for trading decisions
    """
    logger.info("Building main trading decision graph with Reflection Verification")

    # Create main graph builder
    builder = StateGraph(TradingDecisionState)

    # Build subgraphs
    analysis_graph = build_analysis_subgraph()
    debate_graph = build_debate_subgraph()
    position_debate_graph = build_position_debate_subgraph()

    # Add analysis subgraph
    builder.add_node("analysis", analysis_graph)

    # === ENTRY PATH NODES ===
    builder.add_node("entry_debate", debate_graph)
    builder.add_node("entry_reflection", reflection_node)

    # === POSITION PATH NODES ===
    builder.add_node("position_debate", position_debate_graph)
    builder.add_node("position_reflection", position_reflection_node)

    # === SHARED EXECUTOR AGENT (LLM-based) ===
    builder.add_node("executor", executor_agent_node)

    # === EDGES ===

    # START → Analysis
    builder.add_edge(START, "analysis")

    # Analysis → Entry OR Position routing
    builder.add_conditional_edges(
        "analysis",
        route_entry_or_position,
        {
            "entry_debate": "entry_debate",
            "position_debate": "position_debate"
        }
    )

    # === ENTRY PATH EDGES ===
    # Entry Debate → Entry Reflection (always proceed)
    builder.add_edge("entry_debate", "entry_reflection")

    # Entry Reflection → Executor or END (if critical issues)
    builder.add_conditional_edges(
        "entry_reflection",
        route_after_reflection,
        {
            "executor": "executor",
            "end": END
        }
    )

    # === POSITION PATH EDGES ===
    # Position Debate → Position Reflection (always proceed)
    builder.add_edge("position_debate", "position_reflection")

    # Position Reflection → Executor or END (if critical issues)
    builder.add_conditional_edges(
        "position_reflection",
        route_after_position_reflection,
        {
            "executor": "executor",
            "end": END
        }
    )

    # === EXECUTOR → END ===
    builder.add_edge("executor", END)

    logger.info("Main graph structure built successfully with Reflection Verification")

    # Compile with optional checkpointer
    if checkpointer:
        compiled = builder.compile(checkpointer=checkpointer)
    else:
        compiled = builder.compile()

    if debug:
        _log_graph_structure(builder)

    return compiled


def _log_graph_structure(builder: StateGraph):
    """Log the graph structure for debugging."""
    logger.debug("=== Graph Structure ===")
    logger.debug("Nodes: analysis, entry_debate, entry_reflection, "
                 "position_debate, position_reflection, executor")
    logger.debug("Edges:")
    logger.debug("  START → analysis")
    logger.debug("  analysis → route_entry_or_position")
    logger.debug("  --- ENTRY PATH ---")
    logger.debug("  entry_debate → entry_reflection")
    logger.debug("  entry_reflection → executor (if verified)")
    logger.debug("  entry_reflection → END (if critical issues)")
    logger.debug("  --- POSITION PATH ---")
    logger.debug("  position_debate → position_reflection")
    logger.debug("  position_reflection → executor (if verified)")
    logger.debug("  position_reflection → END (if critical issues)")
    logger.debug("  --- SHARED ---")
    logger.debug("  executor → END")


def get_graph_description() -> str:
    """Get description of the main graph for documentation."""
    return """
Main Trading Decision Graph - Multi-Layer Verification System

Overview:
  Unified five-stage pipeline combining market analysis with adversarial debate,
  LLM-based reflection verification, and LLM-based execution for robust trading decisions.

Architecture:
  Entry Path: Analysis → Debate → Reflection → Executor Agent → END
  Position Path: Analysis → Position Debate → Position Reflection → Executor Agent → END

Stage 1: Analysis Subgraph (Parallel)
  - IndicatorAgent: RSI, MACD, ADX, Stochastic analysis
  - TrendAgent: EMA structure, support/resistance (vision-capable)
  - SentimentAgent: Funding rate, OI, Fear & Greed
  - PatternAgent: K-line pattern recognition (vision-capable)
  - Aggregator: Weighted consensus calculation

Stage 2: Debate Subgraph (Sequential) - Layer 3 Verification
  - BullAgent: Makes strongest case FOR the trade
  - BearAgent: Finds every possible flaw
  - JudgeAgent: Evaluates arguments, renders verdict

Stage 3: Reflection Verification (NEW - LLM-based CoVe)
  - Uses Chain-of-Verification to validate analysis
  - Verifies direction correctness (is the trade direction right?)
  - Verifies entry timing (is now a good time to enter?)
  - Verifies risk assessment (are risks properly identified?)
  - Provides natural language corrections and suggestions
  - Better handles diverse LLM output formats vs code-based grounding

Stage 4: Executor Agent (LLM-based)
  - Receives all analysis, debate, and reflection results
  - Uses reflection suggestions for final decision
  - Applies risk management rules
  - Produces final trading decision with SL/TP levels

Token Usage:
  - 4 analysis agents (parallel) = 4 LLM calls
  - 3 debate agents (sequential) = 3 LLM calls
  - 1 reflection agent = 1 LLM call
  - 1 executor agent = 1 LLM call
  - Total: 9 LLM calls per decision

Reflection Verification Details:
  - Direction Verification: Is the recommended direction correct?
  - Entry Timing: Is this a good entry point?
  - Risk Assessment: Are risks properly identified?
  - Logical Consistency: Is the analysis self-consistent?
  - Provides confidence adjustment and suggested corrections

Agent Weights:
  - TrendAgent: 1.2 (highest - trend is king)
  - PatternAgent: 1.1 (high - technical patterns)
  - IndicatorAgent: 1.0 (baseline)
  - SentimentAgent: 0.8 (lowest - supplementary)

Executor Agent Actions:
  Entry Path:
    - signal_entry_long: Open long position
    - signal_entry_short: Open short position
    - signal_wait: Wait for better opportunity
  
  Position Path:
    - signal_hold: Continue holding position
    - signal_exit: Close entire position
    - adjust_position: Scale in/partial exit with adjustment_pct
"""


class TradingGraphRunner:
    """
    Convenience class for running the trading graph.

    Handles graph compilation, state management, and result formatting.
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        risk_config: Optional[Dict[str, Any]] = None,
        experience_config: Optional[Dict[str, Any]] = None,
        use_checkpointer: bool = False,
        debug: bool = False
    ):
        """
        Initialize the graph runner.

        Args:
            llm_config: LLM configuration
            risk_config: Risk management configuration
            experience_config: Experience/logging configuration
            use_checkpointer: Whether to use memory checkpointer
            debug: Enable debug logging
        """
        self.llm_config = llm_config
        self.risk_config = risk_config or {}
        self.experience_config = experience_config or {}
        self.debug = debug

        checkpointer = MemorySaver() if use_checkpointer else None

        # Build and compile graph
        self.graph = build_main_graph(
            checkpointer=checkpointer,
            debug=debug
        )

        # Initialize decision logger
        log_decisions = self.experience_config.get("log_decisions", True)
        log_path = self.experience_config.get(
            "graph_decision_log_path",
            "./user_data/logs/graph_decisions.jsonl"
        )
        self.decision_logger = GraphDecisionLogger(
            log_path=log_path,
            enabled=log_decisions
        )

        logger.info(f"TradingGraphRunner initialized with Reflection Verification (decision_logging={log_decisions})")

    def run(
        self,
        pair: str,
        current_price: float,
        market_context: str,
        timeframe: str = "30m",
        ohlcv_data: Any = None,
        ohlcv_data_htf: Any = None,
        timeframe_htf: Optional[str] = None,
        has_position: bool = False,
        position_side: Optional[str] = None,
        position_profit_pct: float = 0.0,
        position_metrics: Optional[Dict] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the trading decision graph.

        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            current_price: Current market price
            market_context: Formatted market context string
            timeframe: Primary candle timeframe
            ohlcv_data: Optional OHLCV data for chart generation
            ohlcv_data_htf: Optional OHLCV data for higher timeframe
            timeframe_htf: Higher timeframe label
            has_position: Whether there's an existing position
            position_side: Side of existing position
            position_profit_pct: Current position profit %
            position_metrics: Position tracking metrics (MFE/MAE/etc)
            thread_id: Optional thread ID for checkpointing

        Returns:
            Execution result dictionary
        """
        from datetime import datetime

        # Build initial state
        initial_state = TradingDecisionState(
            pair=pair,
            current_price=current_price,
            market_context=market_context,
            timeframe=timeframe,
            ohlcv_data=ohlcv_data,
            ohlcv_data_htf=ohlcv_data_htf,
            timeframe_htf=timeframe_htf,
            has_position=has_position,
            position_side=position_side,
            position_profit_pct=position_profit_pct,
            position_metrics=position_metrics,
            llm_config=self.llm_config,
            risk_config=self.risk_config,
            thread_id=thread_id or f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            errors=[],
            warnings=[]
        )

        # Build config
        config = {
            "configurable": {
                "llm_config": self.llm_config,
                "risk_config": self.risk_config,
                "thread_id": initial_state["thread_id"]
            }
        }

        logger.info(f"Running trading graph for {pair}")

        start_time = time.time()
        
        try:
            # Run graph
            result = self.graph.invoke(initial_state, config)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Extract execution result
            execution_result = result.get("execution_result", {})

            if not execution_result:
                # Build default result from state
                execution_result = {
                    "action": result.get("final_action", "signal_wait"),
                    "pair": pair,
                    "confidence_score": result.get("final_confidence", 0.0),
                    "reason": result.get("final_reason", ""),
                    "leverage": result.get("final_leverage"),
                    "current_price": current_price,
                    "stop_loss": result.get("stop_loss_price"),
                    "take_profit": result.get("take_profit_price"),
                    "key_support": result.get("key_support"),
                    "key_resistance": result.get("key_resistance"),
                    "source": "langgraph_executor_agent",
                    "adjustment_pct": result.get("adjustment_pct"),
                    # Include reflection results
                    "reflection_summary": result.get("reflection_summary"),
                    "reflection_should_proceed": result.get("reflection_should_proceed"),
                }

            # Log the complete decision chain
            self.decision_logger.log_graph_decision(
                state=result,
                execution_result=execution_result,
                execution_time_ms=execution_time_ms
            )

            logger.info(
                f"Graph completed for {pair}: action={execution_result.get('action')} "
                f"(took {execution_time_ms:.0f}ms)"
            )

            return execution_result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Graph execution failed for {pair}: {e}")
            
            error_result = {
                "action": "signal_wait",
                "pair": pair,
                "confidence_score": 0.0,
                "reason": f"Graph execution error: {str(e)}",
                "source": "langgraph_executor_agent",
                "error": str(e)
            }
            
            self.decision_logger.log_graph_decision(
                state=initial_state,
                execution_result=error_result,
                execution_time_ms=execution_time_ms
            )
            
            return error_result

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state for a thread.

        Args:
            thread_id: Thread ID to retrieve

        Returns:
            Current state or None if not found
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return self.graph.get_state(config)
        except Exception as e:
            logger.error(f"Failed to get state for thread {thread_id}: {e}")
            return None
