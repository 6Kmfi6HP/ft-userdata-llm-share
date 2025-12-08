"""
LangGraph Client for LLM Trading System.

Main entry point that replaces ConsensusClient with Bull vs Bear debate system.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Stage 1: 专业 Agent 并行分析                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  IndicatorAgent → RSI, MACD, ADX, Stochastic 分析               │
    │  TrendAgent → EMA 结构、支撑阻力、价格结构分析 (视觉支持)         │
    │  SentimentAgent → 资金费率、多空比、OI、恐惧贪婪分析              │
    │  PatternAgent → K线形态识别 (视觉支持)                           │
    │           ↓                                                     │
    │  Aggregator → 加权共识聚合                                       │
    └─────────────────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Stage 2: Bull vs Bear 对抗辩论               │
    ├─────────────────────────────────────────────────────────────────┤
    │  Bull Agent: 机会发现者 - 为交易辩护                            │
    │  Bear Agent: 魔鬼代言人 - 找出所有缺陷                          │
    │  Judge Agent: 公正裁判 - 评估双方论点并裁决                      │
    │           ↓                                                     │
    │  Verdict: APPROVE / REJECT / ABSTAIN                           │
    └─────────────────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Stage 3: 决策验证与执行                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Validator → 风险管理检查                                        │
    │  Executor → 准备交易执行参数                                     │
    └─────────────────────────────────────────────────────────────────┘
"""

import logging
import re
import copy
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LangGraphClient:
    """
    LangGraph-based trading decision client.

    Replaces ConsensusClient with Bull vs Bear debate system using LangGraph.
    Provides interface compatibility with the existing strategy.
    """

    # Action mapping for compatibility
    ACTION_MAPPING = {
        "signal_entry_long": "enter_long",
        "signal_entry_short": "enter_short",
        "signal_exit": "exit",
        "signal_hold": "hold",
        "signal_wait": "wait",
    }

    def __init__(
        self,
        llm_config: Dict[str, Any],
        function_executor=None,  # Kept for compatibility, not used
        consensus_config: Optional[Dict[str, Any]] = None,
        trading_tools=None,
        experience_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LangGraph client.

        Args:
            llm_config: LLM configuration (api_base, model, temperature, etc.)
            function_executor: Legacy parameter, not used
            consensus_config: Configuration for debate system
            trading_tools: Trading tools instance for signal management
            experience_config: Experience/logging configuration
        """
        self.llm_config = llm_config
        self.trading_tools = trading_tools
        self.experience_config = experience_config or {}

        # Debate configuration
        config = consensus_config or {}
        self.enabled = config.get("enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 50)

        # Debate-specific config
        debate_config = llm_config.get("debate_config", {})
        self.debate_enabled = debate_config.get("enabled", True)
        self.min_debate_quality = debate_config.get("min_debate_quality", 60)
        self.confidence_calibration = debate_config.get("confidence_calibration", True)
        self.fallback_on_error = debate_config.get("fallback_on_error", True)

        # Risk config
        self.risk_config = {
            "min_confidence": config.get("confidence_threshold", 40),
            "max_leverage": llm_config.get("risk_management", {}).get("max_leverage", 100),
            "default_leverage": llm_config.get("risk_management", {}).get("default_leverage", 10),
        }

        # OHLCV data cache (for visual analysis)
        self._current_ohlcv = None
        self._current_ohlcv_htf = None  # Higher timeframe OHLCV
        self._current_timeframe = None
        self._current_timeframe_htf = None  # Higher timeframe
        self._current_pair = None

        # Last analysis state cache
        self._last_state = None

        # Initialize graph runner
        self._graph_runner = None
        self._init_graph_runner()

        logger.info(
            f"LangGraphClient initialized: enabled={self.enabled}, "
            f"debate_enabled={self.debate_enabled}, "
            f"confidence_threshold={self.confidence_threshold}"
        )

    def _init_graph_runner(self):
        """Initialize the LangGraph runner."""
        try:
            from ..trading_graph import TradingGraphRunner

            self._graph_runner = TradingGraphRunner(
                llm_config=self.llm_config,
                risk_config=self.risk_config,
                experience_config=self.experience_config,
                use_checkpointer=False,  # Disabled - can't serialize DataFrame
                debug=False
            )
            logger.info("TradingGraphRunner initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import TradingGraphRunner: {e}")
            self._graph_runner = None
        except Exception as e:
            logger.error(f"Failed to initialize TradingGraphRunner: {e}")
            self._graph_runner = None


    def set_current_ohlcv(
        self,
        dataframe,
        timeframe: str,
        pair: str = None,
        dataframe_htf=None,
        timeframe_htf: str = None
    ):
        """
        Set current OHLCV data for visual analysis agents.

        Args:
            dataframe: pandas DataFrame with OHLCV data (primary timeframe)
            timeframe: Timeframe string (e.g., "15m", "30m")
            pair: Trading pair (optional, for logging)
            dataframe_htf: pandas DataFrame with higher timeframe OHLCV data (e.g., 1h)
            timeframe_htf: Higher timeframe string (e.g., "1h")
        """
        self._current_ohlcv = dataframe
        self._current_timeframe = timeframe
        self._current_pair = pair
        self._current_ohlcv_htf = dataframe_htf
        self._current_timeframe_htf = timeframe_htf
        logger.debug(
            f"OHLCV data set: {pair}, timeframe={timeframe}, "
            f"rows={len(dataframe) if dataframe is not None else 0}"
            f"{f', htf={timeframe_htf} ({len(dataframe_htf)} rows)' if dataframe_htf is not None else ''}"
        )

    def clear_current_ohlcv(self):
        """Clear current OHLCV data cache (both primary and HTF)."""
        self._current_ohlcv = None
        self._current_timeframe = None
        self._current_pair = None
        self._current_ohlcv_htf = None
        self._current_timeframe_htf = None

    def call_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Main entry point for trading decisions.

        Interface compatible with ConsensusClient.call_with_functions().

        Args:
            messages: Message list with system and user messages
            functions: Available functions (not used in LangGraph mode)
            max_iterations: Max iterations (not used in LangGraph mode)

        Returns:
            Response dictionary with function_calls and metadata
        """
        if not self.enabled or not self._graph_runner:
            # Fallback to simple response if disabled
            return self._create_fallback_response("LangGraph disabled or not initialized")

        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("🔄 LangGraph Bull vs Bear Debate System Starting")
        logger.info("=" * 60)

        try:
            # Extract context from messages
            market_context = self._extract_market_context(messages)
            pair = self._extract_pair(messages) or self._current_pair or "UNKNOWN"
            current_price = self._extract_current_price(messages)
            has_position = self._extract_has_position(messages)
            position_side = self._extract_position_side(messages)
            position_profit = self._extract_position_profit(messages)

            logger.info(f"📊 Trading pair: {pair}")
            logger.info(f"   Current price: {current_price}")
            logger.info(f"   Has position: {has_position}")
            if has_position:
                logger.info(f"   Position side: {position_side}")
                logger.info(f"   Position profit: {position_profit:.2f}%")

            # Skip confidence check (post-validation)
            if self.trading_tools:
                self.trading_tools.set_skip_confidence_check(True)

            try:
                # Run the LangGraph trading decision
                # Vision mode: Pass OHLCV data for chart generation in PatternAgent/TrendAgent
                # Note: DataFrame serialization is handled by not using checkpointer
                execution_result = self._graph_runner.run(
                    pair=pair,
                    current_price=current_price,
                    market_context=market_context,
                    timeframe=self._current_timeframe or "30m",
                    ohlcv_data=self._current_ohlcv,  # Enable Vision mode
                    ohlcv_data_htf=self._current_ohlcv_htf,  # Higher timeframe for multi-TF analysis
                    timeframe_htf=self._current_timeframe_htf,  # Higher timeframe label
                    has_position=has_position,
                    position_side=position_side,
                    position_profit_pct=position_profit
                )

                # Convert to legacy format
                result = self._convert_to_legacy_format(execution_result, pair)

                # Post-validate confidence
                result = self._post_validate_confidence(result, pair)

            finally:
                # Restore confidence check
                if self.trading_tools:
                    self.trading_tools.set_skip_confidence_check(False)

        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            if self.fallback_on_error:
                result = self._create_fallback_response(str(e))
            else:
                raise

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️  LangGraph decision time: {elapsed:.2f}s")
        logger.info("=" * 60)

        return result

    def _convert_to_legacy_format(
        self,
        execution_result: Dict[str, Any],
        pair: str
    ) -> Dict[str, Any]:
        """
        Convert LangGraph execution result to legacy format.

        The legacy format has:
        - success: bool
        - function_calls: List[Dict] with function, arguments, result
        - consensus_confidence: float
        - merged_reason: str
        """
        action = execution_result.get("action", "signal_wait")
        confidence = execution_result.get("confidence_score", 0.0)
        reason = execution_result.get("reason", "")
        leverage = execution_result.get("leverage")

        # Extract RSI and trend strength from analysis results
        rsi_value = self._extract_rsi_from_result(execution_result)
        trend_strength = self._extract_trend_strength_from_result(execution_result)

        # Build function call
        function_call = {
            "function": action,
            "arguments": {
                "pair": pair,
                "confidence_score": confidence,
                "reason": reason,
            },
            "result": {
                "success": True,
                "action": self.ACTION_MAPPING.get(action, action),
                "pair": pair,
            }
        }

        # Add action-specific arguments
        if action in ("signal_entry_long", "signal_entry_short"):
            function_call["arguments"]["leverage"] = leverage or 10
            # Get current price for sensible default calculations
            current_price = execution_result.get("current_price", 0)
            key_support = execution_result.get("key_support")
            key_resistance = execution_result.get("key_resistance")
            # Use sensible defaults based on current price if not provided
            if key_support is None or key_support == 0:
                key_support = current_price * 0.95 if current_price > 0 else 0
            if key_resistance is None or key_resistance == 0:
                key_resistance = current_price * 1.05 if current_price > 0 else 0
            function_call["arguments"]["key_support"] = key_support
            function_call["arguments"]["key_resistance"] = key_resistance
            function_call["arguments"]["rsi_value"] = rsi_value
            function_call["arguments"]["trend_strength"] = trend_strength

        # Store signal in trading_tools cache (critical fix!)
        self._store_signal_in_cache(
            action=action,
            pair=pair,
            confidence=confidence,
            reason=reason,
            leverage=leverage,
            rsi_value=rsi_value,
            trend_strength=trend_strength,
            execution_result=execution_result
        )

        result = {
            "success": True,
            "function_calls": [function_call],
            "consensus_type": "langgraph_debate",
            "consensus_confidence": confidence,
            "merged_reason": reason,
            "source": "langgraph",
        }

        # Add debate metadata if available
        if "debate_verdict" in execution_result:
            result["debate_verdict"] = execution_result["debate_verdict"]
            result["winning_argument"] = execution_result.get("winning_argument")
            result["debate_confidence"] = execution_result.get("debate_confidence")

        # Cache for debugging
        self._last_state = execution_result

        return result

    def _store_signal_in_cache(
        self,
        action: str,
        pair: str,
        confidence: float,
        reason: str,
        leverage: Optional[int],
        rsi_value: float,
        trend_strength: str,
        execution_result: Dict[str, Any]
    ):
        """
        Store signal in trading_tools cache for strategy to retrieve.

        This is critical for LangGraph integration - without this,
        LLMFunctionStrategy.get_signal() returns None and shows
        "未提供明确信号" (no clear signal provided).
        """
        if not self.trading_tools:
            logger.warning("trading_tools not available, cannot store signal")
            return

        try:
            # Get current price from execution_result, or try to extract from pair
            current_price = execution_result.get("current_price", 0)
            if current_price == 0:
                # Try to get from cached pair info (fallback)
                current_price = self._extract_current_price([{"content": execution_result.get("market_context", "")}])

            # Calculate sensible defaults if key levels not provided
            # Default: support = 95% of current price, resistance = 105% of current price
            key_support = execution_result.get("key_support")
            key_resistance = execution_result.get("key_resistance")

            if key_support is None or key_support == 0:
                key_support = current_price * 0.95 if current_price > 0 else 0
            if key_resistance is None or key_resistance == 0:
                key_resistance = current_price * 1.05 if current_price > 0 else 0

            if action == "signal_entry_long":
                self.trading_tools.signal_entry_long(
                    pair=pair,
                    leverage=leverage or 10,
                    confidence_score=confidence,
                    key_support=key_support,
                    key_resistance=key_resistance,
                    rsi_value=rsi_value,
                    trend_strength=trend_strength,
                    reason=reason
                )
            elif action == "signal_entry_short":
                self.trading_tools.signal_entry_short(
                    pair=pair,
                    leverage=leverage or 10,
                    confidence_score=confidence,
                    key_support=key_support,
                    key_resistance=key_resistance,
                    rsi_value=rsi_value,
                    trend_strength=trend_strength,
                    reason=reason
                )
            elif action == "signal_exit":
                self.trading_tools.signal_exit(
                    pair=pair,
                    confidence_score=confidence,
                    rsi_value=rsi_value,
                    trade_score=50,  # Default score for LangGraph exits
                    reason=reason
                )
            elif action == "signal_hold":
                self.trading_tools.signal_hold(
                    pair=pair,
                    confidence_score=confidence,
                    rsi_value=rsi_value,
                    reason=reason
                )
            elif action == "signal_wait":
                self.trading_tools.signal_wait(
                    pair=pair,
                    confidence_score=max(1, confidence),  # Ensure min 1
                    rsi_value=rsi_value,
                    reason=reason
                )
            elif action == "adjust_position":
                # Position adjustment from position management subgraph
                adjustment_pct = execution_result.get("adjustment_pct", 0)
                self.trading_tools.adjust_position(
                    pair=pair,
                    adjustment_pct=adjustment_pct,
                    confidence_score=confidence,
                    key_support=key_support,
                    key_resistance=key_resistance,
                    reason=reason
                )
                logger.info(
                    f"Position adjustment signal stored: {pair} -> {adjustment_pct:+.0f}%"
                )
            else:
                logger.warning(f"Unknown action '{action}', storing as wait signal")
                self.trading_tools.signal_wait(
                    pair=pair,
                    confidence_score=max(1, confidence),
                    rsi_value=rsi_value,
                    reason=f"Unknown action: {action}. {reason}"
                )

            logger.debug(f"Signal stored in cache: {pair} -> {action}")

        except Exception as e:
            logger.error(f"Failed to store signal in cache: {e}")

    def _post_validate_confidence(
        self,
        result: Dict[str, Any],
        pair: str
    ) -> Dict[str, Any]:
        """
        Post-validate confidence against threshold.

        If confidence is too low, clear the signal.
        """
        if not self.trading_tools:
            return result

        if not result.get("success"):
            return result

        # Get consensus confidence
        avg_confidence = result.get("consensus_confidence", 0)

        # Extract action from function calls
        function_calls = result.get("function_calls", [])
        action = None

        for call in function_calls:
            func_name = call.get("function", "")
            if func_name in ("signal_entry_long", "signal_entry_short"):
                action = "enter_long" if "long" in func_name else "enter_short"
                break

        if not action:
            # Not an entry signal, no need to validate
            return result

        logger.info(f"📊 Post-validation for {pair}")
        logger.info(f"   Confidence: {avg_confidence:.1f}, threshold: {self.confidence_threshold}")

        if avg_confidence >= self.confidence_threshold:
            # Pass threshold, update signal confidence
            merged_reason = result.get("merged_reason", "")
            self.trading_tools.update_signal_confidence(pair, avg_confidence, merged_reason)
            logger.info("   ✅ Validation passed, signal valid")
        else:
            # Below threshold, clear signal
            self.trading_tools.clear_signal_for_pair(pair)
            logger.warning(
                f"   ❌ Confidence {avg_confidence:.1f} < {self.confidence_threshold}, "
                f"signal cleared"
            )

            result["success"] = False
            result["confidence_rejected"] = True
            result["error"] = (
                f"Confidence {avg_confidence:.1f} below threshold "
                f"{self.confidence_threshold}, entry signal cancelled"
            )

        return result

    def _create_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback response when graph fails."""
        return {
            "success": True,
            "function_calls": [{
                "function": "signal_wait",
                "arguments": {
                    "pair": self._current_pair or "UNKNOWN",
                    "confidence_score": 0,
                    "rsi_value": 50,
                    "reason": f"LangGraph fallback: {error_msg}"
                },
                "result": {"success": True, "action": "wait"}
            }],
            "consensus_type": "langgraph_fallback",
            "consensus_confidence": 0,
            "merged_reason": f"LangGraph fallback: {error_msg}",
            "error": error_msg
        }

    # ========== Message Extraction Methods ==========

    def _extract_market_context(self, messages: List[Dict[str, str]]) -> str:
        """Extract market context from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "<market_data>" in content or len(content) > 500:
                    return content
        return ""

    def _extract_pair(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Extract trading pair from messages."""
        for msg in messages:
            content = msg.get("content", "")

            # Match "交易对: XXX/USDT:USDT" format
            match = re.search(r'交易对[:\s]+([A-Z]+/USDT(?::USDT)?)', content)
            if match:
                return match.group(1)

            # Match "pair: XXX/USDT" format
            match = re.search(r'pair[:\s]+([A-Z]+/USDT(?::USDT)?)', content, re.IGNORECASE)
            if match:
                return match.group(1)

            # Match "## 交易对: XXX" format
            match = re.search(r'##\s*交易对[:\s]+([A-Z]+/USDT(?::USDT)?)', content)
            if match:
                return match.group(1)

        return None

    def _extract_current_price(self, messages: List[Dict[str, str]]) -> float:
        """Extract current price from messages."""
        for msg in messages:
            content = msg.get("content", "")

            # Match "当前价格: 12345.67" format
            match = re.search(r'当前价格[:\s]+([0-9.]+)', content)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

            # Match "current_price: 12345.67" format
            match = re.search(r'current_price[:\s]+([0-9.]+)', content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return 0.0

    def _extract_has_position(self, messages: List[Dict[str, str]]) -> bool:
        """Extract whether there's an existing position."""
        for msg in messages:
            content = msg.get("content", "")

            # Method 1: Check <positions> tag from context_builder
            # context_builder outputs: "<positions>\n### 持仓情况\n  持仓#1: 做多 10x杠杆"
            if "<positions>" in content:
                # If <positions> tag exists and contains "持仓#", there's a position
                if "持仓#" in content or "Position#" in content:
                    return True
                # Empty positions tag means no position
                if "</positions>" in content:
                    # Check if content between tags is substantial
                    start = content.find("<positions>")
                    end = content.find("</positions>")
                    if start != -1 and end != -1:
                        positions_content = content[start:end]
                        # If contains "持仓情况" but no actual position, check for details
                        if "持仓#" in positions_content:
                            return True
                return False

            # Method 2: Legacy keywords (fallback)
            if "持仓状态" in content or "position" in content.lower():
                if "无持仓" in content or "no position" in content.lower():
                    return False
                if "持仓中" in content or "holding" in content.lower():
                    return True
                if "当前方向" in content:
                    return True

            # Method 3: Check for position-specific patterns
            # "做多" or "做空" followed by leverage indicates position
            if ("做多" in content or "做空" in content) and "杠杆" in content:
                if "开仓价" in content or "当前盈亏" in content:
                    return True

        return False

    def _extract_position_side(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Extract position side from messages."""
        for msg in messages:
            content = msg.get("content", "")

            # Match position side
            if "多单" in content or "LONG" in content.upper():
                return "long"
            if "空单" in content or "SHORT" in content.upper():
                return "short"

            # Match "当前方向: long/short"
            match = re.search(r'当前方向[:\s]*(long|short)', content, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return None

    def _extract_position_profit(self, messages: List[Dict[str, str]]) -> float:
        """Extract position profit percentage from messages."""
        for msg in messages:
            content = msg.get("content", "")

            # Match "未实现盈亏: 5.23%" or "盈亏: -2.5%"
            match = re.search(r'(?:未实现)?盈亏[:\s]*([+-]?[0-9.]+)%', content)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

            # Match "profit: 5.23%"
            match = re.search(r'profit[:\s]*([+-]?[0-9.]+)%', content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return 0.0

    def _extract_rsi_from_result(self, execution_result: Dict[str, Any]) -> float:
        """
        Extract RSI value from execution result.

        Tries to get RSI from indicator_report or market context.
        Returns 50.0 (neutral) as default.
        """
        # Try to get from indicator report
        indicator_report = execution_result.get("indicator_report")
        if indicator_report:
            # If it's an AgentReport object
            if hasattr(indicator_report, 'signals'):
                for signal in indicator_report.signals:
                    if hasattr(signal, 'value') and 'rsi' in str(signal).lower():
                        try:
                            return float(signal.value)
                        except (ValueError, TypeError):
                            pass
            # If it's a dict
            elif isinstance(indicator_report, dict):
                # Check for direct RSI value
                if 'rsi' in indicator_report:
                    try:
                        return float(indicator_report['rsi'])
                    except (ValueError, TypeError):
                        pass
                # Check in analysis text for RSI mention
                analysis = indicator_report.get('analysis', '')
                if analysis:
                    rsi_match = re.search(r'RSI[:\s]*([0-9.]+)', analysis, re.IGNORECASE)
                    if rsi_match:
                        try:
                            return float(rsi_match.group(1))
                        except (ValueError, TypeError):
                            pass

        # Try to extract from market context stored in result
        market_context = execution_result.get("market_context", "")
        if market_context:
            rsi_match = re.search(r'RSI[:\s]*([0-9.]+)', market_context, re.IGNORECASE)
            if rsi_match:
                try:
                    return float(rsi_match.group(1))
                except (ValueError, TypeError):
                    pass

        # Default: neutral RSI
        return 50.0

    def _extract_trend_strength_from_result(self, execution_result: Dict[str, Any]) -> str:
        """
        Extract trend strength from execution result.

        Tries to get from trend_report or consensus confidence.
        Returns "moderate" as default.
        """
        # Map confidence to trend strength
        confidence = execution_result.get("confidence_score", 50.0)

        # Try to get from trend report first
        trend_report = execution_result.get("trend_report")
        if trend_report:
            # If it's an AgentReport object
            if hasattr(trend_report, 'confidence'):
                confidence = max(confidence, trend_report.confidence)
            elif isinstance(trend_report, dict):
                report_confidence = trend_report.get('confidence', 0)
                if report_confidence:
                    confidence = max(confidence, report_confidence)
                # Check for explicit trend strength in analysis
                analysis = trend_report.get('analysis', '')
                if analysis:
                    if any(word in analysis.lower() for word in ['strong trend', 'powerful', '强劲', '强势']):
                        return "strong"
                    elif any(word in analysis.lower() for word in ['weak trend', '弱势', 'sideways', '盘整']):
                        return "weak"

        # Check consensus confidence for trend strength
        consensus_confidence = execution_result.get("consensus_confidence", confidence)
        if consensus_confidence >= 75:
            return "strong"
        elif consensus_confidence >= 50:
            return "moderate"
        else:
            return "weak"

    # ========== Debug and Statistics Methods ==========

    def get_last_agent_state(self) -> Optional[Dict[str, Any]]:
        """
        Get last analysis state for debugging.

        Returns:
            Last execution result or None
        """
        return self._last_state

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "client_type": "langgraph",
            "enabled": self.enabled,
            "debate_enabled": self.debate_enabled,
            "confidence_threshold": self.confidence_threshold,
            "min_debate_quality": self.min_debate_quality,
            "confidence_calibration": self.confidence_calibration,
        }

    # ========== Compatibility Methods ==========

    def simple_call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """
        Simple LLM call for non-trading use.

        Falls back to direct LLM call.
        """
        try:
            from ..lc_integration.llm_factory import LLMFactory

            llm = LLMFactory.create_chat_model(
                self.llm_config,
                task_type="default",
                temperature=temperature
            )

            response = llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Simple call failed: {e}")
            return None

    def manage_context_window(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 6000
    ) -> List[Dict[str, str]]:
        """
        Manage context window size.

        Simple implementation that truncates old messages.
        """
        # Estimate tokens (rough: 4 chars = 1 token)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= max_tokens:
            return messages

        # Keep system message and recent messages
        result = []
        char_budget = max_tokens * 4

        # Always keep system message
        for msg in messages:
            if msg.get("role") == "system":
                result.append(msg)
                char_budget -= len(msg.get("content", ""))
                break

        # Add recent messages that fit
        for msg in reversed(messages):
            if msg.get("role") == "system":
                continue
            msg_len = len(msg.get("content", ""))
            if char_budget - msg_len > 0:
                result.insert(1, msg)
                char_budget -= msg_len
            else:
                break

        return result

    def add_to_history(self, role: str, content: str):
        """Placeholder for history management."""
        pass

    def clear_history(self):
        """Placeholder for history management."""
        pass

    def get_history(self, include_timestamp: bool = False) -> List[Dict[str, Any]]:
        """Placeholder for history management."""
        return []
