"""
LLM Function Calling Strategy
基于LLM函数调用的智能交易策略

作者: Claude Code
版本: 1.0.0
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy,
    informative,
    merge_informative_pair,
    stoploss_from_absolute,
)
from llm_modules.experience.experience_manager import ExperienceManager
from llm_modules.experience.trade_logger import TradeLogger

from llm_modules.indicators.indicator_calculator import IndicatorCalculator
from llm_modules.llm.function_executor import FunctionExecutor
from llm_modules.llm.llm_client import LLMClient
from llm_modules.llm.consensus_client import ConsensusClient
from llm_modules.tools.trading_tools import TradingTools

from llm_modules.utils.config_loader import ConfigLoader
from llm_modules.utils.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

from llm_modules.analysis.exit_reason_generator import ExitReasonGenerator
from llm_modules.experience.trade_reviewer import TradeReviewer

from llm_modules.learning.historical_query import HistoricalQueryEngine
from llm_modules.learning.decision_query import DecisionQueryEngine
from llm_modules.learning.pattern_analyzer import PatternAnalyzer
from llm_modules.learning.reward_learning import RewardLearningSystem
from llm_modules.learning.self_reflection import SelfReflectionEngine
from llm_modules.learning.trade_evaluator import TradeEvaluator
from llm_modules.utils.decision_checker import DecisionQualityChecker
from llm_modules.utils.exit_metadata_manager import ExitMetadataManager
from llm_modules.utils.market_comparator import MarketStateComparator

from llm_modules.utils.position_tracker import PositionTracker
from llm_modules.utils.stoploss_calculator import StoplossCalculator

# 学术论文整合模块 (Kelly公式 + 组合风险管理)
from llm_modules.utils.kelly_calculator import KellyCalculator
from llm_modules.utils.portfolio_risk_manager import PortfolioRiskManager


class LLMFunctionStrategy(IStrategy):
    """
    LLM函数调用策略

    特性:
    - OpenAI Function Calling 完整交易控制
    - 支持期货、多空双向、动态杠杆
    - 经验学习和持续优化
    """

    # 策略基本配置
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "15m"  # 15分钟K线，更细粒度的数据

    # 启动需要的历史数据
    startup_candle_count = 1000  # 15分钟*1000 = 约10.4天数据（确保EMA200稳定）

    # 启用分层止损保护：硬止损 + 动态追踪止损 + LLM 决策
    stoploss = (
        -0.10
    )  # 10% 硬止损，防止爆仓（与config.json一致，期货10倍杠杆下价格空间1.0%）
    use_custom_stoploss = True  # 启用自定义动态追踪止损

    # 仓位调整
    position_adjustment_enable = True
    max_entry_position_adjustment = 10

    # 订单类型 - 全部使用市价单
    order_types = {
        "entry": "market",
        "exit": "market",
    }

    # 最小持仓时间硬约束
    MIN_HOLDING_MINUTES = 120  # 最小持仓 120 分钟（8 根 15分钟 K 线）
    MIN_HOLDING_EXCEPTION_LOSS_PCT = -0.08  # 仅 -8% 以上亏损可提前退出

    def __init__(self, config: dict) -> None:
        """初始化策略"""
        super().__init__(config)

        logger.info("=" * 60)
        logger.info("LLM Function Calling Strategy - 正在初始化...")
        logger.info("=" * 60)

        try:
            # 1. 加载配置
            self.config_loader = ConfigLoader()
            self.llm_config = self.config_loader.get_llm_config()
            self.risk_config = self.config_loader.get_risk_config()
            self.experience_config = self.config_loader.get_experience_config()
            self.context_config = self.config_loader.get_context_config()

            # 2. 初始化自我学习系统
            trade_log_path = self.experience_config.get(
                "trade_log_path", "./user_data/logs/trade_experience.jsonl"
            )
            self.historical_query = HistoricalQueryEngine(trade_log_path)
            self.pattern_analyzer = PatternAnalyzer(min_sample_size=5)
            self.self_reflection = SelfReflectionEngine()
            self.trade_evaluator = TradeEvaluator()

            # 2.1 初始化决策查询引擎（用于获取上次分析决策）
            decision_log_path = self.experience_config.get(
                "decision_log_path", "./user_data/logs/llm_decisions.jsonl"
            )
            decision_query_config = {
                "previous_decision_max_age_hours": self.context_config.get("previous_decision_max_age_hours", 24),
                "previous_decision_max_chars": self.context_config.get("previous_decision_max_chars", 1500),
            }
            self.decision_query = DecisionQueryEngine(decision_log_path, decision_query_config)

            # 初始化奖励学习系统
            reward_config = {
                "storage_path": "./user_data/logs/reward_learning.json",
                "learning_rate": 0.1,
                "discount_factor": 0.95,
            }
            self.reward_learning = RewardLearningSystem(reward_config)

            logger.info(
                "✓ 自我学习系统已初始化 (HistoricalQuery, DecisionQuery, PatternAnalyzer, SelfReflection, TradeEvaluator, RewardLearning)"
            )

            # 2.5. 初始化学术论文整合模块 (Kelly + 组合风险管理)
            kelly_config = config.get("kelly_config", {})
            self.kelly_calculator = KellyCalculator(kelly_config) if kelly_config.get("enabled", True) else None

            portfolio_risk_config = config.get("portfolio_risk_config", {})
            self.portfolio_risk_manager = PortfolioRiskManager(portfolio_risk_config) if portfolio_risk_config.get("enabled", True) else None

            if self.kelly_calculator:
                logger.info("✓ Kelly公式仓位计算器已初始化 (基于Busseti et al. 2016)")
            if self.portfolio_risk_manager:
                logger.info("✓ 组合风险管理器已初始化 (软性警告模式)")

            # 3. 初始化上下文构建器（注入学习组件 + 学术论文模块）
            # 🔧 修复M8+M9: 传入止损配置，避免 ContextBuilder 中硬编码
            self.context_builder = ContextBuilder(
                context_config=self.context_config,
                historical_query_engine=self.historical_query,
                pattern_analyzer=self.pattern_analyzer,
                tradable_balance_ratio=config.get("tradable_balance_ratio", 1.0),
                max_open_trades=config.get("max_open_trades", 1),
                stoploss_config=config.get("custom_stoploss_config", {}),
                hard_stoploss_pct=abs(self.stoploss)
                * 100,  # 从策略的硬止损值转换为百分比
                kelly_calculator=self.kelly_calculator,
                portfolio_risk_manager=self.portfolio_risk_manager,
                decision_query_engine=self.decision_query,
            )

            # 4. 初始化函数执行器
            self.function_executor = FunctionExecutor()

            # 5. 初始化交易工具（简化版 - 只保留交易控制工具）
            self.trading_tools = TradingTools(self)

            # 6. 初始化LLM客户端（支持共识模式）
            consensus_config = self.llm_config.get("consensus_config", {})
            if consensus_config.get("enabled", False):
                self.llm_client = ConsensusClient(
                    self.llm_config,
                    self.function_executor,
                    consensus_config,
                    trading_tools=self.trading_tools  # 传入交易工具用于后置置信度验证
                )
                logger.info("✓ 双重决策共识客户端已启用（后置置信度验证）")
            else:
                self.llm_client = LLMClient(self.llm_config, self.function_executor)

            # 8. 注册所有工具函数
            self._register_all_tools()

            # 9. 初始化经验系统（注入反思引擎）
            self.trade_logger = TradeLogger(self.experience_config)

            self.experience_manager = ExperienceManager(
                trade_logger=self.trade_logger,
                self_reflection_engine=self.self_reflection,
                trade_evaluator=self.trade_evaluator,
                reward_learning=self.reward_learning,
            )

            # 10. 缓存
            self._leverage_cache = {}
            self._position_adjustment_cache = {}
            self._stake_request_cache = {}
            self._model_score_cache = {}  # 存储模型对交易的自我评分

            # 10.5 LLM调用节流状态（使用系统时间，仅live/dry_run生效）
            self._last_llm_entry_call: Dict[str, datetime] = {}
            self._last_llm_exit_call: Dict[str, datetime] = {}

            # 从config加载节流间隔
            throttle_config = config.get("llm_throttle_config", {})
            self.llm_entry_interval = throttle_config.get("entry_interval_minutes", 60)
            self.llm_exit_interval = throttle_config.get("exit_interval_minutes", 60)
            logger.info(f"✓ LLM节流: 开仓间隔={self.llm_entry_interval}分钟, 平仓间隔={self.llm_exit_interval}分钟")

            # 11. 初始化增强模块
            self.position_tracker = PositionTracker()
            self.market_comparator = MarketStateComparator()
            self.decision_checker = DecisionQualityChecker()
            self.trade_reviewer = TradeReviewer()
            logger.info(
                "✓ 增强模块已初始化 (PositionTracker, MarketStateComparator, DecisionChecker, TradeReviewer)"
            )

            # 11.5 初始化退出分析系统
            self.exit_metadata_manager = ExitMetadataManager()
            self.exit_reason_generator = ExitReasonGenerator(
                self.llm_client, config, context_builder=self.context_builder
            )
            logger.info(
                "✓ 退出分析系统已初始化 (ExitMetadataManager, ExitReasonGenerator + ContextBuilder)"
            )

            # 12. 系统提示词（两套：开仓和持仓）
            self.entry_system_prompt = self.context_builder.build_entry_system_prompt()
            self.position_system_prompt = (
                self.context_builder.build_position_system_prompt()
            )
            logger.info("✓ 已加载两套系统提示词（开仓/持仓管理）")

            logger.info("✓ 策略初始化完成")
            logger.info(f"  - LLM模型: {self.llm_config.get('model')}")
            logger.info(
                f"  - 交易工具已注册: {len(self.function_executor.list_functions())} 个"
            )
            logger.info(f"  - 自我学习系统: 已启用（历史查询+模式分析+自我反思）")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"策略初始化失败: {e}", exc_info=True)
            raise

    def _get_system_prompt(self, has_position: bool) -> str:
        """
        根据是否有仓位选择系统提示词

        Args:
            has_position: 是否有仓位

        Returns:
            对应的系统提示词
        """
        if has_position:
            return self.position_system_prompt
        else:
            return self.entry_system_prompt

    def _should_run_llm_analysis(self, pair: str, analysis_type: str) -> bool:
        """
        检查是否应该运行LLM分析（基于时间间隔）
        仅在live/dry_run模式生效，回测不节流

        Args:
            pair: 交易对
            analysis_type: "entry" 或 "exit"

        Returns:
            True 如果应该运行LLM分析，False 如果应该跳过
        """
        # 回测模式不节流
        runmode = self.config.get("runmode")
        if runmode and runmode.value not in ("live", "dry_run"):
            return True

        now = datetime.now(timezone.utc)

        if analysis_type == "entry":
            interval = self.llm_entry_interval
            cache = self._last_llm_entry_call
        else:
            interval = self.llm_exit_interval
            cache = self._last_llm_exit_call

        last_call = cache.get(pair)
        if last_call is None:
            return True  # 首次调用

        elapsed = (now - last_call).total_seconds() / 60
        if elapsed >= interval:
            return True

        logger.debug(f"⏳ {pair} | {analysis_type} 跳过 | {elapsed:.1f}分钟 < {interval}分钟")
        return False

    def _record_llm_call(self, pair: str, analysis_type: str) -> None:
        """
        记录LLM调用时间

        Args:
            pair: 交易对
            analysis_type: "entry" 或 "exit"
        """
        now = datetime.now(timezone.utc)
        if analysis_type == "entry":
            self._last_llm_entry_call[pair] = now
        else:
            self._last_llm_exit_call[pair] = now

    def _register_all_tools(self):
        """注册所有工具函数（简化版 - 只注册交易控制工具）"""
        # 只注册交易工具（市场数据、账户信息已在context中提供）
        if self.trading_tools:
            self.function_executor.register_tools_from_instance(
                self.trading_tools, self.trading_tools.get_tools_schema()
            )
            logger.debug(
                f"已注册 {len(self.trading_tools.get_tools_schema())} 个交易控制函数"
            )

    def _collect_multi_timeframe_history(self, pair: str) -> Dict[str, pd.DataFrame]:
        """根据ContextBuilder配置获取多时间框架K线数据"""
        if not getattr(self.context_builder, "include_multi_timeframe_data", True):
            return {}

        if not hasattr(self, "dp") or not self.dp:
            return {}

        if not hasattr(self.context_builder, "get_multi_timeframe_history_config"):
            return {}

        tf_config = self.context_builder.get_multi_timeframe_history_config()
        if not tf_config:
            return {}

        history: Dict[str, pd.DataFrame] = {}

        for timeframe, cfg in tf_config.items():
            candles = cfg.get("candles", 0)
            fields = cfg.get("fields", [])
            tf_df = self._fetch_timeframe_dataframe(pair, timeframe, candles, fields)
            if tf_df is not None and not tf_df.empty:
                history[timeframe] = tf_df

        return history

    def _fetch_timeframe_dataframe(
        self, pair: str, timeframe: str, candles: int, fields: List[str]
    ) -> Optional[pd.DataFrame]:
        if candles <= 0:
            return None

        try:
            raw_df = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        except Exception as e:
            logger.warning(f"获取{timeframe}数据失败: {e}")
            return None

        if raw_df is None or raw_df.empty:
            return None

        padding = max(candles + 100, 200)
        df = raw_df.tail(padding).copy()

        self._append_indicator_columns(df, fields)

        return df.tail(candles)

    def _append_indicator_columns(self, dataframe: pd.DataFrame, fields: List[str]):
        """
        在给定dataframe上补齐所需指标列
        使用统一的 IndicatorCalculator 简化逻辑
        """
        if not fields:
            return

        # 简单粗暴：直接添加所有指标（IndicatorCalculator会跳过已存在的列）
        # 这比之前的逐个判断更简洁，且计算成本可忽略
        IndicatorCalculator.add_all_indicators(dataframe)

    def bot_start(self, **kwargs) -> None:
        """
        策略启动时调用（此时dp和wallets已初始化）
        """
        logger.info("✓ Bot已启动，策略运行中...")
        logger.info(
            f"✓ 交易工具: {len(self.function_executor.list_functions())} 个函数可用"
        )

        # 启动清算数据追踪器（WebSocket后台收集）
        try:
            # 获取配置的交易对列表
            trading_pairs = self.config.get("exchange", {}).get("pair_whitelist", [])
            if trading_pairs and hasattr(self.context_builder, "sentiment"):
                # 转换为Binance格式的symbol（如 BTC/USDT:USDT -> BTCUSDT）
                symbols = []
                for pair in trading_pairs:
                    # 处理期货格式 BTC/USDT:USDT -> BTCUSDT
                    symbol = pair.replace("/", "").replace(":USDT", "")
                    symbols.append(symbol)

                self.context_builder.sentiment.start_liquidation_tracker(symbols)
                logger.info(f"✓ 清算数据追踪器已启动，监控 {len(symbols)} 个交易对")
        except Exception as e:
            logger.warning(f"启动清算数据追踪器失败: {e}")

    def bot_cleanup(self) -> None:
        """
        策略清理时调用（Bot关闭前）
        """
        logger.info("正在清理策略资源...")

        # 停止清算数据追踪器
        try:
            if hasattr(self, "context_builder") and hasattr(self.context_builder, "sentiment"):
                self.context_builder.sentiment.stop_liquidation_tracker()
                logger.info("✓ 清算数据追踪器已停止")
        except Exception as e:
            logger.warning(f"停止清算数据追踪器失败: {e}")

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        开仓确认回调 - 保存市场状态到 MarketComparator

        注意：此时 trade 对象还未创建，无法获取 trade_id
        暂时先获取技术指标，等 trade 创建后再关联
        """
        try:
            # 获取最新的dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return True

            latest = dataframe.iloc[-1]

            # 提取技术指标
            indicators = {
                "atr": latest.get("atr", 0),
                "rsi": latest.get("rsi", 50),
                "ema_20": latest.get("ema_20", 0),
                "ema_50": latest.get("ema_50", 0),
                "macd": latest.get("macd", 0),
                "macd_signal": latest.get("macd_signal", 0),
                "adx": latest.get("adx", 0),
            }

            # 暂存开仓信息（将在下一次 populate 中关联 trade_id）
            # 使用 pair+rate 作为临时key
            temp_key = f"{pair}_{rate}"
            self._pending_entry_states = getattr(self, "_pending_entry_states", {})
            self._pending_entry_states[temp_key] = {
                "pair": pair,
                "rate": rate,
                "indicators": indicators,
                "entry_tag": entry_tag or "",
                "side": side,
                "time": current_time,
            }

            logger.debug(f"开仓确认: {pair} @ {rate}, 等待trade_id关联")

        except Exception as e:
            logger.error(f"confirm_trade_entry 失败: {e}")

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Any,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        平仓确认回调 - 生成交易复盘
        """
        try:
            # 获取持仓追踪数据
            position_metrics = self.position_tracker.get_position_metrics(trade.id)

            # 获取市场状态变化
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                latest = dataframe.iloc[-1]
                current_indicators = {
                    "atr": latest.get("atr", 0),
                    "rsi": latest.get("rsi", 50),
                    "ema_20": latest.get("ema_20", 0),
                    "ema_50": latest.get("ema_50", 0),
                    "macd": latest.get("macd", 0),
                    "adx": latest.get("adx", 0),
                }
                market_changes = self.market_comparator.compare_with_entry(
                    trade_id=trade.id,
                    current_price=rate,
                    current_indicators=current_indicators,
                )
            else:
                market_changes = {}

            # 手动计算盈亏百分比（因为此时 trade.close_profit 可能为 None）
            if trade.is_short:
                profit_pct = (
                    (trade.open_rate - rate) / trade.open_rate * trade.leverage * 100
                )
            else:
                profit_pct = (
                    (rate - trade.open_rate) / trade.open_rate * trade.leverage * 100
                )

            # 计算持仓时长（处理时区兼容性）
            if trade.open_date.tzinfo is None:
                # trade.open_date 是 naive，current_time 也应该是 naive
                exit_time = (
                    current_time.replace(tzinfo=None)
                    if current_time.tzinfo
                    else current_time
                )
            else:
                # trade.open_date 是 aware，current_time 也应该是 aware
                exit_time = (
                    current_time
                    if current_time.tzinfo
                    else current_time.replace(tzinfo=timezone.utc)
                )

            duration_minutes = int((exit_time - trade.open_date).total_seconds() / 60)

            # 生成交易复盘（如果 TradeReviewer 可用）
            if self.trade_reviewer:
                review = self.trade_reviewer.generate_trade_review(
                    pair=pair,
                    side="short" if trade.is_short else "long",
                    entry_price=trade.open_rate,
                    exit_price=rate,
                    entry_reason=getattr(trade, "enter_tag", "") or "",
                    exit_reason=exit_reason,
                    profit_pct=profit_pct,
                    duration_minutes=duration_minutes,
                    leverage=trade.leverage,
                    position_metrics=position_metrics,
                    market_changes=market_changes,
                )

                # 输出复盘报告
                report = self.trade_reviewer.format_review_report(review)
                logger.info(f"\n{report}")

            # ✅ 新增：生成 LLM 退出原因（统一所有退出场景的数据结构）
            trade_score = None
            confidence_score = None
            final_exit_reason = exit_reason

            # 1. 检查是否有退出元数据（Layer 1/2/4 自动退出）
            exit_metadata = self.exit_metadata_manager.get_and_clear(pair)

            # 🔧 修复：如果退出元数据为空，但退出原因是 stop_loss，说明是 Layer 1（交易所硬止损）
            if exit_metadata is None and exit_reason in [
                "stop_loss",
                "stoploss_on_exchange",
            ]:
                # Layer 1 触发：记录退出元数据，供后续 LLM 分析使用
                logger.info(
                    f"[退出分析] {pair} 触发 Layer 1 交易所硬止损，记录退出元数据"
                )
                self.exit_metadata_manager.record_exit(
                    pair=pair,
                    layer="layer1",
                    trigger_profit=profit_pct / 100,  # 转换为小数
                    exit_reason=exit_reason,
                )
                # 重新获取退出元数据
                exit_metadata = self.exit_metadata_manager.get_and_clear(pair)

            if exit_metadata is not None:
                # 自动退出场景：调用 LLM 生成详细原因
                try:
                    logger.info(
                        f"[退出分析] {pair} 触发 {exit_metadata['layer']} 自动退出，调用 LLM 生成原因"
                    )

                    # 将 trade 对象添加到 exit_metadata 中，供 context_builder 使用
                    exit_metadata_with_trade = {
                        **exit_metadata,
                        'trade': trade
                    }

                    llm_exit_result = self.exit_reason_generator.generate_exit_reason(
                        pair=pair,
                        exit_layer=exit_metadata["layer"],
                        exit_metadata=exit_metadata_with_trade,
                        current_dataframe=dataframe,
                    )

                    # 使用 LLM 生成的详细原因
                    final_exit_reason = llm_exit_result["reason"]
                    trade_score = llm_exit_result["trade_score"]
                    confidence_score = llm_exit_result["confidence_score"]
                    lesson = llm_exit_result.get("lesson")  # 可选的交易教训

                    log_msg = (
                        f"[退出分析] {pair} LLM 分析完成: "
                        f"score={trade_score}, confidence={confidence_score}"
                    )
                    if lesson:
                        log_msg += f"\n  📚 教训: {lesson}"
                    logger.info(log_msg)

                except Exception as e:
                    logger.error(f"[退出分析] {pair} LLM 分析失败: {e}", exc_info=True)
                    # 降级：使用原始 exit_reason
                    final_exit_reason = exit_reason

            elif exit_reason in ["exit_signal", "exit"]:
                # Layer 3 (LLM 主动退出)：从缓存中获取原因
                if pair in self._signal_cache:
                    cached_signal = self._signal_cache.get(pair, {})
                    if "reason" in cached_signal:
                        final_exit_reason = cached_signal.get("reason", exit_reason)
                        trade_score = cached_signal.get("trade_score", None)
                        confidence_score = cached_signal.get("confidence_score", None)
                        logger.info(
                            f"[退出分析] {pair} 使用 LLM 主动退出原因 (Layer 3)"
                        )

            # 记录交易到历史日志（供未来决策参考）
            if self.experience_manager:
                # 格式化持仓时间
                if duration_minutes < 60:
                    duration_str = f"{duration_minutes}分钟"
                elif duration_minutes < 1440:
                    duration_str = f"{duration_minutes / 60:.1f}小时"
                else:
                    duration_str = f"{duration_minutes / 1440:.1f}天"

                # 记录交易
                max_loss_pct = (
                    position_metrics.get("max_loss_pct", 0) if position_metrics else 0
                )
                max_profit_pct = (
                    position_metrics.get("max_profit_pct", 0) if position_metrics else 0
                )

                # 获取模型评分（优先使用 LLM 退出分析的 trade_score）
                model_score = (
                    trade_score
                    if trade_score is not None
                    else self._model_score_cache.pop(pair, None)
                )
                model_score_str = (
                    f"模型评分 {model_score:.0f}/100" if model_score else ""
                )
                market_condition = f"MFE {max_profit_pct:+.2f}% / MAE {max_loss_pct:+.2f}% / 持仓 {duration_str} / {model_score_str}"

                # 统一时区：确保 entry_time 和 exit_time 时区一致
                entry_time_unified = trade.open_date
                exit_time_unified = exit_time
                if entry_time_unified.tzinfo is None and exit_time_unified.tzinfo is not None:
                    entry_time_unified = entry_time_unified.replace(tzinfo=timezone.utc)
                elif entry_time_unified.tzinfo is not None and exit_time_unified.tzinfo is None:
                    exit_time_unified = exit_time_unified.replace(tzinfo=timezone.utc)
                
                self.experience_manager.log_trade_completion(
                    trade_id=trade.id,
                    pair=pair,
                    side="short" if trade.is_short else "long",
                    entry_time=entry_time_unified,  # ✅ 使用统一后的时间
                    entry_price=trade.open_rate,
                    entry_reason=getattr(trade, "enter_tag", "") or "未记录",
                    exit_time=exit_time_unified,  # ✅ 使用统一后的时间
                    exit_price=rate,
                    exit_reason=final_exit_reason,  # ✅ 使用 LLM 生成的详细原因
                    profit_pct=profit_pct,
                    profit_abs=trade.stake_amount * profit_pct / 100,
                    leverage=trade.leverage,
                    stake_amount=trade.stake_amount,
                    max_drawdown=max_loss_pct,
                    market_condition=market_condition,
                    position_metrics=position_metrics,  # 【新增】传递持仓指标
                    market_changes=market_changes,  # 【新增】传递市场变化
                    trade_score=trade_score,  # ✅ 新增：LLM 评分
                    confidence_score=confidence_score,  # ✅ 新增：LLM 置信度
                )
                logger.info(f"✓ 交易 {trade.id} 已记录到历史日志")

            # === 学术论文整合: 更新组合风险管理器 ===
            if self.portfolio_risk_manager:
                self.portfolio_risk_manager.record_trade_result(
                    profit_pct=profit_pct,
                    trade_info={
                        "pair": pair,
                        "side": "short" if trade.is_short else "long",
                        "leverage": trade.leverage
                    }
                )

            # 清理追踪数据
            if trade.id in self.position_tracker.positions:
                del self.position_tracker.positions[trade.id]
            if trade.id in self.market_comparator.entry_states:
                del self.market_comparator.entry_states[trade.id]

        except Exception as e:
            logger.error(f"生成交易复盘失败: {e}", exc_info=True)

        return True

    # 多时间框架数据支持
    @informative("1h")
    def populate_indicators_1h(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """1小时数据指标 - 使用统一的 IndicatorCalculator"""
        return IndicatorCalculator.add_all_indicators(dataframe)

    @informative("4h")
    def populate_indicators_4h(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """4小时数据指标 - 使用统一的 IndicatorCalculator"""
        return IndicatorCalculator.add_all_indicators(dataframe)

    @informative("1d")
    def populate_indicators_1d(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """日线数据指标（注意：8天数据只有8根日线K线，EMA50勉强可用，已删除EMA200）"""
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)  # 需要200天数据，删除
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_lower"] = bollinger["lowerband"]
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        计算技术指标（15分钟基础数据）- 使用统一的 IndicatorCalculator
        """
        return IndicatorCalculator.add_all_indicators(dataframe)

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        开仓信号 - 由LLM决策
        """
        pair = metadata["pair"]

        # === 学术论文整合: 更新组合风险管理器的余额 ===
        if self.portfolio_risk_manager and self.wallets:
            try:
                current_balance = self.wallets.get_total('USDT')
                if current_balance > 0:
                    self.portfolio_risk_manager.update_balance(current_balance)
            except Exception as e:
                logger.debug(f"更新组合风险管理器余额失败: {e}")

        # 默认不开仓
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        # 只在最新的K线上做决策
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # 获取当前所有持仓（用于传给context_builder）
            from freqtrade.persistence import Trade

            current_trades = Trade.get_open_trades()

            # 🔧 修复：检查当前交易对是否已有持仓
            # 如果已有持仓，跳过开仓分析（由 populate_exit_trend 进行持仓管理）
            pair_has_position = any(t.pair == pair for t in current_trades)
            if pair_has_position:
                logger.debug(f"⏭️  {pair} | 已有持仓，跳过开仓分析")
                return dataframe

            # LLM调用节流检查（开仓分析）
            if not self._should_run_llm_analysis(pair, "entry"):
                return dataframe

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, "dp") and self.dp:
                if hasattr(self.dp, "_exchange"):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, "exchange"):
                    exchange = self.dp.exchange

            multi_tf_history = (
                self._collect_multi_timeframe_history(pair)
                if getattr(self.context_builder, "include_multi_timeframe_data", True)
                else {}
            )

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=self.wallets,
                current_trades=current_trades,
                exchange=exchange,
                position_tracker=self.position_tracker,
                market_comparator=self.market_comparator,
                multi_timeframe_data=multi_tf_history,
            )

            # 构建决策请求
            decision_request = self.context_builder.build_decision_request(
                action_type="entry",
                market_context=market_context,
                position_context="",  # 已包含在market_context中
            )

            # 调用LLM决策（使用开仓提示词）
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(has_position=False),
                },
                {"role": "user", "content": decision_request},
            ]

            # 设置 OHLCV 数据供视觉分析 Agent 使用
            if hasattr(self.llm_client, "set_current_ohlcv"):
                self.llm_client.set_current_ohlcv(dataframe, self.timeframe, pair)

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10,  # 限制迭代次数，防止无限循环
            )

            # 清除 OHLCV 缓存
            if hasattr(self.llm_client, "clear_current_ohlcv"):
                self.llm_client.clear_current_ohlcv()

            # 处理响应
            if response.get("success"):
                function_calls = response.get("function_calls", [])
                llm_message = response.get("message", "")

                # 检查是否有交易信号
                signal = self.trading_tools.get_signal(pair)

                # 提取置信度用于记录决策
                confidence = signal.get("confidence_score", 50) / 100 if signal else 0.5

                # 记录决策
                self.experience_manager.log_decision_with_context(
                    pair=pair,
                    action="entry",
                    decision=llm_message,
                    reasoning=str(function_calls),
                    confidence=confidence,
                    market_context={"indicators": market_context},
                    function_calls=function_calls,
                )

                if signal:
                    action = signal.get("action")
                    reason = signal.get("reason", llm_message)

                    # 提取新增参数
                    confidence_score = signal.get("confidence_score", 0)
                    key_support = signal.get("key_support", 0)
                    key_resistance = signal.get("key_resistance", 0)
                    rsi_value = signal.get("rsi_value", 0)
                    trend_strength = signal.get("trend_strength", "未知")
                    stake_amount = signal.get("stake_amount")

                    # 🛡️ 置信度门槛过滤（硬编码 70）
                    MIN_CONFIDENCE_THRESHOLD = 80
                    if action in ["enter_long", "enter_short"]:
                        if confidence_score < MIN_CONFIDENCE_THRESHOLD:
                            logger.warning(
                                f"⚠️ {pair} | 置信度 {confidence_score} < {MIN_CONFIDENCE_THRESHOLD}, "
                                f"信号被过滤，转为 signal_wait"
                            )
                            # 清空信号，不开仓
                            self.trading_tools.clear_signal_for_pair(pair)
                            return dataframe

                    if stake_amount and stake_amount > 0:
                        self._stake_request_cache[pair] = stake_amount

                    if action == "enter_long":
                        dataframe.loc[dataframe.index[-1], "enter_long"] = 1
                        dataframe.loc[dataframe.index[-1], "enter_tag"] = reason
                        logger.info(f"📈 {pair} | 做多 | 置信度: {confidence_score}")
                        logger.info(f"   支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "enter_short":
                        dataframe.loc[dataframe.index[-1], "enter_short"] = 1
                        dataframe.loc[dataframe.index[-1], "enter_tag"] = reason
                        logger.info(f"📉 {pair} | 做空 | 置信度: {confidence_score}")
                        logger.info(f"   支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "hold":
                        logger.info(
                            f"🔒 {pair} | 保持持仓 | 置信度: {confidence_score} | RSI: {rsi_value}"
                        )
                        logger.info(f"   理由: {reason}")
                    elif action == "wait":
                        logger.info(
                            f"⏸️  {pair} | 空仓等待 | 置信度: {confidence_score} | RSI: {rsi_value}"
                        )
                        logger.info(f"   理由: {reason}")
                else:
                    # 没有交易信号 = 观望，显示LLM的完整分析
                    logger.info(f"⏸️  {pair} | 未提供明确信号\n{llm_message}")

                # 🔧 修复C4: 清空当前交易对的信号缓存（避免竞态条件）
                self.trading_tools.clear_signal_for_pair(pair)

                # 记录LLM调用时间（用于节流）
                self._record_llm_call(pair, "entry")

        except Exception as e:
            logger.error(f"开仓决策失败 {pair}: {e}")

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        平仓信号 - 由LLM决策
        """
        pair = metadata["pair"]

        # 默认不平仓
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        # 只在最新的K线上做决策
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # 获取当前所有持仓
            from freqtrade.persistence import Trade

            current_trades = Trade.get_open_trades()

            # 检查当前交易对是否有持仓
            pair_has_position = any(t.pair == pair for t in current_trades)
            if not pair_has_position:
                return dataframe  # 无持仓，不需要决策

            # LLM调用节流检查（平仓分析）
            if not self._should_run_llm_analysis(pair, "exit"):
                return dataframe

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, "dp") and self.dp:
                if hasattr(self.dp, "_exchange"):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, "exchange"):
                    exchange = self.dp.exchange

            multi_tf_history = (
                self._collect_multi_timeframe_history(pair)
                if getattr(self.context_builder, "include_multi_timeframe_data", True)
                else {}
            )

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=self.wallets,
                current_trades=current_trades,
                exchange=exchange,
                position_tracker=self.position_tracker,
                market_comparator=self.market_comparator,
                multi_timeframe_data=multi_tf_history,
            )

            # 更新 PositionTracker 和关联 MarketComparator
            pair_trades = [t for t in current_trades if t.pair == pair]

            # 检查dataframe是否为空
            if dataframe.empty:
                logger.warning(f"{pair} dataframe为空，跳过持仓追踪更新")
                return dataframe

            current_price = dataframe.iloc[-1]["close"]

            for trade in pair_trades:
                try:
                    # 更新持仓追踪数据（仅更新 MFE/MAE，决策在 LLM 返回后记录）
                    self.position_tracker.update_position(
                        trade_id=trade.id,
                        pair=pair,
                        current_price=current_price,
                        open_price=trade.open_rate,
                        is_short=trade.is_short,
                        leverage=trade.leverage,
                        decision_type="price_update",  # 价格更新（非决策）
                        decision_reason="",  # 仅更新价格，决策在 LLM 返回后记录
                    )

                    # 关联待定的开仓状态（如果存在）
                    temp_key = f"{pair}_{trade.open_rate}"
                    if (
                        hasattr(self, "_pending_entry_states")
                        and temp_key in self._pending_entry_states
                    ):
                        pending = self._pending_entry_states[temp_key]
                        # 保存到 MarketComparator
                        self.market_comparator.save_entry_state(
                            trade_id=trade.id,
                            pair=pair,
                            price=trade.open_rate,
                            indicators=pending["indicators"],
                            entry_reason=pending["entry_tag"],
                            trend_alignment="",
                            market_sentiment="",
                        )
                        # 清除待定状态
                        del self._pending_entry_states[temp_key]
                        logger.debug(f"已关联开仓状态到 trade_id={trade.id}")

                except Exception as e:
                    logger.debug(f"更新持仓追踪失败: {e}")

            # 构建决策请求
            decision_request = self.context_builder.build_decision_request(
                action_type="exit",
                market_context=market_context,
                position_context="",  # 已包含在market_context中
            )

            # 调用LLM决策（使用持仓管理提示词）
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt(has_position=True),
                },
                {"role": "user", "content": decision_request},
            ]

            # 设置 OHLCV 数据供视觉分析 Agent 使用
            if hasattr(self.llm_client, "set_current_ohlcv"):
                self.llm_client.set_current_ohlcv(dataframe, self.timeframe, pair)

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10,  # 限制迭代次数，防止无限循环
            )

            # 清除 OHLCV 缓存
            if hasattr(self.llm_client, "clear_current_ohlcv"):
                self.llm_client.clear_current_ohlcv()

            if response.get("success"):
                llm_message = response.get("message", "")
                signal = self.trading_tools.get_signal(pair)
                if signal and signal.get("action") == "exit":
                    reason = signal.get("reason", llm_message)

                    # 提取新增参数
                    confidence_score = signal.get("confidence_score", 0)
                    rsi_value = signal.get("rsi_value", 0)
                    trade_score = signal.get("trade_score", None)  # 模型自我评分

                    # 缓存模型评分（在 confirm_trade_exit 中使用）
                    if trade_score is not None:
                        self._model_score_cache[pair] = trade_score

                    dataframe.loc[dataframe.index[-1], "exit_long"] = 1
                    dataframe.loc[dataframe.index[-1], "exit_short"] = 1
                    dataframe.loc[dataframe.index[-1], "exit_tag"] = reason
                    logger.info(
                        f"🔚 {pair} | 平仓 | 置信度: {confidence_score} | 自我评分: {trade_score}/100"
                    )
                    logger.info(f"   RSI: {rsi_value}")
                    logger.info(f"   理由: {reason}")

                    # 【立即生成交易复盘】- 在平仓信号发出时
                    if pair_trades and self.trade_reviewer:
                        try:
                            trade = pair_trades[0]

                            # 获取持仓追踪数据
                            position_metrics = (
                                self.position_tracker.get_position_metrics(trade.id)
                            )

                            # 获取市场状态变化
                            latest = dataframe.iloc[-1]
                            current_indicators = {
                                "atr": latest.get("atr", 0),
                                "rsi": latest.get("rsi", 50),
                                "ema_20": latest.get("ema_20", 0),
                                "ema_50": latest.get("ema_50", 0),
                                "macd": latest.get("macd", 0),
                                "adx": latest.get("adx", 0),
                            }
                            market_changes = self.market_comparator.compare_with_entry(
                                trade_id=trade.id,
                                current_price=current_price,
                                current_indicators=current_indicators,
                            )

                            # 计算持仓时长（分钟）
                            now = (
                                datetime.utcnow()
                                if trade.open_date.tzinfo is None
                                else datetime.now(timezone.utc)
                            )
                            duration_minutes = int(
                                (now - trade.open_date).total_seconds() / 60
                            )

                            # 计算预期平仓盈亏（使用当前市价）
                            exit_price = current_price
                            if trade.is_short:
                                profit_pct = (
                                    (trade.open_rate - exit_price)
                                    / trade.open_rate
                                    * trade.leverage
                                    * 100
                                )
                            else:
                                profit_pct = (
                                    (exit_price - trade.open_rate)
                                    / trade.open_rate
                                    * trade.leverage
                                    * 100
                                )

                            # 生成交易复盘
                            review = self.trade_reviewer.generate_trade_review(
                                pair=pair,
                                side="short" if trade.is_short else "long",
                                entry_price=trade.open_rate,
                                exit_price=exit_price,
                                entry_reason=getattr(trade, "enter_tag", "") or "",
                                exit_reason=reason,
                                profit_pct=profit_pct,
                                duration_minutes=duration_minutes,
                                leverage=trade.leverage,
                                position_metrics=position_metrics,
                                market_changes=market_changes,
                            )

                            # 输出复盘报告
                            report = self.trade_reviewer.format_review_report(review)
                            logger.info(f"\n{report}")

                        except Exception as e:
                            logger.error(f"生成交易复盘失败: {e}", exc_info=True)

                else:
                    logger.info(f"💎 {pair} | 继续持有\n{llm_message}")

                # 记录决策到 DecisionChecker（用于检测重复模式和盈利回撤）
                if signal:
                    action = signal.get("action")
                    reason = signal.get("reason", llm_message)

                    # 计算当前盈亏（用于决策质量分析）
                    if pair_trades:
                        trade = pair_trades[0]
                        if trade.is_short:
                            profit_pct = (
                                (trade.open_rate - current_price)
                                / trade.open_rate
                                * trade.leverage
                                * 100
                            )
                        else:
                            profit_pct = (
                                (current_price - trade.open_rate)
                                / trade.open_rate
                                * trade.leverage
                                * 100
                            )

                        # 记录决策
                        decision_type = "exit" if action == "exit" else "hold"

                        # 🔧 修复：更新 PositionTracker 的决策历史（包含真实的 reason）
                        try:
                            self.position_tracker.update_position(
                                trade_id=trade.id,
                                pair=pair,
                                current_price=current_price,
                                open_price=trade.open_rate,
                                is_short=trade.is_short,
                                leverage=trade.leverage,
                                decision_type=decision_type,
                                decision_reason=reason[:200] if reason else ""
                            )
                        except Exception as e:
                            logger.debug(f"更新持仓追踪决策失败: {e}")

                        try:
                            quality_check = self.decision_checker.record_decision(
                                pair=pair,
                                decision_type=decision_type,
                                reason=reason,
                                profit_pct=profit_pct,
                            )

                            # 如果有警告，记录到日志（不阻止交易）
                            if quality_check.get("warnings"):
                                for warning in quality_check["warnings"]:
                                    if warning.get("level") == "high":
                                        logger.warning(
                                            f"[决策质量警告] {warning.get('message')}"
                                        )
                                        if warning.get("suggestion"):
                                            logger.warning(
                                                f"  建议: {warning.get('suggestion')}"
                                            )

                        except Exception as e:
                            logger.debug(f"决策质量检查失败: {e}")

                # 🔧 修复C4: 清空当前交易对的信号缓存（避免竞态条件）
                self.trading_tools.clear_signal_for_pair(pair)

                # 记录LLM调用时间（用于节流）
                self._record_llm_call(pair, "exit")

        except Exception as e:
            logger.error(f"平仓决策失败 {pair}: {e}")

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        第4层止盈: 极端情况保护 (最小化干预LLM决策)

        【重要】只在极端情况下触发，绝大多数情况交给LLM决策：
        - ROI > 80% + 趋势减弱 = 保护暴利（趋势强度检查，而非RSI极端）
        - ROI > 100% = 无条件强制保护

        优化说明（2025-01-23）：
        - 提高阈值避免过早止盈：80% ROI（10x杠杆=8%价格波动）更符合趋势跟踪策略
        - 移除RSI条件：RSI极端在强趋势中可能持续，不应作为止盈信号
        - 改为趋势强度检查：ADX<20或(ADX<25且MACD柱状图<0)表示趋势减弱

        新增（2025-11-27）：
        - 最小持仓时间检查：硬编码 120 分钟约束
        - 解决短持仓导致亏损严重的问题

        杠杆处理：
        - 阈值直接表示ROI百分比 (current_profit已包含杠杆效应)
        - 例如：10x杠杆下，8%价格波动 = 80% ROI

        Returns:
            止盈理由字符串,或None(交给LLM决策)
        """
        try:
            # 获取配置
            exit_config = self.config.get("custom_exit_config", {})

            # ============ 🛡️ 最小持仓时间硬约束（优先使用类属性硬编码值） ============
            # 硬编码值优先于配置，确保最小持仓保护始终生效
            min_holding_minutes = self.MIN_HOLDING_MINUTES  # 硬编码 120 分钟
            exception_loss_pct = self.MIN_HOLDING_EXCEPTION_LOSS_PCT  # 硬编码 -8%

            # 计算持仓时间
            if hasattr(trade, "open_date_utc") and trade.open_date_utc:
                holding_duration = current_time - trade.open_date_utc
                holding_minutes = holding_duration.total_seconds() / 60
            else:
                holding_minutes = 0

            # 检查是否处于短持仓期间
            is_short_holding = holding_minutes < min_holding_minutes

            if is_short_holding:
                # 🛡️ 硬约束检查（始终启用 - 硬编码）
                # 例外情况：亏损超过阈值（如-8%）时允许提前退出
                is_severe_loss = current_profit < exception_loss_pct
                if not is_severe_loss:
                    # 仅 debug 级别记录，避免日志刷屏
                    logger.debug(
                        f"🛡️ {pair} | 持仓 {holding_minutes:.0f}分钟 < {min_holding_minutes}分钟，"
                        f"阻止退出"
                    )
                    return None  # 阻止LLM退出决策

            # ============ 原有的极端止盈保护逻辑 ============
            # 获取技术指标
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return None

            latest = dataframe.iloc[-1]
            adx = latest.get("adx", 0)
            macd = latest.get("macd", 0)
            macd_signal = latest.get("macd_signal", 0)
            macd_hist = macd - macd_signal  # MACD柱状图
            # 阈值直接表示ROI百分比 (current_profit已包含杠杆效应)
            extreme_profit_threshold = exit_config.get(
                "extreme_profit_threshold", 0.80
            )  # 从0.50提高到0.80
            exceptional_profit_threshold = exit_config.get(
                "exceptional_profit_threshold", 1.00
            )  # 从0.70提高到1.00
            trend_weak_threshold = exit_config.get(
                "trend_weak_threshold", 20
            )  # ADX趋势减弱阈值
            trend_weak_confirmation = exit_config.get(
                "trend_weak_confirmation", 25
            )  # ADX确认阈值

            # 浮点数比较容差（防止精度问题导致意外触发）
            PROFIT_EPSILON = exit_config.get(
                "profit_epsilon", 0.001
            )  # 可配置的容差，默认0.1%

            # 情况1: 超高利润(ROI>80%) + 趋势减弱 = 极端止盈保护
            # ✅ 优化：移除RSI条件，改为趋势强度检查（ADX + MACD）
            if current_profit >= (extreme_profit_threshold - PROFIT_EPSILON):
                # 检查趋势是否减弱：ADX<20 或 (ADX<25 且 MACD柱状图<0)
                trend_weakening = adx < trend_weak_threshold or (
                    adx < trend_weak_confirmation and macd_hist < 0
                )

                if trend_weakening:
                    logger.warning(
                        f"[第4层-极端止盈] {pair} {'做空' if trade.is_short else '做多'} | "
                        f"ROI {current_profit * 100:.2f}% > {extreme_profit_threshold * 100:.0f}% "
                        f"+ 趋势减弱(ADX={adx:.1f}, MACD_hist={macd_hist:.4f}) - 强制保护"
                    )
                    # ✅ 记录 Layer 4 退出元数据
                    self.exit_metadata_manager.record_exit(
                        pair=pair,
                        layer="layer4",
                        trigger_profit=current_profit,
                        adx_value=adx,
                        macd_hist=macd_hist,
                        profit_threshold=extreme_profit_threshold,
                        exit_reason="trend_weakening_protection",
                    )
                    return "trend_weakening_protection"

            # 情况2: 暴利(ROI>100%) = 无条件保护（已经是优秀交易）
            # ✅ 优化：提高阈值到100%，避免过早止盈
            if current_profit >= (exceptional_profit_threshold - PROFIT_EPSILON):
                logger.warning(
                    f"[第4层-暴利保护] {pair} {'做空' if trade.is_short else '做多'} | "
                    f"ROI {current_profit * 100:.2f}% > {exceptional_profit_threshold * 100:.0f}% "
                    f"- 已达暴利水平，强制保护"
                )
                # ✅ 记录 Layer 4 退出元数据
                self.exit_metadata_manager.record_exit(
                    pair=pair,
                    layer="layer4",
                    trigger_profit=current_profit,
                    adx_value=adx,
                    macd_hist=macd_hist,
                    profit_threshold=exceptional_profit_threshold,
                    exit_reason="exceptional_profit_protection",
                )
                return "exceptional_profit_protection"

            # 其他所有情况: 完全交给LLM智能决策
            return None

        except Exception as e:
            logger.debug(f"{pair} custom_exit检查失败: {e}")
            return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        动态杠杆 - 由LLM决定或使用缓存值
        """
        # 🔧 修复H12: 确保缓存已初始化（防御性编程）
        if not hasattr(self, "_leverage_cache"):
            logger.warning(f"{pair} _leverage_cache 未初始化，重新创建")
            self._leverage_cache = {}

        # 🔧 修复C6: 使用原子操作获取并删除缓存（避免竞态条件）
        leverage_value = self._leverage_cache.pop(pair, None)
        if leverage_value is not None:
            return min(leverage_value, max_leverage)

        # 默认杠杆
        default_leverage = self.risk_config.get("default_leverage", 10)

        # 动态调整最大允许杠杆，防止止损位置低于强平线
        # 假设强平线在 margin/leverage 位置，安全系数 0.8
        # stoploss = 0.06, 意味着最大安全杠杆约为 1 / (0.06 * 1.2) ≈ 13.8x
        # 所以如果当前 stoploss 是 6%，不应允许 20x 杠杆
        safe_max_leverage = 1.0 / (abs(self.stoploss) * 1.1)  # 留10%安全缓冲

        if max_leverage > safe_max_leverage:
            logger.warning(
                f"{pair} 配置的最大杠杆 {max_leverage}x 风险过高(止损{self.stoploss})，已限制为安全值 {safe_max_leverage:.1f}x"
            )
            max_leverage = safe_max_leverage

        return min(default_leverage, max_leverage)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        动态仓位大小 - 可由LLM调整
        """
        stake_request = None
        if hasattr(self, "_stake_request_cache"):
            stake_request = self._stake_request_cache.pop(pair, None)

        if stake_request is None:
            return proposed_stake

        desired = stake_request

        # 只检查最小值，不限制最大值（由tradable_balance_ratio自然限制）
        if min_stake and desired < min_stake:
            logger.warning(
                f"{pair} 指定投入 {stake_request:.2f} USDT 低于最小要求 {min_stake:.2f}，已调整为最小值"
            )
            desired = min_stake

        logger.info(
            f"{pair} 使用LLM指定仓位: {desired:.2f} USDT (请求 {stake_request:.2f})"
        )
        return desired

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        仓位调整 - 允许LLM加仓或减仓

        Args:
            trade: 当前交易对象
            current_rate: 当前价格
            其他参数...

        Returns:
            Optional[float]: 要增加的stake金额（正数=加仓，负数=减仓），None=不调整
        """
        pair = trade.pair

        # 🔧 修复C6: 使用原子操作获取并删除缓存（避免竞态条件）
        adjustment_info = self._position_adjustment_cache.pop(pair, None)
        if adjustment_info is None:
            return None  # 无调整

        adjustment_pct = adjustment_info.get("adjustment_pct", 0)
        reason = adjustment_info.get("reason", "")

        # 计算调整金额
        current_stake = trade.stake_amount
        adjustment_stake = current_stake * (adjustment_pct / 100)

        if adjustment_pct > 0:
            # 加仓
            adjustment_stake = min(adjustment_stake, max_stake)
            if min_stake and adjustment_stake < min_stake:
                logger.warning(
                    f"{pair} 加仓金额 {adjustment_stake} 低于最小stake {min_stake}"
                )
                return None

            logger.info(
                f"{pair} 加仓 {adjustment_pct:.1f}% = {adjustment_stake:.2f} USDT | {reason}"
            )
            return adjustment_stake

        elif adjustment_pct < 0:
            # 减仓
            # 🔧 修复M5: 验证减仓后剩余仓位是否满足最小stake要求
            remaining_stake = (
                current_stake + adjustment_stake
            )  # adjustment_stake 是负数

            if min_stake and 0 < remaining_stake < min_stake:
                logger.warning(
                    f"{pair} 减仓后剩余仓位 {remaining_stake:.2f} USDT 低于最小要求 {min_stake:.2f} USDT. "
                    f"拒绝减仓操作，建议全平或调整减仓幅度."
                )
                return None  # 拒绝无效的减仓

            max_reduce = -current_stake * 0.99  # 最多减99%（保留一点避免完全平仓）
            adjustment_stake = max(adjustment_stake, max_reduce)

            logger.info(
                f"{pair} 减仓 {abs(adjustment_pct):.1f}% = {adjustment_stake:.2f} USDT "
                f"(剩余{remaining_stake:.2f}) | {reason}"
            )

            # ✅ 新增：记录部分平仓到交易日志
            # 由于 Freqtrade 不会为 partial_exit 调用 confirm_trade_exit()
            # 我们需要在这里手动记录减仓事件
            if self.experience_manager:
                try:
                    # === 1. 准备日志数据 ===
                    exit_stake = abs(adjustment_stake)
                    exit_pct = abs(adjustment_pct)
                    confidence_score = adjustment_info.get("confidence_score", None)

                    # === 2. 计算持仓时长（用于日志统计）===
                    if current_time.tzinfo is None:
                        current_time = current_time.replace(tzinfo=timezone.utc)
                    if trade.open_date.tzinfo is None:
                        open_time = trade.open_date.replace(tzinfo=timezone.utc)
                    else:
                        open_time = trade.open_date
                    duration_minutes = int(
                        (current_time - open_time).total_seconds() / 60
                    )

                    # === 3. 获取持仓追踪数据（用于日志统计）===
                    position_metrics = (
                        self.position_tracker.get_position_metrics(trade.id)
                        if hasattr(self, "position_tracker")
                        else {}
                    )
                    max_loss_pct = (
                        position_metrics.get("max_loss_pct", 0)
                        if position_metrics
                        else 0
                    )
                    max_profit_pct = (
                        position_metrics.get("max_profit_pct", 0)
                        if position_metrics
                        else 0
                    )

                    # === 4. 构建市场状态字符串（用于日志展示）===
                    if duration_minutes < 60:
                        duration_str = f"{duration_minutes}分钟"
                    elif duration_minutes < 1440:
                        duration_str = f"{duration_minutes / 60:.1f}小时"
                    else:
                        duration_str = f"{duration_minutes / 1440:.1f}天"

                    market_condition = f"MFE {max_profit_pct:+.2f}% / MAE {max_loss_pct:+.2f}% / 持仓 {duration_str}"
                    if confidence_score:
                        market_condition += f" / 置信度 {confidence_score}/100"

                    # === 5. 构建 LLM 生成的退出原因（直接使用 reason，无需二次生成）===
                    llm_exit_reason = f"[部分平仓 {exit_pct:.0f}%] {reason}"

                    # === 6. 写入日志 ===
                    # 使用统一后的时间对象（已在1325-1330行统一时区）
                    self.experience_manager.log_trade_completion(
                        trade_id=trade.id,
                        pair=pair,
                        side="short" if trade.is_short else "long",
                        entry_time=open_time,  # ✅ 使用统一后的 open_time
                        entry_price=trade.open_rate,
                        entry_reason=getattr(trade, "enter_tag", "") or "未记录",
                        exit_time=current_time,  # ✅ 使用统一后的 current_time
                        exit_price=current_rate,
                        exit_reason=llm_exit_reason,  # ✅ 使用 LLM 生成的原因
                        profit_pct=current_profit
                        * 100,  # current_profit 已经是小数形式
                        profit_abs=exit_stake * current_profit,  # 减仓部分的盈亏
                        leverage=trade.leverage,
                        stake_amount=exit_stake,  # 记录减仓金额
                        max_drawdown=max_loss_pct,
                        market_condition=market_condition,
                        position_metrics=position_metrics,
                        market_changes={},  # 部分平仓不需要市场变化分析
                        trade_score=None,  # 部分平仓可以添加评分逻辑
                        confidence_score=confidence_score,  # ✅ LLM 的置信度
                    )
                    logger.info(
                        f"✓ 部分平仓 {trade.id} 已记录到历史日志 (减仓 {exit_pct:.0f}%)"
                    )

                except Exception as e:
                    logger.error(f"记录部分平仓失败: {e}", exc_info=True)

            return adjustment_stake

        # 无调整
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        第2层：ATR动态追踪止损 + 时间衰减 + 趋势适应

        基于2024-2025加密货币ATR止损最佳实践优化：
        - 来源: Flipster, LuxAlgo, TrendSpider, Freqtrade Docs
        - 加密货币推荐止损距离: 8-15% (vs 股票3-5%)
        - 避免1×ATR内止损 (防止whipsaw假突破震出)

        策略逻辑（使用平滑过渡避免跳变）：
        - 盈利 ≤2%: 使用硬止损 (self.stoploss)
        - 盈利 2-6%: 追踪距离 = 2.0×ATR, 最小4% (保护初始盈利)
        - 盈利 6-15%: 追踪距离 = 2.0×ATR 平滑过渡, 最小5%
        - 盈利 >15%: 追踪距离 = 3.0×ATR, 最小8% (让利润奔跑)
        - 盈利 >80%: 追踪距离 = 4.0×ATR, 最小10% (极端放宽给Layer4)

        增强特性：
        - 时间衰减: 持仓>4小时未达6%利润,收紧止损15%
        - 趋势适应: ADX>25时,放宽追踪距离25%

        返回值：
        - 相对于当前价格的止损百分比（负数），如 -0.05 表示当前价格下方5%
        - None 表示使用硬止损 (self.stoploss)

        重要说明：
        - Freqtrade 自动确保返回值不会比 self.stoploss 更宽松（硬止损作为绝对底线）
        - 使用 StoplossCalculator.calculate_stoploss_price 计算绝对止损价格
        - 使用 stoploss_from_absolute 转换为 Freqtrade 要求的格式（相对于当前价格）
        - 不需要手动与 self.stoploss 比较，Freqtrade 引擎会自动执行此检查
        """
        from datetime import timedelta

        # 获取当前市场数据
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                logger.warning(f"[第2层-ATR止损] {pair} dataframe为空，使用硬止损")
                return None

            latest = dataframe.iloc[-1]
            atr = latest.get("atr", 0)
            adx = latest.get("adx", 0)

            # 计算ATR百分比并应用合理边界
            # 防止极端ATR值导致不合理的止损设置
            # 从配置中获取ATR上限，默认10%
            # 🔧 修复: 使用实时价格current_rate而非过时的K线收盘价，确保价格一致性
            stoploss_config = self.config.get("custom_stoploss_config", {})
            MIN_ATR_PCT = 0.001  # 0.1% 最小ATR
            MAX_ATR_PCT = stoploss_config.get("max_atr_pct", 0.10)  # 可配置的ATR上限
            DEFAULT_ATR_PCT = 0.01  # 1% 默认值

            if current_rate > 0 and atr > 0:
                atr_pct = (
                    atr / current_rate
                )  # ✓ 使用实时价格，与后续stoploss_from_absolute()计算一致
                # 应用边界限制
                atr_pct = max(MIN_ATR_PCT, min(atr_pct, MAX_ATR_PCT))

                if atr_pct == MAX_ATR_PCT:
                    logger.warning(
                        f"[第2层-ATR止损] {pair} ATR过大被限制: "
                        f"原始={atr / current_rate * 100:.2f}%, 限制为{MAX_ATR_PCT * 100:.0f}%"
                    )
            else:
                atr_pct = DEFAULT_ATR_PCT
                logger.debug(
                    f"[第2层-ATR止损] {pair} ATR数据无效，使用默认值 {DEFAULT_ATR_PCT * 100}%"
                )

        except Exception as e:
            logger.debug(f"[第2层-ATR止损] {pair} 获取数据失败: {e}, 使用硬止损")
            return None

        # 使用 StoplossCalculator 计算目标止损价格（绝对值）
        # 确保时间计算的时区安全性 - 统一转换为UTC
        from datetime import timezone

        # 确保 current_time 是 UTC 时区aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        # 确保 trade.open_date_utc 是 UTC 时区aware
        if trade.open_date_utc.tzinfo is None:
            trade_open = trade.open_date_utc.replace(tzinfo=timezone.utc)
        else:
            trade_open = trade.open_date_utc.astimezone(timezone.utc)

        hold_duration = current_time - trade_open

        # 加载自定义止损配置，并确保启用平滑过渡
        custom_stoploss_config = self.config.get("custom_stoploss_config", {}).copy()
        custom_stoploss_config["use_smooth_transition"] = True

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ✅ 重要设计：Layer 2 与 Layer 4 协同机制
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #
        # 在高盈利区间（>80%），Layer 2 (custom_stoploss) 故意大幅放宽止损（4.0×ATR，最小8%），
        # 目的是将退出决策交给 Layer 4 (custom_exit) 的趋势强度检查（基于 ADX + MACD）。
        #
        # 【工作原理】：
        # 1. Freqtrade 回调执行顺序：custom_exit → custom_stoploss → exchange stop
        # 2. 当利润 >80% 时，Layer 2 主动放宽止损距离（4.0×ATR vs 常规的 0.8-2.0×ATR）
        # 3. 这为 Layer 4 留出足够的"安全空间"，使其能够：
        #    - 检测趋势是否减弱（ADX < 20 或 MACD 柱状图转负）
        #    - 在趋势仍强时继续持仓，让利润奔跑
        #    - 在趋势减弱时智能退出，而非被机械的 ATR 止损过早打掉
        #
        # 【依赖关系】：
        # - 此机制依赖 Freqtrade 的 custom_exit 优先执行
        # - 不要"优化"这个放宽逻辑，这是有意设计！
        # - 删除或收紧此处止损将破坏 Layer 4 的趋势跟踪能力
        #
        # 【实际效果】：
        # - 在强趋势中，80%+ 利润可以继续增长而不被止损
        # - 在趋势减弱时，Layer 4 会主动退出保护利润
        # - 避免了"坐电梯"现象（利润大幅回撤后才触发 ATR 止损）
        #
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if current_profit >= 0.80:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 盈利>80%：极端高盈利区间特殊处理
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 【最佳实践】趋势市场应使用4.0×ATR，最小10-12%
            # 目的：给 Layer 4 (custom_exit) 的趋势强度检查留出执行空间
            # 让利润在强趋势中继续奔跑，避免过早被止损打出
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if atr_pct < 0.025:  # ATR < 2.5%（低波动环境）
                min_distance_for_high_profit = 0.10  # 固定10%
            else:
                min_distance_for_high_profit = max(
                    4.0 * atr_pct, 0.10
                )  # 4.0×ATR，最小10%

            # 临时覆盖配置中的最小距离，仅用于极端高盈利区间
            original_min_distances = custom_stoploss_config.get(
                "min_distances", [0.04, 0.05, 0.08]  # 更新默认值匹配新配置
            )
            custom_stoploss_config["min_distances"] = [
                original_min_distances[0],  # 2-6%区间保持不变
                original_min_distances[1],  # 6-15%区间保持不变
                min_distance_for_high_profit,  # >15%区间使用极端放宽值
            ]
            logger.debug(
                f"[第2层-ATR止损] {pair} 盈利{current_profit * 100:.1f}% > 80%，"
                f"放宽止损到{min_distance_for_high_profit * 100:.1f}%，让利润奔跑+给Layer4执行空间"
            )

        # 1. 计算目标止损价格（基于当前价格和ATR动态距离）
        target_stop_price = StoplossCalculator.calculate_stoploss_price(
            current_price=current_rate,
            current_profit=current_profit,
            atr_pct=atr_pct,
            adx=adx,
            hold_duration_hours=hold_duration.total_seconds() / 3600,
            is_short=trade.is_short,
            open_price=trade.open_rate,
            config=custom_stoploss_config,
        )

        if target_stop_price is None:
            return None

        # 2. 验证止损价格的方向性（防止计算错误）
        # 做多：止损价必须低于当前价（止损在下方）
        # 做空：止损价必须高于当前价（止损在上方）
        # 🔧 修复H1: 移除容差，使用直接方向检查

        if trade.is_short:
            if target_stop_price <= current_rate:
                logger.error(
                    f"[第2层-ATR止损] {pair} 做空止损价格错误: "
                    f"止损价 {target_stop_price:.4f} <= 当前价 {current_rate:.4f} "
                    f"(做空止损应该在当前价上方)"
                )
                return None
        else:  # 做多
            if target_stop_price >= current_rate:
                logger.error(
                    f"[第2层-ATR止损] {pair} 做多止损价格错误: "
                    f"止损价 {target_stop_price:.4f} >= 当前价 {current_rate:.4f} "
                    f"(做多止损应该在当前价下方)"
                )
                return None

        # 3. 🔧 修复M7: 防御性验证 current_rate 有效性
        if not current_rate or current_rate <= 0:
            logger.error(
                f"[第2层-ATR止损] {pair} current_rate 无效: {current_rate}，使用硬止损"
            )
            return None

        # 4. 检查止损价格与当前价的距离是否合理（防止极端值）
        price_distance_pct = abs(target_stop_price - current_rate) / current_rate
        MIN_STOP_DISTANCE = stoploss_config.get(
            "min_stop_distance", 0.0001
        )  # 0.01% 最小距离
        MAX_STOP_DISTANCE = stoploss_config.get(
            "max_stop_distance", 0.50
        )  # 50% 最大距离

        if price_distance_pct < MIN_STOP_DISTANCE:
            logger.warning(
                f"[第2层-ATR止损] {pair} 止损距离过小: {price_distance_pct * 100:.4f}% < {MIN_STOP_DISTANCE * 100}%，使用硬止损"
            )
            return None
        elif price_distance_pct > MAX_STOP_DISTANCE:
            logger.warning(
                f"[第2层-ATR止损] {pair} 止损距离过大: {price_distance_pct * 100:.2f}% > {MAX_STOP_DISTANCE * 100}%，使用硬止损"
            )
            return None

        # 🔧 修复H9: 验证杠杆值的有效性（防止除零或类型错误）
        leverage = getattr(trade, "leverage", 0.0)
        if not isinstance(leverage, (int, float)) or leverage <= 0:
            logger.error(f"[第2层-ATR止损] {pair} 无效的杠杆值: {leverage}，使用硬止损")
            return None

        # 5. 转换为 Freqtrade 要求的相对比例（使用官方helper函数）
        # stoploss_from_absolute 会自动处理做多/做空和杠杆的计算
        new_stoploss = stoploss_from_absolute(
            target_stop_price, current_rate, is_short=trade.is_short, leverage=leverage
        )

        # 🔧 修复C5: 完整的返回值验证（包括 NaN/Inf 检查）
        # 5a. 检查是否为 None
        if new_stoploss is None:
            logger.debug(f"[第2层-ATR止损] {pair} 止损计算返回 None，使用硬止损")
            return None

        # 5b. 检查是否为有限数字（排除 NaN 和 Inf）
        if not math.isfinite(new_stoploss):
            logger.error(
                f"[第2层-ATR止损] {pair} 止损值非有限数: {new_stoploss} "
                f"(可能由于极端市场条件或计算错误)，使用硬止损"
            )
            return None

        # 5c. 检查符号正确性（止损应该是负数）
        if new_stoploss >= 0:
            logger.debug(
                f"[第2层-ATR止损] {pair} 计算的止损值无效 ({new_stoploss})，使用硬止损"
            )
            return None

        logger.debug(
            f"[第2层-ATR止损] {pair} 动态追踪止损: {new_stoploss * 100:.2f}% "
            f"(当前盈利: {current_profit * 100:.2f}%, 目标价: {target_stop_price:.4f})"
        )

        # 记录止损更新（注意：这是计算止损，不是触发止损）
        # Freqtrade 引擎负责实际触发，此处只是返回计算结果
        if new_stoploss is not None:
            logger.debug(
                f"[第2层-ATR止损更新] {pair} | "
                f"止损价: {target_stop_price:.6f} | "
                f"当前盈利: {current_profit * 100:+.2f}% | "
                f"止损比例: {new_stoploss * 100:.2f}%"
            )

        # ✅ 新增：记录 Layer 2 退出元数据（供后续 LLM 分析使用）
        if new_stoploss is not None:
            # 确定盈利区间
            profit_thresholds = custom_stoploss_config.get(
                "profit_thresholds", [0.0, 0.02, 0.06, 0.15]
            )
            if current_profit < profit_thresholds[1]:
                profit_zone = "<2%"
            elif current_profit < profit_thresholds[2]:
                profit_zone = "2-6%"
            elif current_profit < profit_thresholds[3]:
                profit_zone = "6-15%"
            else:
                profit_zone = "15%+"

            # 获取ATR倍数
            atr_multipliers = custom_stoploss_config.get(
                "atr_multipliers", [2.0, 1.5, 1.0, 0.8]
            )
            atr_multiplier = StoplossCalculator._get_atr_multiplier(
                current_profit, custom_stoploss_config
            )

            self.exit_metadata_manager.record_exit(
                pair=pair,
                layer="layer2",
                trigger_profit=current_profit,
                profit_zone=profit_zone,
                atr_pct=atr_pct,
                atr_multiplier=atr_multiplier,
                adx_value=adx,
            )

        return new_stoploss
