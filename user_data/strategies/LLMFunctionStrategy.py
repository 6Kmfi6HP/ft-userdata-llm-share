"""
LLM Function Calling Strategy
基于LLM函数调用的智能交易策略

作者: Claude Code
版本: 1.0.0
"""

import logging
import math
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from freqtrade.strategy import IStrategy, informative, merge_informative_pair, stoploss_from_absolute
import talib.abstract as ta

# 导入自定义模块
from llm_modules.utils.config_loader import ConfigLoader
from llm_modules.utils.context_builder import ContextBuilder
from llm_modules.tools.trading_tools import TradingTools
from llm_modules.llm.llm_client import LLMClient
from llm_modules.llm.function_executor import FunctionExecutor
from llm_modules.experience.trade_logger import TradeLogger
from llm_modules.experience.experience_manager import ExperienceManager

# 导入新的指标计算器
from llm_modules.indicators.indicator_calculator import IndicatorCalculator

# 初始化 logger（必须在使用前定义）
logger = logging.getLogger(__name__)

# 历史上下文系统
from llm_modules.experience.trade_reviewer import TradeReviewer

# 增强模块导入
from llm_modules.utils.position_tracker import PositionTracker
from llm_modules.utils.market_comparator import MarketStateComparator
from llm_modules.utils.decision_checker import DecisionQualityChecker
from llm_modules.utils.stoploss_calculator import StoplossCalculator

# 自我学习系统导入
from llm_modules.learning.historical_query import HistoricalQueryEngine
from llm_modules.learning.pattern_analyzer import PatternAnalyzer
from llm_modules.learning.self_reflection import SelfReflectionEngine
from llm_modules.learning.trade_evaluator import TradeEvaluator
from llm_modules.learning.reward_learning import RewardLearningSystem


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
    timeframe = '30m'  # 30分钟K线，减少噪音，提高信号质量

    # 启动需要的历史数据
    startup_candle_count = 400  # 30分钟*400 = 约8.3天数据（确保4小时框架EMA50稳定）

    # 启用分层止损保护：硬止损 + 动态追踪止损 + LLM 决策
    stoploss = -0.10  # 10% 硬止损，防止爆仓（与config.json一致，期货10倍杠杆下价格空间1.0%）
    use_custom_stoploss = True  # 启用自定义动态追踪止损

    # 仓位调整
    position_adjustment_enable = True
    max_entry_position_adjustment = 10

    # 订单类型 - 全部使用市价单
    order_types = {
        'entry': 'market',
        'exit': 'market',
    }

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
            trade_log_path = self.experience_config.get('trade_log_path', './user_data/logs/trade_experience.jsonl')
            self.historical_query = HistoricalQueryEngine(trade_log_path)
            self.pattern_analyzer = PatternAnalyzer(min_sample_size=5)
            self.self_reflection = SelfReflectionEngine()
            self.trade_evaluator = TradeEvaluator()

            # 初始化奖励学习系统
            reward_config = {
                'storage_path': './user_data/logs/reward_learning.json',
                'learning_rate': 0.1,
                'discount_factor': 0.95
            }
            self.reward_learning = RewardLearningSystem(reward_config)

            logger.info("✓ 自我学习系统已初始化 (HistoricalQuery, PatternAnalyzer, SelfReflection, TradeEvaluator, RewardLearning)")

            # 3. 初始化上下文构建器（注入学习组件）
            # 🔧 修复M8+M9: 传入止损配置，避免 ContextBuilder 中硬编码
            self.context_builder = ContextBuilder(
                context_config=self.context_config,
                historical_query_engine=self.historical_query,
                pattern_analyzer=self.pattern_analyzer,
                tradable_balance_ratio=config.get('tradable_balance_ratio', 1.0),
                max_open_trades=config.get('max_open_trades', 1),
                stoploss_config=config.get('custom_stoploss_config', {}),
                hard_stoploss_pct=abs(self.stoploss) * 100  # 从策略的硬止损值转换为百分比
            )

            # 4. 初始化函数执行器
            self.function_executor = FunctionExecutor()

            # 5. 初始化交易工具（简化版 - 只保留交易控制工具）
            self.trading_tools = TradingTools(self)

            # 6. 初始化LLM客户端
            self.llm_client = LLMClient(self.llm_config, self.function_executor)

            # 8. 注册所有工具函数
            self._register_all_tools()

            # 9. 初始化经验系统（注入反思引擎）
            self.trade_logger = TradeLogger(self.experience_config)

            self.experience_manager = ExperienceManager(
                trade_logger=self.trade_logger,
                self_reflection_engine=self.self_reflection,
                trade_evaluator=self.trade_evaluator,
                reward_learning=self.reward_learning
            )

            # 10. 缓存
            self._leverage_cache = {}
            self._position_adjustment_cache = {}
            self._stake_request_cache = {}
            self._model_score_cache = {}  # 存储模型对交易的自我评分

            # 11. 初始化增强模块
            self.position_tracker = PositionTracker()
            self.market_comparator = MarketStateComparator()
            self.decision_checker = DecisionQualityChecker()
            self.trade_reviewer = TradeReviewer()
            logger.info("✓ 增强模块已初始化 (PositionTracker, MarketStateComparator, DecisionChecker, TradeReviewer)")

            # 12. 系统提示词（两套：开仓和持仓）
            self.entry_system_prompt = self.context_builder.build_entry_system_prompt()
            self.position_system_prompt = self.context_builder.build_position_system_prompt()
            logger.info("✓ 已加载两套系统提示词（开仓/持仓管理）")

            logger.info("✓ 策略初始化完成")
            logger.info(f"  - LLM模型: {self.llm_config.get('model')}")
            logger.info(f"  - 交易工具已注册: {len(self.function_executor.list_functions())} 个")
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

    def _register_all_tools(self):
        """注册所有工具函数（简化版 - 只注册交易控制工具）"""
        # 只注册交易工具（市场数据、账户信息已在context中提供）
        if self.trading_tools:
            self.function_executor.register_tools_from_instance(
                self.trading_tools,
                self.trading_tools.get_tools_schema()
            )
            logger.debug(f"已注册 {len(self.trading_tools.get_tools_schema())} 个交易控制函数")

    def _collect_multi_timeframe_history(self, pair: str) -> Dict[str, pd.DataFrame]:
        """根据ContextBuilder配置获取多时间框架K线数据"""
        if not getattr(self.context_builder, 'include_multi_timeframe_data', True):
            return {}

        if not hasattr(self, 'dp') or not self.dp:
            return {}

        if not hasattr(self.context_builder, 'get_multi_timeframe_history_config'):
            return {}

        tf_config = self.context_builder.get_multi_timeframe_history_config()
        if not tf_config:
            return {}

        history: Dict[str, pd.DataFrame] = {}

        for timeframe, cfg in tf_config.items():
            candles = cfg.get('candles', 0)
            fields = cfg.get('fields', [])
            tf_df = self._fetch_timeframe_dataframe(pair, timeframe, candles, fields)
            if tf_df is not None and not tf_df.empty:
                history[timeframe] = tf_df

        return history

    def _fetch_timeframe_dataframe(
        self,
        pair: str,
        timeframe: str,
        candles: int,
        fields: List[str]
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
        logger.info(f"✓ 交易工具: {len(self.function_executor.list_functions())} 个函数可用")

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
        **kwargs
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
                'atr': latest.get('atr', 0),
                'rsi': latest.get('rsi', 50),
                'ema_20': latest.get('ema_20', 0),
                'ema_50': latest.get('ema_50', 0),
                'macd': latest.get('macd', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'adx': latest.get('adx', 0)
            }

            # 暂存开仓信息（将在下一次 populate 中关联 trade_id）
            # 使用 pair+rate 作为临时key
            temp_key = f"{pair}_{rate}"
            self._pending_entry_states = getattr(self, '_pending_entry_states', {})
            self._pending_entry_states[temp_key] = {
                'pair': pair,
                'rate': rate,
                'indicators': indicators,
                'entry_tag': entry_tag or '',
                'side': side,
                'time': current_time
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
        **kwargs
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
                    'atr': latest.get('atr', 0),
                    'rsi': latest.get('rsi', 50),
                    'ema_20': latest.get('ema_20', 0),
                    'ema_50': latest.get('ema_50', 0),
                    'macd': latest.get('macd', 0),
                    'adx': latest.get('adx', 0)
                }
                market_changes = self.market_comparator.compare_with_entry(
                    trade_id=trade.id,
                    current_price=rate,
                    current_indicators=current_indicators
                )
            else:
                market_changes = {}

            # 手动计算盈亏百分比（因为此时 trade.close_profit 可能为 None）
            if trade.is_short:
                profit_pct = (trade.open_rate - rate) / trade.open_rate * trade.leverage * 100
            else:
                profit_pct = (rate - trade.open_rate) / trade.open_rate * trade.leverage * 100

            # 计算持仓时长（处理时区兼容性）
            # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
            from datetime import timezone
            if trade.open_date.tzinfo is None:
                # trade.open_date 是 naive，current_time 也应该是 naive
                exit_time = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
            else:
                # trade.open_date 是 aware，current_time 也应该是 aware
                exit_time = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)

            duration_minutes = int((exit_time - trade.open_date).total_seconds() / 60)

            # 生成交易复盘（如果 TradeReviewer 可用）
            if self.trade_reviewer:
                review = self.trade_reviewer.generate_trade_review(
                    pair=pair,
                    side='short' if trade.is_short else 'long',
                    entry_price=trade.open_rate,
                    exit_price=rate,
                    entry_reason=getattr(trade, 'enter_tag', '') or '',
                    exit_reason=exit_reason,
                    profit_pct=profit_pct,
                    duration_minutes=duration_minutes,
                    leverage=trade.leverage,
                    position_metrics=position_metrics,
                    market_changes=market_changes
                )

                # 输出复盘报告
                report = self.trade_reviewer.format_review_report(review)
                logger.info(f"\n{report}")

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
                max_loss_pct = position_metrics.get('max_loss_pct', 0) if position_metrics else 0
                max_profit_pct = position_metrics.get('max_profit_pct', 0) if position_metrics else 0

                # 获取模型评分
                model_score = self._model_score_cache.pop(pair, None)
                model_score_str = f"模型评分 {model_score:.0f}/100" if model_score else ""
                market_condition = f"MFE {max_profit_pct:+.2f}% / MAE {max_loss_pct:+.2f}% / 持仓 {duration_str} / {model_score_str}"

                self.experience_manager.log_trade_completion(
                    trade_id=trade.id,
                    pair=pair,
                    side='short' if trade.is_short else 'long',
                    entry_time=trade.open_date,
                    entry_price=trade.open_rate,
                    entry_reason=getattr(trade, 'enter_tag', '') or '未记录',
                    exit_time=exit_time,
                    exit_price=rate,
                    exit_reason=exit_reason,
                    profit_pct=profit_pct,
                    profit_abs=trade.stake_amount * profit_pct / 100,
                    leverage=trade.leverage,
                    stake_amount=trade.stake_amount,
                    max_drawdown=max_loss_pct,
                    market_condition=market_condition,
                    position_metrics=position_metrics,  # 【新增】传递持仓指标
                    market_changes=market_changes      # 【新增】传递市场变化
                )
                logger.info(f"✓ 交易 {trade.id} 已记录到历史日志")

            # 清理追踪数据
            if trade.id in self.position_tracker.positions:
                del self.position_tracker.positions[trade.id]
            if trade.id in self.market_comparator.entry_states:
                del self.market_comparator.entry_states[trade.id]

        except Exception as e:
            logger.error(f"生成交易复盘失败: {e}", exc_info=True)

        return True

    # 多时间框架数据支持
    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """1小时数据指标 - 使用统一的 IndicatorCalculator"""
        return IndicatorCalculator.add_all_indicators(dataframe)

    @informative('4h')
    def populate_indicators_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """4小时数据指标 - 使用统一的 IndicatorCalculator"""
        return IndicatorCalculator.add_all_indicators(dataframe)

    @informative('1d')
    def populate_indicators_1d(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """日线数据指标（注意：8天数据只有8根日线K线，EMA50勉强可用，已删除EMA200）"""
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)  # 需要200天数据，删除
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算技术指标（30分钟基础数据）- 使用统一的 IndicatorCalculator
        """
        return IndicatorCalculator.add_all_indicators(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        开仓信号 - 由LLM决策
        """
        pair = metadata['pair']

        # 默认不开仓
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        # 只在最新的K线上做决策
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # 获取当前所有持仓（用于传给context_builder）
            from freqtrade.persistence import Trade
            current_trades = Trade.get_open_trades()

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, 'dp') and self.dp:
                if hasattr(self.dp, '_exchange'):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, 'exchange'):
                    exchange = self.dp.exchange

            multi_tf_history = (
                self._collect_multi_timeframe_history(pair)
                if getattr(self.context_builder, 'include_multi_timeframe_data', True)
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
                multi_timeframe_data=multi_tf_history
            )

            # 构建决策请求
            decision_request = self.context_builder.build_decision_request(
                action_type="entry",
                market_context=market_context,
                position_context=""  # 已包含在market_context中
            )

            # 调用LLM决策（使用开仓提示词）
            messages = [
                {"role": "system", "content": self._get_system_prompt(has_position=False)},
                {"role": "user", "content": decision_request}
            ]

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10  # 限制迭代次数，防止无限循环
            )

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
                    function_calls=function_calls
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

                    if stake_amount and stake_amount > 0:
                        self._stake_request_cache[pair] = stake_amount

                    if action == "enter_long":
                        dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = reason
                        logger.info(f"📈 {pair} | 做多 | 置信度: {confidence_score}")
                        logger.info(f"   支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "enter_short":
                        dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = reason
                        logger.info(f"📉 {pair} | 做空 | 置信度: {confidence_score}")
                        logger.info(f"   支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "hold":
                        logger.info(f"🔒 {pair} | 保持持仓 | 置信度: {confidence_score} | RSI: {rsi_value}")
                        logger.info(f"   理由: {reason}")
                    elif action == "wait":
                        logger.info(f"⏸️  {pair} | 空仓等待 | 置信度: {confidence_score} | RSI: {rsi_value}")
                        logger.info(f"   理由: {reason}")
                else:
                    # 没有交易信号 = 观望，显示LLM的完整分析
                    logger.info(f"⏸️  {pair} | 未提供明确信号\n{llm_message}")

                # 🔧 修复C4: 清空当前交易对的信号缓存（避免竞态条件）
                self.trading_tools.clear_signal_for_pair(pair)

        except Exception as e:
            logger.error(f"开仓决策失败 {pair}: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        平仓信号 - 由LLM决策
        """
        pair = metadata['pair']

        # 默认不平仓
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''

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

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, 'dp') and self.dp:
                if hasattr(self.dp, '_exchange'):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, 'exchange'):
                    exchange = self.dp.exchange

            multi_tf_history = (
                self._collect_multi_timeframe_history(pair)
                if getattr(self.context_builder, 'include_multi_timeframe_data', True)
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
                multi_timeframe_data=multi_tf_history
            )

            # 更新 PositionTracker 和关联 MarketComparator
            pair_trades = [t for t in current_trades if t.pair == pair]

            # 检查dataframe是否为空
            if dataframe.empty:
                logger.warning(f"{pair} dataframe为空，跳过持仓追踪更新")
                return dataframe

            current_price = dataframe.iloc[-1]['close']

            for trade in pair_trades:
                try:
                    # 更新持仓追踪数据
                    self.position_tracker.update_position(
                        trade_id=trade.id,
                        pair=pair,
                        current_price=current_price,
                        open_price=trade.open_rate,
                        is_short=trade.is_short,
                        leverage=trade.leverage,
                        decision_type='check',  # 正在检查是否平仓
                        decision_reason=''  # 稍后在决策后更新
                    )

                    # 关联待定的开仓状态（如果存在）
                    temp_key = f"{pair}_{trade.open_rate}"
                    if hasattr(self, '_pending_entry_states') and temp_key in self._pending_entry_states:
                        pending = self._pending_entry_states[temp_key]
                        # 保存到 MarketComparator
                        self.market_comparator.save_entry_state(
                            trade_id=trade.id,
                            pair=pair,
                            price=trade.open_rate,
                            indicators=pending['indicators'],
                            entry_reason=pending['entry_tag'],
                            trend_alignment='',
                            market_sentiment=''
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
                position_context=""  # 已包含在market_context中
            )

            # 调用LLM决策（使用持仓管理提示词）
            messages = [
                {"role": "system", "content": self._get_system_prompt(has_position=True)},
                {"role": "user", "content": decision_request}
            ]

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10  # 限制迭代次数，防止无限循环
            )

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

                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
                    dataframe.loc[dataframe.index[-1], 'exit_tag'] = reason
                    logger.info(f"🔚 {pair} | 平仓 | 置信度: {confidence_score} | 自我评分: {trade_score}/100")
                    logger.info(f"   RSI: {rsi_value}")
                    logger.info(f"   理由: {reason}")

                    # 【立即生成交易复盘】- 在平仓信号发出时
                    if pair_trades and self.trade_reviewer:
                        try:
                            trade = pair_trades[0]

                            # 获取持仓追踪数据
                            position_metrics = self.position_tracker.get_position_metrics(trade.id)

                            # 获取市场状态变化
                            latest = dataframe.iloc[-1]
                            current_indicators = {
                                'atr': latest.get('atr', 0),
                                'rsi': latest.get('rsi', 50),
                                'ema_20': latest.get('ema_20', 0),
                                'ema_50': latest.get('ema_50', 0),
                                'macd': latest.get('macd', 0),
                                'adx': latest.get('adx', 0)
                            }
                            market_changes = self.market_comparator.compare_with_entry(
                                trade_id=trade.id,
                                current_price=current_price,
                                current_indicators=current_indicators
                            )

                            # 计算持仓时长（分钟）
                            from datetime import datetime, timezone
                            now = datetime.utcnow() if trade.open_date.tzinfo is None else datetime.now(timezone.utc)
                            duration_minutes = int((now - trade.open_date).total_seconds() / 60)

                            # 计算预期平仓盈亏（使用当前市价）
                            exit_price = current_price
                            if trade.is_short:
                                profit_pct = (trade.open_rate - exit_price) / trade.open_rate * trade.leverage * 100
                            else:
                                profit_pct = (exit_price - trade.open_rate) / trade.open_rate * trade.leverage * 100

                            # 生成交易复盘
                            review = self.trade_reviewer.generate_trade_review(
                                pair=pair,
                                side='short' if trade.is_short else 'long',
                                entry_price=trade.open_rate,
                                exit_price=exit_price,
                                entry_reason=getattr(trade, 'enter_tag', '') or '',
                                exit_reason=reason,
                                profit_pct=profit_pct,
                                duration_minutes=duration_minutes,
                                leverage=trade.leverage,
                                position_metrics=position_metrics,
                                market_changes=market_changes
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
                            profit_pct = (trade.open_rate - current_price) / trade.open_rate * trade.leverage * 100
                        else:
                            profit_pct = (current_price - trade.open_rate) / trade.open_rate * trade.leverage * 100

                        # 记录决策
                        decision_type = 'exit' if action == 'exit' else 'hold'
                        try:
                            quality_check = self.decision_checker.record_decision(
                                pair=pair,
                                decision_type=decision_type,
                                reason=reason,
                                profit_pct=profit_pct
                            )

                            # 如果有警告，记录到日志（不阻止交易）
                            if quality_check.get('warnings'):
                                for warning in quality_check['warnings']:
                                    if warning.get('level') == 'high':
                                        logger.warning(f"[决策质量警告] {warning.get('message')}")
                                        if warning.get('suggestion'):
                                            logger.warning(f"  建议: {warning.get('suggestion')}")

                        except Exception as e:
                            logger.debug(f"决策质量检查失败: {e}")

                # 🔧 修复C4: 清空当前交易对的信号缓存（避免竞态条件）
                self.trading_tools.clear_signal_for_pair(pair)

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
        **kwargs
    ) -> Optional[str]:
        """
        第4层止盈: 极端情况保护 (最小化干预LLM决策)
        
        【重要】只在极端情况下触发，绝大多数情况交给LLM决策：
        - ROI > 50% + 极度超买/超卖 = 保护暴利
        - ROI > 70% = 无条件强制保护
        
        杠杆处理：
        - 默认自动按杠杆调整阈值（因为 current_profit 包含杠杆效应）
        - 例如：10x杠杆下，5%价格波动 = 50% ROI
        - 配置中的阈值会乘以杠杆倍数以匹配实际价格波动
        - 可通过 custom_exit_config.use_leverage_adjusted_thresholds=False 禁用
        
        Returns:
            止盈理由字符串,或None(交给LLM决策)
        """
        try:
            # 获取技术指标
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return None
            
            latest = dataframe.iloc[-1]
            rsi = latest.get('rsi', 50)
            
            # 获取配置
            exit_config = self.config.get('custom_exit_config', {})
            # 阈值直接表示ROI百分比 (current_profit已包含杠杆效应)
            extreme_profit_threshold = exit_config.get('extreme_profit_threshold', 0.50)
            exceptional_profit_threshold = exit_config.get('exceptional_profit_threshold', 0.70)
            rsi_threshold = exit_config.get('extreme_rsi_threshold', 90)
            rsi_lower_bound = 100 - rsi_threshold
            
            # 浮点数比较容差（防止精度问题导致意外触发）
            PROFIT_EPSILON = exit_config.get('profit_epsilon', 0.001)  # 可配置的容差，默认0.1%
            
            # 情况1: 超高利润(ROI>50%) + 极端超买/超卖 = 极端止盈保护
            # 🔧 修复H2+H4: 使用减法提前触发保护 + 使用 >= 确保边界一致性
            if current_profit >= (extreme_profit_threshold - PROFIT_EPSILON):
                # 做空交易：RSI<10（超卖）时保护
                if trade.is_short and rsi < rsi_lower_bound:
                    logger.warning(
                        f"[第4层-极端止盈] {pair} 做空 | "
                        f"ROI {current_profit*100:.2f}% > {extreme_profit_threshold*100:.0f}% "
                        f"+ RSI {rsi:.1f} < {rsi_lower_bound} - 强制保护"
                    )
                    return "extreme_profit_protection_short"
                # 做多交易：RSI>90（超买）时保护
                elif not trade.is_short and rsi > rsi_threshold:
                    logger.warning(
                        f"[第4层-极端止盈] {pair} 做多 | "
                        f"ROI {current_profit*100:.2f}% > {extreme_profit_threshold*100:.0f}% "
                        f"+ RSI {rsi:.1f} > {rsi_threshold} - 强制保护"
                    )
                    return "extreme_profit_protection_long"
            
            # 情况2: 暴利(ROI>70%) = 无条件保护（已经是优秀交易）
            # 🔧 修复H2+H4: 使用减法提前触发保护 + 使用 >= 确保边界一致性
            if current_profit >= (exceptional_profit_threshold - PROFIT_EPSILON):
                logger.warning(
                    f"[第4层-暴利保护] {pair} {'做空' if trade.is_short else '做多'} | "
                    f"ROI {current_profit*100:.2f}% > {exceptional_profit_threshold*100:.0f}% "
                    f"- 已达暴利水平，强制保护"
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
        **kwargs
    ) -> float:
        """
        动态杠杆 - 由LLM决定或使用缓存值
        """
        # 🔧 修复H12: 确保缓存已初始化（防御性编程）
        if not hasattr(self, '_leverage_cache'):
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
        safe_max_leverage = 1.0 / (abs(self.stoploss) * 1.1) # 留10%安全缓冲
        
        if max_leverage > safe_max_leverage:
            logger.warning(f"{pair} 配置的最大杠杆 {max_leverage}x 风险过高(止损{self.stoploss})，已限制为安全值 {safe_max_leverage:.1f}x")
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
        **kwargs
    ) -> float:
        """
        动态仓位大小 - 可由LLM调整
        """
        stake_request = None
        if hasattr(self, '_stake_request_cache'):
            stake_request = self._stake_request_cache.pop(pair, None)

        if stake_request is None:
            return proposed_stake

        desired = stake_request

        # 只检查最小值，不限制最大值（由tradable_balance_ratio自然限制）
        if min_stake and desired < min_stake:
            logger.warning(f"{pair} 指定投入 {stake_request:.2f} USDT 低于最小要求 {min_stake:.2f}，已调整为最小值")
            desired = min_stake

        logger.info(f"{pair} 使用LLM指定仓位: {desired:.2f} USDT (请求 {stake_request:.2f})")
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
        **kwargs
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
                logger.warning(f"{pair} 加仓金额 {adjustment_stake} 低于最小stake {min_stake}")
                return None

            logger.info(f"{pair} 加仓 {adjustment_pct:.1f}% = {adjustment_stake:.2f} USDT | {reason}")
            return adjustment_stake

        elif adjustment_pct < 0:
            # 减仓
            # 🔧 修复M5: 验证减仓后剩余仓位是否满足最小stake要求
            remaining_stake = current_stake + adjustment_stake  # adjustment_stake 是负数

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
        **kwargs
    ) -> Optional[float]:
        """
        第2层：ATR动态追踪止损 + 时间衰减 + 趋势适应
        
        策略逻辑（使用平滑过渡避免跳变）：
        - 盈利 ≤2%: 使用硬止损 -6% (self.stoploss)
        - 盈利 2-6%: 追踪距离 = 1.5×ATR%, 最小1.5%
        - 盈利 6-15%: 追踪距离 = 1.5x平滑过渡到1.0×ATR (线性插值)
        - 盈利 >15%: 追踪距离 = 0.8×ATR%, 最小0.5%
        
        增强特性：
        - 时间衰减: 持仓>2小时未达6%利润,收紧止损20%
        - 趋势适应: ADX>25时,放宽追踪距离20%
        
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

        # ✅ P0级新增：杠杆自适应硬止损验证
        leverage = trade.leverage

        # 计算杠杆自适应的安全止损阈值
        # 公式：ROI止损% = 价格止损% × 杠杆
        # 例如：10x杠杆下，-10% ROI 对应 -1% 价格波动
        leverage_adjusted_hard_stop_price_pct = self.stoploss / leverage  # -0.10 / 10 = -0.01 (-1%价格)

        logger.debug(
            f"[第2层-ATR止损] {pair} {leverage}x杠杆 | "
            f"ROI硬止损={self.stoploss*100:.1f}% | "
            f"对应价格止损={leverage_adjusted_hard_stop_price_pct*100:.2f}%"
        )

        # 获取当前市场数据
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                logger.warning(f"[第2层-ATR止损] {pair} dataframe为空，使用硬止损")
                return None
            
            latest = dataframe.iloc[-1]
            atr = latest.get('atr', 0)
            adx = latest.get('adx', 0)
            
            # 计算ATR百分比并应用合理边界
            # 防止极端ATR值导致不合理的止损设置
            # 从配置中获取ATR上限，默认10%
            # 🔧 修复: 使用实时价格current_rate而非过时的K线收盘价，确保价格一致性
            stoploss_config = self.config.get('custom_stoploss_config', {})
            MIN_ATR_PCT = 0.001  # 0.1% 最小ATR
            MAX_ATR_PCT = stoploss_config.get('max_atr_pct', 0.10)  # 可配置的ATR上限
            DEFAULT_ATR_PCT = 0.01  # 1% 默认值
            
            if current_rate > 0 and atr > 0:
                atr_pct = atr / current_rate  # ✓ 使用实时价格，与后续stoploss_from_absolute()计算一致
                # 应用边界限制
                atr_pct = max(MIN_ATR_PCT, min(atr_pct, MAX_ATR_PCT))
                
                if atr_pct == MAX_ATR_PCT:
                    logger.warning(
                        f"[第2层-ATR止损] {pair} ATR过大被限制: "
                        f"原始={atr/current_rate*100:.2f}%, 限制为{MAX_ATR_PCT*100:.0f}%"
                    )
            else:
                atr_pct = DEFAULT_ATR_PCT
                logger.debug(f"[第2层-ATR止损] {pair} ATR数据无效，使用默认值 {DEFAULT_ATR_PCT*100}%")
            
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
        custom_stoploss_config = self.config.get('custom_stoploss_config', {}).copy()
        custom_stoploss_config['use_smooth_transition'] = True

        # 1. 计算目标止损价格（基于当前价格和ATR动态距离）
        target_stop_price = StoplossCalculator.calculate_stoploss_price(
            current_price=current_rate,
            current_profit=current_profit,
            atr_pct=atr_pct,
            adx=adx,
            hold_duration_hours=hold_duration.total_seconds() / 3600,
            is_short=trade.is_short,
            open_price=trade.open_rate,
            config=custom_stoploss_config
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
        MIN_STOP_DISTANCE = stoploss_config.get('min_stop_distance', 0.0001)  # 0.01% 最小距离
        MAX_STOP_DISTANCE = stoploss_config.get('max_stop_distance', 0.50)    # 50% 最大距离
        
        if price_distance_pct < MIN_STOP_DISTANCE:
            logger.warning(
                f"[第2层-ATR止损] {pair} 止损距离过小: {price_distance_pct*100:.4f}% < {MIN_STOP_DISTANCE*100}%，使用硬止损"
            )
            return None
        elif price_distance_pct > MAX_STOP_DISTANCE:
            logger.warning(
                f"[第2层-ATR止损] {pair} 止损距离过大: {price_distance_pct*100:.2f}% > {MAX_STOP_DISTANCE*100}%，使用硬止损"
            )
            return None

        # 🔧 修复H9: 验证杠杆值的有效性（防止除零或类型错误）
        leverage = getattr(trade, 'leverage', 0.0)
        if not isinstance(leverage, (int, float)) or leverage <= 0:
            logger.error(
                f"[第2层-ATR止损] {pair} 无效的杠杆值: {leverage}，使用硬止损"
            )
            return None

        # 5. 转换为 Freqtrade 要求的相对比例（使用官方helper函数）
        # stoploss_from_absolute 会自动处理做多/做空和杠杆的计算
        new_stoploss = stoploss_from_absolute(
            target_stop_price,
            current_rate,
            is_short=trade.is_short,
            leverage=leverage
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
            f"[第2层-ATR止损] {pair} 动态追踪止损: {new_stoploss*100:.2f}% "
            f"(当前盈利: {current_profit*100:.2f}%, 目标价: {target_stop_price:.4f})"
        )

        # 🔧 修复H5: 添加止损触发的 INFO 级别日志，提高可观测性
        if new_stoploss is not None:
            logger.info(
                f"[第2层-ATR止损触发] {pair} | "
                f"止损价: {target_stop_price:.6f} | "
                f"当前盈利: {current_profit*100:+.2f}% | "
                f"止损比例: {new_stoploss*100:.2f}%"
            )

        return new_stoploss
