"""
上下文构建器模块
负责构建LLM决策所需的市场上下文（重构版：使用模块化组件）
"""
import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
from .market_sentiment import MarketSentiment

# 导入新的模块化组件
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from context.data_formatter import DataFormatter
from context.prompt_builder import PromptBuilder
# 🔧 修复M6: 导入浮点比较容差常量
from .stoploss_calculator import StoplossCalculator, PROFIT_EPSILON

logger = logging.getLogger(__name__)


class ContextBuilder:
    """LLM上下文构建器（门面类，协调各个模块）"""

    def __init__(
        self,
        context_config: Dict[str, Any],
        historical_query_engine=None,
        pattern_analyzer=None,
        tradable_balance_ratio=1.0,
        max_open_trades=1,
        stoploss_config: Optional[Dict[str, Any]] = None,
        hard_stoploss_pct: Optional[float] = None,
        kelly_calculator=None,
        portfolio_risk_manager=None
    ):
        """
        初始化上下文构建器

        Args:
            context_config: 上下文配置
            historical_query_engine: 历史查询引擎实例（可选）
            pattern_analyzer: 模式分析器实例（可选）
            tradable_balance_ratio: 可交易余额比例
            max_open_trades: 最大开仓数
            stoploss_config: 止损配置（🔧 修复M8: 从配置读取利润阈值）
            hard_stoploss_pct: 硬止损百分比（🔧 修复M9: 从策略读取硬止损值）
            kelly_calculator: Kelly公式仓位计算器（可选）
            portfolio_risk_manager: 组合风险管理器（可选）
        """
        self.config = context_config
        # 学术论文整合: Kelly公式和组合风险管理
        self.kelly_calculator = kelly_calculator
        self.portfolio_risk_manager = portfolio_risk_manager

        # 🔧 修复M8+M9: 存储止损相关配置
        self.stoploss_config = stoploss_config or {}
        self.profit_threshold_1 = self.stoploss_config.get('profit_thresholds', [0.02, 0.06, 0.15])[0]
        self.hard_stoploss_pct = hard_stoploss_pct if hard_stoploss_pct is not None else 6.0
        self.max_tokens = context_config.get("max_context_tokens", 6000)
        self.sentiment = MarketSentiment()  # 初始化市场情绪获取器
        self.tradable_balance_ratio = tradable_balance_ratio
        self.max_open_trades = max_open_trades

        # 学习系统组件
        self.historical_query = historical_query_engine
        self.pattern_analyzer = pattern_analyzer
        self.enable_learning = historical_query_engine is not None

        # 先初始化新的模块化组件（在使用它们之前）
        self.formatter = DataFormatter()
        self.prompt_builder = PromptBuilder(
            include_timeframe_guidance=context_config.get("include_timeframe_guidance", True)
        )

        self.include_timeframe_guidance = context_config.get(
            "include_timeframe_guidance",
            True
        )
        self.raw_kline_history_points = max(0, context_config.get("raw_kline_history_points", 0))
        self.raw_kline_max_rows = max(
            1,
            context_config.get(
                "raw_kline_max_rows",
                self.raw_kline_history_points or 1
            )
        )
        self.raw_kline_extra_fields = self._ensure_list(
            context_config.get("raw_kline_extra_fields", [])
        )
        self.raw_kline_compact = context_config.get("raw_kline_compact_format", True)
        self.raw_kline_stride = max(1, context_config.get("raw_kline_stride", 1))
        self.indicator_history_points = max(1, context_config.get("indicator_history_points", 20))
        self.indicator_history_lookback = max(
            self.indicator_history_points,
            context_config.get("indicator_history_lookback", 100)
        )
        self.include_multi_timeframe_data = context_config.get("include_multi_timeframe_data", True)
        # 现在可以安全地调用 formatter 方法了
        self.multi_timeframe_history = self._normalize_multi_timeframe_config(
            context_config.get("multi_timeframe_history", {})
        ) if self.include_multi_timeframe_data else {}
        self.multi_timeframe_compact = context_config.get(
            "multi_timeframe_compact_format",
            self.raw_kline_compact
        )
        self.multi_timeframe_max_rows = max(
            1,
            context_config.get("multi_timeframe_max_rows", 120)
        )

    def build_market_context(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
        wallets: Any = None,
        current_trades: Optional[List[Any]] = None,
        exchange: Any = None,
        position_tracker: Any = None,
        market_comparator: Any = None,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> str:
        """
        构建完整的市场上下文（一次性提供所有数据）

        Args:
            dataframe: OHLCV数据和所有技术指标
            metadata: 交易对元数据
            wallets: 钱包对象（用于获取账户余额）
            current_trades: 当前所有持仓列表
            exchange: 交易所对象（用于获取资金费率）
            position_tracker: PositionTracker实例，提供持仓表现
            market_comparator: MarketStateComparator实例，用于对比
            multi_timeframe_data: 其他时间框架的K线与指标数据

        Returns:
            格式化的完整上下文字符串
        """
        pair = metadata.get('pair', 'UNKNOWN')

        # 获取最新数据
        if dataframe.empty:
            return f"市场数据: {pair} - 无数据"

        latest = dataframe.iloc[-1]
        prev = dataframe.iloc[-2] if len(dataframe) > 1 else latest

        # 获取K线时间并格式化（确保显示完整的时区信息）
        candle_time = latest['date'] if 'date' in latest else datetime.now(timezone.utc)
        if hasattr(candle_time, 'strftime'):
            candle_time_str = candle_time.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            candle_time_str = str(candle_time)
        
        context_parts = [
            "<market_data>",
            f"## 交易对: {pair}",
            f"最新完成K线时间: {candle_time_str}",
            f"当前UTC时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "### 价格信息",
            f"  当前价格: {latest['close']:.8f}",
            f"  开盘: {latest['open']:.8f}  最高: {latest['high']:.8f}  最低: {latest['low']:.8f}",
            f"  成交量: {latest['volume']:.2f}",
            f"  价格变化: {((latest['close'] - prev['close']) / prev['close'] * 100):.2f}%",
        ]

        # EMA200距离计算 - 用于均值回归策略判断
        if 'ema_200' in latest and 'atr' in latest and pd.notna(latest['ema_200']) and pd.notna(latest['atr']) and latest['atr'] > 0:
            distance_to_ema200 = (latest['close'] - latest['ema_200']) / latest['atr']
            position = "上方" if distance_to_ema200 > 0 else "下方"
            context_parts.append(f"  EMA200距离: {abs(distance_to_ema200):.2f} ATR ({position})")

        # 暂存市场情绪数据，稍后在末尾添加
        sentiment_parts = []
        sentiment_parts.append("")
        sentiment_parts.append("<sentiment>")
        sentiment_parts.append("### 市场情绪")
        if exchange:
            try:
                sentiment_data = self.sentiment.get_combined_sentiment(exchange, pair)

                # Fear & Greed Index - 显示完整历史
                if sentiment_data.get('fear_greed'):
                    fg = sentiment_data['fear_greed']
                    sentiment_parts.append(f"  恐惧与贪婪指数: {fg['value']}/100 ({fg['classification']})")

                    # 显示30天完整历史（原始数据，不做处理）
                    if fg.get('history_30d'):
                        sentiment_parts.append("  ")
                        sentiment_parts.append("  历史30天（原始数据）：")
                        history = fg['history_30d']

                        # 策略：显示最近3天每日 + 每周关键点
                        # 最近3天
                        for i, h in enumerate(history[:3]):
                            if i == 0:
                                time_desc = "今天"
                            elif i == 1:
                                time_desc = "昨天"
                            else:
                                time_desc = f"{i}天前"
                            sentiment_parts.append(
                                f"    {h['date']} ({time_desc}): {h['value']} ({h['classification']})"
                            )

                        # 每周关键点：第7天、第14天、第21天、第30天
                        key_points = [7, 14, 21, 29]  # 索引从0开始，29=第30天
                        for idx in key_points:
                            if idx < len(history):
                                h = history[idx]
                                days_ago = idx
                                sentiment_parts.append(
                                    f"    {h['date']} ({days_ago}天前): {h['value']} ({h['classification']})"
                                )

                # Funding Rate
                if sentiment_data.get('funding_rate'):
                    fr = sentiment_data['funding_rate']
                    sentiment_parts.append("  ")
                    sentiment_parts.append(f"  资金费率: {fr['rate_pct']:.4f}% ({fr['interpretation']})")

                # Long/Short Ratio - 显示最近几天的趋势
                if sentiment_data.get('long_short'):
                    ls = sentiment_data['long_short']
                    sentiment_parts.append("  ")
                    sentiment_parts.append(f"  多空比: {ls['current_ratio']:.2f} (多{ls['long_pct']:.1f}% / 空{ls['short_pct']:.1f}%)")
                    sentiment_parts.append(f"    状态: {ls['extreme_level']} | 趋势: {ls['trend']}")

                    # 显示最近7天的多空比（每12小时一个点）
                    if ls.get('history_30d'):
                        history = ls['history_30d']
                        # 取最近7天（168小时）的数据，每12小时一个点 = 14个点
                        recent_7d = history[-168:]
                        sampled = [recent_7d[i] for i in range(0, len(recent_7d), 12)][-14:]

                        if sampled:
                            sentiment_parts.append("    最近7天多空比变化（每12小时）：")
                            for h in reversed(sampled):  # 从旧到新
                                time_str = datetime.fromtimestamp(h['timestamp'] / 1000).strftime('%m-%d %H:00')
                                sentiment_parts.append(
                                    f"      {time_str}: {h['ratio']:.2f} (多{h['long_pct']:.0f}%/空{h['short_pct']:.0f}%)"
                                )

            except Exception as e:
                logger.error(f"获取市场情绪失败: {e}")


        # 自动提取所有技术指标（排除基础列）
        excluded_cols = {'date', 'open', 'high', 'low', 'close', 'volume',
                        'enter_long', 'enter_short', 'enter_tag',
                        'exit_long', 'exit_short', 'exit_tag'}

        # 按时间框架分组
        indicators_30m = []  # 改为30分钟
        indicators_1h = []
        indicators_4h = []
        indicators_1d = []

        for col in latest.index:
            if col in excluded_cols:
                continue

            value = latest[col]
            if pd.isna(value):
                continue

            # 分类指标
            if '_1h' in col:
                indicators_1h.append((col, value))
            elif '_4h' in col:
                indicators_4h.append((col, value))
            elif '_1d' in col:
                indicators_1d.append((col, value))
            else:
                indicators_30m.append((col, value))

        # 按照从大到小的时间框架顺序呈现（高时间框架更重要）
        # 日线 > 4H > 1H > 30M

        # 输出日线指标 - 最高优先级，决定大势方向（不可逆转）
        if self.include_multi_timeframe_data and indicators_1d:
            context_parts.append("")
            context_parts.append("### 技术指标 - 日线（决定大势方向，不可逆转）")
            context_parts.append("  ⚠️ 日线趋势是大资金意志，顺势而为是唯一正确选择")
            for ind, val in indicators_1d:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出4小时指标 - 确定方向和节奏
        if self.include_multi_timeframe_data and indicators_4h:
            context_parts.append("")
            context_parts.append("### 技术指标 - 4小时（确定方向和节奏）")
            for ind, val in indicators_4h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出1小时指标 - 确认中期趋势
        if self.include_multi_timeframe_data and indicators_1h:
            context_parts.append("")
            context_parts.append("### 技术指标 - 1小时（确认中期趋势）")
            for ind, val in indicators_1h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出30分钟指标 - 寻找入场时机
        if indicators_30m:
            context_parts.append("")
            context_parts.append("### 技术指标 - 30分钟（寻找入场时机）")
            for ind, val in indicators_30m:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 添加账户信息
        if wallets:
            context_parts.append("</market_data>")
            context_parts.append("")
            context_parts.append("<account>")
            context_parts.append("### 账户信息")
            try:
                total = wallets.get_total('USDT')
                free = wallets.get_free('USDT')
                used = wallets.get_used('USDT')

                # 计算实际可用交易余额（考虑tradable_balance_ratio和max_open_trades）
                tradable_total = total * self.tradable_balance_ratio
                tradable_free = tradable_total - used
                per_trade_avg = tradable_total / self.max_open_trades if self.max_open_trades > 0 else tradable_total

                context_parts.extend([
                    f"  总余额: {total:.2f} USDT",
                    f"  可交易余额: {tradable_total:.2f} USDT ({self.tradable_balance_ratio*100:.0f}%资金)",
                    f"  当前可用: {tradable_free:.2f} USDT",
                    f"  已用资金: {used:.2f} USDT",
                    f"  最多{self.max_open_trades}个仓位，平均每个约 {per_trade_avg:.2f} USDT"
                ])

                # === 学术论文整合: Kelly建议仓位 ===
                # 基于 Busseti et al. 2016 "Risk-Constrained Kelly Gambling"
                if self.kelly_calculator and self.historical_query:
                    try:
                        pair_summary = self.historical_query.get_pair_summary(pair, days=30)
                        # 获取当前回撤（如果有组合风险管理器）
                        current_dd = 0.0
                        if self.portfolio_risk_manager:
                            risk_status = self.portfolio_risk_manager.get_risk_status()
                            current_dd = risk_status.get("drawdown_pct", 0.0)

                        kelly_suggestion = self.kelly_calculator.get_kelly_suggestion(
                            pair_summary=pair_summary,
                            available_balance=tradable_free,
                            current_drawdown_pct=current_dd
                        )
                        kelly_text = self.kelly_calculator.format_for_context(kelly_suggestion)
                        if kelly_text:
                            context_parts.append("")
                            context_parts.append(kelly_text)
                    except Exception as e:
                        logger.debug(f"[Kelly] 计算建议失败: {e}")

            except Exception as e:
                context_parts.append(f"  无法获取账户信息: {e}")

        # === 学术论文整合: 组合风险警告 ===
        # 基于多篇高引用风险管理论文
        if self.portfolio_risk_manager:
            try:
                risk_text = self.portfolio_risk_manager.format_for_context()
                if risk_text:
                    context_parts.append("")
                    context_parts.append(risk_text)
            except Exception as e:
                logger.debug(f"[PortfolioRisk] 获取风险状态失败: {e}")

        # 添加持仓信息
        context_parts.append("</account>")
        context_parts.append("")
        context_parts.append("<positions>")
        context_parts.append("### 持仓情况")
        if not current_trades:
            context_parts.append("  当前无持仓")
        else:
            # 筛选当前交易对的持仓
            pair_trades = [t for t in current_trades if getattr(t, 'pair', '') == pair]

            if not pair_trades:
                context_parts.append(f"  {pair}: 无持仓")
            else:
                current_price = latest['close']
                for i, trade in enumerate(pair_trades, 1):
                    is_short = getattr(trade, 'is_short', False)
                    open_rate = getattr(trade, 'open_rate', 0)
                    stake = getattr(trade, 'stake_amount', 0)
                    leverage = getattr(trade, 'leverage', 1)
                    enter_tag = getattr(trade, 'enter_tag', '')
                    open_date = getattr(trade, 'open_date', None)

                    # 计算当前盈亏
                    if is_short:
                        profit_pct = (open_rate - current_price) / open_rate * leverage * 100
                    else:
                        profit_pct = (current_price - open_rate) / open_rate * leverage * 100

                    # 计算持仓时间（初始化默认值，防止未定义错误）
                    hours = 0  # 默认值
                    time_str = "未知"

                    if open_date:
                        if isinstance(open_date, datetime):
                            # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
                            now = datetime.utcnow() if open_date.tzinfo is None else datetime.now(timezone.utc)
                            holding_time = now - open_date
                            hours = holding_time.total_seconds() / 3600
                            if hours < 1:
                                time_str = f"{int(hours * 60)}分钟"
                            elif hours < 24:
                                time_str = f"{hours:.1f}小时"
                            else:
                                time_str = f"{hours / 24:.1f}天"
                        else:
                            time_str = "未知"
                    else:
                        time_str = "未知"

                    # 基本信息
                    context_parts.append(f"  持仓#{i}: {'做空' if is_short else '做多'} {leverage}x杠杆")
                    context_parts.append(f"    开仓价: {open_rate:.6f}")
                    context_parts.append(f"    当前价: {current_price:.6f}")
                    context_parts.append(f"    当前盈亏: {profit_pct:+.2f}% ({profit_pct * stake / 100:+.2f}U)")
                    context_parts.append(f"    持仓时间: {time_str}")
                    context_parts.append(f"    投入: {stake:.2f}U")
                    
                    # 添加动态止损位信息(第2层ATR追踪止损) - 使用统一的StoplossCalculator
                    try:
                        # 🔧 修复M6+M8: 使用Epsilon容差 + 从配置读取阈值（而非硬编码2%）
                        if (profit_pct / 100) > (self.profit_threshold_1 + PROFIT_EPSILON):
                            atr = latest.get('atr', 0)
                            adx = latest.get('adx', 0)
                            atr_pct = (atr / current_price) if current_price > 0 and atr > 0 else 0.01
                            
                            # 使用 StoplossCalculator 统一计算止损价格
                            stop_price = StoplossCalculator.calculate_stoploss_price(
                                current_price=current_price,
                                current_profit=profit_pct / 100,  # 转换为小数
                                atr_pct=atr_pct,
                                adx=adx,
                                hold_duration_hours=hours,
                                is_short=is_short,
                                open_price=open_rate,
                                config={'use_smooth_transition': True}
                            )
                            
                            if stop_price is not None:
                                # 计算止损距离百分比
                                if is_short:
                                    distance_pct = (stop_price - current_price) / current_price * 100
                                else:
                                    distance_pct = (current_price - stop_price) / current_price * 100
                                
                                # 判断利润区间
                                if profit_pct > 15.0:
                                    level = ">15%"
                                elif profit_pct > 6.0:
                                    level = "6-15%"
                                else:
                                    level = "2-6%"
                                
                                # 添加增强特性说明
                                enhancements = []
                                if hours > 2 and profit_pct < 6.0:
                                    enhancements.append("时间衰减-20%")
                                if adx > 25:
                                    enhancements.append(f"强趋势ADX={adx:.0f},+20%")
                                
                                enhancement_msg = f" ({', '.join(enhancements)})" if enhancements else ""
                                
                                context_parts.append(f"    动态止损: {stop_price:.6f} (距离{distance_pct:.2f}%{enhancement_msg})")
                                context_parts.append(f"      └─ 基于{level}利润区间 + ATR追踪 (平滑过渡)")
                            else:
                                # 🔧 修复M9: 从配置读取硬止损百分比（而非硬编码6.0）
                                # StoplossCalculator返回None，表示应使用硬止损
                                if is_short:
                                    stop_price = open_rate * (1 + self.hard_stoploss_pct / 100)
                                else:
                                    stop_price = open_rate * (1 - self.hard_stoploss_pct / 100)
                                context_parts.append(f"    硬止损: {stop_price:.6f} (-{self.hard_stoploss_pct:.1f}%)")
                                context_parts.append(f"      └─ 盈利≤{self.profit_threshold_1*100:.1f}%时使用交易所硬止损")
                        else:
                            # 🔧 修复M9: 从配置读取硬止损百分比（而非硬编码6.0）
                            # 使用硬止损
                            if is_short:
                                stop_price = open_rate * (1 + self.hard_stoploss_pct / 100)
                            else:
                                stop_price = open_rate * (1 - self.hard_stoploss_pct / 100)
                            context_parts.append(f"    硬止损: {stop_price:.6f} (-{self.hard_stoploss_pct:.1f}%)")
                            context_parts.append(f"      └─ 盈利≤{self.profit_threshold_1*100:.1f}%时使用交易所硬止损")
                    except Exception as e:
                        # 🔧 修复H6: 异常日志级别从 DEBUG 提升为 WARNING
                        logger.warning(f"[上下文构建] 计算止损位失败: {e}")

                    # 添加PositionTracker的追踪数据
                    if position_tracker:
                        try:
                            trade_id = getattr(trade, 'id', None)
                            if trade_id:
                                metrics = position_tracker.get_position_metrics(trade_id)
                                if metrics:
                                    context_parts.append("")
                                    context_parts.append("    #### 持仓追踪数据")
                                    context_parts.append(f"      最大浮盈(MFE): {metrics['max_profit_pct']:+.2f}%")
                                    context_parts.append(f"      最大浮亏(MAE): {metrics['max_loss_pct']:+.2f}%")
                                    if metrics['drawdown_from_peak_pct'] < -1:
                                        context_parts.append(f"      盈利回撤: {metrics['drawdown_from_peak_pct']:+.2f}% (从峰值{metrics['max_profit_pct']:+.2f}%)")
                                    context_parts.append(f"      hold次数: {metrics['hold_count']}次")

                                    # hold模式记录（不添加评价）
                                    hold_pattern = metrics.get('hold_pattern', {})
                                    if hold_pattern.get('pattern') == 'stuck_in_loop':
                                        context_parts.append(f"      连续{hold_pattern['repeat_count']}次使用相似理由hold")
                                        context_parts.append(f"      重复理由: \"{hold_pattern['repeated_reason']}\"")
                                    elif hold_pattern.get('pattern') == 'repeated_reasoning':
                                        context_parts.append(f"      理由重复度: {hold_pattern['repeat_count']}/{hold_pattern['total_holds']}")

                                    # 最近决策（完整显示，不截断）
                                    if metrics.get('recent_decisions'):
                                        context_parts.append("      最近3次决策:")
                                        for d in metrics['recent_decisions'][-3:]:
                                            time_str_short = d['time'].strftime("%H:%M")
                                            context_parts.append(f"        [{time_str_short}] {d['type']}: {d['reason']}")
                        except Exception as e:
                            pass  # 静默失败，不影响主流程

                    # 开仓理由（完整显示，不限制字符）
                    if enter_tag:
                        context_parts.append("")
                        context_parts.append("    开仓理由:")
                        # 分行显示，完整保留
                        for line in enter_tag.split('\n'):
                            if line.strip():
                                context_parts.append(f"      {line.strip()}")

            # 显示其他交易对的持仓
            other_trades = [t for t in current_trades if getattr(t, 'pair', '') != pair]
            if other_trades:
                context_parts.append(f"  其他交易对持仓数: {len(other_trades)}")

        # 添加市场状态对比（如果有持仓且提供了market_comparator）
        context_parts.append("</positions>")
        if current_trades and market_comparator and pair_trades:
            context_parts.append("")
            context_parts.append("<market_comparison>")
            for trade in pair_trades:
                trade_id = getattr(trade, 'id', None)
                if trade_id:
                    try:
                        # 获取当前指标
                        current_indicators = {
                            'atr': latest.get('atr', 0),
                            'rsi': latest.get('rsi', 50),
                            'ema_20': latest.get('ema_20', 0),
                            'ema_50': latest.get('ema_50', 0),
                            'macd': latest.get('macd', 0),
                            'macd_signal': latest.get('macd_signal', 0),
                            'adx': latest.get('adx', 0)
                        }

                        # 生成对比文本
                        comparison_text = market_comparator.generate_comparison_text(
                            trade_id=trade_id,
                            current_price=latest['close'],
                            current_indicators=current_indicators
                        )

                        context_parts.append(comparison_text)
                    except Exception as e:
                        pass  # 静默失败
            context_parts.append("</market_comparison>")

        # 添加关键指标历史序列（使用新的 DataFormatter）
        context_parts.append("")
        context_parts.append("<indicator_history>")
        context_parts.append(f"### 关键指标历史（最近{self.indicator_history_points}根K线）")
        indicator_history = self.formatter.get_indicator_history(
            dataframe,
            lookback=self.indicator_history_lookback,
            display_points=self.indicator_history_points
        )
        if indicator_history:
            for ind_name, values in indicator_history.items():
                if values and any(v is not None for v in values):
                    values_str = ", ".join([f"{v}" if v is not None else "N/A" for v in values])
                    context_parts.append(f"  {ind_name}: [{values_str}]")
        else:
            context_parts.append("  数据不足")

        # 提供原始K线历史（使用新的 DataFormatter）
        if self.raw_kline_history_points > 0:
            raw_history = self.formatter.get_raw_kline_history(
                dataframe,
                self.raw_kline_history_points,
                extra_fields=self.raw_kline_extra_fields,
                compact=self.raw_kline_compact,
                stride=self.raw_kline_stride,
                max_rows=self.raw_kline_max_rows
            )
            raw_rows = raw_history.get('rows', [])
            if raw_rows:
                context_parts.append("")
                header_note = []
                if raw_history.get('header'):
                    header_note.append(f"列: {raw_history['header']}")
                stride_used = raw_history.get('stride', 1)
                if stride_used > 1:
                    header_note.append(f"步长:{stride_used}")
                header_text = f"（{'，'.join(header_note)}）" if header_note else ""
                context_parts.append("</indicator_history>")
                context_parts.append("")
                context_parts.append("<kline_history>")
                context_parts.append(f"### K线历史（最近{len(raw_rows)}根）{header_text}")
                for entry in raw_rows:
                    context_parts.append(f"  {entry}")

        # 多时间框架原始数据
        if self.multi_timeframe_history and multi_timeframe_data:
            context_parts.append("</kline_history>")
            context_parts.append("")
            context_parts.append("<multi_timeframe>")
            context_parts.append("### 多时间框架K线数据")
            for tf, cfg in self.multi_timeframe_history.items():
                tf_df = multi_timeframe_data.get(tf)
                candles = cfg.get('candles', 0)
                if tf_df is None or candles <= 0:
                    continue

                compact_tf = cfg.get('compact', self.multi_timeframe_compact)
                if compact_tf is None:
                    compact_tf = self.multi_timeframe_compact

                stride_tf = cfg.get('stride', 1)
                try:
                    stride_tf = int(stride_tf)
                except (TypeError, ValueError):
                    stride_tf = 1
                stride_tf = max(1, stride_tf)

                tf_history = self.formatter.get_raw_kline_history(
                    tf_df,
                    candles,
                    extra_fields=cfg.get('fields', []),
                    compact=compact_tf,
                    stride=stride_tf,
                    max_rows=cfg.get('max_rows', self.multi_timeframe_max_rows)
                )
                tf_rows = tf_history.get('rows', [])
                if tf_rows:
                    header_note = []
                    if tf_history.get('header'):
                        header_note.append(f"列: {tf_history['header']}")
                    stride_used = tf_history.get('stride', 1)
                    if stride_used > 1:
                        header_note.append(f"步长:{stride_used}")
                    header_text = f"（{'，'.join(header_note)}）" if header_note else ""
                    context_parts.append(f"  [{tf}] 最近{len(tf_rows)}根K线{header_text}")
                    for entry in tf_rows:
                        context_parts.append(f"    {entry}")

        # 在最后添加市场情绪参考（弱化显示）
        sentiment_parts.append("</sentiment>")
        context_parts.append("</multi_timeframe>")
        context_parts.extend(sentiment_parts)

        # 添加历史经验和模式分析（自我学习系统）
        if self.enable_learning:
            try:
                context_parts.append("")
                # 获取最近交易
                recent_trades_text = self.historical_query.format_recent_trades_for_context(
                    pair=pair,
                    limit=10
                )
                context_parts.append(recent_trades_text)

                # 获取统计摘要
                context_parts.append("")
                summary_text = self.historical_query.format_pair_summary_for_context(
                    pair=pair,
                    days=30
                )
                context_parts.append(summary_text)

                # 获取模式分析
                if self.pattern_analyzer:
                    recent_trades = self.historical_query.query_recent_trades(pair=pair, limit=50)
                    if len(recent_trades) >= 5:
                        context_parts.append("")
                        patterns_text = self.pattern_analyzer.format_patterns_for_context(
                            pair=pair,
                            trades=recent_trades
                        )
                        context_parts.append(patterns_text)

            except Exception as e:
                logger.error(f"添加历史经验失败: {e}")

        return "\n".join(context_parts)

    def build_entry_system_prompt(self) -> str:
        """
        构建开仓决策专用系统提示词（使用新的 PromptBuilder）

        Returns:
            开仓决策系统提示词字符串
        """
        return self.prompt_builder.build_entry_prompt()

    def build_position_system_prompt(self) -> str:
        """
        构建持仓管理系统提示词（使用新的 PromptBuilder）

        Returns:
            持仓管理系统提示词字符串
        """
        return self.prompt_builder.build_position_prompt()

    def build_decision_request(
        self,
        action_type: str,
        market_context: str,
        position_context: str
    ) -> str:
        """
        构建决策请求

        Args:
            action_type: 决策类型 (entry/exit)
            market_context: 市场上下文（已包含历史经验）
            position_context: 持仓上下文

        Returns:
            完整的决策请求字符串
        """
        action_desc = {
            'entry': '是否应该开仓(做多或做空)',
            'exit': '是否应该平仓'
        }

        request_parts = [
            f"请分析当前情况，决策: {action_desc.get(action_type, action_type)}",
            "",
            "=" * 50,
            market_context,
            "",
            "=" * 50,
            position_context,
            "",
            "决策后调用一个函数，立即停止"
        ]

        return "\n".join(request_parts)

    def get_multi_timeframe_history_config(self) -> Dict[str, Dict[str, Any]]:
        """对外提供多时间框架配置，供策略层决定需要拉取的数据"""
        return self.multi_timeframe_history

    def _normalize_multi_timeframe_config(self, cfg: Any) -> Dict[str, Dict[str, Any]]:
        """将多时间框架配置标准化为 {tf: {candles:int, fields:list}}（委托给 DataFormatter）"""
        return self.formatter.normalize_multi_timeframe_config(cfg, self.config)

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
