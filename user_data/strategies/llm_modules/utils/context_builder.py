"""
上下文构建器模块
负责构建LLM决策所需的市场上下文（重构版：使用模块化组件）
"""
import logging
import math
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from .market_sentiment import MarketSentiment

# 导入新的模块化组件
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from context.data_formatter import DataFormatter
from context.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ContextBuilder:
    """LLM上下文构建器（门面类，协调各个模块）"""

    def __init__(self, context_config: Dict[str, Any], historical_query_engine=None, pattern_analyzer=None):
        """
        初始化上下文构建器

        Args:
            context_config: 上下文配置
            historical_query_engine: 历史查询引擎实例（可选）
            pattern_analyzer: 模式分析器实例（可选）
        """
        self.config = context_config
        self.max_tokens = context_config.get("max_context_tokens", 6000)
        self.sentiment = MarketSentiment()  # 初始化市场情绪获取器

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

    def _format_vision_analysis(self, result: Dict[str, Any]) -> str:
        """
        格式化视觉分析结果为可读文本
        
        Args:
            result: vision_tools.analyze_image_with_gemini 返回的结果
            
        Returns:
            格式化后的文本
        """
        if not result or not isinstance(result, dict):
            return ""
        
        # 检查是否是错误响应
        summary = result.get('summary', '')
        if summary.startswith('VISION_CALL_FAILED'):
            return f"⚠️ 视觉分析失败: {summary}"
        
        # 构建格式化文本
        lines = []
        
        # 1. 总结
        if summary:
            lines.append(f"📊 **分析总结**：{summary}")
            lines.append("")
        
        # 2. 趋势判断
        judgement = result.get('judgement', {})
        if judgement:
            direction_emoji = {
                'up': '📈 上涨',
                'down': '📉 下跌',
                'sideways': '➡️ 横盘'
            }
            direction = judgement.get('direction', 'unknown')
            confidence = judgement.get('confidence', 0.0)
            
            lines.append(f"🎯 **趋势判断**：{direction_emoji.get(direction, direction)} (置信度: {confidence:.1%})")
            
            # 证据列表
            evidence = judgement.get('evidence', [])
            if evidence:
                lines.append("   **支撑证据**：")
                for i, item in enumerate(evidence, 1):
                    lines.append(f"   {i}. {item}")
            lines.append("")
        
        # 3. 识别的形态
        patterns = result.get('patterns', [])
        if patterns:
            lines.append("🔍 **识别的K线形态**：")
            for pattern in patterns:
                name = pattern.get('name', 'Unknown')
                conf = pattern.get('confidence', 0.0)
                lines.append(f"   • {name} (置信度: {conf:.1%})")
            lines.append("")
        
        # 4. 风险提示
        risks = result.get('risks', [])
        if risks:
            lines.append("⚠️ **风险提示**：")
            for i, risk in enumerate(risks, 1):
                lines.append(f"   {i}. {risk}")
            lines.append("")
        
        # 5. 任务类型（调试信息）
        vision_task = result.get('vision_task', '')
        if vision_task:
            lines.append(f"_（分析类型: {vision_task}）_")
        
        return "\n".join(lines)

    def build_market_context_with_image(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
        wallets: Any = None,
        current_trades: Optional[List[Any]] = None,
        exchange: Any = None,
        position_tracker: Any = None,
        market_comparator: Any = None,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None,
        chart_image_b64: Optional[str] = None,
        vision_tools: Any = None
    ) -> Dict[str, Any]:
        """
        构建完整的市场上下文（包含可选的Gemini视觉分析）

        Args:
            dataframe: OHLCV数据和所有技术指标
            metadata: 交易对元数据
            wallets: 钱包对象（用于获取账户余额）
            current_trades: 当前所有持仓列表
            exchange: 交易所对象（用于获取资金费率）
            position_tracker: PositionTracker实例，提供持仓表现
            market_comparator: MarketStateComparator实例，用于对比
            multi_timeframe_data: 其他时间框架的K线与指标数据
            chart_image_b64: 可选的base64编码图片
            vision_tools: VisionTools实例，用于Gemini视觉分析

        Returns:
            {
                "text_context": str,              # 文本上下文（包含视觉分析结果）
                "has_vision_analysis": bool       # 是否包含视觉分析
            }
        """
        # 构建文本上下文（调用原方法）
        text_context = self.build_market_context(
            dataframe=dataframe,
            metadata=metadata,
            wallets=wallets,
            current_trades=current_trades,
            exchange=exchange,
            position_tracker=position_tracker,
            market_comparator=market_comparator,
            multi_timeframe_data=multi_timeframe_data
        )
        
        # 如果有图片且提供了 vision_tools，调用 Gemini 分析
        vision_analysis = ""
        if chart_image_b64 and vision_tools:
            try:
                logger.info("📸 调用 Gemini 视觉分析...")
                result = vision_tools.analyze_image_with_gemini(
                    image_b64=chart_image_b64,
                    task="trend",  # 趋势分析
                    time_frame=metadata.get('timeframe', '30m'),
                    pair=metadata.get('pair'),  # 交易对名称
                    return_format="json"
                )
                
                # 格式化结构化分析结果（充分利用 Pydantic 模型返回的数据）
                vision_analysis = self._format_vision_analysis(result)
                
                logger.info("✅ Gemini 视觉分析完成")
                logger.info(f"参数: {json.dumps(result, indent=4)}")
                logger.debug(f"视觉分析长度: {len(vision_analysis)} 字符")
            except Exception as e:
                logger.error(f"❌ Gemini 视觉分析失败: {e}")
                vision_analysis = ""
        
        # 将视觉分析结果添加到文本上下文
        if vision_analysis:
            text_context += f"\n\n{'='*60}\n【K线图视觉分析】\n{'='*60}\n{vision_analysis}\n{'='*60}"
        
        # 返回结构化数据
        return {
            "text_context": text_context,
            "has_vision_analysis": bool(vision_analysis)
        }
    
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

        context_parts = [
            f"=" * 60,
            f"交易对: {pair}",
            f"时间: {latest['date'] if 'date' in latest else datetime.now()}",
            f"=" * 60,
            "",
            "【价格信息】",
            f"  当前价格: {latest['close']:.8f}",
            f"  开盘: {latest['open']:.8f}  最高: {latest['high']:.8f}  最低: {latest['low']:.8f}",
            f"  成交量: {latest['volume']:.2f}",
            f"  价格变化: {((latest['close'] - prev['close']) / prev['close'] * 100):.2f}%",
        ]

        # 暂存市场情绪数据，稍后在末尾添加
        sentiment_parts = []
        sentiment_parts.append("")
        sentiment_parts.append("【市场情绪】")
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
                                from datetime import datetime
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

        # 输出4小时指标 - 最重要，决定大趋势
        if self.include_multi_timeframe_data and indicators_4h:
            context_parts.append("")
            context_parts.append("【技术指标 - 4小时】(决定大趋势)")
            for ind, val in indicators_4h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出1小时指标 - 次重要，确认中期趋势
        if self.include_multi_timeframe_data and indicators_1h:
            context_parts.append("")
            context_parts.append("【技术指标 - 1小时】(确认中期趋势)")
            for ind, val in indicators_1h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出30分钟指标 - 寻找入场时机
        if indicators_30m:
            context_parts.append("")
            context_parts.append("【技术指标 - 30分钟】(寻找入场时机)")
            for ind, val in indicators_30m:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出1天指标（可选）
        if self.include_multi_timeframe_data and indicators_1d:
            context_parts.append("")
            context_parts.append("【技术指标 - 1天】")
            for ind, val in indicators_1d:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 添加账户信息
        if wallets:
            context_parts.append("")
            context_parts.append("【账户信息】")
            try:
                total = wallets.get_total('USDT')
                free = wallets.get_free('USDT')
                used = wallets.get_used('USDT')
                context_parts.extend([
                    f"  总余额: {total:.2f} USDT",
                    f"  可用余额: {free:.2f} USDT",
                    f"  已用资金: {used:.2f} USDT",
                    f"  资金利用率: {(used/total*100):.1f}%" if total > 0 else "  资金利用率: 0%"
                ])
            except Exception as e:
                context_parts.append(f"  无法获取账户信息: {e}")

        # 添加持仓信息
        context_parts.append("")
        context_parts.append("【持仓情况】")
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

                    # 计算持仓时间
                    if open_date:
                        from datetime import datetime, timezone
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

                    # 添加PositionTracker的追踪数据
                    if position_tracker:
                        try:
                            trade_id = getattr(trade, 'id', None)
                            if trade_id:
                                metrics = position_tracker.get_position_metrics(trade_id)
                                if metrics:
                                    context_parts.append("")
                                    context_parts.append("    【持仓追踪数据】")
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
        if current_trades and market_comparator and pair_trades:
            context_parts.append("")
            context_parts.append("=" * 60)
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
            context_parts.append("=" * 60)

        # 添加关键指标历史序列（使用新的 DataFormatter）
        context_parts.append("")
        context_parts.append(f"【关键指标历史（最近{self.indicator_history_points}根K线）】")
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
                context_parts.append(f"【K线历史（最近{len(raw_rows)}根）】{header_text}")
                for entry in raw_rows:
                    context_parts.append(f"  {entry}")

        # 多时间框架原始数据
        if self.multi_timeframe_history and multi_timeframe_data:
            context_parts.append("")
            context_parts.append("【多时间框架K线数据】")
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
        context_parts.extend(sentiment_parts)

        # 添加历史经验和模式分析（自我学习系统）
        if self.enable_learning:
            try:
                context_parts.append("")
                # 获取最近交易
                recent_trades_text = self.historical_query.format_recent_trades_for_context(
                    pair=pair,
                    limit=5
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

        context_parts.append("")
        context_parts.append("=" * 60)

        return "\n".join(context_parts)

    def build_position_context(
        self,
        current_trades: List[Any],
        pair: str
    ) -> str:
        """
        构建当前持仓上下文

        Args:
            current_trades: 当前交易列表（可以是Trade对象或字典）
            pair: 交易对

        Returns:
            格式化的持仓上下文字符串
        """
        if not current_trades:
            return f"{pair} 当前无持仓"

        # 查找当前交易对的持仓（兼容dict和对象）
        pair_trades = []
        for t in current_trades:
            t_pair = t.get('pair') if isinstance(t, dict) else getattr(t, 'pair', None)
            if t_pair == pair:
                pair_trades.append(t)

        if not pair_trades:
            return f"{pair} 当前无持仓"

        context_parts = [f"\n{'='*50}", f"{pair} 【持仓详情】", f"{'='*50}"]

        for i, trade in enumerate(pair_trades, 1):
            # 兼容字典和对象
            if isinstance(trade, dict):
                direction = "做空" if trade.get('side') == 'short' else "做多"
                open_rate = trade.get('open_rate', 0)
                current_rate = trade.get('current_rate', open_rate)
                amount = trade.get('amount', 0)
                stake_amount = trade.get('stake_amount', 0)
                leverage = trade.get('leverage', 1)
                profit_pct = trade.get('profit_pct', 0)
                profit_abs = stake_amount * profit_pct / 100 if stake_amount else 0
                stop_loss = trade.get('stop_loss')
                duration = trade.get('duration_minutes', 0)

                # 计算持仓天数和小时
                days = duration // 1440
                hours = (duration % 1440) // 60
                mins = duration % 60

                context_parts.extend([
                    f"\n持仓 #{i}:",
                    f"  交易方向: {direction} {leverage}x杠杆",
                    f"  开仓价格: {open_rate:.2f}",
                    f"  当前价格: {current_rate:.2f}",
                    f"  持仓数量: {amount:.4f}",
                    f"  投入资金: {stake_amount:.2f} USDT",
                    f"  当前盈亏: {profit_pct:.2f}% ({profit_abs:+.2f} USDT)",
                    f"  持仓时间: {days}天{hours}小时{mins}分钟",
                ])
            else:
                # 对象格式
                direction = "做空" if getattr(trade, 'is_short', False) else "做多"
                current_rate = getattr(trade, 'close_rate', None) or getattr(trade, 'open_rate', 0)
                profit_pct = trade.calc_profit_ratio(current_rate) * 100 if hasattr(trade, 'calc_profit_ratio') else 0
                profit_abs = getattr(trade, 'stake_amount', 0) * profit_pct / 100
                leverage = getattr(trade, 'leverage', 1)

                # 持仓时间
                from datetime import datetime, timezone
                open_date = getattr(trade, 'open_date', None)
                if open_date:
                    # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
                    now = datetime.utcnow() if open_date.tzinfo is None else datetime.now(timezone.utc)
                    duration = (now - open_date).total_seconds() / 60
                    days = int(duration // 1440)
                    hours = int((duration % 1440) // 60)
                    mins = int(duration % 60)
                    duration_str = f"{days}天{hours}小时{mins}分钟"
                else:
                    duration_str = "未知"

                context_parts.extend([
                    f"\n持仓 #{i}:",
                    f"  交易方向: {direction} {leverage}x杠杆",
                    f"  开仓价格: {trade.open_rate:.2f}",
                    f"  当前价格: {current_rate:.2f}",
                    f"  持仓数量: {trade.amount:.4f}",
                    f"  投入资金: {getattr(trade, 'stake_amount', 0):.2f} USDT",
                    f"  当前盈亏: {profit_pct:.2f}% ({profit_abs:+.2f} USDT)",
                    f"  持仓时间: {duration_str}",
                ])

        context_parts.append(f"{'='*50}\n")
        return "\n".join(context_parts)

    def build_system_prompt(self) -> str:
        """
        构建系统提示词

        Returns:
            系统提示词字符串
        """
        parts = [
            "你是专业的加密货币永续合约交易员。目标：通过做多/做空永续合约获利。",
            "",
            "【核心交易原则】",
            "  1. 技术面和价格位置是决策的主要依据",
            "  2. 风险收益比必须≥1.5，否则观望",
            "  3. 情绪指标仅作为辅助参考，不应单独作为开仓理由",
            "  4. 趋势不明确时宁可等待，不要强行开仓",
            "  5. 你是趋势交易员，让利润奔跑，不要做只赚蚂蚁肉的短线交易员",
            "  6. 你的决策间隔是15分钟，下次决策在15分钟后，不要对单K线的指标波动过度反应",
            "",
            "【市场结构】",
            "  - 摆动高点（swing high）= 阻力位，价格多次在此受阻",
            "  - 摆动低点（swing low）= 支撑位，价格多次在此止跌",
            "  - 价格位置相对关键位的距离决定风险收益比",
        ]

        if self.include_timeframe_guidance:
            parts.extend([
                "",
                "【多时间框架优先级】",
                "  - 大周期 > 小周期：1d > 4h > 1h > 15m",
                "  - 框架冲突时跟随大周期，小周期仅作为入场时机参考",
            ])
        else:
            parts.extend([
                "",
                "【趋势判断】",
                "  - 仅使用上下文提供的真实K线/指标来评估趋势，不要臆测未提供的数据",
                "  - 如果不同时间框架数据矛盾，以风险收益和关键价位为先，宁可观望",
            ])

        parts.extend([
            "",
            "【交易工具（函数调用）】",
            "  空仓状态：",
            "    • signal_entry_long  - 做多开仓 (limit_price, leverage, stoploss_pct, stake_amount, confidence_score, key_support/resistance, rsi, trend_strength, reason)",
            "    • signal_entry_short - 做空开仓 (参数同上)",
            "    • signal_wait        - 观望不开仓 (confidence_score, rsi, reason)",
            "  持仓状态：",
            "    • signal_exit     - 平仓 (limit_price, confidence_score, rsi, reason)",
            "    • adjust_position - 加/减仓 (adjustment_pct, limit_price, confidence_score, key_support/resistance, reason)",
            "    • signal_hold     - 维持持仓 (confidence_score, rsi, reason)",
            "  注意：先确认【持仓情况】，调用一个函数后立即停止",
            "",
            "【入场决策分析顺序】",
            "  第一步：价格位置分析",
            "    • 当前价格相对支撑/阻力的位置",
            "    • 距离关键位的百分比（至少2-3%空间）",
            "    • 价格是否处于技术形态的关键点位",
            "  第二步：技术指标确认",
        ])

        if self.include_timeframe_guidance:
            parts.append("    • 多时间框架趋势是否对齐（大周期 > 小周期）")
        else:
            parts.append("    • 依据提供的K线和指标推断趋势方向")

        parts.extend([
            "    • RSI是否处于合理区间（超买>70，超卖<30）",
            "    • MACD/EMA等趋势指标是否支持",
            "    • 多个指标是否共振",
            "  第三步：风险收益评估",
            "    • 计算风险距离（到关键支撑/阻力）和目标距离",
            "    • 风险收益比必须≥1.5",
            "    • 风险收益比<1.5时必须选择signal_wait",
            "  第四步：辅助参考（可选）",
            "    • 情绪指标是否与技术面一致",
            "    • 成交量是否配合",
            "  决策：前三步都满足才考虑开仓，任一步不满足则观望",
            "",
            "【市场情绪参考】",
            "  - 仅作为辅助参考，不应作为主要开仓依据",
            "  - 极端情绪可持续数周甚至数月",
            "  - 强趋势中单边情绪是常态",
            "",
            "【平仓考虑因素】",
        ])

        if self.include_timeframe_guidance:
            parts.append("  - 关键支撑/阻力的突破情况，多时间框架趋势")
        else:
            parts.append("  - 关键支撑/阻力的突破情况，实际走势与开仓逻辑的对比")

        parts.extend([
            "  - 开仓逻辑是否仍然成立（价格是否跌破关键支撑/突破阻力）",
            "  - 趋势是否发生实质性反转（不是指标波动，而是价格结构改变）",
            "  - 盈利是否接近预设目标",
            "  重要提醒：",
            "    • 你15分钟才能决策一次，ADX下降、MACD转负等单K线指标变化是正常波动",
            "    • 只有价格跌破EMA20/关键支撑位，才是趋势反转的信号",
            "    • 不要因为指标短期波动就恐慌平仓，让利润奔跑",
            "",
            "【持仓管理考虑】",
            "  - 大趋势是否仍然成立（价格与EMA20的位置关系）",
            "  - 盈利是否接近预设目标（至少70%以上）",
            "  - 给趋势时间发展，15分钟K线需要更长的持仓时间",
            "  - 让利润奔跑，不要赚一点点就跑",
            "",
            "【订单类型】",
            "  - Limit Order：做多limit<当前价，做空limit>当前价（等待更好价格）",
            "  - Market Order：做多limit≥当前价，做空limit≤当前价（立即入场）",
            "  选择原则：",
            "    • 趋势明确+量能放大+技术突破 → 市价单立即入场（错过机会成本高）",
            "    • 震荡行情+接近支撑阻力 → 限价单等待更好价格",
            "    • 当决定开仓时，优先考虑市价单，避免因等待而错过行情",
            "",
            "【风险收益比计算】",
            "  - 风险距离 = 价格到关键支撑/阻力的距离百分比",
            "  - 目标距离 = 价格到目标位的距离百分比",
            "  - 风险收益比 = 目标距离 / 风险距离",
            "  - 风险收益比必须≥1.5，否则必须调用signal_wait",
            "",
            "【杠杆选择考虑因素】",
            "  - 市场波动（ATR占比）",
            "  - 信号质量（多个指标是否一致）",
            "  - 趋势强度（ADX值）",
            "  - 价格位置（是否在关键位附近）",
            "  - 账户可用余额",
            "",
            "【决策检查清单】",
            "  ✓ 技术面分析完整（价格位置+指标确认）",
            "  ✓ 风险收益比≥1.5",
            "  ✓ 趋势方向明确",
            "  ✓ 关键支撑/阻力位明确",
            "  如任一项不满足，调用signal_wait并说明理由",
            "",
            "综合分析后独立判断，调用一个函数后停止。",
        ])

        return "\n".join(parts)

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

    def _build_entry_system_prompt_old(self) -> str:
        """
        旧版开仓提示词（已废弃，保留用于参考）

        Returns:
            开仓决策系统提示词字符串
        """
        parts = [
            "你是专业的加密货币趋势交易员。任务：评估是否开仓。",
            "",
            "【核心理念】",
            "",
            "趋势交易的本质：识别趋势方向并顺势而为。",
            "大周期决定方向，小周期寻找入场时机。",
            "入场时机有多种形式：回调、突破、延续。",
            "市场是动态的，要灵活应对，不要机械执行规则。",
            "",
            "【多时间框架分析】",
            "",
            "按顺序分析，但要综合判断：",
            "",
            "第一步：看4小时图 - 确定主要方向",
            "  EMA排列显示趋势：",
            "    EMA20 > EMA50 > EMA100：上涨趋势，偏向做多",
            "    EMA20 < EMA50 < EMA100：下跌趋势，偏向做空",
            "    EMA交织：震荡，谨慎",
            "  MACD柱状图显示动能：",
            "    持续为正：多头动能",
            "    持续为负：空头动能",
            "    频繁翻转：无方向，观察",
            "",
            "第二步：看1小时图 - 确认或识别回调",
            "  理想：1小时和4小时方向一致",
            "  如果相反：可能是大趋势中的回调（机会）",
            "  如果长期背离：大趋势可能在改变",
            "",
            "第三步：看30分钟图 - 识别入场时机",
            "  确认趋势方向后，寻找入场点：",
            "    回调后的反弹 / 反弹后的下跌",
            "    关键位突破后的延续",
            "    趋势加速中的跟随",
            "",
            "注意：",
            "  逆势交易风险极高（如4小时空头时做多）",
            "  但也要识别趋势反转，不要盲目相信大周期",
            "  30分钟信号单独不够，但结合其他维度就有效",
            "",
            "【技术指标理解】",
            "",
            "EMA - 趋势的骨架：",
            "  三线排列显示趋势基础：",
            "    多头（EMA20>50>100）：上涨趋势信号",
            "    空头（EMA20<50<100）：下跌趋势信号",
            "    排列越清晰 = 趋势越强",
            "  ",
            "  多种入场时机方式：",
            "    经典：回调到EMA20反弹（稳健，适用于回调入场）",
            "    激进：价格在EMA20上方奔跑时入场（适用于强趋势）",
            "    突破：价格突破并站稳EMA20后入场（适用于趋势启动）",
            "  不要死盯EMA20 - 有时EMA50也是强支撑",
            "",
            "MACD - 动能温度计：",
            "  关注柱状图（hist），它反映动能变化。",
            "  ",
            "  入场信号：",
            "    柱状图从负转正：多头动能恢复（经典回调买点）",
            "    柱状图扩大：趋势加速（可以追）",
            "    柱状图为负但收窄：动能衰竭，可能反转",
            "  ",
            "  不要机械等MACD转正 - 有时行情提前启动。",
            "",
            "RSI - 超买超卖参考：",
            "  辅助指标，不是主要依据。",
            "  ",
            "  理解：",
            "    从30-40上升：回调可能结束",
            "    从60-70下降：反弹可能结束",
            "    强趋势中RSI可以长期>70或<30",
            "  ",
            "  不要单纯因为RSI超买就做空，或超卖就做多。",
            "",
            "布林带 - 波动率边界：",
            "  趋势市场：价格沿上轨（多头）或下轨（空头）运行",
            "  震荡市场：价格在带内来回",
            "  突破市场：价格突破上轨并继续",
            "",
            "ADX - 趋势强度表：",
            "  理解ADX数值：",
            "    >30：强趋势，适合趋势交易",
            "    20-30：中等趋势，需要更多确认",
            "    <20：弱趋势或震荡，谨慎",
            "  ",
            "  不要把ADX当硬门槛：",
            "    ADX 23可能是好趋势的开始",
            "    ADX从15升到23意味着趋势在增强",
            "    结合其他指标综合判断",
            "",
            "【成交量理解】",
            "",
            "成交量是确认工具：",
            "  突破时成交量放大：突破可能有效",
            "  突破时成交量萎缩：突破可能假",
            "  不要机械要求'1.5倍放量' - 市场是动态的",
            "",
            "【市场情绪使用】",
            "",
            "情绪是参考，不是决定性因素。",
            "",
            "多空比：",
            "  极端比值（>2.5或<0.4）：市场情绪偏向一边",
            "  可能含义：",
            "    强趋势中的正常现象（继续跟随）",
            "    趋势末端的反转信号（需价格确认）",
            "  不要单纯基于比值开仓，结合价格行为",
            "",
            "资金费率：",
            "  极端费率反映极端情绪",
            "  但强趋势中费率可长期极端",
            "  作为参考，不是硬信号",
            "",
            "【多种入场方式】",
            "",
            "趋势交易有多种入场时机 - 要灵活：",
            "",
            "方式1 - 回调入场（稳健型）：",
            "  何时：趋势确立，等待回调",
            "  信号：价格回调到EMA20/50，MACD柱状图收窄后转向",
            "  优点：风险低，入场位置好",
            "  缺点：可能没有回调",
            "",
            "方式2 - 突破追入（激进型）：",
            "  何时：价格突破关键支撑/阻力",
            "  信号：突破后站稳，成交量放大，ADX上升",
            "  优点：不会错过强劲走势",
            "  缺点：入场位置可能不是最优",
            "",
            "方式3 - 趋势延续（跟随型）：",
            "  何时：趋势明确，价格沿EMA运行",
            "  信号：价格在EMA20上方，MACD持续为正，ADX>25",
            "  优点：搭上强趋势",
            "  缺点：入场点不是最优",
            "",
            "方式4 - 趋势反转（反向型）：",
            "  何时：识别趋势反转",
            "  信号：EMA转向，价格突破EMA，结构改变",
            "  优点：抓住大趋势开始",
            "  缺点：难度高，容易错",
            "",
            "根据当前市场状态选择合适方式，不要固守一种。",
            "",
            "【开仓决策框架】",
            "",
            "综合评估，不是机械执行：",
            "",
            "核心问题：",
            "  1. 趋势方向是什么？（看4小时EMA排列）",
            "  2. 趋势强度如何？（ADX、MACD持续性）",
            "  3. 当前处于趋势的什么阶段？（启动、发展、衰竭）",
            "  4. 有什么入场时机？（回调、突破、延续、反转）",
            "  5. 风险收益比如何？（关键支撑/阻力位、目标位）",
            "",
            "综合判断：",
            "  多个维度都指向同一方向 = 高置信度，可以开仓",
            "  部分维度确认，部分矛盾 = 中等置信度，谨慎开仓或等待",
            "  维度冲突严重 = 低置信度，观望",
            "",
            "不要机械要求'必须3个维度'或'ADX必须>25'：",
            "  有时2个强信号比3个弱信号更好",
            "  ADX 23且在上升，可能比ADX 26但在下降更好",
            "  市场是动态的，灵活判断",
            "",
            "【仓位大小】",
            "",
            "根据置信度决定投入金额：",
            "",
            "高置信度：多维度确认，趋势强劲",
            "  可考虑投入300-400 USDT（假设余额1000 USDT）",
            "  例如：4h+1h趋势一致，ADX>25且上升，突破确认",
            "",
            "中等置信度：部分确认，趋势中等",
            "  可考虑投入200-300 USDT（假设余额1000 USDT）",
            "  例如：4h趋势明确，1h有入场信号，ADX 20-25",
            "",
            "低置信度：信号较弱，但有机会",
            "  可考虑投入100-200 USDT（假设余额1000 USDT）",
            "  例如：趋势初现，但未完全确认",
            "",
            "通过stake_amount参数指定具体USDT金额：",
            "  查看【账户信息】中的可用余额",
            "  根据你的置信度和余额计算",
            "  如果不设置，系统使用默认仓位",
            "",
            "【要避免的思维陷阱】",
            "",
            "不要只看单一指标：",
            "  只看MACD转正，忽略EMA死叉 = 片面",
            "  只看多空比极端，忽略价格走势 = 误导",
            "  综合多个维度，但不机械要求数量",
            "",
            "不要过度依赖规则：",
            "  ADX 23 vs 25，看趋势是否在增强",
            "  回调未到EMA20，但其他信号强烈，也可考虑",
            "  市场是动态的，灵活应对",
            "",
            "不要逆势而为（除非确认反转）：",
            "  4小时空头时做多风险极高",
            "  但要识别趋势转折点",
            "  结构改变 + 多维度确认 = 可能的反转",
            "",
            "【核心理念】",
            "",
            "开仓理由是持仓管理的锚点，必须清晰。",
            "多维度验证，但灵活判断，不机械执行。",
            "趋势是朋友，但要识别趋势的不同阶段。",
            "有疑虑时宁可等待，机会永远都有。",
            "",
            "分析完毕后，调用一个函数做决策。",
        ]

        return "\n".join(parts)

    def _build_position_system_prompt_old(self) -> str:
        """
        旧版持仓提示词（已废弃，保留用于参考）

        Returns:
            持仓管理系统提示词字符串
        """
        parts = [
            "你是专业的加密货币趋势交易员。任务：管理现有持仓。",
            "",
            "【核心原则】",
            "",
            "目标：吃完整趋势，不要半路下车。",
            "开仓理由是你的锚点，只有锚点被破坏时才考虑平仓。",
            "",
            "【开仓锚点分析】",
            "",
            "首先回顾开仓理由：",
            "  查看'开仓理由'，提取核心依据：",
            "    是基于EMA突破？",
            "    是基于趋势形成？",
            "    是基于关键支撑位？",
            "    是基于多时间框架对齐？",
            "",
            "判断锚点是否还有效：",
            "  如果开仓理由是'4小时多头趋势，价格回踩EMA20反弹'：",
            "    检查：4小时EMA排列是否还是多头（EMA20>50>100）？",
            "    检查：价格是否还在EMA20上方？",
            "    如果都满足 = 锚点有效 = 继续持有",
            "    如果任一不满足 = 锚点破坏 = 考虑平仓",
            "",
            "  如果开仓理由是'突破关键阻力位XXX'：",
            "    检查：价格是否还在阻力位上方？",
            "    检查：是否站稳至少3根K线？",
            "    如果跌破并站稳下方 = 假突破 = 立即平仓",
            "",
            "【区分回调与反转】",
            "",
            "趋势的结构特征：",
            "  多头趋势 = Higher Highs + Higher Lows（每次回调低点都比上次高）",
            "  空头趋势 = Lower Highs + Lower Lows（每次反弹高点都比上次低）",
            "",
            "判断方法：",
            "  看K线历史数据，识别最近的高点和低点。",
            "  多头趋势中，如果出现Lower High（新高点低于前一高点）= 趋势可能反转",
            "  空头趋势中，如果出现Higher Low（新低点高于前一低点）= 趋势可能反转",
            "",
            "回调的特征：",
            "  价格短暂回撤，但不破坏趋势结构",
            "  支撑位（多头）或阻力位（空头）守住",
            "  EMA排列不变",
            "  ADX虽然下降但仍>20",
            "  回调后价格重新朝趋势方向运动",
            "",
            "反转的特征：",
            "  价格结构被破坏（出现Lower High或Higher Low）",
            "  关键支撑/阻力位被突破并站稳",
            "  EMA开始拐头（EMA20穿越EMA50）",
            "  MACD在零轴附近金叉/死叉",
            "  ADX持续下降至<20",
            "",
            "【指标解读】",
            "",
            "单根K线的指标波动不重要，看趋势。",
            "",
            "MACD：",
            "  多头趋势中，MACD柱状图转负是正常回调，等它重新转正",
            "  只有MACD在零轴附近死叉且价格跌破EMA20 = 警惕反转",
            "  空头趋势中，MACD柱状图转正是正常反弹，等它重新转负",
            "",
            "RSI：",
            "  强趋势中RSI可以长期超买（>70）或超卖（<30）",
            "  多头趋势中RSI从80降到60 = 正常回调，不是卖出信号",
            "  空头趋势中RSI从20升到40 = 正常反弹，不是买入信号",
            "",
            "ADX：",
            "  ADX>25：趋势强劲，继续持有",
            "  ADX 20-25：趋势减弱但仍在，观察",
            "  ADX<20：趋势消失，考虑平仓",
            "",
            "EMA：",
            "  价格回踩EMA20但不破 = 健康回调",
            "  价格跌破EMA20并站稳2根K线以上 = 趋势减弱",
            "  EMA20跌破EMA50（死叉）= 趋势反转信号",
            "",
            "【加仓策略】",
            "",
            "加仓（adjust_position）- 降低平均成本或扩大盈利。",
            "",
            "允许加仓的情况：",
            "  趋势强劲（ADX>30）且开仓锚点更加强化",
            "  当前持仓盈利>3%，且价格回调到强支撑位",
            "  例如：做多时，4小时EMA金叉后，价格回踩1小时EMA20反弹 = 可加仓",
            "",
            "加仓幅度：",
            "  根据趋势强度和当前盈利决定：",
            "    趋势强（ADX>30）+ 当前盈利>5% = 可加仓50-100%（相对当前仓位）",
            "    趋势中等（ADX 25-30）+ 当前盈利3-5% = 可加仓30-50%（相对当前仓位）",
            "  例如：当前仓位100 USDT，加仓50%，则增加50 USDT",
            "",
            "不允许加仓的情况：",
            "  当前持仓亏损 = 绝对不加仓（不要摊平亏损）",
            "  趋势减弱（ADX<25）",
            "  持仓时间<1小时（趋势未确认）",
            "",
            "加仓方式：",
            "  调用adjust_position函数，参数：",
            "    pair: 交易对",
            "    adjustment_pct: 加仓百分比（50表示加仓50%）",
            "    limit_price: 期望的加仓价格",
            "    confidence_score: 置信度（1-100）",
            "    key_support: 关键支撑位",
            "    key_resistance: 关键阻力位",
            "    reason: 加仓理由",
            "",
            "【持仓时间与盈利】",
            "",
            "趋势需要时间发展。",
            "  持仓<1小时 = 趋势未展开，不要因为小幅波动就平仓",
            "  持仓1-3小时 = 趋势发展中，只要锚点有效就持有",
            "  持仓>3小时 = 如果盈利达标（>5%）且趋势减弱，可以获利了结",
            "",
            "盈利回撤处理：",
            "  查看'盈利回撤'数据：",
            "    如果从峰值回撤>30%（如从+8%回撤到+5.5% = 回撤31%）",
            "    且ADX<25，考虑获利了结",
            "",
            "【决策流程】",
            "",
            "按顺序执行：",
            "",
            "1. 回顾开仓锚点",
            "   开仓理由中的核心依据还成立吗？",
            "   如果不成立 = 调用signal_exit",
            "",
            "2. 判断趋势状态",
            "   看价格结构（Higher Highs/Lower Lows）",
            "   看EMA排列、MACD方向、ADX强度",
            "   如果趋势反转 = 调用signal_exit",
            "   如果是回调 = 继续持有",
            "",
            "3. 评估加仓机会",
            "   如果盈利>3% 且 趋势强化 且 价格回调到支撑",
            "   = 考虑调用adjust_position加仓",
            "",
            "4. 评估获利了结",
            "   如果盈利>5% 且（趋势减弱 或 持仓>3小时 或 达到目标位）",
            "   = 调用signal_exit",
            "",
            "5. 如果以上都不满足",
            "   = 调用signal_hold，说明持有理由（锚点有效、趋势未变）",
            "",
            "【常见错误】",
            "",
            "必须避免：",
            "  被单根K线的指标变化吓跑（MACD转负、RSI回落）",
            "  正常回调时过早平仓（价格回踩EMA20就跑）",
            "  持仓时间太短（<30分钟就想获利）",
            "  亏损时加仓（试图摊平成本）",
            "  盈利时过度贪婪（趋势已反转还不走）",
            "  忘记开仓锚点，只看当前指标",
            "",
            "【记住】",
            "",
            "你是趋势交易员，不是短线客。",
            "给趋势时间，不要被短期波动干扰。",
            "只有锚点破坏或趋势真正反转时才离场。",
            "",
            "分析完毕后，调用一个函数做决策。",
        ]

        return "\n".join(parts)

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

    def optimize_token_usage(self, text: str, max_tokens: int) -> str:
        """
        优化文本以适应token限制

        Args:
            text: 原始文本
            max_tokens: 最大token数

        Returns:
            优化后的文本
        """
        # 简单估算: 1 token ≈ 4 字符 (英文) 或 1.5 字符 (中文)
        estimated_tokens = len(text) / 2.5

        if estimated_tokens <= max_tokens:
            return text

        # 需要截断
        target_chars = int(max_tokens * 2.5)
        truncated = text[:target_chars]

        # 在最后一个换行符处截断，避免截断句子
        last_newline = truncated.rfind('\n')
        if last_newline > target_chars * 0.8:  # 如果不会损失太多
            truncated = truncated[:last_newline]

        return truncated + "\n... (内容已截断)"

    def _analyze_trend(self, prices: List[float]) -> Dict[str, Any]:
        """
        分析价格趋势（改进版）

        Returns:
            Dict包含: direction (方向), strength (强度), change_pct (变化百分比)
        """
        if len(prices) < 2:
            return {
                "direction": "未知",
                "strength": "无",
                "change_pct": 0
            }

        # 计算价格变化
        total_change = (prices[-1] - prices[0]) / prices[0] * 100

        # 确定方向（降低门槛：从±1%降到±0.5%）
        if total_change > 0.5:
            direction = "上涨"
        elif total_change < -0.5:
            direction = "下跌"
        else:
            direction = "横盘"

        # 计算趋势强度（基于幅度）
        abs_change = abs(total_change)
        if abs_change > 3:
            strength = "强势"
        elif abs_change > 1:
            strength = "中等"
        elif abs_change > 0.5:
            strength = "弱势"
        else:
            strength = "极弱"

        # 计算趋势一致性（价格是否朝同一方向移动）
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if direction != "横盘":
            same_direction = sum(1 for c in changes if (c > 0 and direction == "上涨") or (c < 0 and direction == "下跌"))
            consistency = same_direction / len(changes) * 100
        else:
            consistency = 0

        return {
            "direction": direction,
            "strength": strength,
            "change_pct": total_change,
            "consistency": consistency
        }

    def _format_duration(self, start_time: datetime) -> str:
        """格式化持续时间"""
        if not start_time:
            return "未知"

        from datetime import timezone, datetime as dt
        # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
        now = dt.utcnow() if start_time.tzinfo is None else dt.now(timezone.utc)
        duration = now - start_time
        hours = duration.total_seconds() / 3600

        if hours < 1:
            return f"{int(hours * 60)}分钟"
        elif hours < 24:
            return f"{hours:.1f}小时"
        else:
            return f"{hours / 24:.1f}天"

    def _analyze_market_structure(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """分析市场结构（支撑/阻力/趋势）- lookback=100根K线(25小时)能找到更长期的支撑阻力"""
        if len(dataframe) < lookback:
            return {"structure": "数据不足"}

        recent = dataframe.tail(lookback)

        # 计算摆动高低点
        highs = recent['high']
        lows = recent['low']
        closes = recent['close']

        swing_high = highs.max()
        swing_low = lows.min()
        current_price = closes.iloc[-1]

        # 判断市场结构
        # 检查是否在创新高/新低
        prev_highs = dataframe['high'].tail(lookback*2).head(lookback)
        prev_lows = dataframe['low'].tail(lookback*2).head(lookback)

        is_higher_high = swing_high > prev_highs.max() if len(prev_highs) > 0 else False
        is_lower_low = swing_low < prev_lows.min() if len(prev_lows) > 0 else False

        # 确定结构
        if is_higher_high and not is_lower_low:
            structure = "上升结构（Higher Highs）"
        elif is_lower_low and not is_higher_high:
            structure = "下降结构（Lower Lows）"
        elif is_higher_high and is_lower_low:
            structure = "扩张结构（震荡加剧）"
        else:
            structure = "盘整结构（震荡）"

        # 距离关键位的百分比
        distance_to_high = ((swing_high - current_price) / current_price) * 100
        distance_to_low = ((current_price - swing_low) / current_price) * 100

        return {
            "structure": structure,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "distance_to_high_pct": distance_to_high,
            "distance_to_low_pct": distance_to_low,
            "range_pct": ((swing_high - swing_low) / swing_low) * 100
        }

    def _analyze_indicator_trends(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """分析指标变化趋势"""
        if len(dataframe) < 5:
            return {}

        latest = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]
        prev_5 = dataframe.iloc[-5]

        trends = {}

        # EMA交叉状态
        if 'ema_20' in latest and 'ema_50' in latest:
            ema20_now = latest['ema_20']
            ema50_now = latest['ema_50']
            ema20_prev = prev['ema_20']
            ema50_prev = prev['ema_50']

            if ema20_now > ema50_now and ema20_prev <= ema50_prev:
                trends['ema_cross'] = "刚刚金叉"
            elif ema20_now < ema50_now and ema20_prev >= ema50_prev:
                trends['ema_cross'] = "刚刚死叉"
            elif ema20_now > ema50_now:
                trends['ema_cross'] = "金叉持续中"
            else:
                trends['ema_cross'] = "死叉持续中"

        # MACD柱状图趋势
        if 'macd_hist' in latest:
            macd_hist_now = latest['macd_hist']
            macd_hist_prev = prev['macd_hist']

            if macd_hist_now > macd_hist_prev:
                trends['macd_histogram'] = "增强（多头）" if macd_hist_now > 0 else "减弱（空头弱化）"
            else:
                trends['macd_histogram'] = "减弱（多头弱化）" if macd_hist_now > 0 else "增强（空头）"

        # RSI趋势
        if 'rsi' in latest:
            rsi_now = latest['rsi']
            rsi_5ago = prev_5['rsi']

            if rsi_now > rsi_5ago + 5:
                trends['rsi_trend'] = "上升（动能增强）"
            elif rsi_now < rsi_5ago - 5:
                trends['rsi_trend'] = "下降（动能减弱）"
            else:
                trends['rsi_trend'] = "平稳"

        # ADX趋势强度
        if 'adx' in latest:
            adx_now = latest['adx']
            adx_prev = prev['adx']

            if adx_now > adx_prev:
                trends['adx_direction'] = f"上升（趋势增强，当前{adx_now:.1f}）"
            else:
                trends['adx_direction'] = f"下降（趋势减弱，当前{adx_now:.1f}）"

        return trends

    def _analyze_timeframe_alignment(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        分析多时间框架趋势对齐（改进版，支持横盘）
        """
        if len(dataframe) < 2:
            return {"alignment": "数据不足", "trends": {}}

        latest = dataframe.iloc[-1]
        trends = {}

        # 横盘判断阈值：EMA差距小于0.3%视为横盘
        consolidation_threshold = 0.003

        # 15分钟趋势
        if 'ema_20' in latest and 'ema_50' in latest:
            ema_diff = (latest['ema_20'] - latest['ema_50']) / latest['ema_50']
            if abs(ema_diff) < consolidation_threshold:
                trends['15m'] = "横盘"
            elif ema_diff > 0:
                trends['15m'] = "上涨"
            else:
                trends['15m'] = "下跌"

        # 1小时趋势
        if 'ema_20_1h' in latest and 'ema_50_1h' in latest:
            ema_diff = (latest['ema_20_1h'] - latest['ema_50_1h']) / latest['ema_50_1h']
            if abs(ema_diff) < consolidation_threshold:
                trends['1h'] = "横盘"
            elif ema_diff > 0:
                trends['1h'] = "上涨"
            else:
                trends['1h'] = "下跌"

        # 4小时趋势
        if 'ema_20_4h' in latest and 'ema_50_4h' in latest:
            ema_diff = (latest['ema_20_4h'] - latest['ema_50_4h']) / latest['ema_50_4h']
            if abs(ema_diff) < consolidation_threshold:
                trends['4h'] = "横盘"
            elif ema_diff > 0:
                trends['4h'] = "上涨"
            else:
                trends['4h'] = "下跌"

        # 1天趋势
        if 'ema_20_1d' in latest and 'ema_50_1d' in latest:
            ema_diff = (latest['ema_20_1d'] - latest['ema_50_1d']) / latest['ema_50_1d']
            if abs(ema_diff) < consolidation_threshold:
                trends['1d'] = "横盘"
            elif ema_diff > 0:
                trends['1d'] = "上涨"
            else:
                trends['1d'] = "下跌"

        # 判断对齐情况
        if not trends:
            return {"alignment": "无趋势数据", "trends": {}}

        uptrend_count = sum(1 for t in trends.values() if t == "上涨")
        downtrend_count = sum(1 for t in trends.values() if t == "下跌")
        consolidation_count = sum(1 for t in trends.values() if t == "横盘")
        total_count = len(trends)

        if uptrend_count == total_count:
            alignment = "完全对齐 - 强势上涨"
        elif downtrend_count == total_count:
            alignment = "完全对齐 - 强势下跌"
        elif consolidation_count == total_count:
            alignment = "完全对齐 - 横盘整理"
        elif consolidation_count >= total_count / 2:
            alignment = f"横盘为主（{consolidation_count}/{total_count}）- 等待突破"
        elif uptrend_count > downtrend_count:
            alignment = f"多数上涨（{uptrend_count}/{total_count}）"
        elif downtrend_count > uptrend_count:
            alignment = f"多数下跌（{downtrend_count}/{total_count}）"
        else:
            alignment = "趋势分歧（震荡）"

        return {
            "trends": trends,
            "alignment": alignment,
            "strength": uptrend_count / total_count if uptrend_count > downtrend_count else -downtrend_count / total_count
        }

    def _analyze_volume_trend(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """分析成交量趋势 - lookback=100根K线(25小时)能更好识别放量/缩量趋势"""
        if len(dataframe) < lookback:
            return {"trend": "数据不足"}

        recent = dataframe.tail(lookback)
        volumes = recent['volume']

        # 计算成交量移动平均
        volume_ma = volumes.mean()
        current_volume = volumes.iloc[-1]

        # 成交量趋势
        first_half_avg = volumes.head(lookback//2).mean()
        second_half_avg = volumes.tail(lookback//2).mean()

        if second_half_avg > first_half_avg * 1.2:
            trend = "持续放量"
        elif second_half_avg < first_half_avg * 0.8:
            trend = "持续缩量"
        else:
            trend = "平稳"

        # 当前成交量相对于平均值
        volume_ratio = current_volume / volume_ma

        if volume_ratio > 1.5:
            current_status = "异常放量"
        elif volume_ratio > 1.2:
            current_status = "明显放量"
        elif volume_ratio < 0.7:
            current_status = "明显缩量"
        else:
            current_status = "正常"

        return {
            "trend": trend,
            "current_status": current_status,
            "volume_ratio": volume_ratio,
            "current_vs_avg": f"{(volume_ratio - 1) * 100:+.1f}%"
        }

    def _get_indicator_history(
        self,
        dataframe: pd.DataFrame,
        lookback: int = 100,
        display_points: int = 20
    ) -> Dict[str, List]:
        """获取关键指标的历史序列 - 展示可配置数量的最新值"""
        if len(dataframe) < lookback:
            lookback = len(dataframe)

        recent = dataframe.tail(lookback)
        history = {}

        # 选择关键指标提供历史序列
        key_indicators = ['close', 'rsi', 'macd', 'macd_hist', 'ema_20', 'ema_50', 'adx', 'volume']

        for ind in key_indicators:
            if ind in recent.columns:
                values = recent[ind].tolist()
                # 只保留最近的值，格式化为简洁形式
                trimmed = values[-display_points:]
                history[f"{ind}_recent"] = [
                    round(float(v), 2) if pd.notna(v) else None
                    for v in trimmed
                ]

        return history

    def _get_raw_kline_history(
        self,
        dataframe: pd.DataFrame,
        count: int,
        extra_fields: Optional[List[str]] = None,
        compact: bool = False,
        stride: int = 1,
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """返回指定数量的原始K线数据文本/表格"""
        result = {"rows": [], "header": None}

        if count <= 0 or dataframe.empty:
            return result

        available_cols = set(dataframe.columns)
        base_fields = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in available_cols]
        if len(base_fields) < 4:
            return result

        extra = [
            field for field in self._ensure_list(extra_fields)
            if field in available_cols and field not in base_fields
        ]

        columns = ['time'] + base_fields + extra

        stride = max(1, stride)
        total_available = min(len(dataframe), count)
        if max_rows:
            max_rows = max(1, max_rows)
            if total_available > max_rows:
                stride = max(stride, math.ceil(total_available / max_rows))

        fetch_count = total_available * stride if stride > 1 else total_available
        subset = dataframe.tail(fetch_count)

        if subset.empty:
            return result

        if stride > 1 and len(subset) > count:
            subset = subset.iloc[::-1][::stride].iloc[::-1]

        subset = subset.tail(total_available)
        if subset.empty:
            return result

        if compact:
            result['header'] = ",".join(columns)
        result['stride'] = stride

        for _, row in subset.iterrows():
            time_str = self._format_timestamp(row.get('date'), row.name)

            if compact:
                values = [time_str]
                for col in columns[1:]:
                    values.append(self._format_number(row.get(col)))
                result['rows'].append(",".join(values))
            else:
                pieces = [
                    f"O:{self._format_number(row.get('open'))}",
                    f"H:{self._format_number(row.get('high'))}",
                    f"L:{self._format_number(row.get('low'))}",
                    f"C:{self._format_number(row.get('close'))}"
                ]

                if 'volume' in base_fields:
                    pieces.append(f"V:{self._format_number(row.get('volume'))}")

                if extra:
                    extras = [
                        f"{field}:{self._format_number(row.get(field))}"
                        for field in extra
                    ]
                    pieces.append(" ".join(extras))

                result['rows'].append(f"{time_str} | {' '.join(pieces)}")

        return result

    def get_multi_timeframe_history_config(self) -> Dict[str, Dict[str, Any]]:
        """对外提供多时间框架配置，供策略层决定需要拉取的数据"""
        return self.multi_timeframe_history

    def _normalize_multi_timeframe_config(self, cfg: Any) -> Dict[str, Dict[str, Any]]:
        """将多时间框架配置标准化为 {tf: {candles:int, fields:list}}（委托给 DataFormatter）"""
        return self.formatter.normalize_multi_timeframe_config(cfg, self.config)

    def _normalize_multi_timeframe_config_old(self, cfg: Any) -> Dict[str, Dict[str, Any]]:
        """旧版配置标准化（已废弃，保留用于参考）"""
        normalized: Dict[str, Dict[str, Any]] = {}
        if not cfg:
            return normalized

        default_fields = self._ensure_list(self.config.get("default_multi_timeframe_fields", []))

        if isinstance(cfg, dict):
            items = cfg.items()
        elif isinstance(cfg, list):
            items = []
            for entry in cfg:
                if isinstance(entry, dict) and entry.get('timeframe'):
                    tf = entry['timeframe']
                    data = entry.copy()
                    data.pop('timeframe', None)
                    items.append((tf, data))
        else:
            return normalized

        for tf, settings in items:
            if not tf:
                continue

            candles = 0
            fields = default_fields
            stride = 1
            compact = None
            max_rows = None

            if isinstance(settings, dict):
                candles = settings.get('candles') or settings.get('count') or settings.get('points') or settings.get('length') or 0
                fields = self._ensure_list(settings.get('fields') or settings.get('extra_fields') or default_fields)
                stride = settings.get('stride', 1)
                compact = settings.get('compact')
                max_rows = settings.get('max_rows')
            else:
                candles = settings
                fields = default_fields

            try:
                candles = int(candles)
            except (TypeError, ValueError):
                candles = 0

            try:
                stride = int(stride)
            except (TypeError, ValueError):
                stride = 1

            stride = max(1, stride)

            normalized[str(tf)] = {
                'candles': max(0, candles),
                'fields': fields,
                'stride': stride,
                'compact': compact,
                'max_rows': max_rows
            }

        return normalized

    def _format_timestamp(self, timestamp: Any, fallback_index: Any) -> str:
        if isinstance(timestamp, pd.Timestamp):
            ts = timestamp.to_pydatetime()
        elif isinstance(timestamp, datetime):
            ts = timestamp
        elif timestamp is None or (isinstance(timestamp, float) and pd.isna(timestamp)):
            if isinstance(fallback_index, pd.Timestamp):
                ts = fallback_index.to_pydatetime()
            elif isinstance(fallback_index, datetime):
                ts = fallback_index
            else:
                ts = None
        else:
            return str(timestamp)

        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M")

        return str(timestamp if timestamp is not None else fallback_index)

    def _format_number(self, value: Any, decimals: int = 2) -> str:
        if value is None or pd.isna(value):
            return "-"

        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        formatted = f"{number:.{decimals}f}"
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted or "0"

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
