"""
市场情绪数据获取模块
获取免费的市场情绪指标：Fear & Greed Index, Funding Rate, OI监控, 反转信号
"""

import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
from threading import Thread
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)


# ==================== OI监控常量 ====================
OI_CHANGE_THRESHOLD = 0.05          # OI变化阈值 5%
PRICE_DEVIATION_THRESHOLD = 0.02    # 价格偏离阈值 2%
VOLUME_SURGE_RATIO = 2.0            # 成交量放大倍数 2x
BREAKOUT_OI_THRESHOLD = 0.03        # 突破OI阈值 3%
OI_CACHE_DURATION = 60              # OI缓存时间 60秒

# ==================== 反转监控常量 ====================
CONFIRMATION_WINDOW = 3             # 确认窗口 3根K线
BB_PERIOD = 20                      # 布林带周期
BB_STD = 2.0                        # 布林带标准差倍数
MIN_ANCHOR_VOLUME_RATIO = 1.5       # 最小锚定成交量比率

# ==================== Order Book 常量 ====================
ORDERBOOK_CACHE_DURATION = 30       # 盘口缓存时间 30秒
ORDERBOOK_DEPTH = 20                # 默认盘口深度
WALL_THRESHOLD_MULTIPLIER = 3.0     # 流动性墙检测倍数
MIN_WALL_SIZE_USDT = 50000          # 最小流动性墙规模

# ==================== Order Flow 常量 ====================
ORDERFLOW_CACHE_DURATION = 60       # 订单流缓存时间 60秒
LARGE_TRADE_THRESHOLD_USDT = 100000 # 大单阈值 (用户指定)
TRADE_FETCH_LIMIT = 100             # 获取成交记录数量

# ==================== Liquidation 常量 ====================
LIQUIDATION_BUFFER_MINUTES = 5      # 清算数据缓冲窗口
SIGNIFICANT_LIQ_USDT = 50000        # 显著清算阈值
LIQ_IMBALANCE_ALERT_RATIO = 2.0     # 清算失衡告警比率


class OIMonitor:
    """OI (Open Interest) 监控器 - 使用 CCXT 统一接口"""

    def __init__(self):
        self._oi_cache: Dict[str, Dict] = {}          # {symbol: {data, time}}
        self._oi_history: Dict[str, deque] = {}       # {symbol: deque of {oi, ts}}
        self._price_ma_cache: Dict[str, float] = {}   # {symbol: price_ma}

    def fetch_current_oi(self, exchange, symbol: str) -> Optional[Dict]:
        """
        使用 CCXT 获取当前OI

        Args:
            exchange: CCXT exchange 对象
            symbol: 统一格式交易对，如 'BTC/USDT:USDT'
        """
        # 检查缓存
        if symbol in self._oi_cache:
            cached = self._oi_cache[symbol]
            if datetime.now() - cached['time'] < timedelta(seconds=OI_CACHE_DURATION):
                return cached['data']

        try:
            # 获取底层 CCXT exchange
            ccxt_exchange = getattr(exchange, '_api', exchange)

            # 检查交易所是否支持 fetchOpenInterest
            if not ccxt_exchange.has.get('fetchOpenInterest', False):
                logger.warning(f"交易所不支持 fetchOpenInterest")
                return None

            # 使用 CCXT 统一接口获取当前 OI
            oi_data = ccxt_exchange.fetch_open_interest(symbol)

            # 安全获取数值，处理 None 情况
            oi_amount = oi_data.get('openInterestAmount')
            oi_value = oi_data.get('openInterestValue')
            ts = oi_data.get('timestamp')

            result = {
                'symbol': symbol,
                'open_interest': float(oi_amount) if oi_amount is not None else 0.0,
                'open_interest_value': float(oi_value) if oi_value is not None else 0.0,
                'timestamp': int(ts) if ts is not None else int(datetime.now().timestamp() * 1000)
            }

            # 更新缓存
            self._oi_cache[symbol] = {'data': result, 'time': datetime.now()}

            # 更新历史
            if symbol not in self._oi_history:
                self._oi_history[symbol] = deque(maxlen=12)  # 3小时的15分钟数据
            self._oi_history[symbol].append({
                'oi': result['open_interest'],
                'ts': result['timestamp']
            })

            return result

        except Exception as e:
            logger.error(f"CCXT 获取OI失败 {symbol}: {e}")
            return None

    def fetch_oi_history(self, exchange, symbol: str, timeframe: str = '15m', limit: int = 12) -> List[Dict]:
        """
        使用 CCXT 获取历史OI数据

        Args:
            exchange: CCXT exchange 对象
            symbol: 统一格式交易对
            timeframe: 时间周期 '15m', '1h', '4h', '1d'
            limit: 数据点数量
        """
        try:
            # 获取底层 CCXT exchange
            ccxt_exchange = getattr(exchange, '_api', exchange)

            # 检查交易所是否支持 fetchOpenInterestHistory
            if not ccxt_exchange.has.get('fetchOpenInterestHistory', False):
                logger.warning(f"交易所不支持 fetchOpenInterestHistory")
                return []

            # 使用 CCXT 统一接口获取 OI 历史
            oi_history = ccxt_exchange.fetch_open_interest_history(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            # 转换为统一格式（兼容旧代码的 sumOpenInterest 字段）
            # 安全处理 None 值
            result = []
            for item in oi_history:
                oi_amount = item.get('openInterestAmount')
                oi_value = item.get('openInterestValue')
                ts = item.get('timestamp')
                result.append({
                    'sumOpenInterest': str(oi_amount) if oi_amount is not None else '0',
                    'sumOpenInterestValue': str(oi_value) if oi_value is not None else '0',
                    'timestamp': str(ts) if ts is not None else '0'
                })
            return result

        except Exception as e:
            logger.error(f"CCXT 获取OI历史失败 {symbol}: {e}")
            return []

    def get_oi_alerts(self, exchange, symbol: str, current_price: float,
                      volume_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        主入口 - 返回指定币种的所有OI警报

        参数:
            exchange: CCXT exchange 对象
            symbol: 交易对，如 'BTC/USDT:USDT'
            current_price: 当前价格
            volume_data: 可选的成交量数据 {'volume_ratio': float}

        返回:
            {
                'symbol': str,
                'alerts': List[Dict],  # OI异动和突破信号列表
                'current_oi': float,
                'timestamp': str
            }
        """
        alerts = []

        # 获取当前OI（使用 CCXT）
        current_oi_data = self.fetch_current_oi(exchange, symbol)
        if not current_oi_data:
            return {'symbol': symbol, 'alerts': [], 'error': '无法获取OI数据'}

        # 获取历史OI用于比较（使用 CCXT）
        oi_history = self.fetch_oi_history(exchange, symbol, '15m', 2)
        if len(oi_history) >= 2:
            current_oi = float(oi_history[-1].get('sumOpenInterest', 0))
            previous_oi = float(oi_history[-2].get('sumOpenInterest', 0))

            if previous_oi > 0:
                oi_change_pct = (current_oi - previous_oi) / previous_oi

                # 检测OI异动
                anomaly = self._detect_oi_anomaly(
                    symbol, current_price, oi_change_pct, current_oi, previous_oi
                )
                if anomaly:
                    alerts.append(anomaly)

                # 检测突破信号
                if volume_data:
                    breakout = self._detect_breakout_signal(
                        symbol, volume_data.get('volume_ratio', 1.0), oi_change_pct
                    )
                    if breakout:
                        alerts.append(breakout)

        return {
            'symbol': symbol,
            'alerts': alerts,
            'current_oi': current_oi_data.get('open_interest', 0),
            'timestamp': datetime.now().isoformat()
        }

    def _detect_oi_anomaly(self, symbol: str, current_price: float,
                           oi_change_pct: float, current_oi: float,
                           previous_oi: float) -> Optional[Dict]:
        """
        检测OI异动条件
        条件: OI变化 > 5% AND 价格偏离 > 2%
        """
        # 获取价格MA（如果有缓存）
        baseline_price = self._price_ma_cache.get(symbol, current_price)
        price_deviation_pct = (current_price - baseline_price) / baseline_price if baseline_price > 0 else 0

        # 检查是否同时满足两个条件
        if abs(oi_change_pct) >= OI_CHANGE_THRESHOLD and \
           abs(price_deviation_pct) >= PRICE_DEVIATION_THRESHOLD:

            direction = 'bullish' if price_deviation_pct > 0 else 'bearish'

            return {
                'type': 'oi_anomaly',
                'symbol': symbol,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'oi_change_pct': round(oi_change_pct * 100, 2),
                'price_deviation_pct': round(price_deviation_pct * 100, 2),
                'oi_current': current_oi,
                'oi_previous': previous_oi,
                'baseline_price': baseline_price,
                'current_price': current_price,
                'direction': direction,
                'interpretation': self._interpret_oi_anomaly(
                    oi_change_pct, price_deviation_pct, direction
                )
            }

        return None

    def _detect_breakout_signal(self, symbol: str, volume_ratio: float,
                                 oi_delta_pct: float) -> Optional[Dict]:
        """
        检测突破/假突破信号
        条件: 成交量 > 2x 均量 AND |OI Delta| > 3%

        真突破: 放量 + OI增加 (新仓位进入)
        假突破: 放量 + OI减少 (平仓离场)
        """
        if volume_ratio >= VOLUME_SURGE_RATIO and \
           abs(oi_delta_pct) >= BREAKOUT_OI_THRESHOLD:

            if oi_delta_pct > 0:
                signal = 'breakout'
                interpretation = f"成交量{volume_ratio:.1f}x + OI增{oi_delta_pct*100:.1f}%，真突破 - 新仓位入场"
            else:
                signal = 'fake_breakout'
                interpretation = f"成交量{volume_ratio:.1f}x但OI降{abs(oi_delta_pct)*100:.1f}%，假突破 - 平仓离场"

            return {
                'type': 'breakout_signal',
                'symbol': symbol,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'volume_ratio': round(volume_ratio, 2),
                'oi_delta_pct': round(oi_delta_pct * 100, 2),
                'signal': signal,
                'interpretation': interpretation
            }

        return None

    def _interpret_oi_anomaly(self, oi_change: float, price_dev: float,
                              direction: str) -> str:
        """生成人类可读的OI异动解释"""
        oi_action = "增" if oi_change > 0 else "降"
        price_action = "高于" if price_dev > 0 else "低于"

        return (f"OI{oi_action}{abs(oi_change)*100:.1f}%，价格{price_action}MA "
                f"{abs(price_dev)*100:.1f}%，{direction}动能")

    def update_price_ma(self, symbol: str, price_ma: float):
        """更新价格MA缓存（由context_builder调用）"""
        self._price_ma_cache[symbol] = price_ma


class VolumeReversalMonitor:
    """15分钟成交量反转监控器 - 锚定K线模式"""

    def __init__(self):
        self._anchor_cache: Dict[str, Dict] = {}  # {symbol: anchor_data}

    def get_reversal_signals(self, symbol: str,
                              dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        主入口 - 处理DataFrame并返回反转信号

        参数:
            symbol: 交易对
            dataframe: 15分钟K线数据（需包含 open, high, low, close, volume）

        返回:
            {
                'symbol': str,
                'signal': Optional[Dict],  # Buy/Sell信号
                'anchor': Optional[Dict],  # 当前锚定K线
                'status': str  # pending/confirmed/expired
            }
        """
        if dataframe is None or dataframe.empty or len(dataframe) < BB_PERIOD + 5:
            return {'symbol': symbol, 'signal': None, 'anchor': None}

        # 计算布林带（如果不存在）
        if 'bb_upper' not in dataframe.columns:
            dataframe = self._calculate_bollinger_bands(dataframe)

        # 计算成交量MA（如果不存在）
        if 'volume_ma' not in dataframe.columns:
            dataframe['volume_ma'] = dataframe['volume'].rolling(window=20).mean()

        # 获取最新K线
        latest = dataframe.iloc[-1]

        # 检查现有锚定
        if symbol in self._anchor_cache:
            anchor = self._anchor_cache[symbol]

            # 检查反转确认
            confirmation = self._check_reversal_confirmation(symbol, latest, anchor)
            if confirmation:
                # 确认后清除锚定
                del self._anchor_cache[symbol]
                return confirmation

            # 更新经过的K线数
            anchor['candles_elapsed'] += 1

            # 检查是否超时
            if anchor['candles_elapsed'] >= anchor['confirmation_deadline']:
                logger.debug(f"{symbol} 锚定超时未确认")
                del self._anchor_cache[symbol]
                return {'symbol': symbol, 'signal': None, 'anchor': None,
                        'status': 'anchor_expired'}

        # 尝试识别新锚定
        new_anchor = self._identify_anchor(symbol, latest, dataframe)
        if new_anchor:
            self._anchor_cache[symbol] = new_anchor
            return {
                'symbol': symbol,
                'signal': None,
                'anchor': new_anchor,
                'status': 'anchor_identified',
                'interpretation': f"锚定K线识别: {new_anchor['anchor_type']}，监控反转中"
            }

        return {'symbol': symbol, 'signal': None, 'anchor': None}

    def _calculate_bollinger_bands(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        dataframe['bb_middle'] = dataframe['close'].rolling(window=BB_PERIOD).mean()
        rolling_std = dataframe['close'].rolling(window=BB_PERIOD).std()
        dataframe['bb_upper'] = dataframe['bb_middle'] + (BB_STD * rolling_std)
        dataframe['bb_lower'] = dataframe['bb_middle'] - (BB_STD * rolling_std)
        return dataframe

    def _identify_anchor(self, symbol: str, candle: pd.Series,
                         dataframe: pd.DataFrame) -> Optional[Dict]:
        """
        识别锚定K线
        条件: 价格突破布林带 + 成交量 > 1.5x 均量
        """
        try:
            close = candle['close']
            bb_upper = candle.get('bb_upper', 0)
            bb_lower = candle.get('bb_lower', 0)
            volume = candle['volume']
            volume_ma = candle.get('volume_ma', volume)

            # 检查成交量要求
            volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
            if volume_ratio < MIN_ANCHOR_VOLUME_RATIO:
                return None

            # 检查布林带突破
            anchor_type = None
            if close > bb_upper:
                anchor_type = 'upper_extreme'
            elif close < bb_lower:
                anchor_type = 'lower_extreme'
            else:
                return None

            # 获取K线时间戳
            candle_time = candle.get('date', candle.name)
            if isinstance(candle_time, pd.Timestamp):
                timestamp = int(candle_time.timestamp() * 1000)
            else:
                timestamp = int(datetime.now().timestamp() * 1000)

            return {
                'symbol': symbol,
                'anchor_time': timestamp,
                'anchor_type': anchor_type,
                'anchor_open': float(candle['open']),
                'anchor_high': float(candle['high']),
                'anchor_low': float(candle['low']),
                'anchor_close': float(close),
                'anchor_volume': float(volume),
                'volume_ratio': round(volume_ratio, 2),
                'bb_upper': float(bb_upper),
                'bb_lower': float(bb_lower),
                'confirmation_deadline': CONFIRMATION_WINDOW,
                'candles_elapsed': 0,
                'status': 'pending'
            }

        except Exception as e:
            logger.error(f"识别锚定K线失败 {symbol}: {e}")
            return None

    def _check_reversal_confirmation(self, symbol: str, candle: pd.Series,
                                      anchor: Dict) -> Optional[Dict]:
        """
        检查反转确认
        - 上极端锚定 → 价格跌破 anchor_low → SELL
        - 下极端锚定 → 价格突破 anchor_high → BUY
        """
        try:
            current_low = candle['low']
            current_high = candle['high']
            current_close = candle['close']

            # 上极端: 价格需跌破锚定K线低点
            if anchor['anchor_type'] == 'upper_extreme':
                if current_low < anchor['anchor_low']:
                    reversal_size = (anchor['anchor_close'] - current_close) / anchor['anchor_close'] * 100
                    confidence = self._calculate_confidence(anchor, reversal_size)

                    return {
                        'type': 'volume_reversal',
                        'symbol': symbol,
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'signal': 'sell',
                        'anchor_time': anchor['anchor_time'],
                        'anchor_close': anchor['anchor_close'],
                        'confirmation_candle': anchor['candles_elapsed'] + 1,
                        'current_price': float(current_close),
                        'reversal_size_pct': round(reversal_size, 2),
                        'anchor_volume_ratio': anchor['volume_ratio'],
                        'confidence': confidence,
                        'interpretation': f"上轨突破后第{anchor['candles_elapsed']+1}根K线反转，{confidence}卖出信号"
                    }

            # 下极端: 价格需突破锚定K线高点
            elif anchor['anchor_type'] == 'lower_extreme':
                if current_high > anchor['anchor_high']:
                    reversal_size = (current_close - anchor['anchor_close']) / anchor['anchor_close'] * 100
                    confidence = self._calculate_confidence(anchor, reversal_size)

                    return {
                        'type': 'volume_reversal',
                        'symbol': symbol,
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'signal': 'buy',
                        'anchor_time': anchor['anchor_time'],
                        'anchor_close': anchor['anchor_close'],
                        'confirmation_candle': anchor['candles_elapsed'] + 1,
                        'current_price': float(current_close),
                        'reversal_size_pct': round(reversal_size, 2),
                        'anchor_volume_ratio': anchor['volume_ratio'],
                        'confidence': confidence,
                        'interpretation': f"下轨突破后第{anchor['candles_elapsed']+1}根K线反转，{confidence}买入信号"
                    }

            return None

        except Exception as e:
            logger.error(f"检查反转确认失败 {symbol}: {e}")
            return None

    def _calculate_confidence(self, anchor: Dict, reversal_size: float) -> str:
        """
        计算信号置信度
        基于: 锚定成交量、确认速度、反转幅度
        """
        score = 0

        # 成交量因子（锚定成交量越大越可信）
        if anchor['volume_ratio'] >= 2.5:
            score += 2
        elif anchor['volume_ratio'] >= 2.0:
            score += 1

        # 速度因子（越快确认越可信）
        if anchor['candles_elapsed'] == 0:  # 同一根K线
            score += 2
        elif anchor['candles_elapsed'] == 1:
            score += 1

        # 反转幅度因子
        if abs(reversal_size) >= 2.0:
            score += 1

        if score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'

    def get_active_anchors(self) -> Dict[str, Dict]:
        """获取所有活跃锚定（用于调试/监控）"""
        return self._anchor_cache.copy()


class OrderBookAnalyzer:
    """盘口深度分析器 - 分析买卖盘深度、流动性墙、盘口失衡"""

    def __init__(self, config: Optional[Dict] = None):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, datetime] = {}

        # 从配置加载参数
        cfg = config or {}
        self.cache_duration = cfg.get('cache_seconds', ORDERBOOK_CACHE_DURATION)
        self.depth = cfg.get('depth', ORDERBOOK_DEPTH)
        self.wall_multiplier = cfg.get('wall_threshold_multiplier', WALL_THRESHOLD_MULTIPLIER)
        self.min_wall_size = cfg.get('min_wall_size_usdt', MIN_WALL_SIZE_USDT)

    def analyze(self, orderbook: Dict, pair: str, current_price: float) -> Optional[Dict]:
        """
        分析盘口深度

        参数:
            orderbook: Freqtrade dp.orderbook() 返回的数据
                       格式: {'bids': [[price, amount], ...], 'asks': [[price, amount], ...]}
            pair: 交易对
            current_price: 当前价格

        返回:
            {
                'bid_depth': float,        # 买盘总深度 (USDT)
                'ask_depth': float,        # 卖盘总深度 (USDT)
                'imbalance_ratio': float,  # 买卖失衡比 (>1 买强, <1 卖强)
                'bid_walls': List[Dict],   # 买盘流动性墙
                'ask_walls': List[Dict],   # 卖盘流动性墙
                'spread_pct': float,       # 买卖价差百分比
                'best_bid': float,
                'best_ask': float,
                'interpretation': str
            }
        """
        # 检查缓存
        if pair in self._cache and pair in self._cache_time:
            if datetime.now() - self._cache_time[pair] < timedelta(seconds=self.cache_duration):
                return self._cache[pair]

        try:
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return None

            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return None

            # 计算买卖盘深度 (USDT)
            bid_depth = sum(price * amount for price, amount in bids)
            ask_depth = sum(price * amount for price, amount in asks)

            # 计算失衡比
            imbalance_ratio = bid_depth / ask_depth if ask_depth > 0 else 999

            # 最佳买卖价
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0

            # 价差
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0

            # 检测流动性墙
            bid_walls = self._detect_walls(bids, 'bid', current_price)
            ask_walls = self._detect_walls(asks, 'ask', current_price)

            # 生成解释
            interpretation = self._interpret_orderbook(
                imbalance_ratio, bid_walls, ask_walls, spread_pct
            )

            # 计算压力方向
            if imbalance_ratio > 1.2:
                pressure = 'buy'
            elif imbalance_ratio < 0.8:
                pressure = 'sell'
            else:
                pressure = 'neutral'

            # 合并买卖墙为统一列表，添加side字段
            liquidity_walls = []
            for wall in bid_walls:
                wall_copy = wall.copy()
                wall_copy['side'] = 'bid'
                wall_copy['size'] = wall_copy.pop('size_usdt', 0)
                wall_copy['pct_of_side'] = wall_copy.pop('size_ratio', 0) * 10  # 近似百分比
                liquidity_walls.append(wall_copy)
            for wall in ask_walls:
                wall_copy = wall.copy()
                wall_copy['side'] = 'ask'
                wall_copy['size'] = wall_copy.pop('size_usdt', 0)
                wall_copy['pct_of_side'] = wall_copy.pop('size_ratio', 0) * 10
                liquidity_walls.append(wall_copy)
            # 按规模排序
            liquidity_walls.sort(key=lambda x: x['size'], reverse=True)

            result = {
                'bid_volume': round(bid_depth, 2),  # context_builder 期望的字段名
                'ask_volume': round(ask_depth, 2),
                'imbalance_ratio': round(imbalance_ratio, 3),
                'pressure': pressure,  # 新增：压力方向
                'liquidity_walls': liquidity_walls,  # 合并后的流动性墙列表
                'bid_walls': bid_walls,  # 保留原始数据
                'ask_walls': ask_walls,
                'spread_pct': round(spread_pct, 4),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'interpretation': interpretation
            }

            # 更新缓存
            self._cache[pair] = result
            self._cache_time[pair] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"分析盘口深度失败 {pair}: {e}")
            return None

    def _detect_walls(self, orders: List, side: str, current_price: float) -> List[Dict]:
        """检测流动性墙 (异常大单)"""
        if not orders or len(orders) < 3:
            return []

        walls = []

        # 计算平均订单规模
        order_sizes = [price * amount for price, amount in orders]
        avg_size = sum(order_sizes) / len(order_sizes) if order_sizes else 0

        for price, amount in orders:
            size_usdt = price * amount
            # 检测条件: 规模 > 平均的N倍 且 > 最小规模
            if size_usdt > avg_size * self.wall_multiplier and size_usdt >= self.min_wall_size:
                # 计算距离当前价格的百分比
                distance_pct = abs(price - current_price) / current_price * 100

                walls.append({
                    'price': price,
                    'size_usdt': round(size_usdt, 2),
                    'size_ratio': round(size_usdt / avg_size, 1),
                    'distance_pct': round(distance_pct, 2),
                    'side': side
                })

        # 按规模排序，返回最大的3个
        walls.sort(key=lambda x: x['size_usdt'], reverse=True)
        return walls[:3]

    def _interpret_orderbook(self, imbalance: float, bid_walls: List,
                              ask_walls: List, spread: float) -> str:
        """生成盘口解释"""
        parts = []

        # 失衡分析
        if imbalance > 1.5:
            parts.append(f"买盘强势({imbalance:.2f})")
        elif imbalance < 0.67:
            parts.append(f"卖盘强势({imbalance:.2f})")
        else:
            parts.append(f"买卖平衡({imbalance:.2f})")

        # 流动性墙
        if bid_walls:
            wall = bid_walls[0]
            parts.append(f"买墙@{wall['price']:.2f}({wall['size_usdt']/1000:.0f}K)")
        if ask_walls:
            wall = ask_walls[0]
            parts.append(f"卖墙@{wall['price']:.2f}({wall['size_usdt']/1000:.0f}K)")

        # 价差
        if spread > 0.1:
            parts.append(f"价差较大({spread:.3f}%)")

        return ", ".join(parts) if parts else "盘口正常"


class OrderFlowTracker:
    """订单流分析器 - 分析成交记录、大单检测、主动买卖比"""

    def __init__(self, config: Optional[Dict] = None):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, datetime] = {}

        # 从配置加载参数
        cfg = config or {}
        self.cache_duration = cfg.get('cache_seconds', ORDERFLOW_CACHE_DURATION)
        self.large_trade_threshold = cfg.get('large_trade_threshold_usdt', LARGE_TRADE_THRESHOLD_USDT)
        self.trade_limit = cfg.get('trade_limit', TRADE_FETCH_LIMIT)

    def analyze(self, exchange, pair: str) -> Optional[Dict]:
        """
        分析近期成交记录

        参数:
            exchange: Freqtrade exchange 对象
            pair: 交易对

        返回:
            {
                'buy_volume': float,       # 主动买入量 (USDT)
                'sell_volume': float,      # 主动卖出量 (USDT)
                'buy_sell_ratio': float,   # 买卖比 (>1 买压强)
                'large_trades': List[Dict], # 大单列表
                'trade_count': int,        # 成交笔数
                'trade_frequency': float,  # 成交频率 (笔/分钟)
                'avg_trade_size': float,   # 平均成交额
                'interpretation': str
            }
        """
        # 检查缓存
        if pair in self._cache and pair in self._cache_time:
            if datetime.now() - self._cache_time[pair] < timedelta(seconds=self.cache_duration):
                return self._cache[pair]

        try:
            # 获取近期成交记录 - 使用底层 CCXT exchange
            # Freqtrade 的 exchange 包装器通过 _api 属性访问底层 CCXT
            ccxt_exchange = getattr(exchange, '_api', exchange)
            trades = ccxt_exchange.fetch_trades(pair, limit=self.trade_limit)

            if not trades:
                return None

            # 分析成交数据
            buy_volume = 0.0
            sell_volume = 0.0
            large_trades = []
            timestamps = []
            large_buy_volume = 0.0
            large_sell_volume = 0.0

            for trade in trades:
                price = float(trade.get('price', 0))
                amount = float(trade.get('amount', 0))
                side = trade.get('side', 'buy')
                timestamp = trade.get('timestamp', 0)
                trade_value = price * amount

                timestamps.append(timestamp)

                if side == 'buy':
                    buy_volume += trade_value
                else:
                    sell_volume += trade_value

                # 检测大单
                if trade_value >= self.large_trade_threshold:
                    large_trades.append({
                        'side': side.lower(),  # context_builder 期望小写
                        'price': price,
                        'amount': amount,
                        'value': round(trade_value, 2),  # context_builder 期望 'value'
                        'timestamp': timestamp
                    })
                    # 累计大单买卖量
                    if side == 'buy':
                        large_buy_volume += trade_value
                    else:
                        large_sell_volume += trade_value

            # 计算买卖比
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 999

            # 计算净流入和方向
            net_flow = buy_volume - sell_volume
            if net_flow > buy_volume * 0.1:  # 净买入超过10%
                flow_direction = 'buy'
            elif net_flow < -sell_volume * 0.1:  # 净卖出超过10%
                flow_direction = 'sell'
            else:
                flow_direction = 'neutral'

            # 计算成交频率
            trade_count = len(trades)
            if timestamps and len(timestamps) >= 2:
                time_span_minutes = (max(timestamps) - min(timestamps)) / 60000
                trade_frequency = trade_count / time_span_minutes if time_span_minutes > 0 else 0
            else:
                trade_frequency = 0

            # 平均成交额
            total_volume = buy_volume + sell_volume
            avg_trade_size = total_volume / trade_count if trade_count > 0 else 0

            # 按时间排序大单（最新在前）
            large_trades.sort(key=lambda x: x['timestamp'], reverse=True)

            # 生成解释
            interpretation = self._interpret_orderflow(
                buy_sell_ratio, large_trades, trade_frequency
            )

            result = {
                'buy_volume': round(buy_volume, 2),
                'sell_volume': round(sell_volume, 2),
                'buy_sell_ratio': round(buy_sell_ratio, 3),
                'net_flow': round(net_flow, 2),  # 新增：净流入
                'flow_direction': flow_direction,  # 新增：流动方向
                'large_trades': large_trades[:5],  # 只保留最近5笔
                'large_buy_volume': round(large_buy_volume, 2),  # 新增
                'large_sell_volume': round(large_sell_volume, 2),  # 新增
                'trade_count': trade_count,
                'trade_frequency': round(trade_frequency, 2),
                'avg_trade_size': round(avg_trade_size, 2),
                'interpretation': interpretation
            }

            # 更新缓存
            self._cache[pair] = result
            self._cache_time[pair] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"分析订单流失败 {pair}: {e}")
            return None

    def _interpret_orderflow(self, ratio: float, large_trades: List,
                              frequency: float) -> str:
        """生成订单流解释"""
        parts = []

        # 买卖比分析
        if ratio > 2.0:
            parts.append("强力买入")
        elif ratio > 1.3:
            parts.append("买入占优")
        elif ratio < 0.5:
            parts.append("强力卖出")
        elif ratio < 0.77:
            parts.append("卖出占优")
        else:
            parts.append("买卖平衡")

        # 大单分析
        if large_trades:
            buy_count = sum(1 for t in large_trades if t['side'] == 'BUY')
            sell_count = len(large_trades) - buy_count
            parts.append(f"大单: {buy_count}买/{sell_count}卖")

        # 活跃度
        if frequency > 20:
            parts.append("高频交易")
        elif frequency < 2:
            parts.append("低频交易")

        return ", ".join(parts) if parts else "订单流正常"


class EnhancedOIAnalyzer:
    """增强型 OI 分析器 - 多周期趋势分析"""

    def __init__(self, oi_monitor: OIMonitor, config: Optional[Dict] = None):
        self.oi_monitor = oi_monitor
        self._trend_cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, datetime] = {}

        cfg = config or {}
        self.cache_duration = cfg.get('cache_seconds', 300)
        self.divergence_threshold = cfg.get('divergence_threshold', 0.03)

    def get_multi_timeframe_trend(self, exchange, symbol: str,
                                   current_price: float,
                                   price_change_24h: Optional[float] = None) -> Optional[Dict]:
        """
        获取多周期 OI 趋势

        参数:
            exchange: CCXT exchange 对象
            symbol: 交易对（统一格式，如 'BTC/USDT:USDT'）
            current_price: 当前价格
            price_change_24h: 24小时价格变化百分比

        返回:
            {
                'trend_15m': str,      # rising/falling/stable
                'trend_4h': str,
                'trend_daily': str,
                'oi_change_15m': float,
                'oi_change_4h': float,
                'oi_change_daily': float,
                'current_oi': float,
                'divergence': Optional[str],  # price_up_oi_down / price_down_oi_up
                'interpretation': str
            }
        """
        # 检查缓存
        if symbol in self._trend_cache and symbol in self._cache_time:
            if datetime.now() - self._cache_time[symbol] < timedelta(seconds=self.cache_duration):
                return self._trend_cache[symbol]

        try:
            # 使用 CCXT 获取不同周期的OI历史（无需转换symbol格式）
            oi_15m = self._fetch_oi_history(exchange, symbol, '15m', 4)
            oi_4h = self._fetch_oi_history(exchange, symbol, '4h', 6)
            oi_daily = self._fetch_oi_history(exchange, symbol, '1d', 7)

            # 分析各周期趋势
            trend_15m, change_15m = self._analyze_trend(oi_15m)
            trend_4h, change_4h = self._analyze_trend(oi_4h)
            trend_daily, change_daily = self._analyze_trend(oi_daily)

            # 获取当前OI（使用 CCXT）
            current_oi_data = self.oi_monitor.fetch_current_oi(exchange, symbol)
            current_oi = current_oi_data['open_interest'] if current_oi_data else 0

            # 检测OI-价格背离
            divergence = None
            if price_change_24h is not None and change_daily != 0:
                divergence = self._detect_divergence(price_change_24h, change_daily)

            # 生成解释
            interpretation = self._interpret_multi_tf(
                trend_15m, trend_4h, trend_daily, divergence
            )

            result = {
                'trend_15m': trend_15m,
                'trend_4h': trend_4h,
                'trend_daily': trend_daily,
                'oi_change_15m': round(change_15m * 100, 2),
                'oi_change_4h': round(change_4h * 100, 2),
                'oi_change_daily': round(change_daily * 100, 2),
                'current_oi': current_oi,
                'divergence': divergence,
                'interpretation': interpretation
            }

            # 更新缓存
            self._trend_cache[symbol] = result
            self._cache_time[symbol] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"获取多周期OI趋势失败 {symbol}: {e}")
            return None

    def _fetch_oi_history(self, exchange, symbol: str, timeframe: str, limit: int) -> List[float]:
        """
        使用 CCXT 获取指定周期的OI历史数据

        Args:
            exchange: CCXT exchange 对象
            symbol: 统一格式交易对，如 'BTC/USDT:USDT'
            timeframe: 时间周期 '15m', '4h', '1d' 等
            limit: 获取数据点数量

        Returns:
            OI值列表（按时间正序，旧->新）
        """
        try:
            # 获取底层 CCXT exchange
            ccxt_exchange = getattr(exchange, '_api', exchange)

            # 检查交易所是否支持 fetchOpenInterestHistory
            if not ccxt_exchange.has.get('fetchOpenInterestHistory', False):
                logger.warning(f"交易所不支持 fetchOpenInterestHistory")
                return []

            # 使用 CCXT 统一接口获取 OI 历史
            oi_history = ccxt_exchange.fetch_open_interest_history(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            # 提取 openInterestAmount（以币为单位），安全处理 None
            result = []
            for item in oi_history:
                oi_amount = item.get('openInterestAmount')
                if oi_amount is not None:
                    result.append(float(oi_amount))
                else:
                    result.append(0.0)
            return result

        except Exception as e:
            logger.warning(f"CCXT 获取OI历史失败 {symbol} {timeframe}: {e}")
            return []

    def _analyze_trend(self, oi_values: List[float]) -> tuple:
        """分析OI趋势"""
        if not oi_values or len(oi_values) < 2:
            return 'unknown', 0.0

        latest = oi_values[-1]
        earliest = oi_values[0]

        if earliest == 0:
            return 'unknown', 0.0

        change = (latest - earliest) / earliest

        if change > self.divergence_threshold:
            return 'rising', change
        elif change < -self.divergence_threshold:
            return 'falling', change
        else:
            return 'stable', change

    def _detect_divergence(self, price_change: float, oi_change: float) -> Optional[str]:
        """检测价格与OI背离"""
        price_up = price_change > self.divergence_threshold * 100
        price_down = price_change < -self.divergence_threshold * 100
        oi_up = oi_change > self.divergence_threshold
        oi_down = oi_change < -self.divergence_threshold

        if price_up and oi_down:
            return "price_up_oi_down"  # 价涨OI降：空头平仓拉升，假突破风险
        elif price_down and oi_up:
            return "price_down_oi_up"  # 价跌OI升：新空开仓，下跌动能增强
        return None

    def _interpret_multi_tf(self, t15m: str, t4h: str, tdaily: str,
                            divergence: Optional[str]) -> str:
        """生成多周期解释"""
        parts = []

        # 趋势一致性
        trends = [t15m, t4h, tdaily]
        rising_count = trends.count('rising')
        falling_count = trends.count('falling')

        if rising_count == 3:
            parts.append("全周期OI上升，仓位持续增加")
        elif falling_count == 3:
            parts.append("全周期OI下降，仓位持续减少")
        elif rising_count >= 2:
            parts.append("OI整体上升趋势")
        elif falling_count >= 2:
            parts.append("OI整体下降趋势")

        # 背离信号
        if divergence == 'price_up_oi_down':
            parts.append("⚠️ 价涨OI降：空头平仓推升，假突破风险")
        elif divergence == 'price_down_oi_up':
            parts.append("⚠️ 价跌OI升：新空入场，下跌动能增强")

        return ", ".join(parts) if parts else "OI趋势正常"


class LiquidationTracker:
    """清算数据追踪器 - WebSocket 后台收集"""

    def __init__(self, config: Optional[Dict] = None):
        self._buffers: Dict[str, deque] = {}  # {symbol: deque of liquidations}
        self._running = False
        self._thread = None
        self._symbols: List[str] = []

        cfg = config or {}
        self.buffer_minutes = cfg.get('buffer_minutes', LIQUIDATION_BUFFER_MINUTES)
        self.significant_threshold = cfg.get('significant_liq_usdt', SIGNIFICANT_LIQ_USDT)
        self.imbalance_alert_ratio = cfg.get('imbalance_alert_ratio', LIQ_IMBALANCE_ALERT_RATIO)

    def start(self, symbols: List[str]):
        """启动后台清算收集"""
        if self._running:
            return

        self._symbols = [s.replace('/USDT:USDT', 'USDT').replace('/', '') for s in symbols]
        for s in self._symbols:
            self._buffers[s] = deque(maxlen=1000)

        self._running = True
        self._thread = Thread(target=self._run_collector, daemon=True)
        self._thread.start()
        logger.info(f"清算收集器已启动，监控币种: {self._symbols}")

    def stop(self):
        """停止收集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("清算收集器已停止")

    def _run_collector(self):
        """运行收集循环"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._collect_loop())

    async def _collect_loop(self):
        """WebSocket 收集循环"""
        import websockets

        uri = "wss://fstream.binance.com/ws/!forceOrder@arr"

        while self._running:
            try:
                async with websockets.connect(uri) as ws:
                    logger.info("清算 WebSocket 已连接")
                    while self._running:
                        try:
                            import asyncio as aio
                            msg = await aio.wait_for(ws.recv(), timeout=30)
                            self._process_liquidation(msg)
                        except Exception:
                            # 超时或其他错误，继续循环
                            pass
            except Exception as e:
                if self._running:
                    logger.warning(f"清算 WebSocket 断开: {e}, 5秒后重连")
                    import asyncio as aio
                    await aio.sleep(5)

    def _process_liquidation(self, msg: str):
        """处理清算消息"""
        import json
        try:
            data = json.loads(msg)
            order = data.get('o', {})
            symbol = order.get('s', '')

            if symbol in self._buffers:
                # SELL = 多头被清算，BUY = 空头被清算
                side = order.get('S', '')
                price = float(order.get('p', 0))
                qty = float(order.get('q', 0))
                value = price * qty

                self._buffers[symbol].append({
                    'side': 'long' if side == 'SELL' else 'short',
                    'price': price,
                    'qty': qty,
                    'value_usdt': value,
                    'time': int(order.get('T', 0))
                })
        except Exception as e:
            logger.debug(f"处理清算消息失败: {e}")

    def get_recent_liquidations(self, pair: str) -> Optional[Dict]:
        """
        获取近期清算数据

        返回:
            {
                'total_long_liq': float,     # 多头清算总量 (USDT)
                'total_short_liq': float,    # 空头清算总量 (USDT)
                'long_count': int,           # 多头清算笔数
                'short_count': int,          # 空头清算笔数
                'largest_liq': Optional[Dict], # 最大清算
                'liq_imbalance': float,      # 清算失衡比
                'interpretation': str
            }
        """
        try:
            symbol = pair.replace('/USDT:USDT', 'USDT').replace('/', '')

            if symbol not in self._buffers:
                return {
                    'total_count': 0,
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'long_volume': 0,
                    'short_volume': 0,
                    'imbalance_ratio': 1.0,
                    'significant_liquidations': [],
                    'interpretation': f"近{self.buffer_minutes}分钟无清算"
                }

            # 获取时间窗口内的清算
            cutoff = int(datetime.now().timestamp() * 1000) - (self.buffer_minutes * 60 * 1000)
            recent = [l for l in self._buffers[symbol] if l['time'] >= cutoff]

            if not recent:
                return {
                    'total_count': 0,
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'long_volume': 0,
                    'short_volume': 0,
                    'imbalance_ratio': 1.0,
                    'significant_liquidations': [],
                    'interpretation': f"近{self.buffer_minutes}分钟无清算"
                }

            # 统计
            long_liqs = [l for l in recent if l['side'] == 'long']
            short_liqs = [l for l in recent if l['side'] == 'short']

            total_long = sum(l['value_usdt'] for l in long_liqs)
            total_short = sum(l['value_usdt'] for l in short_liqs)

            # 计算失衡比
            if total_short > 0:
                imbalance = total_long / total_short
            elif total_long > 0:
                imbalance = 999
            else:
                imbalance = 1.0

            # 筛选显著清算事件（超过阈值的）
            significant = []
            for l in recent:
                if l['value_usdt'] >= self.significant_threshold:
                    significant.append({
                        'side': l['side'],
                        'price': l['price'],
                        'value': round(l['value_usdt'], 2),
                        'timestamp': l['time']
                    })
            # 按时间排序
            significant.sort(key=lambda x: x['timestamp'], reverse=True)

            # 生成解释
            interpretation = self._interpret_liquidations(
                total_long, total_short, len(long_liqs), len(short_liqs)
            )

            return {
                'total_count': len(recent),
                'long_liquidations': len(long_liqs),
                'short_liquidations': len(short_liqs),
                'long_volume': round(total_long, 2),
                'short_volume': round(total_short, 2),
                'imbalance_ratio': round(imbalance, 2),
                'significant_liquidations': significant[:5],  # 最多5个
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"获取清算数据失败 {pair}: {e}")
            return None

    def _interpret_liquidations(self, long_liq: float, short_liq: float,
                                 long_count: int, short_count: int) -> str:
        """解释清算数据"""
        if long_liq == 0 and short_liq == 0:
            return "无显著清算"

        parts = []

        # 清算失衡分析
        if long_liq > short_liq * self.imbalance_alert_ratio:
            parts.append(f"多头大量清算({long_liq/1000:.0f}K)，可能触发空头挤压反弹")
        elif short_liq > long_liq * self.imbalance_alert_ratio:
            parts.append(f"空头大量清算({short_liq/1000:.0f}K)，可能加速下跌")
        else:
            parts.append(f"清算相对平衡(多:{long_liq/1000:.0f}K/空:{short_liq/1000:.0f}K)")

        return ", ".join(parts)

    def is_running(self) -> bool:
        """检查收集器是否运行中"""
        return self._running


class MarketSentiment:
    """市场情绪数据获取器 - 包含完整的市场微观结构数据"""

    def __init__(self, config: Optional[Dict] = None):
        self.fear_greed_cache = None
        self.fear_greed_cache_time = None
        self.long_short_cache = {}  # {pair: {data, time}}
        self.cache_duration = 300  # 5分钟缓存

        # 获取配置
        ms_config = config or {}
        orderbook_config = ms_config.get('orderbook', {})
        orderflow_config = ms_config.get('orderflow', {})
        oi_trend_config = ms_config.get('oi_trend', {})
        liquidation_config = ms_config.get('liquidation', {})

        # 原有监控器
        self.oi_monitor = OIMonitor()
        self.reversal_monitor = VolumeReversalMonitor()

        # 新增分析器
        self.orderbook_analyzer = OrderBookAnalyzer(orderbook_config)
        self.orderflow_tracker = OrderFlowTracker(orderflow_config)
        self.enhanced_oi_analyzer = EnhancedOIAnalyzer(self.oi_monitor, oi_trend_config)
        self.liquidation_tracker = LiquidationTracker(liquidation_config)

        # 配置开关
        self.orderbook_enabled = orderbook_config.get('enabled', True)
        self.orderflow_enabled = orderflow_config.get('enabled', True)
        self.oi_trend_enabled = oi_trend_config.get('enabled', True)
        self.liquidation_enabled = liquidation_config.get('enabled', True)

    def get_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """
        获取加密货币恐惧与贪婪指数
        来源: Alternative.me (免费API)

        返回:
        {
            'value': 50,  # 0-100
            'classification': 'Neutral',  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
            'timestamp': 1234567890,
            'trend': 'rising'  # rising, falling, stable
        }
        """
        # 检查缓存
        if self.fear_greed_cache and self.fear_greed_cache_time:
            if datetime.now() - self.fear_greed_cache_time < timedelta(seconds=self.cache_duration):
                return self.fear_greed_cache

        try:
            # Alternative.me Fear & Greed Index API
            # 获取最近30天数据，让模型看到完整的情绪变化历史
            url = "https://api.alternative.me/fng/?limit=30"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            if 'data' not in data or len(data['data']) == 0:
                logger.warning("Fear & Greed Index API 返回空数据")
                return None

            # 当前值
            current = data['data'][0]
            value = int(current['value'])
            classification = current['value_classification']
            timestamp = int(current['timestamp'])

            # 计算短期趋势（最近2天）
            trend = 'stable'
            if len(data['data']) >= 2:
                previous = int(data['data'][1]['value'])
                diff = value - previous
                if diff > 5:
                    trend = 'rising'
                elif diff < -5:
                    trend = 'falling'

            # 构建完整历史（带时间戳和分类）
            history_with_time = []
            for d in data['data']:
                history_with_time.append({
                    'value': int(d['value']),
                    'classification': d['value_classification'],
                    'timestamp': int(d['timestamp']),
                    'date': datetime.fromtimestamp(int(d['timestamp'])).strftime('%Y-%m-%d')
                })

            # 分析情绪持续时间
            duration_days = 0
            history_values = [int(d['value']) for d in data['data']]

            # 判断当前情绪持续了多少天（同一区间：极度恐惧<25, 恐惧25-45, 中性45-55, 贪婪55-75, 极度贪婪>75）
            def get_zone(v):
                if v < 25: return 'extreme_fear'
                elif v < 45: return 'fear'
                elif v < 55: return 'neutral'
                elif v < 75: return 'greed'
                else: return 'extreme_greed'

            current_zone = get_zone(value)
            for v in history_values:
                if get_zone(v) == current_zone:
                    duration_days += 1
                else:
                    break

            # 计算30天变化趋势
            if len(history_values) >= 7:
                week_change = history_values[0] - history_values[6]
                week_trend = 'rising' if week_change > 10 else ('falling' if week_change < -10 else 'stable')
            else:
                week_trend = trend

            if len(history_values) >= 30:
                month_change = history_values[0] - history_values[-1]
                month_trend = 'rising' if month_change > 15 else ('falling' if month_change < -15 else 'stable')
            else:
                month_trend = week_trend

            result = {
                'value': value,
                'classification': classification,
                'timestamp': timestamp,
                'trend': trend,  # 短期趋势（1-2天）
                'week_trend': week_trend,  # 周趋势
                'month_trend': month_trend,  # 月趋势
                'duration_days': duration_days,  # 当前情绪持续天数
                'history_30d': history_with_time,  # 最近30天历史（带时间戳）
                'history_values': history_values,  # 仅数值（用于快速趋势判断）
                'interpretation': self._interpret_fear_greed(value, trend, duration_days)
            }

            # 更新缓存
            self.fear_greed_cache = result
            self.fear_greed_cache_time = datetime.now()

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"获取 Fear & Greed Index 失败: {e}")
            return None
        except Exception as e:
            logger.error(f"处理 Fear & Greed Index 数据失败: {e}")
            return None

    def get_funding_rate(self, exchange, pair: str) -> Optional[Dict[str, Any]]:
        """
        获取资金费率

        参数:
            exchange: Freqtrade exchange 对象
            pair: 交易对，如 'BTC/USDT:USDT'

        返回:
        {
            'rate': 0.0001,  # 当前资金费率
            'rate_pct': 0.01,  # 百分比形式
            'next_funding_time': 1234567890,
            'interpretation': '多头极度过热(0.010%，多头付费给空头)'
        }
        """
        try:
            # 使用 ccxt 的 fetch_funding_rate 方法
            if hasattr(exchange, 'fetch_funding_rate'):
                funding_info = exchange.fetch_funding_rate(pair)

                if not funding_info or 'fundingRate' not in funding_info:
                    logger.warning(f"无法获取 {pair} 的资金费率")
                    return None

                rate = float(funding_info['fundingRate'])
                rate_pct = rate * 100  # 转换为百分比

                next_funding_time = funding_info.get('fundingTimestamp', 0)

                result = {
                    'rate': rate,
                    'rate_pct': rate_pct,
                    'next_funding_time': next_funding_time,
                    'interpretation': self._interpret_funding_rate(rate_pct)
                }

                return result
            else:
                logger.warning(f"交易所不支持 fetch_funding_rate")
                return None

        except Exception as e:
            logger.error(f"获取资金费率失败 {pair}: {e}")
            return None

    def _interpret_fear_greed(self, value: int, trend: str, duration_days: int = 0) -> str:
        """解释恐惧与贪婪指数 - 仅提供客观描述，不做主观判断"""
        interpretations = []

        # 当前状态 - 只描述情绪，不提供交易建议
        if value <= 20:
            interpretations.append("极度恐惧")
        elif value <= 40:
            interpretations.append("恐惧")
        elif value <= 60:
            interpretations.append("中性")
        elif value <= 80:
            interpretations.append("贪婪")
        else:
            interpretations.append("极度贪婪")

        # 持续时间分析 - 客观描述
        if duration_days > 0:
            interpretations.append(f"已持续{duration_days}天")
            if duration_days >= 5:
                if value <= 40:
                    interpretations.append("长期处于恐惧状态")
                elif value >= 60:
                    interpretations.append("长期处于贪婪状态")

        # 短期趋势
        if trend == 'rising':
            interpretations.append("情绪回升中")
        elif trend == 'falling':
            interpretations.append("情绪下降中")

        return ", ".join(interpretations)

    def _interpret_funding_rate(self, rate_pct: float) -> str:
        """
        解释资金费率 - 仅提供客观描述，不做主观判断

        资金费率含义:
        - 正值: 多头付费给空头 (多头过热)
        - 负值: 空头付费给多头 (空头过热)
        - 一般范围: -0.1% 到 0.1%
        """
        if rate_pct > 0.1:
            return f"多头极度过热({rate_pct:.3f}%，多头付费给空头)"
        elif rate_pct > 0.05:
            return f"多头过热({rate_pct:.3f}%)"
        elif rate_pct > -0.05:
            return f"市场平衡({rate_pct:.3f}%)"
        elif rate_pct > -0.1:
            return f"空头过热({rate_pct:.3f}%)"
        else:
            return f"空头极度过热({rate_pct:.3f}%，空头付费给多头)"

    def get_long_short_ratio(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        获取币安多空比历史数据（30天，1小时间隔）

        参数:
            pair: 交易对，如 'BTC/USDT:USDT'

        返回:
        {
            'current_ratio': 1.2,  # 当前多空比
            'trend': 'bullish',  # bullish/bearish/neutral
            'interpretation': '多头占优',
            'history_24h': [...],  # 最近24小时数据
            'history_7d': [...],  # 最近7天数据
            'history_30d': [...],  # 最近30天数据（最多720个点）
            'extreme_level': 'normal'  # extreme_long/extreme_short/normal
        }
        """
        # 检查缓存
        if pair in self.long_short_cache:
            cache_entry = self.long_short_cache[pair]
            if datetime.now() - cache_entry['time'] < timedelta(seconds=self.cache_duration):
                return cache_entry['data']

        try:
            # 转换交易对格式：BTC/USDT:USDT -> BTCUSDT
            symbol = pair.replace('/USDT:USDT', 'USDT').replace('/', '')

            # 币安API：全局多空账户比
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': '1h',  # 1小时间隔
                'limit': 500  # 币安API最大限制，约20天数据
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data or len(data) == 0:
                logger.warning(f"币安多空比API返回空数据: {symbol}")
                return None

            # 解析数据：[{longShortRatio, longAccount, shortAccount, timestamp}, ...]
            # Binance API返回的数据是按时间正序的（旧->新），timestamp递增
            # 注意：longAccount和shortAccount是小数比例（如0.654代表65.4%），需要乘100
            history = []
            for item in data:  # 直接使用正序数据（旧->新）
                ratio = float(item.get('longShortRatio', 0))
                if ratio > 0:
                    history.append({
                        'ratio': ratio,
                        'long_pct': float(item.get('longAccount', 0)) * 100,  # 转换为百分比
                        'short_pct': float(item.get('shortAccount', 0)) * 100,  # 转换为百分比
                        'timestamp': int(item.get('timestamp', 0))
                    })

            if len(history) == 0:
                return None

            # 当前多空比
            current = history[-1]
            current_ratio = current['ratio']

            # 分析趋势（最近24小时）
            if len(history) >= 24:
                ratio_24h_ago = history[-24]['ratio']
                change_24h = current_ratio - ratio_24h_ago

                if change_24h > 0.1:
                    trend = 'bullish'  # 多头增强
                elif change_24h < -0.1:
                    trend = 'bearish'  # 空头增强
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'

            # 判断极端情绪
            if current_ratio > 2.0:
                extreme_level = 'extreme_long'  # 多头过热
            elif current_ratio < 0.5:
                extreme_level = 'extreme_short'  # 空头过热
            elif current_ratio > 1.5:
                extreme_level = 'high_long'
            elif current_ratio < 0.7:
                extreme_level = 'high_short'
            else:
                extreme_level = 'normal'

            # 计算持续时间（当前情绪区间持续多久）
            def get_sentiment_zone(ratio):
                if ratio > 2.0: return 'extreme_long'
                elif ratio > 1.5: return 'high_long'
                elif ratio > 1.2: return 'slight_long'
                elif ratio > 0.8: return 'balanced'
                elif ratio > 0.5: return 'slight_short'
                else: return 'extreme_short'

            current_zone = get_sentiment_zone(current_ratio)
            duration_hours = 1  # 至少1小时
            for i in range(len(history) - 2, -1, -1):
                if get_sentiment_zone(history[i]['ratio']) == current_zone:
                    duration_hours += 1
                else:
                    break

            result = {
                'current_ratio': current_ratio,
                'long_pct': current['long_pct'],
                'short_pct': current['short_pct'],
                'trend': trend,
                'extreme_level': extreme_level,
                'duration_hours': duration_hours,
                'history_24h': [h['ratio'] for h in history[-24:]],
                'history_7d': [h['ratio'] for h in history[-168:]] if len(history) >= 168 else [h['ratio'] for h in history],
                'history_30d': history,  # 完整历史（包含ratio, long_pct, short_pct, timestamp）
                'interpretation': self._interpret_long_short_ratio(current_ratio, trend, extreme_level, duration_hours)
            }

            # 更新缓存
            self.long_short_cache[pair] = {
                'data': result,
                'time': datetime.now()
            }

            return result

        except Exception as e:
            logger.error(f"获取多空比失败 {pair}: {e}")
            return None

    def _interpret_long_short_ratio(self, ratio: float, trend: str, extreme: str, duration_hours: int) -> str:
        """解释多空比"""
        interpretations = []

        # 当前状态 - 只描述多空力量对比，不提供交易建议
        if ratio > 2.0:
            interpretations.append(f"多头极度过热({ratio:.2f})")
        elif ratio > 1.5:
            interpretations.append(f"多头偏强({ratio:.2f})")
        elif ratio > 1.2:
            interpretations.append(f"多头略占优({ratio:.2f})")
        elif ratio > 0.8:
            interpretations.append(f"多空平衡({ratio:.2f})")
        elif ratio > 0.5:
            interpretations.append(f"空头略占优({ratio:.2f})")
        else:
            interpretations.append(f"空头极度过热({ratio:.2f})")

        # 持续时间 - 客观描述
        if duration_hours >= 24:
            interpretations.append(f"已持续{duration_hours}小时")
            if extreme in ['extreme_long', 'extreme_short']:
                interpretations.append("长期处于极端状态")

        # 趋势
        if trend == 'bullish':
            interpretations.append("多头增强中")
        elif trend == 'bearish':
            interpretations.append("空头增强中")

        return ", ".join(interpretations)

    # ==================== 新增: 市场微观结构方法 ====================

    def start_liquidation_tracker(self, symbols: List[str]):
        """启动清算数据收集器"""
        if self.liquidation_enabled:
            self.liquidation_tracker.start(symbols)

    def stop_liquidation_tracker(self):
        """停止清算数据收集器"""
        self.liquidation_tracker.stop()

    def get_orderbook_analysis(self, exchange, pair: str, current_price: float) -> Optional[Dict]:
        """
        获取盘口深度分析

        参数:
            exchange: Freqtrade exchange 对象 (CCXT)
            pair: 交易对
            current_price: 当前价格

        返回:
            盘口分析数据，包含买卖深度、失衡比、流动性墙
        """
        if not self.orderbook_enabled:
            return None

        try:
            # 使用底层 CCXT exchange 的 fetch_order_book 方法
            # Freqtrade 的 exchange 包装器通过 _api 属性访问底层 CCXT
            ccxt_exchange = getattr(exchange, '_api', exchange)
            orderbook = ccxt_exchange.fetch_order_book(pair, limit=self.orderbook_analyzer.depth)
            return self.orderbook_analyzer.analyze(orderbook, pair, current_price)
        except Exception as e:
            logger.debug(f"获取盘口分析失败 {pair}: {e}")
            return None

    def get_orderflow_analysis(self, exchange, pair: str) -> Optional[Dict]:
        """
        获取订单流分析

        参数:
            exchange: Freqtrade exchange 对象
            pair: 交易对

        返回:
            订单流分析数据，包含买卖比、大单、成交频率
        """
        if not self.orderflow_enabled:
            return None

        return self.orderflow_tracker.analyze(exchange, pair)

    def get_oi_trend_analysis(self, exchange, pair: str, current_price: float,
                               price_change_24h: Optional[float] = None) -> Optional[Dict]:
        """
        获取多周期OI趋势分析

        参数:
            exchange: CCXT exchange 对象
            pair: 交易对
            current_price: 当前价格
            price_change_24h: 24小时价格变化百分比

        返回:
            多周期OI趋势数据，包含背离检测
        """
        if not self.oi_trend_enabled:
            return None

        return self.enhanced_oi_analyzer.get_multi_timeframe_trend(
            exchange, pair, current_price, price_change_24h
        )

    def get_liquidation_data(self, pair: str) -> Optional[Dict]:
        """
        获取近期清算数据

        参数:
            pair: 交易对

        返回:
            清算数据，包含多空清算量、失衡比
        """
        if not self.liquidation_enabled:
            return None

        if not self.liquidation_tracker.is_running():
            return None

        return self.liquidation_tracker.get_recent_liquidations(pair)

    def get_microstructure_data(self, exchange, pair: str,
                                 current_price: float,
                                 price_change_24h: Optional[float] = None) -> Dict[str, Any]:
        """
        获取完整的市场微观结构数据

        参数:
            exchange: Freqtrade exchange 对象 (CCXT)
            pair: 交易对
            current_price: 当前价格
            price_change_24h: 24小时价格变化百分比

        返回:
            {
                'orderbook': {...},     # 盘口深度分析
                'orderflow': {...},     # 订单流分析
                'oi_trend': {...},      # OI多周期趋势
                'liquidation': {...}    # 清算数据
            }
        """
        return {
            'orderbook': self.get_orderbook_analysis(exchange, pair, current_price),
            'orderflow': self.get_orderflow_analysis(exchange, pair),
            'oi_trend': self.get_oi_trend_analysis(exchange, pair, current_price, price_change_24h),
            'liquidation': self.get_liquidation_data(pair)
        }

    def get_oi_alerts(self, exchange, symbol: str, current_price: float,
                      volume_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        获取OI监控警报

        参数:
            exchange: CCXT exchange 对象
            symbol: 交易对，如 'BTC/USDT:USDT'
            current_price: 当前价格
            volume_data: 可选的成交量数据 {'volume_ratio': float}

        返回:
            OI警报数据，包含异动和突破信号
        """
        return self.oi_monitor.get_oi_alerts(exchange, symbol, current_price, volume_data)

    def get_reversal_signals(self, symbol: str,
                              dataframe_15m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        获取成交量反转信号

        参数:
            symbol: 交易对
            dataframe_15m: 15分钟K线数据

        返回:
            反转信号数据
        """
        return self.reversal_monitor.get_reversal_signals(symbol, dataframe_15m)

    def update_price_ma(self, symbol: str, price_ma: float):
        """更新价格MA缓存（用于OI异动检测的价格偏离计算）"""
        self.oi_monitor.update_price_ma(symbol, price_ma)

    def get_combined_sentiment(self, exchange, pair: str,
                                current_price: float = None,
                                dataframe_15m: pd.DataFrame = None,
                                volume_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取综合市场情绪（包含OI警报和反转信号）

        参数:
            exchange: Freqtrade exchange 对象
            pair: 交易对
            current_price: 当前价格（用于OI监控）
            dataframe_15m: 15分钟K线数据（用于反转监控）
            volume_data: 成交量数据 {'volume_ratio': float}

        返回:
        {
            'fear_greed': {...},
            'funding_rate': {...},
            'long_short': {...},
            'oi_alerts': {...},        # 新增: OI监控警报
            'reversal_signals': {...}, # 新增: 反转信号
            'overall_signal': 'bullish/bearish/neutral',
            'confidence': 'high/medium/low'
        }
        """
        fear_greed = self.get_fear_greed_index()
        funding_rate = self.get_funding_rate(exchange, pair)
        long_short = self.get_long_short_ratio(pair)

        # 新增: OI警报（使用 CCXT）
        oi_alerts = None
        if current_price:
            oi_alerts = self.get_oi_alerts(exchange, pair, current_price, volume_data)

        # 新增: 反转信号
        reversal_signals = None
        if dataframe_15m is not None and not dataframe_15m.empty:
            reversal_signals = self.get_reversal_signals(pair, dataframe_15m)

        # 综合判断 - 基于逆向思维的信号聚合
        # 注意：此信号仅供参考，实际决策需结合趋势、结构、持仓状态等完整上下文
        signals = []

        if fear_greed:
            fg_value = fear_greed['value']
            if fg_value <= 20:
                signals.append('bullish')  # 极度恐惧（逆向信号）
            elif fg_value >= 80:
                signals.append('bearish')  # 极度贪婪（逆向信号）

        if funding_rate:
            fr_pct = funding_rate['rate_pct']
            if fr_pct > 0.1:
                signals.append('bearish')  # 多头极度过热
            elif fr_pct < -0.1:
                signals.append('bullish')  # 空头极度过热

        if long_short:
            ls_ratio = long_short['current_ratio']
            if ls_ratio > 2.0:
                signals.append('bearish')  # 多头极度过热
            elif ls_ratio < 0.5:
                signals.append('bullish')  # 空头极度过热

        # 综合信号
        if len(signals) == 0:
            overall_signal = 'neutral'
            confidence = 'low'
        elif signals.count('bullish') > signals.count('bearish'):
            overall_signal = 'bullish'
            confidence = 'high' if len(signals) == 2 else 'medium'
        elif signals.count('bearish') > signals.count('bullish'):
            overall_signal = 'bearish'
            confidence = 'high' if len(signals) == 2 else 'medium'
        else:
            overall_signal = 'neutral'
            confidence = 'medium'

        return {
            'fear_greed': fear_greed,
            'funding_rate': funding_rate,
            'long_short': long_short,
            'oi_alerts': oi_alerts,              # 新增
            'reversal_signals': reversal_signals, # 新增
            'overall_signal': overall_signal,
            'confidence': confidence
        }
