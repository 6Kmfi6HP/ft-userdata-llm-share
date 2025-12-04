"""
微观结构分析模块单元测试
测试 OrderBookAnalyzer, OrderFlowTracker, EnhancedOIAnalyzer, LiquidationTracker
"""
import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / "strategies"))

from llm_modules.utils.market_sentiment import (
    OrderBookAnalyzer,
    OrderFlowTracker,
    EnhancedOIAnalyzer,
    LiquidationTracker,
    ORDERBOOK_DEPTH,
    LARGE_TRADE_THRESHOLD_USDT,
    SIGNIFICANT_LIQ_USDT,
)


class TestOrderBookAnalyzer:
    """盘口深度分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.config = {
            "depth": 20,
            "cache_duration_seconds": 30,
            "wall_threshold_multiplier": 3.0,
            "min_wall_size_usdt": 50000,
        }
        self.analyzer = OrderBookAnalyzer(self.config)

    def test_init(self):
        """测试初始化"""
        assert self.analyzer.depth == 20
        assert self.analyzer.cache_duration == 30
        assert self.analyzer.wall_threshold == 3.0
        assert self.analyzer.min_wall_size == 50000

    def test_analyze_orderbook_empty(self):
        """测试空盘口数据"""
        # OrderBookAnalyzer.analyze() 接收 orderbook dict，不是 exchange
        orderbook = {"bids": [], "asks": []}

        result = self.analyzer.analyze(orderbook, "BTC/USDT:USDT", 50000.0)

        # 空盘口返回 None
        assert result is None

    def test_analyze_orderbook_buy_pressure(self):
        """测试买盘压力"""
        # 买盘深度 > 卖盘深度
        orderbook = {
            "bids": [
                [49900, 2.0],  # 价格, 数量
                [49800, 3.0],
                [49700, 5.0],
            ],
            "asks": [
                [50100, 1.0],
                [50200, 1.0],
            ],
        }

        result = self.analyzer.analyze(orderbook, "BTC/USDT:USDT", 50000.0)

        # 买盘深度 = 49900*2 + 49800*3 + 49700*5 = 99800 + 149400 + 248500 = 497700
        # 卖盘深度 = 50100*1 + 50200*1 = 100300
        assert result["bid_volume"] > result["ask_volume"]
        assert result["imbalance_ratio"] > 1  # imbalance_ratio = bid/ask, >1 表示买盘强
        assert result["pressure"] == "buy"

    def test_analyze_orderbook_sell_pressure(self):
        """测试卖盘压力"""
        # 卖盘深度 > 买盘深度
        orderbook = {
            "bids": [
                [49900, 1.0],
            ],
            "asks": [
                [50100, 5.0],
                [50200, 5.0],
                [50300, 5.0],
            ],
        }

        result = self.analyzer.analyze(orderbook, "BTC/USDT:USDT", 50000.0)

        assert result["ask_volume"] > result["bid_volume"]
        assert result["imbalance_ratio"] < 1  # imbalance_ratio = bid/ask, <1 表示卖盘强
        assert result["pressure"] == "sell"

    def test_detect_liquidity_wall(self):
        """测试流动性墙检测"""
        # 创建一个明显的买墙
        orderbook = {
            "bids": [
                [49900, 0.5],
                [49800, 0.5],
                [49700, 10.0],  # 大单 = 497000 USDT
                [49600, 0.5],
            ],
            "asks": [
                [50100, 0.5],
                [50200, 0.5],
            ],
        }

        result = self.analyzer.analyze(orderbook, "BTC/USDT:USDT", 50000.0)

        # 应该检测到流动性墙
        walls = result.get("liquidity_walls", [])
        assert len(walls) > 0

    def test_cache_mechanism(self):
        """测试缓存机制"""
        orderbook1 = {
            "bids": [[49900, 1.0]],
            "asks": [[50100, 1.0]],
        }

        # 第一次调用
        result1 = self.analyzer.analyze(orderbook1, "BTC/USDT:USDT", 50000.0)

        # 第二次调用使用不同的 orderbook 数据
        orderbook2 = {
            "bids": [[49900, 2.0]],  # 改变数量
            "asks": [[50100, 2.0]],
        }

        # 第二次调用（应该使用缓存，因为 pair 相同且在缓存有效期内）
        result2 = self.analyzer.analyze(orderbook2, "BTC/USDT:USDT", 50000.0)

        # 两次结果应该相同（使用缓存）
        assert result1["bid_volume"] == result2["bid_volume"]


class TestOrderFlowTracker:
    """订单流分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.config = {
            "cache_duration_seconds": 60,
            "large_trade_threshold_usdt": 100000,
            "trade_fetch_limit": 100,
        }
        self.tracker = OrderFlowTracker(self.config)

    def test_init(self):
        """测试初始化"""
        assert self.tracker.cache_duration == 60
        assert self.tracker.large_trade_threshold == 100000
        assert self.tracker.fetch_limit == 100

    def test_analyze_empty_trades(self):
        """测试空交易数据"""
        mock_exchange = MagicMock()
        mock_exchange.fetch_trades.return_value = []

        # OrderFlowTracker.analyze() 只接收 exchange 和 pair 两个参数
        result = self.tracker.analyze(mock_exchange, "BTC/USDT:USDT")

        # 空交易返回 None
        assert result is None

    def test_analyze_buy_dominated(self):
        """测试买入主导"""
        mock_exchange = MagicMock()
        mock_exchange.fetch_trades.return_value = [
            {"side": "buy", "amount": 2.0, "price": 50000.0, "timestamp": 1000},
            {"side": "buy", "amount": 1.0, "price": 50100.0, "timestamp": 1001},
            {"side": "sell", "amount": 0.5, "price": 49900.0, "timestamp": 1002},
        ]

        result = self.tracker.analyze(mock_exchange, "BTC/USDT:USDT")

        # 买入量 = 2*50000 + 1*50100 = 150100
        # 卖出量 = 0.5*49900 = 24950
        assert result["buy_volume"] > result["sell_volume"]
        assert result["buy_sell_ratio"] > 1.0
        assert result["net_flow"] > 0
        assert result["flow_direction"] == "buy"

    def test_analyze_sell_dominated(self):
        """测试卖出主导"""
        mock_exchange = MagicMock()
        mock_exchange.fetch_trades.return_value = [
            {"side": "sell", "amount": 3.0, "price": 50000.0, "timestamp": 1000},
            {"side": "buy", "amount": 0.5, "price": 50100.0, "timestamp": 1001},
        ]

        result = self.tracker.analyze(mock_exchange, "BTC/USDT:USDT")

        assert result["sell_volume"] > result["buy_volume"]
        assert result["buy_sell_ratio"] < 1.0
        assert result["net_flow"] < 0
        assert result["flow_direction"] == "sell"

    def test_large_trade_detection(self):
        """测试大单检测"""
        mock_exchange = MagicMock()
        # 创建一个超过阈值的大单
        mock_exchange.fetch_trades.return_value = [
            {"side": "buy", "amount": 3.0, "price": 50000.0, "timestamp": 1000},  # 150000 USDT
            {"side": "sell", "amount": 0.5, "price": 50000.0, "timestamp": 1001},  # 25000 USDT
        ]

        result = self.tracker.analyze(mock_exchange, "BTC/USDT:USDT")

        # 应该检测到大单
        large_trades = result.get("large_trades", [])
        assert len(large_trades) == 1
        assert large_trades[0]["side"] == "buy"
        assert large_trades[0]["value"] == 150000


class TestEnhancedOIAnalyzer:
    """增强OI分析器测试"""

    def setup_method(self):
        """测试前设置"""
        self.mock_oi_monitor = MagicMock()
        self.config = {
            "cache_seconds": 300,
            "divergence_threshold": 0.02,
        }
        self.analyzer = EnhancedOIAnalyzer(self.mock_oi_monitor, self.config)

        # 创建 mock exchange 对象
        self.mock_exchange = MagicMock()
        self.mock_ccxt = MagicMock()
        self.mock_exchange._api = self.mock_ccxt
        self.mock_ccxt.has = {'fetchOpenInterestHistory': True}

    def test_init(self):
        """测试初始化"""
        assert self.analyzer.divergence_threshold == 0.02
        assert self.analyzer.cache_duration == 300

    def test_analyze_rising_trend(self):
        """测试上升趋势"""
        # Mock CCXT fetch_open_interest_history 返回 OI 上升趋势
        self.mock_ccxt.fetch_open_interest_history.return_value = [
            {"openInterestAmount": 100000},
            {"openInterestAmount": 105000},
            {"openInterestAmount": 110000},
            {"openInterestAmount": 115000},
        ]

        # Mock fetch_current_oi
        self.mock_oi_monitor.fetch_current_oi.return_value = {"open_interest": 115000}

        result = self.analyzer.get_multi_timeframe_trend(
            self.mock_exchange, "BTC/USDT:USDT", current_price=50000.0
        )

        # 应该检测到上升趋势
        assert result is not None
        assert result.get("trend_15m") == "rising"

    def test_analyze_falling_trend(self):
        """测试下降趋势"""
        # Mock CCXT fetch_open_interest_history 返回 OI 下降趋势
        self.mock_ccxt.fetch_open_interest_history.return_value = [
            {"openInterestAmount": 110000},
            {"openInterestAmount": 105000},
            {"openInterestAmount": 100000},
            {"openInterestAmount": 95000},
        ]

        # Mock fetch_current_oi
        self.mock_oi_monitor.fetch_current_oi.return_value = {"open_interest": 95000}

        result = self.analyzer.get_multi_timeframe_trend(
            self.mock_exchange, "BTC/USDT:USDT", current_price=50000.0
        )

        # 应该检测到下降趋势
        assert result is not None
        assert result.get("trend_15m") in ["falling", "stable"]

    def test_no_data(self):
        """测试无数据情况"""
        # Mock CCXT fetch_open_interest_history 返回空数据
        self.mock_ccxt.fetch_open_interest_history.return_value = []

        # Mock fetch_current_oi
        self.mock_oi_monitor.fetch_current_oi.return_value = None

        result = self.analyzer.get_multi_timeframe_trend(
            self.mock_exchange, "BTC/USDT:USDT", current_price=50000.0
        )

        # 应该返回 None 或包含 unknown 趋势
        assert result is None or result.get("trend_15m") == "unknown"


class TestLiquidationTracker:
    """清算数据追踪器测试"""

    def setup_method(self):
        """测试前设置"""
        self.config = {
            "buffer_minutes": 5,
            "significant_liq_usdt": 50000,
            "imbalance_alert_ratio": 2.0,
        }
        self.tracker = LiquidationTracker(self.config)

    def test_init(self):
        """测试初始化"""
        assert self.tracker.buffer_minutes == 5
        assert self.tracker.significant_threshold == 50000
        assert self.tracker.imbalance_alert_ratio == 2.0
        assert self.tracker._running is False

    def test_get_recent_liquidations_empty(self):
        """测试空清算数据"""
        result = self.tracker.get_recent_liquidations("BTC/USDT:USDT")

        assert result["total_count"] == 0
        assert result["long_liquidations"] == 0
        assert result["short_liquidations"] == 0
        assert result["long_volume"] == 0
        assert result["short_volume"] == 0

    def test_add_and_get_liquidation(self):
        """测试添加和获取清算数据"""
        # 模拟添加清算数据（使用实际类内部的格式）
        current_time = int(time.time() * 1000)
        # LiquidationTracker 使用 _buffers 属性，数据格式为处理后的格式
        from collections import deque
        self.tracker._buffers["BTCUSDT"] = deque([
            {
                "side": "long",  # 多头被清算
                "price": 50000.0,
                "qty": 2.0,
                "value_usdt": 100000.0,  # 2 * 50000
                "time": current_time,
            },
            {
                "side": "short",  # 空头被清算
                "price": 50000.0,
                "qty": 1.0,
                "value_usdt": 50000.0,  # 1 * 50000
                "time": current_time,
            },
        ], maxlen=1000)

        result = self.tracker.get_recent_liquidations("BTC/USDT:USDT")

        assert result["total_count"] == 2
        assert result["long_liquidations"] == 1  # 多头被清算
        assert result["short_liquidations"] == 1  # 空头被清算
        assert result["long_volume"] == 100000  # 2 * 50000
        assert result["short_volume"] == 50000  # 1 * 50000

    def test_significant_liquidation_detection(self):
        """测试显著清算事件检测"""
        current_time = int(time.time() * 1000)
        # 添加一个超过阈值的清算
        from collections import deque
        self.tracker._buffers["BTCUSDT"] = deque([
            {
                "side": "long",
                "price": 50000.0,
                "qty": 2.0,
                "value_usdt": 100000.0,  # > 50000阈值
                "time": current_time,
            },
        ], maxlen=1000)

        result = self.tracker.get_recent_liquidations("BTC/USDT:USDT")

        significant = result.get("significant_liquidations", [])
        assert len(significant) == 1
        assert significant[0]["value"] == 100000

    def test_imbalance_ratio(self):
        """测试清算失衡比例"""
        current_time = int(time.time() * 1000)
        # 添加不平衡的清算数据（多头清算远多于空头）
        from collections import deque
        self.tracker._buffers["BTCUSDT"] = deque([
            {
                "side": "long",  # 多头被清算
                "price": 50000.0,
                "qty": 3.0,
                "value_usdt": 150000.0,  # 3 * 50000
                "time": current_time,
            },
            {
                "side": "short",  # 空头被清算
                "price": 50000.0,
                "qty": 0.5,
                "value_usdt": 25000.0,  # 0.5 * 50000
                "time": current_time,
            },
        ], maxlen=1000)

        result = self.tracker.get_recent_liquidations("BTC/USDT:USDT")

        # 失衡比例 = 150000 / 25000 = 6
        assert result["imbalance_ratio"] > 2.0  # 应该触发警报

    def test_start_stop(self):
        """测试启动和停止"""
        # 由于实际WebSocket需要网络，我们只测试状态变化
        assert self.tracker._running is False
        assert self.tracker.is_running() is False

        # 模拟启动（不实际连接）
        with patch.object(self.tracker, "_run_collector"):
            self.tracker.start(["BTC/USDT:USDT"])
            assert self.tracker._running is True
            assert self.tracker.is_running() is True
            assert "BTCUSDT" in self.tracker._symbols

        # 停止
        self.tracker.stop()
        assert self.tracker._running is False
        assert self.tracker.is_running() is False


class TestMarketSentimentIntegration:
    """MarketSentiment集成测试"""

    def test_get_microstructure_data(self):
        """测试获取微观结构数据"""
        from llm_modules.utils.market_sentiment import MarketSentiment

        config = {
            "orderbook": {"enabled": True},
            "orderflow": {"enabled": True},
            "oi_trend": {"enabled": True},
            "liquidation": {"enabled": True},
        }
        sentiment = MarketSentiment(config)

        # Mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book.return_value = {
            "bids": [[49900, 1.0]],
            "asks": [[50100, 1.0]],
        }
        mock_exchange.fetch_trades.return_value = []

        result = sentiment.get_microstructure_data(
            exchange=mock_exchange,
            pair="BTC/USDT:USDT",
            current_price=50000.0,
        )

        # 应该包含所有四个数据类型
        assert "orderbook" in result
        assert "orderflow" in result
        assert "oi_trend" in result
        assert "liquidation" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
