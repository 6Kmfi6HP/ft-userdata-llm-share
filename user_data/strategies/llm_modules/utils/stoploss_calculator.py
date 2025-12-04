"""
止损计算器模块
提供统一的动态追踪止损计算逻辑，避免代码重复

注意：此模块实现的是追踪止损（trailing stop），即止损价格随当前价格移动
不使用 stoploss_from_open（它用于固定止损位置）
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 🔧 修复M6: 浮点数比较容差常量（0.0000001% = 1e-9）
PROFIT_EPSILON = 1e-9


class StoplossCalculator:
    """动态止损计算器（ATR追踪 + 时间衰减 + 趋势适应）"""
    
    @staticmethod
    def _calculate_smooth_distance(
        current_profit: float,
        atr_pct: float,
        cfg: Dict[str, Any]
    ) -> float:
        """
        计算平滑过渡的追踪距离（使用线性插值避免跳变）

        Args:
            current_profit: 当前盈利
            atr_pct: ATR百分比
            cfg: 配置参数

        Returns:
            追踪距离百分比
        """
        thresholds = cfg.get('profit_thresholds', [])
        multipliers = cfg.get('atr_multipliers', [])
        min_distances = cfg.get('min_distances', [])

        # 🔧 修复M12: 验证配置数组长度（防止 IndexError）
        if len(thresholds) < 3 or len(multipliers) < 3 or len(min_distances) < 3:
            logger.error(
                f"止损配置数组长度不足: "
                f"thresholds={len(thresholds)}, multipliers={len(multipliers)}, "
                f"min_distances={len(min_distances)}. 需要至少3个值."
            )
            # 返回一个安全的默认值（1.5% 默认最小距离）
            return 0.015
        
        # 🔧 修复H4+M6: 2-6% 区间：1.5×ATR, 最小1.5% (使用 < 排除边界 + Epsilon容差)
        if current_profit < (thresholds[1] + PROFIT_EPSILON):  # < 6% (不含边界)
            atr_multiplier = multipliers[0]
            min_distance = min_distances[0]

        # 6-15% 区间：线性过渡从1.5x到1.0x
        elif current_profit <= (thresholds[2] + PROFIT_EPSILON):
            # 🔧 修复M10: 防止除零错误（如果配置 thresholds[1] == thresholds[2]）
            denominator = thresholds[2] - thresholds[1]
            if abs(denominator) < 1e-9:  # 避免除零
                progress = 1.0  # 视为已到达阈值边界
            else:
                progress = (current_profit - thresholds[1]) / denominator

            atr_multiplier = multipliers[0] - (multipliers[0] - multipliers[1]) * progress
            min_distance = min_distances[0] - (min_distances[0] - min_distances[1]) * progress
        
        # >15% 区间：0.8×ATR, 最小0.5%
        else:
            atr_multiplier = multipliers[2]
            min_distance = min_distances[2]
        
        return max(atr_multiplier * atr_pct, min_distance)
    
    @staticmethod
    def _calculate_stepped_distance(
        current_profit: float,
        atr_pct: float,
        cfg: Dict[str, Any]
    ) -> float:
        """
        计算阶梯式追踪距离（保留原始逻辑，可能产生跳变）

        Args:
            current_profit: 当前盈利
            atr_pct: ATR百分比
            cfg: 配置参数

        Returns:
            追踪距离百分比
        """
        thresholds = cfg.get('profit_thresholds', [])
        multipliers = cfg.get('atr_multipliers', [])
        min_distances = cfg.get('min_distances', [])

        # 🔧 修复M12: 验证配置数组长度（防止 IndexError）
        if len(thresholds) < 3 or len(multipliers) < 3 or len(min_distances) < 3:
            logger.error(
                f"止损配置数组长度不足: "
                f"thresholds={len(thresholds)}, multipliers={len(multipliers)}, "
                f"min_distances={len(min_distances)}. 需要至少3个值."
            )
            # 返回一个安全的默认值（1.5% 默认最小距离）
            return 0.015
        
        if current_profit > thresholds[2]:
            # >15%: 0.8×ATR, 最小0.5%
            return max(multipliers[2] * atr_pct, min_distances[2])
        elif current_profit > thresholds[1]:
            # 6-15%: 1.0×ATR, 最小1.0%
            return max(multipliers[1] * atr_pct, min_distances[1])
        else:
            # 2-6%: 1.5×ATR, 最小1.5%
            return max(multipliers[0] * atr_pct, min_distances[0])
    
    @staticmethod
    def calculate_stoploss_price(
        current_price: float,
        current_profit: float,
        atr_pct: float,
        adx: float,
        hold_duration_hours: float,
        is_short: bool,
        open_price: float,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        计算止损价格（用于展示给LLM）
        
        Args:
            current_price: 当前价格
            current_profit: 当前盈利百分比
            atr_pct: ATR百分比
            adx: ADX值
            hold_duration_hours: 持仓小时数
            is_short: 是否做空
            open_price: 开仓价
            config: 配置参数
        
        Returns:
            止损价格，或None（使用硬止损）
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 默认配置 - 基于ATR追踪止损最佳实践 (2024-2025)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #
        # 【最佳实践来源】:
        # - Flipster: ATR Stop Loss Strategy For Crypto Trading
        # - LuxAlgo: 5 ATR Stop-Loss Strategies for Risk Control
        # - TrendSpider: ATR Trailing Stops Guide
        # - Freqtrade Official Documentation
        #
        # 【加密货币特殊考虑】:
        # - 加密货币推荐止损距离: 8-15% (vs 股票3-5%)
        # - 震荡市场ATR倍数: 1.5-2.0×
        # - 趋势市场ATR倍数: 3.0-4.0× (让利润奔跑)
        # - 避免1×ATR内止损 (易被whipsaw假突破震出)
        #
        # 【配置说明】:
        # - profit_thresholds: 盈利阈值边界 [低, 中, 高]
        # - atr_multipliers: 各区间ATR倍数 [2-6%, 6-15%, >15%]
        # - min_distances: 各区间最小止损距离 [2-6%, 6-15%, >15%]
        #
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        default_config = {
            'profit_thresholds': [0.02, 0.06, 0.15],
            # ATR倍数：震荡区间保守(2.0×)，趋势区间宽松(3.0×)让利润奔跑
            'atr_multipliers': [2.0, 2.0, 3.0],
            # 最小距离：加密货币波动大，适当放宽防止whipsaw
            # 2-6%: 4% (保护初始盈利，但不过紧)
            # 6-15%: 5% (中等保护)
            # >15%: 8% (符合加密货币8-15%最佳实践)
            'min_distances': [0.04, 0.05, 0.08],
            'time_decay_hours': 2,
            'time_decay_factor': 0.8,
            'trend_strength_threshold': 25,
            'trend_strength_factor': 1.2,
            'use_smooth_transition': True,
            'max_atr_pct': 0.12  # ATR上限12% (加密货币波动适配)
        }
        
        cfg = {**default_config, **(config or {})}
        
        # 应用ATR上限（防止极端波动导致止损失效）
        max_atr = cfg.get('max_atr_pct', 0.10)
        if atr_pct > max_atr:
            atr_pct = max_atr
        
        # 🔧 修复M6: 未盈利或盈利过低：返回None表示使用硬止损（使用Epsilon容差）
        if current_profit <= (cfg['profit_thresholds'][0] + PROFIT_EPSILON):
            return None
        
        # 计算追踪距离
        if cfg['use_smooth_transition']:
            base_distance_pct = StoplossCalculator._calculate_smooth_distance(
                current_profit, atr_pct, cfg
            )
        else:
            base_distance_pct = StoplossCalculator._calculate_stepped_distance(
                current_profit, atr_pct, cfg
            )
        
        # 🔧 修复H3: 应用趋势适应和时间衰减（允许叠加）
        # 趋势因子和时间衰减因子可以同时生效（例如：1.2 × 0.8 = 0.96）
        adjustment_factor = 1.0

        if adx > cfg['trend_strength_threshold']:
            # 强趋势时放宽止损空间
            adjustment_factor *= cfg['trend_strength_factor']  # 1.2

        if hold_duration_hours > cfg['time_decay_hours'] and \
           current_profit < cfg['profit_thresholds'][1]:
            # 长时间持仓但盈利不足时收紧止损
            adjustment_factor *= cfg['time_decay_factor']  # 0.8

        base_distance_pct *= adjustment_factor
        
        # 计算止损价格
        if is_short:
            stop_price = current_price * (1 + base_distance_pct)
        else:
            stop_price = current_price * (1 - base_distance_pct)
        
        return stop_price

