"""
K线图生成器模块
负责生成合并的K线图（价格图+趋势线 和 标准K线图）
"""
import logging
import os
import base64
import io
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image

logger = logging.getLogger(__name__)


# ==================== 趋势线算法（从源项目迁移） ====================

def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    """
    检查趋势线有效性

    Args:
        support: True为支撑线，False为阻力线
        pivot: 枢轴点索引
        slope: 斜率
        y: 价格数组（numpy array或pandas Series）

    Returns:
        平方误差和（如果有效）或 -1.0（如果无效）
    """
    # 兼容numpy array和pandas Series
    if hasattr(y, 'iloc'):
        pivot_value = y.iloc[pivot]
    else:
        pivot_value = y[pivot]

    intercept = -slope * pivot + pivot_value
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    # 检查约束
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # 返回平方误差和
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> Tuple[float, float]:
    """
    优化趋势线斜率

    Args:
        support: True为支撑线，False为阻力线
        pivot: 枢轴点索引
        init_slope: 初始斜率
        y: 价格数组（numpy array或pandas Series）

    Returns:
        (优化后的斜率, 截距) 元组
    """
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0, f"初始斜率无效: err={best_err}"

    get_derivative = True
    derivative = None

    while curr_step > min_step:
        if get_derivative:
            # 计算导数
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        # 根据导数方向调整斜率
        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # 步长减半
            curr_step *= 0.5
        else:
            # 接受新斜率
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    # 计算截距（兼容numpy和pandas）
    if hasattr(y, 'iloc'):
        intercept = -best_slope * pivot + y.iloc[pivot]
    else:
        intercept = -best_slope * pivot + y[pivot]

    return (best_slope, intercept)


def fit_trendlines_single(data: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    基于收盘价拟合趋势线

    Args:
        data: 收盘价数组（numpy array或pandas Series）

    Returns:
        (支撑线系数, 阻力线系数)，每个系数为 (斜率, 截距) 元组
    """
    x = np.arange(len(data))

    # 使用线性回归找初始趋势线
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]

    # 找到偏离初始趋势线最远的点作为枢轴点
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # 基于枢轴点优化趋势线
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    基于高低点拟合趋势线

    Args:
        high: 最高价数组（numpy array或pandas Series）
        low: 最低价数组（numpy array或pandas Series）
        close: 收盘价数组（numpy array或pandas Series）

    Returns:
        (支撑线系数, 阻力线系数)，每个系数为 (斜率, 截距) 元组
    """
    x = np.arange(len(close))

    # 使用收盘价线性回归找初始趋势线
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]

    # 找到偏离初始趋势线最远的高低点作为枢轴点
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    # 基于枢轴点优化趋势线
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


# ==================== ChartGenerator类 ====================

class ChartGenerator:
    """
    K线图生成器
    生成合并的K线图（上下布局）
    """
    
    def __init__(
        self, 
        runtime_market_cache_getter: Optional[Callable] = None,
        image_dir: str = "./user_data/logs/images",
        save_to_disk: bool = True
    ):
        """
        初始化图表生成器
        
        Args:
            runtime_market_cache_getter: 获取运行时市场数据的回调函数
            image_dir: 图片保存目录
            save_to_disk: 是否保存图片到磁盘
        """
        self.runtime_market_cache_getter = runtime_market_cache_getter
        self.image_dir = image_dir
        self.save_to_disk = save_to_disk
        
        # 只在需要保存时创建目录
        if self.save_to_disk:
            os.makedirs(self.image_dir, exist_ok=True)
            logger.info(f"ChartGenerator已初始化，图片保存目录: {self.image_dir}")
        else:
            logger.info(f"ChartGenerator已初始化（仅内存模式，不保存到磁盘）")
    
    def _ensure_kline_df(
        self, 
        kline_data: Optional[Dict[str, Any]] = None, 
        lookback: int = 50
    ) -> pd.DataFrame:
        """
        确保获取K线数据并转换为DataFrame
        
        Args:
            kline_data: 外部提供的K线数据（可选）
            lookback: 需要的K线数量
            
        Returns:
            格式化的DataFrame
            
        Raises:
            ValueError: 无法获取数据时
        """
        # 1. 如果提供了外部数据，直接使用
        if kline_data is not None:
            logger.debug(f"接收到kline_data，date列前3个值: {kline_data.get('date', [])[:3]}")
            df = pd.DataFrame(kline_data)
            logger.debug(f"DataFrame创建后，date列dtype: {df['date'].dtype}, 前3个值: {df['date'].iloc[:3].tolist()}")
        # 2. 否则尝试从缓存获取
        elif self.runtime_market_cache_getter is not None:
            try:
                cache_data = self.runtime_market_cache_getter()
                if cache_data is None:
                    raise ValueError("运行时市场缓存为空")
                df = pd.DataFrame(cache_data)
            except Exception as e:
                raise ValueError(f"从缓存获取数据失败: {e}")
        else:
            raise ValueError("未提供K线数据且未配置缓存获取器")
        
        # 数据验证
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 截取最后N根K线
        df = df.tail(lookback).copy()
        
        # 转换date为DatetimeIndex（mplfinance要求）
        if 'date' in df.columns:
            # 检测时间戳类型并转换（优化边界判断）
            if df['date'].dtype in ['int64', 'float64']:
                # 如果是数字时间戳，根据数值范围判断单位
                first_val = df['date'].iloc[0]
                logger.debug(f"时间戳转换调试: first_val={first_val}, dtype={df['date'].dtype}")
                
                # 使用更严谨的边界值判断
                if first_val > 1e15:  # 纳秒时间戳 (>2003年的纳秒: 1e15 ≈ 2001-09-09)
                    df['date'] = pd.to_datetime(df['date'], unit='ns')
                    logger.debug(f"使用纳秒转换: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
                elif first_val > 1e11:  # 毫秒时间戳 (>1973年的毫秒: 1e11 ≈ 1973-03-03)
                    df['date'] = pd.to_datetime(df['date'], unit='ms')
                    logger.debug(f"使用毫秒转换: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
                elif first_val > 1e8:  # 秒时间戳 (>1973年的秒: 1e8 ≈ 1973-03-03)
                    df['date'] = pd.to_datetime(df['date'], unit='s')
                    logger.debug(f"使用秒转换: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
                else:
                    # 时间戳值异常小，可能是错误数据
                    raise ValueError(f"时间戳值异常: {first_val}，无法判断时间单位")
            else:
                # 字符串格式，直接转换
                df['date'] = pd.to_datetime(df['date'])
                logger.debug(f"字符串转换: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            df.set_index('date', inplace=True)
        
        return df
    
    def generate_combined_chart(
        self, 
        kline_data: Optional[Dict[str, Any]] = None,
        lookback: int = 50,
        dpi: int = 600,
        figsize: Tuple[int, int] = (12, 5),
        pair: str = "UNKNOWN",
        timeframe: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        生成合并K线图（上下布局）
        
        Args:
            kline_data: K线数据字典（可选）
            lookback: 回溯K线数量
            dpi: 图片DPI
            figsize: 每个子图的尺寸（宽, 高）英寸
            pair: 交易对名称
            timeframe: 时间框架
            
        Returns:
            {
                "combined_image_b64": str,  # base64编码的PNG图片
                "filename": str,            # 保存的文件路径
                "chart_info": {             # 元数据
                    "lookback": int,
                    "timestamp": str,
                    "pair": str,
                    "timeframe": str
                }
            }
            
        Raises:
            Exception: 图表生成失败时
        """
        try:
            logger.debug(f"开始生成K线图: pair={pair}, lookback={lookback}")
            
            # 1. 准备数据
            df = self._ensure_kline_df(kline_data, lookback)
            
            # 2. 拟合趋势线（直接传递pandas Series，与源项目一致）
            support_coefs, resist_coefs = fit_trendlines_high_low(
                df["high"],
                df["low"],
                df["close"]
            )

            # support_coefs和resist_coefs现在是 (slope, intercept) 元组
            support_line = support_coefs[0] * np.arange(len(df)) + support_coefs[1]
            resist_line = resist_coefs[0] * np.arange(len(df)) + resist_coefs[1]
            
            # 3. 生成上半部分图片（价格图 + 趋势线 + EMA + 布林带）
            logger.debug("生成上半部分图片（价格图+趋势线+EMA+布林带）")
            apds = [
                mpf.make_addplot(support_line, color="blue", width=1.5, label="Support"),
                mpf.make_addplot(resist_line, color="red", width=1.5, label="Resistance")
            ]
            
            # 添加 EMA 均线（如果存在）
            if 'ema_20' in df.columns:
                apds.append(mpf.make_addplot(df['ema_20'], color='orange', width=1.2, label='EMA20'))
            if 'ema_50' in df.columns:
                apds.append(mpf.make_addplot(df['ema_50'], color='purple', width=1.2, label='EMA50'))
            
            # 添加布林带（如果存在）
            if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                apds.append(mpf.make_addplot(df['bb_upper'], color='gray', width=0.8, linestyle='--', alpha=0.5))
                apds.append(mpf.make_addplot(df['bb_middle'], color='gray', width=0.8, linestyle='--', alpha=0.5))
                apds.append(mpf.make_addplot(df['bb_lower'], color='gray', width=0.8, linestyle='--', alpha=0.5))
            
            fig1, axes1 = mpf.plot(
                df,
                type="candle",
                style="charles",
                addplot=apds,
                returnfig=True,
                figsize=figsize,
                title="Price Chart with Trendlines & Indicators",
                ylabel="Price",
                volume=True  # 添加成交量面板
            )
            
            # 添加元数据标注（左上角）
            axes1[0].text(
                0.02, 0.98, 
                f"Pair: {pair} | Timeframe: {timeframe} | Candles: {lookback}",
                transform=axes1[0].transAxes, 
                fontsize=9, 
                va='top', 
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # 保存到内存缓冲区
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=dpi, bbox_inches='tight')
            buf1.seek(0)
            img1 = Image.open(buf1)
            plt.close(fig1)  # 释放内存
            
            # 4. 生成下半部分图片（标准K线图 + MACD）
            logger.debug("生成下半部分图片（标准K线图+MACD）")
            
            # 准备 MACD 附加图（如果存在）
            apds_lower = []
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                apds_lower.append(mpf.make_addplot(df['macd'], panel=1, color='blue', width=1.0, ylabel='MACD'))
                apds_lower.append(mpf.make_addplot(df['macd_signal'], panel=1, color='red', width=1.0))
                apds_lower.append(mpf.make_addplot(df['macd_hist'], panel=1, type='bar', color='gray', alpha=0.5))
            
            # 构建 plot 参数（mplfinance 不接受 addplot=None）
            plot_kwargs = {
                'type': 'candle',
                'style': 'charles',
                'returnfig': True,
                'figsize': figsize,
                'title': 'Candlestick Pattern Chart with MACD',
                'ylabel': 'Price',
                'volume': True
            }
            if apds_lower:
                plot_kwargs['addplot'] = apds_lower
            
            fig2, axes2 = mpf.plot(df, **plot_kwargs)
            
            # 添加时间范围标注（左上角）
            start_time = df.index[0].strftime("%Y-%m-%d %H:%M")
            end_time = df.index[-1].strftime("%Y-%m-%d %H:%M")
            axes2[0].text(
                0.02, 0.98, 
                f"Time Range: {start_time} to {end_time}",
                transform=axes2[0].transAxes, 
                fontsize=9, 
                va='top', 
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # 保存到内存缓冲区
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=dpi, bbox_inches='tight')
            buf2.seek(0)
            img2 = Image.open(buf2)
            plt.close(fig2)  # 释放内存
            
            # 5. 使用PIL合并两张图片
            logger.debug("合并两张图片")
            width1, height1 = img1.size
            width2, height2 = img2.size
            
            total_width = max(width1, width2)
            total_height = height1 + height2
            
            # 创建白色背景的新图片
            combined_img = Image.new('RGB', (total_width, total_height), 'white')
            
            # 粘贴上半部分
            combined_img.paste(img1, (0, 0))
            
            # 粘贴下半部分
            combined_img.paste(img2, (0, height1))
            
            # 6. 保存到本地文件（如果启用）
            fname = None
            if self.save_to_disk:
                fname = os.path.join(
                    self.image_dir, 
                    f"combined_{pair.replace('/', '_')}_{int(time.time())}.png"
                )
                combined_img.save(fname, format='PNG')
                logger.info(f"✅ 合并K线图已保存: {fname}")
            else:
                logger.debug(f"跳过保存图片到磁盘（save_to_disk=False）")
            
            # 7. 转换为base64
            buf_final = io.BytesIO()
            combined_img.save(buf_final, format='PNG')
            buf_final.seek(0)
            b64 = base64.b64encode(buf_final.read()).decode("utf-8")
            
            # 8. 返回结果
            return {
                "combined_image_b64": b64,
                "filename": fname,
                "chart_info": {
                    "lookback": lookback,
                    "timestamp": datetime.now().isoformat(),
                    "pair": pair,
                    "timeframe": timeframe,
                    "width": total_width,
                    "height": total_height,
                    "dpi": dpi
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 生成K线图失败: {e}", exc_info=True)
            raise

