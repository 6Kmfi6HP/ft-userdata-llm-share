"""
图表生成模块
生成K线图和趋势线图，用于视觉分析Agent

参考 QuantAgent 的图表生成实现:
- generate_kline_image(): 生成标准K线图
- generate_trend_image(): 生成带趋势线的K线图（含梯度下降优化）

核心算法:
- 梯度下降趋势线优化 (fit_trendlines_high_low, optimize_slope)
- 与 QuantAgent 的 graph_util.py 实现一致

依赖:
- mplfinance: K线图绘制
- matplotlib: 基础绘图
- pandas: 数据处理
- numpy: 数值计算
"""

import logging
import base64
import io
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


# ============================================================================
# 趋势线优化算法 - 参考 QuantAgent graph_util.py
# 使用梯度下降拟合更精准的支撑/阻力趋势线
# ============================================================================

def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    """
    检验趋势线有效性并计算误差

    趋势线必须满足约束:
    - 支撑线: 必须在所有价格点下方
    - 阻力线: 必须在所有价格点上方

    Args:
        support: True 表示支撑线，False 表示阻力线
        pivot: 枢轴点索引（趋势线锚点）
        slope: 趋势线斜率
        y: 价格数据序列 (pandas Series 或 numpy array)

    Returns:
        误差值（平方和），-1.0 表示无效趋势线
    """
    # 计算趋势线截距（通过枢轴点）
    if hasattr(y, 'iloc'):
        pivot_value = y.iloc[pivot]
    else:
        pivot_value = y[pivot]
    intercept = -slope * pivot + pivot_value

    # 计算趋势线上的所有点
    line_vals = slope * np.arange(len(y)) + intercept

    # 计算差值
    if hasattr(y, 'values'):
        diffs = line_vals - y.values
    else:
        diffs = line_vals - y

    # 验证趋势线约束
    if support and diffs.max() > 1e-5:
        # 支撑线约束：线必须在所有价格下方
        return -1.0
    elif not support and diffs.min() < -1e-5:
        # 阻力线约束：线必须在所有价格上方
        return -1.0

    # 计算误差（差值平方和）
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> Tuple[float, float]:
    """
    使用梯度下降优化趋势线斜率

    从初始斜率开始，通过数值微分找到最优斜率，使误差最小化
    同时满足支撑/阻力线的约束条件

    Args:
        support: True 表示支撑线
        pivot: 枢轴点索引
        init_slope: 初始斜率（来自最小二乘法）
        y: 价格数据序列

    Returns:
        (最优斜率, 截距)
    """
    if hasattr(y, 'values'):
        y_arr = y.values
    else:
        y_arr = y

    # 斜率调整单位（基于价格范围）
    slope_unit = (y_arr.max() - y_arr.min()) / len(y_arr)

    # 优化参数
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    # 初始化最佳值
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)

    if best_err < 0:
        # 初始斜率无效，返回原值
        if hasattr(y, 'iloc'):
            intercept = -init_slope * pivot + y.iloc[pivot]
        else:
            intercept = -init_slope * pivot + y[pivot]
        return (init_slope, intercept)

    get_derivative = True
    derivative = None

    # 梯度下降迭代
    while curr_step > min_step:
        if get_derivative:
            # 数值微分：微小增加斜率，观察误差变化
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            # 如果增加斜率失败，尝试减少斜率
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                # 导数计算失败，退出
                break

            get_derivative = False

        # 根据导数方向调整斜率
        if derivative > 0.0:
            # 增加斜率导致误差增大，减小斜率
            test_slope = best_slope - slope_unit * curr_step
        else:
            # 增加斜率导致误差减小，增大斜率
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)

        if test_err < 0 or test_err >= best_err:
            # 新斜率无效或未改善，减小步长
            curr_step *= 0.5
        else:
            # 接受新斜率
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # 重新计算导数

    # 计算最终截距
    if hasattr(y, 'iloc'):
        intercept = -best_slope * pivot + y.iloc[pivot]
    else:
        intercept = -best_slope * pivot + y[pivot]

    return (best_slope, intercept)


def fit_trendlines_single(data: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    对单一价格序列（如收盘价）拟合趋势线

    步骤:
    1. 最小二乘法获取初始斜率
    2. 找到最大偏离点作为枢轴
    3. 梯度下降优化斜率

    Args:
        data: 价格序列 (numpy array 或 pandas Series)

    Returns:
        ((支撑线斜率, 截距), (阻力线斜率, 截距))
    """
    if hasattr(data, 'values'):
        data_arr = data.values
    else:
        data_arr = data

    # 最小二乘法获取初始斜率
    x = np.arange(len(data_arr))
    coefs = np.polyfit(x, data_arr, 1)

    # 计算初始趋势线
    line_points = coefs[0] * x + coefs[1]

    # 找枢轴点（最大偏离点）
    upper_pivot = (data_arr - line_points).argmax()  # 阻力线锚点
    lower_pivot = (data_arr - line_points).argmin()  # 支撑线锚点

    # 梯度下降优化
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    使用高低价拟合趋势线（更精确）

    与 QuantAgent graph_util.py 完全一致的实现

    使用收盘价计算初始斜率，然后:
    - 支撑线基于低价优化
    - 阻力线基于高价优化

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列

    Returns:
        ((支撑线斜率, 截距), (阻力线斜率, 截距))
    """
    # 获取 numpy array
    if hasattr(close, 'values'):
        close_arr = close.values
        high_arr = high.values
        low_arr = low.values
    else:
        close_arr = close
        high_arr = high
        low_arr = low

    # 最小二乘法获取初始斜率（基于收盘价）
    x = np.arange(len(close_arr))
    coefs = np.polyfit(x, close_arr, 1)

    # 计算初始趋势线
    line_points = coefs[0] * x + coefs[1]

    # 找枢轴点
    upper_pivot = (high_arr - line_points).argmax()  # 阻力线锚点（最高点最大偏离）
    lower_pivot = (low_arr - line_points).argmin()   # 支撑线锚点（最低点最小偏离）

    # 梯度下降优化：支撑线用低价，阻力线用高价
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def get_line_points(candles_len: int, slope: float, intercept: float) -> np.ndarray:
    """
    根据斜率和截距计算趋势线所有点的值

    Args:
        candles_len: K线数量
        slope: 趋势线斜率
        intercept: 趋势线截距

    Returns:
        趋势线上各点的价格值
    """
    x = np.arange(candles_len)
    return slope * x + intercept


class ChartGenerator:
    """
    K线图和趋势线图生成器

    参考 QuantAgent 的 graph_util.py 实现
    """

    # 默认图表配置
    DEFAULT_CONFIG = {
        "figsize": (12, 8),           # 图表尺寸
        "style": "charles",            # mplfinance 样式
        "num_candles": 50,             # 显示的K线数量
        "volume": True,                # 是否显示成交量
        "mav": (10, 20, 50),          # 移动平均线周期
        "dpi": 100,                    # 图片分辨率
    }

    # 经典形态参考图案（用于 PatternAgent 对比）
    PATTERN_TEMPLATES = {
        "head_and_shoulders": "头肩顶/底",
        "double_top": "双顶",
        "double_bottom": "双底",
        "triple_top": "三重顶",
        "triple_bottom": "三重底",
        "ascending_triangle": "上升三角形",
        "descending_triangle": "下降三角形",
        "symmetrical_triangle": "对称三角形",
        "rising_wedge": "上升楔形",
        "falling_wedge": "下降楔形",
        "bull_flag": "牛旗",
        "bear_flag": "熊旗",
        "cup_and_handle": "杯柄形态",
        "rounding_bottom": "圆底",
        "rounding_top": "圆顶",
        "channel": "通道"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化图表生成器

        Args:
            config: 图表配置选项
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # 检查依赖
        self._check_dependencies()

        logger.info(f"ChartGenerator 初始化完成: num_candles={self.config['num_candles']}")

    def _check_dependencies(self):
        """Check required dependencies"""
        missing = []
        if not HAS_PANDAS:
            missing.append("pandas")
        if not HAS_MPLFINANCE:
            missing.append("mplfinance")
        if not HAS_MATPLOTLIB:
            missing.append("matplotlib")

        if missing:
            logger.warning(f"Chart generation missing dependencies: {', '.join(missing)}")

    def generate_kline_image(
        self,
        ohlcv_data: pd.DataFrame,
        pair: str = "",
        timeframe: str = "",
        show_volume: bool = True,
        show_mav: bool = True,
        num_candles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        生成K线图

        参考 QuantAgent 的 generate_kline_image 实现

        Args:
            ohlcv_data: OHLCV 数据 DataFrame，需包含列:
                        open, high, low, close, volume (可选)
                        index 为 datetime
            pair: 交易对名称（用于标题）
            timeframe: 时间框架（用于标题）
            show_volume: 是否显示成交量
            show_mav: 是否显示移动平均线
            num_candles: 显示的K线数量（覆盖默认配置）

        Returns:
            {
                "success": bool,
                "image_base64": str,  # PNG 图片的 base64 编码
                "image_description": str,
                "error": str (可选)
            }
        """
        if not HAS_MPLFINANCE or not HAS_PANDAS:
            return {
                "success": False,
                "image_base64": "",
                "error": "缺少 mplfinance 或 pandas 依赖"
            }

        try:
            # 确保数据格式正确
            df = self._prepare_ohlcv_data(ohlcv_data, num_candles)

            if df is None or df.empty:
                return {
                    "success": False,
                    "image_base64": "",
                    "error": "OHLCV 数据为空或格式错误"
                }

            # 构建图表配置
            kwargs = {
                "type": "candle",
                "style": self.config["style"],
                "figsize": self.config["figsize"],
                "datetime_format": "%m-%d %H:%M",
                "returnfig": True,
            }

            # 添加成交量
            if show_volume and "volume" in df.columns:
                kwargs["volume"] = True

            # 添加移动平均线
            if show_mav:
                kwargs["mav"] = self.config["mav"]

            # 添加标题
            title = f"{pair} {timeframe}" if pair else "K-Line Chart"
            kwargs["title"] = title

            # 生成图表
            fig, axes = mpf.plot(df, **kwargs)

            # 转换为 base64
            image_base64 = self._fig_to_base64(fig)

            # 关闭图表释放内存
            plt.close(fig)

            return {
                "success": True,
                "image_base64": image_base64,
                "image_description": f"{pair} {timeframe} Candlestick chart with {len(df)} candles",
                "num_candles": len(df)
            }

        except Exception as e:
            logger.error(f"生成K线图失败: {e}")
            return {
                "success": False,
                "image_base64": "",
                "error": str(e)
            }

    def generate_trend_image(
        self,
        ohlcv_data: pd.DataFrame,
        pair: str = "",
        timeframe: str = "",
        num_candles: Optional[int] = None,
        auto_trendlines: bool = True,
        use_gradient_descent: bool = True
    ) -> Dict[str, Any]:
        """
        生成带趋势线的K线图

        参考 QuantAgent 的 generate_trend_image 实现
        使用梯度下降算法拟合更精准的支撑/阻力趋势线

        Args:
            ohlcv_data: OHLCV 数据 DataFrame
            pair: 交易对名称
            timeframe: 时间框架
            num_candles: 显示的K线数量
            auto_trendlines: 是否自动识别趋势线
            use_gradient_descent: 是否使用梯度下降优化（默认True）

        Returns:
            {
                "success": bool,
                "image_base64": str,
                "support_levels": List[float],
                "resistance_levels": List[float],
                "support_trendline": Dict,  # 支撑趋势线参数
                "resistance_trendline": Dict,  # 阻力趋势线参数
                "error": str (可选)
            }
        """
        if not HAS_MPLFINANCE or not HAS_PANDAS:
            return {
                "success": False,
                "image_base64": "",
                "error": "缺少 mplfinance 或 pandas 依赖"
            }

        try:
            # 准备数据
            df = self._prepare_ohlcv_data(ohlcv_data, num_candles)

            if df is None or df.empty:
                return {
                    "success": False,
                    "image_base64": "",
                    "error": "OHLCV 数据为空或格式错误"
                }

            # 趋势线数据
            support_line = None
            resist_line = None
            support_coefs = None
            resist_coefs = None

            # 使用梯度下降拟合趋势线
            if auto_trendlines and use_gradient_descent and HAS_PANDAS:
                try:
                    # 使用高低价拟合趋势线（QuantAgent 方式）
                    support_coefs, resist_coefs = fit_trendlines_high_low(
                        df['high'], df['low'], df['close']
                    )

                    # 计算趋势线数值
                    support_line = get_line_points(len(df), support_coefs[0], support_coefs[1])
                    resist_line = get_line_points(len(df), resist_coefs[0], resist_coefs[1])

                    logger.debug(
                        f"梯度下降趋势线: 支撑斜率={support_coefs[0]:.6f}, "
                        f"阻力斜率={resist_coefs[0]:.6f}"
                    )
                except Exception as e:
                    logger.warning(f"梯度下降趋势线拟合失败: {e}，使用备选方法")
                    support_line = None
                    resist_line = None

            # 备选：使用简单的关键价位识别
            support_levels, resistance_levels = [], []
            if auto_trendlines:
                support_levels, resistance_levels = self._identify_key_levels(df)

            # 构建附加图（趋势线）
            addplot = []
            if support_line is not None:
                addplot.append(mpf.make_addplot(
                    support_line,
                    color='green',
                    width=1.5,
                    linestyle='--',
                    label='Support'
                ))
            if resist_line is not None:
                addplot.append(mpf.make_addplot(
                    resist_line,
                    color='red',
                    width=1.5,
                    linestyle='--',
                    label='Resistance'
                ))

            # 构建图表
            kwargs = {
                "type": "candle",
                "style": self.config["style"],
                "figsize": self.config["figsize"],
                "datetime_format": "%m-%d %H:%M",
                "returnfig": True,
                "volume": "volume" in df.columns,
                "mav": self.config["mav"],
                "title": f"{pair} {timeframe} Trend Analysis (Gradient Descent)" if pair else "Trend Analysis",
            }

            # 添加趋势线
            if addplot:
                kwargs["addplot"] = addplot

            # 添加水平线（支撑/阻力价位）
            if support_levels or resistance_levels:
                all_levels = support_levels + resistance_levels
                colors = ["green"] * len(support_levels) + ["red"] * len(resistance_levels)
                kwargs["hlines"] = dict(
                    hlines=all_levels,
                    colors=colors,
                    linestyle=":",
                    linewidths=0.8,
                    alpha=0.6
                )

            # 生成图表
            fig, axes = mpf.plot(df, **kwargs)

            # 添加图例说明
            ax = axes[0] if isinstance(axes, list) else axes
            legend_elements = []
            if support_line is not None:
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], color='green', linestyle='--', linewidth=1.5,
                           label='Support Line')
                )
            if resist_line is not None:
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                           label='Resistance Line')
                )
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

            # 转换为 base64
            image_base64 = self._fig_to_base64(fig)
            plt.close(fig)

            # Build result
            result = {
                "success": True,
                "image_base64": image_base64,
                "image_description": f"{pair} {timeframe} Trend Chart (Gradient Descent)",
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "num_candles": len(df)
            }

            # 添加趋势线参数
            if support_coefs is not None:
                result["support_trendline"] = {
                    "slope": float(support_coefs[0]),
                    "intercept": float(support_coefs[1]),
                    "start_price": float(support_line[0]) if support_line is not None else None,
                    "end_price": float(support_line[-1]) if support_line is not None else None
                }
            if resist_coefs is not None:
                result["resistance_trendline"] = {
                    "slope": float(resist_coefs[0]),
                    "intercept": float(resist_coefs[1]),
                    "start_price": float(resist_line[0]) if resist_line is not None else None,
                    "end_price": float(resist_line[-1]) if resist_line is not None else None
                }

            return result

        except Exception as e:
            logger.error(f"生成趋势图失败: {e}")
            return {
                "success": False,
                "image_base64": "",
                "error": str(e)
            }

    def _prepare_ohlcv_data(
        self,
        data: pd.DataFrame,
        num_candles: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        准备 OHLCV 数据格式

        确保 DataFrame 格式符合 mplfinance 要求:
        - index 为 DatetimeIndex
        - 列名为小写: open, high, low, close, volume
        """
        if data is None or data.empty:
            return None

        try:
            df = data.copy()

            # 标准化列名（转小写）
            df.columns = [c.lower() for c in df.columns]

            # 检查必要列
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"缺少必要列: {col}")
                    return None

            # 确保 index 是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                else:
                    # 尝试将 index 转换为 datetime
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        logger.error("无法将 index 转换为 DatetimeIndex")
                        return None

            # 截取指定数量的K线
            n = num_candles or self.config["num_candles"]
            if len(df) > n:
                df = df.tail(n)

            return df

        except Exception as e:
            logger.error(f"准备 OHLCV 数据失败: {e}")
            return None

    def _identify_key_levels(
        self,
        df: pd.DataFrame,
        window: int = 5,
        num_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        识别关键支撑和阻力位

        使用局部极值法识别潜在的支撑和阻力位

        Args:
            df: OHLCV DataFrame
            window: 局部极值窗口大小
            num_levels: 返回的支撑/阻力位数量

        Returns:
            (support_levels, resistance_levels)
        """
        if not HAS_PANDAS:
            return [], []

        try:
            highs = df['high'].values
            lows = df['low'].values
            close = df['close'].values[-1]

            # 找局部最高点（潜在阻力）
            resistance_candidates = []
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    resistance_candidates.append(highs[i])

            # 找局部最低点（潜在支撑）
            support_candidates = []
            for i in range(window, len(lows) - window):
                if lows[i] == min(lows[i-window:i+window+1]):
                    support_candidates.append(lows[i])

            # 筛选：支撑位需要在当前价格下方，阻力位需要在上方
            support_levels = sorted(
                [s for s in support_candidates if s < close],
                reverse=True
            )[:num_levels]

            resistance_levels = sorted(
                [r for r in resistance_candidates if r > close]
            )[:num_levels]

            return support_levels, resistance_levels

        except Exception as e:
            logger.error(f"识别关键价位失败: {e}")
            return [], []

    def _fig_to_base64(self, fig) -> str:
        """
        将 matplotlib figure 转换为 base64 字符串

        Args:
            fig: matplotlib Figure 对象

        Returns:
            base64 编码的 PNG 图片字符串
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.config["dpi"],
                    bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64

    def get_pattern_templates(self) -> Dict[str, str]:
        """获取支持识别的形态列表"""
        return self.PATTERN_TEMPLATES.copy()

    @staticmethod
    def is_available() -> bool:
        """检查图表生成功能是否可用"""
        return HAS_PANDAS and HAS_MPLFINANCE and HAS_MATPLOTLIB


# 便捷函数
def generate_kline_chart(
    ohlcv_data: pd.DataFrame,
    pair: str = "",
    timeframe: str = ""
) -> Dict[str, Any]:
    """
    便捷函数：生成K线图

    Args:
        ohlcv_data: OHLCV 数据
        pair: 交易对
        timeframe: 时间框架

    Returns:
        包含 image_base64 的字典
    """
    generator = ChartGenerator()
    return generator.generate_kline_image(ohlcv_data, pair, timeframe)


def generate_trend_chart(
    ohlcv_data: pd.DataFrame,
    pair: str = "",
    timeframe: str = ""
) -> Dict[str, Any]:
    """
    便捷函数：生成趋势线图

    Args:
        ohlcv_data: OHLCV 数据
        pair: 交易对
        timeframe: 时间框架

    Returns:
        包含 image_base64 和 support/resistance levels 的字典
    """
    generator = ChartGenerator()
    return generator.generate_trend_image(ohlcv_data, pair, timeframe)
