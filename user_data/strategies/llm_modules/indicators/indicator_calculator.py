"""
技术指标计算器模块
统一计算所有技术指标，避免重复代码
"""
import logging
import pandas as pd
import talib.abstract as ta

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def add_trend_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加趋势指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了趋势指标的DataFrame
        """
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)  # EMA200策略核心指标

        return dataframe

    @staticmethod
    def add_momentum_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加动量指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了动量指标的DataFrame
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # ROC (Rate of Change) - 14日动量
        # 基于学术论文: Begušić & Kostanjčar 2019 "Momentum and Liquidity in Cryptocurrencies"
        dataframe['roc_14'] = ta.ROC(dataframe, timeperiod=14)

        # Stochastic 随机指标 - 短期超买超卖判断
        # 参考 QuantAgent 实现: fastk_period=14, slowk_period=3, slowd_period=3
        # %K > 80 超买, %K < 20 超卖; %K与%D交叉为信号
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Williams %R - 类似RSI但更敏感的超买超卖指标
        # 范围 -100 到 0: < -80 超卖, > -20 超买
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def add_liquidity_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加流动性指标

        基于学术论文: Begušić & Kostanjčar 2019
        Amihud非流动性指标: |收益率| / 成交量
        高Amihud值 = 低流动性

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了流动性指标的DataFrame
        """
        # 确保依赖列存在 (volma_20 由 add_volume_indicators 添加)
        if 'volma_20' not in dataframe.columns:
            dataframe['volma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        # 计算日收益率
        dataframe['daily_return'] = dataframe['close'].pct_change()

        # Amihud 非流动性指标
        # 公式: Amihud = |return| / volume (缩放后)
        volume_scaled = dataframe['volume'] / 1e6  # 缩放到百万单位
        # 避免除零
        volume_safe = volume_scaled.replace(0, 1e-10)
        dataframe['amihud_raw'] = abs(dataframe['daily_return']) / volume_safe

        # 14日滚动平均
        dataframe['amihud_14'] = dataframe['amihud_raw'].rolling(window=14).mean()

        # 流动性排名指标 (越高流动性越好，与amihud相反)
        # 使用成交量相对于均值的比率
        dataframe['liquidity_ratio'] = dataframe['volume'] / dataframe['volma_20'].replace(0, 1)

        return dataframe

    @staticmethod
    def add_volatility_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加波动率指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了波动率指标的DataFrame
        """
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def add_trend_strength_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加趋势强度指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了趋势强度指标的DataFrame
        """
        # ADX - 趋势强度 (不含方向)
        # > 25 强趋势, 20-25 中等, < 20 弱趋势/震荡
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # +DI/-DI - 方向性指标 (Directional Indicators)
        # +DI > -DI = 上升趋势主导 (看多信号)
        # -DI > +DI = 下降趋势主导 (看空信号)
        # +DI 与 -DI 交叉 = 趋势反转信号
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def add_volume_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了成交量指标的DataFrame
        """
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['volma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)  # VOLMA20成交量验证 (EMA200策略用)
        dataframe['volume_ma'] = dataframe['volma_20']  # 向后兼容别名

        return dataframe

    @staticmethod
    def add_all_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有技术指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了所有指标的DataFrame
        """
        dataframe = IndicatorCalculator.add_trend_indicators(dataframe)
        dataframe = IndicatorCalculator.add_momentum_indicators(dataframe)
        dataframe = IndicatorCalculator.add_volatility_indicators(dataframe)
        dataframe = IndicatorCalculator.add_trend_strength_indicators(dataframe)
        dataframe = IndicatorCalculator.add_volume_indicators(dataframe)
        # 流动性指标依赖 volma_20，需在volume指标之后添加
        dataframe = IndicatorCalculator.add_liquidity_indicators(dataframe)

        return dataframe
