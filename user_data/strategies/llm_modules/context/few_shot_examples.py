"""
Few-Shot格式示例：仅用于输出格式引导

设计原则（基于Google提示词工程白皮书）：
- 提供极简示例，仅展示输出格式
- 不包含具体交易逻辑，避免过拟合
- 详细的交易示例已被动态历史学习模块替代

Created: 2025-01-23
Updated: 2025-11-28 - 移除未使用的详细示例，仅保留格式引导
"""


def get_format_examples_entry() -> str:
    """
    获取极简的入场格式示例（仅用于格式引导，非交易逻辑）

    设计原则（基于Google提示词工程白皮书）：
    - 提供2-4个极短示例，仅展示输出格式
    - 不包含具体交易逻辑，避免过拟合
    - 覆盖主要决策类型：做多、做空、等待

    Returns:
        格式化的极简示例文本
    """
    return """<format_examples>
### 格式示例（仅展示输出结构）

**示例1 - 趋势做多：**
调用 signal_entry_long
  pair="BTC/USDT:USDT"
  leverage=8
  confidence_score=75
  reason="EMA20>50>200; ADX↑30; MACD柱扩张"
  key_support=41800.0
  key_resistance=44000.0

**示例2 - 信号不足等待：**
调用 signal_wait
  pair="ETH/USDT:USDT"
  confidence_score=40
  reason="4H下降vs30M反弹; 方向冲突; 等待突破"

**示例3 - 空头确认做空：**
调用 signal_entry_short
  pair="BNB/USDT:USDT"
  leverage=6
  confidence_score=70
  reason="EMA20<50<200; MACD死叉; 结构LH+LL"
  key_support=290.0
  key_resistance=315.0
</format_examples>"""


def get_format_examples_position() -> str:
    """
    获取极简的持仓管理格式示例（仅用于格式引导）

    Returns:
        格式化的极简示例文本
    """
    return """<format_examples>
### 格式示例（仅展示输出结构）

**示例1 - 锚点完好持有：**
调用 signal_hold
  pair="BTC/USDT:USDT"
  confidence_score=80
  reason="EMA多头保持; MACD金叉; ADX>25趋势延续"

**示例2 - MFE回撤减仓：**
调用 adjust_position
  pair="ETH/USDT:USDT"
  adjustment_pct=-30
  confidence_score=75
  reason="MFE回撤33%; 动量收窄; 锁定部分利润"
  key_support=2250.0
  key_resistance=2400.0

**示例3 - 锚点破坏平仓：**
调用 signal_exit
  pair="BNB/USDT:USDT"
  trade_score=40
  confidence_score=90
  reason="跌破支撑2285; MACD死叉; 策略失效"
</format_examples>"""
