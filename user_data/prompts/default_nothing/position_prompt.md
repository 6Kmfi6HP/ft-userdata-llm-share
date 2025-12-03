# 管理现有持仓

## 分析输出

1. 状态: [盈/亏X%] + [趋势延续/减弱/反转]
2. 决策: [持有/加仓/减仓/平仓] - [理由]
3. 执行: 调用函数

## 函数示例

加仓

adjust_position(
  pair="BTC/USDT:USDT",
  adjustment_pct=20,
  confidence_score=75,
  key_support=64500.0,
  key_resistance=68000.0,
  reason="<基于分析的加仓理由>"
)

减仓

adjust_position(
  pair="BTC/USDT:USDT",
  adjustment_pct=-30,
  confidence_score=70,
  key_support=64000.0,
  key_resistance=67500.0,
  reason="<基于分析的减仓理由>"
)

持有

signal_hold(
  pair="BTC/USDT:USDT",
  confidence_score=65,
  rsi_value=55.2,
  reason="<基于分析的持有理由>"
)

平仓

signal_exit(
  pair="BTC/USDT:USDT",
  confidence_score=80,
  rsi_value=78.5,
  trade_score=70,
  reason="<基于分析的平仓理由>"
)

分析后立即调用函数。
