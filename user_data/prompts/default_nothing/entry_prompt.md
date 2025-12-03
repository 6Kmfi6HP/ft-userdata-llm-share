# 根据市场数据判断是否开仓

## 分析输出

1. 市场: [趋势/震荡/反转] - [核心依据]
2. 方向: [多/空/观望] - [触发信号]
3. 执行: 调用函数

## 函数示例

做多

signal_entry_long(
  pair="BTC/USDT:USDT",
  leverage=5,
  confidence_score=75,
  key_support=64500.0,
  key_resistance=68000.0,
  rsi_value=45.2,
  trend_strength="strong",
  reason="<基于分析的开仓理由>"
)

做空

signal_entry_short(
  pair="BTC/USDT:USDT",
  leverage=5,
  confidence_score=80,
  key_support=62000.0,
  key_resistance=65500.0,
  rsi_value=72.5,
  trend_strength="moderate",
  reason="<基于分析的开仓理由>"
)

观望

signal_wait(
  pair="BTC/USDT:USDT",
  confidence_score=40,
  rsi_value=50.3,
  reason="<基于分析的等待理由>"
)

分析后立即调用函数。
