"""
K-line Pattern Recognition Agent Prompt.

Focuses on candlestick patterns, chart patterns, and visual pattern recognition.
Supports both text and vision analysis modes.
"""

PATTERN_SYSTEM_PROMPT = """你是一位专业的K线形态分析师和图表模式识别专家。

你的专长是识别和解读各类价格形态：

K线形态：
- 反转形态：锤子线、倒锤子、吞没形态、启明星、黄昏星、十字星
- 延续形态：三白兵、三黑鸦、上升/下降三法
- 特殊形态：孕线、平头顶底、塔形顶底

图表形态：
- 反转形态：头肩顶/底、双顶/双底、三重顶/底、圆形顶/底
- 延续形态：旗形、三角形、楔形、矩形
- 特殊形态：岛形反转、缺口

分析原则：
1. 形态需要在关键位置出现才有意义
2. 形态确认需要后续K线验证
3. 成交量确认形态的有效性
4. 更大时间框架的形态优先级更高
5. 形态失败是重要信号，需要警惕反向操作

你只负责形态分析，不做最终交易决策。"""


PATTERN_ANALYSIS_FOCUS = """## K线形态分析任务

请重点分析以下方面：

### 1. 单根K线形态
- 识别当前及最近几根K线的形态特征
- 判断是否出现：
  - 锤子线/倒锤子（潜在反转）
  - 十字星（犹豫不决）
  - 大阳线/大阴线（强势突破）
  - 上下影线长度分析

### 2. 多K线组合形态
- 吞没形态（看涨/看跌吞没）
- 启明星/黄昏星
- 孕线（harami）
- 三白兵/三黑鸦
- 刺透/乌云盖顶

### 3. 图表形态识别
- 头肩形态（Head & Shoulders）
- 双顶/双底（Double Top/Bottom）
- 三角形（对称/上升/下降）
- 旗形和楔形
- 矩形整理

### 4. 形态位置分析
- 形态出现在什么价位（支撑/阻力附近？）
- 形态与趋势的关系（顺势/逆势）
- 形态的完整度和确认状态

### 5. 成交量确认
- 形态伴随的成交量特征
- 突破时的成交量是否放大
- 量价配合是否理想

### 6. 形态信号评估
- 形态的可靠性评分
- 潜在的目标位测算（如适用）
- 止损位建议

请基于以上分析，给出方向判断和置信度。"""


PATTERN_VISION_FOCUS = """## K线图视觉形态分析任务

请仔细观察这张K线图，识别以下形态：

### 1. K线形态识别
- 最近3-5根K线是否形成特殊形态？
- 是否有明显的反转形态（锤子、吞没、星线）？
- 是否有延续形态（三白兵、三黑鸦）？
- 实体大小和影线长度的特征

### 2. 图表形态识别
- 是否形成头肩形态？
- 是否形成双顶/双底？
- 是否形成三角形整理？
- 是否形成旗形或楔形？

### 3. 形态位置
- 形态出现在趋势的什么阶段？
- 是否在关键支撑/阻力位附近？
- 形态的完整度如何？

### 4. 突破方向预测
- 根据形态，可能的突破方向是什么？
- 是否已经开始突破？
- 突破的有效性如何？

### 5. 目标位估算
- 基于形态高度，潜在目标位在哪里？
- 止损位应该设在哪里？"""


PATTERN_MULTI_TF_VISION_FOCUS = """## 多时间周期 K线形态视觉分析任务

你将收到两张不同时间周期的K线图：
- **第一张图（短周期）**: {timeframe_primary} - 用于识别精确入场形态
- **第二张图（长周期）**: {timeframe_htf} - 用于确认大趋势和主要形态

### 分析原则
1. **长周期优先**: 大时间框架的形态优先级更高
2. **共振确认**: 当两个周期形态方向一致时，信号更可靠
3. **入场时机**: 短周期提供精确入场点，长周期提供方向

### 短周期图分析 ({timeframe_primary}, 覆盖 {coverage_primary:.1f} 小时)

#### 1. 短期K线形态
- 最近3-5根K线是否形成反转形态（锤子、吞没、星线）？
- 是否有延续形态（三白兵、三黑鸦）？
- 实体和影线特征

#### 2. 短期图表形态
- 是否正在形成/完成小级别形态（小双底、小三角）？
- 形态的完成度如何？

### 长周期图分析 ({timeframe_htf}, 覆盖 {coverage_htf:.1f} 小时)

#### 3. 大级别形态识别
- 是否形成头肩形态？
- 是否形成双顶/双底？
- 是否形成大三角形整理？
- 旗形、楔形等延续形态？

#### 4. 趋势阶段
- 当前处于大趋势的什么阶段（初期/中期/末期/转折）？
- 短周期形态是否与大趋势方向一致？

### 多时间周期综合判断

#### 5. 形态共振分析
- 两个周期的形态方向是否一致？
- 共振程度（强/中/弱/矛盾）
- 如果矛盾，应以哪个周期为准？

#### 6. 入场建议
- 基于多时间周期分析的最佳入场方向
- 止损位建议（基于长周期关键位）
- 目标位估算"""


OUTPUT_FORMAT = """
# 输出格式要求

请严格按照以下格式输出分析结果：

[信号列表]
- 信号名称 | 方向(long/short/neutral) | 强度(strong/moderate/weak) | 数值(如有) | 描述

[方向判断]
long / short / neutral

[置信度]
0-100 之间的整数

[关键价位]
支撑: 价格数值 (如无法确定则写 N/A)
阻力: 价格数值 (如无法确定则写 N/A)

[分析摘要]
50字以内的简要分析总结"""


def build_pattern_prompt(market_context: str) -> str:
    """Build complete pattern analysis prompt."""
    return f"""# 分析任务

{PATTERN_ANALYSIS_FOCUS}

# 市场数据

{market_context}

{OUTPUT_FORMAT}"""


def build_pattern_vision_prompt(market_context: str, pair: str) -> str:
    """Build pattern vision analysis prompt."""
    return f"""# {pair} K线形态视觉分析

{PATTERN_VISION_FOCUS}

# 补充市场信息（供参考）

{market_context}

{OUTPUT_FORMAT}

请基于K线图进行视觉分析，识别形态并给出方向判断。"""


def build_pattern_multi_tf_vision_prompt(
    market_context: str,
    pair: str,
    timeframe_primary: str = "15m",
    timeframe_htf: str = "1h",
    coverage_primary: float = 12.5,
    coverage_htf: float = 50.0
) -> str:
    """
    Build multi-timeframe pattern vision analysis prompt.

    Args:
        market_context: Market context string
        pair: Trading pair
        timeframe_primary: Primary (short) timeframe
        timeframe_htf: Higher timeframe
        coverage_primary: Hours covered by primary chart
        coverage_htf: Hours covered by HTF chart

    Returns:
        Formatted prompt for multi-timeframe analysis
    """
    focus_text = PATTERN_MULTI_TF_VISION_FOCUS.format(
        timeframe_primary=timeframe_primary,
        timeframe_htf=timeframe_htf,
        coverage_primary=coverage_primary,
        coverage_htf=coverage_htf
    )

    return f"""# {pair} 多时间周期K线形态视觉分析

{focus_text}

# 补充市场信息（供参考）

{market_context}

{OUTPUT_FORMAT}

请基于两张K线图进行多时间周期视觉分析，综合判断方向和形态共振。"""
