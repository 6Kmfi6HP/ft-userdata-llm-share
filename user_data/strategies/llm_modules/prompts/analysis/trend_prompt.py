"""
Trend Structure Analysis Agent Prompt.

Focuses on price structure, trend direction, support/resistance levels.
Supports both text and vision analysis modes.
"""

TREND_SYSTEM_PROMPT = """你是一位专业的加密货币趋势分析师。

你的专长是分析价格结构和趋势，包括：
- EMA均线系统：分析EMA20/50/200的排列和距离
- 价格结构：识别更高高点(HH)、更高低点(HL)、更低高点(LH)、更低低点(LL)
- 支撑阻力：识别关键价格区域和转折点
- 趋势阶段：判断趋势的初期、中期、末期或转折期
- 突破确认：判断价格突破的有效性
- 趋势线分析：识别上升/下降趋势线、通道

分析原则：
1. 趋势是你的朋友，顺势而为
2. 更高时间框架的趋势优先级更高
3. 支撑阻力位需要多次验证才更可靠
4. 关注趋势的动量和结构变化
5. 保持客观，识别趋势但不预测转折点
6. 趋势线突破需要成交量确认

你只负责趋势分析，不做最终交易决策。"""


TREND_ANALYSIS_FOCUS = """## 趋势结构分析任务

请重点分析以下方面：

### 1. EMA均线结构
- EMA20/50/200的相对位置
- 是否形成多头排列（价格>EMA20>EMA50>EMA200）或空头排列
- 价格与各均线的距离（用ATR衡量）
- 均线的斜率和方向

### 2. 价格结构分析
- 识别最近的重要高点和低点
- 判断是否形成更高高点(HH)/更高低点(HL)（上升趋势）
- 或更低高点(LH)/更低低点(LL)（下降趋势）
- 当前价格在结构中的位置

### 3. 支撑与阻力
- 识别关键支撑位（多次反弹的价格区域）
- 识别关键阻力位（多次受阻的价格区域）
- 评估当前价格距离关键位置的距离
- 这些关键位是否被测试或突破

### 4. 趋势阶段判断
- 初期：刚形成，动量强
- 中期：稳定运行，可能有回调
- 末期：动量减弱，可能反转
- 转折期：趋势正在改变
- 震荡：无明显趋势

### 5. 多时间框架分析（如有数据）
- 更高时间框架的趋势方向
- 是否与当前时间框架一致
- 时间框架间的支撑阻力对齐

### 6. 突破分析
- 是否存在突破关键位的迹象
- 突破的有效性判断（成交量确认、回测确认）

请基于以上分析，给出方向判断、关键价位和置信度。"""


TREND_VISION_FOCUS = """## 趋势线视觉分析任务

请仔细观察这张带趋势线的K线图，分析以下方面：

### 1. 趋势线分析
- **支撑趋势线（绿色）**: 斜率如何？是否有效支撑价格？
- **阻力趋势线（红色）**: 斜率如何？是否有效压制价格？
- **通道识别**: 是否形成上升/下降/横盘通道？
- **趋势线角度**: 陡峭（强势）还是平缓（弱势）？

### 2. 价格与趋势线关系
- 当前价格距离支撑线多远？
- 当前价格距离阻力线多远？
- 价格是否正在测试趋势线？
- 是否有突破趋势线的迹象？

### 3. 趋势方向判断
- 支撑线和阻力线是否同向（平行通道）？
- 是否收敛（三角形）或发散？
- 主趋势方向是什么？

### 4. 关键价位识别
- 从图中识别最近的支撑价位
- 从图中识别最近的阻力价位
- 趋势线与当前价格的交汇点

### 5. 均线系统（如图中显示）
- 均线排列顺序
- 价格与均线的位置关系"""


TREND_MULTI_TF_VISION_FOCUS = """## 多时间周期趋势线视觉分析任务

你将收到两张不同时间周期的趋势线图：
- **第一张图（短周期）**: {timeframe_primary} - 短期趋势和精确入场位
- **第二张图（长周期）**: {timeframe_htf} - 主趋势方向和关键支撑阻力

### 分析原则
1. **大趋势优先**: 长周期确定主要方向，短周期找入场点
2. **趋势共振**: 两个周期趋势方向一致时更可靠
3. **关键位对齐**: 长周期的支撑阻力比短周期更重要

### 短周期趋势分析 ({timeframe_primary}, 覆盖 {coverage_primary:.1f} 小时)

#### 1. 短期趋势线
- 支撑趋势线（绿色）斜率和有效性
- 阻力趋势线（红色）斜率和有效性
- 短期通道方向

#### 2. 短期价格结构
- 是否形成更高高点(HH)/更高低点(HL)？
- 还是更低高点(LH)/更低低点(LL)？
- 短期趋势阶段（初期/中期/末期）

### 长周期趋势分析 ({timeframe_htf}, 覆盖 {coverage_htf:.1f} 小时)

#### 3. 主趋势线
- 长期支撑趋势线：斜率、测试次数、有效性
- 长期阻力趋势线：斜率、测试次数、有效性
- 主通道类型（上升/下降/横盘）

#### 4. 主趋势结构
- 主趋势方向（上升/下降/震荡）
- 当前处于主趋势的什么阶段？
- 是否接近主趋势的关键转折点？

### 多时间周期综合判断

#### 5. 趋势共振分析
- 短周期趋势与长周期趋势是否一致？
- 共振程度评估（强共振/弱共振/矛盾）
- 矛盾时的处理建议

#### 6. 关键价位综合
- 长周期支撑位（更重要）
- 长周期阻力位（更重要）
- 短周期入场参考位

#### 7. 趋势线状态综合
- 长周期趋势线方向
- 短周期趋势线方向
- 是否存在突破信号？"""


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


def build_trend_prompt(market_context: str) -> str:
    """Build complete trend analysis prompt."""
    return f"""# 分析任务

{TREND_ANALYSIS_FOCUS}

# 市场数据

{market_context}

{OUTPUT_FORMAT}"""


def build_trend_vision_prompt(market_context: str, pair: str, trendline_info: dict = None) -> str:
    """Build trend vision analysis prompt."""
    trendline_context = ""
    if trendline_info:
        support_tl = trendline_info.get("support_trendline")
        resist_tl = trendline_info.get("resistance_trendline")

        if support_tl or resist_tl:
            trendline_context = "\n## 趋势线参数（算法计算结果）\n"
            if support_tl:
                trendline_context += f"- 支撑线: 斜率={support_tl.get('slope', 0):.6f}\n"
            if resist_tl:
                trendline_context += f"- 阻力线: 斜率={resist_tl.get('slope', 0):.6f}\n"

    return f"""# {pair} 趋势线视觉分析

{TREND_VISION_FOCUS}
{trendline_context}

# 补充市场信息（供参考）

{market_context}

{OUTPUT_FORMAT}

请基于趋势线图进行视觉分析，判断趋势方向和关键价位。"""


def build_trend_multi_tf_vision_prompt(
    market_context: str,
    pair: str,
    timeframe_primary: str = "15m",
    timeframe_htf: str = "1h",
    coverage_primary: float = 12.5,
    coverage_htf: float = 50.0,
    trendline_info_primary: dict = None,
    trendline_info_htf: dict = None
) -> str:
    """
    Build multi-timeframe trend vision analysis prompt.

    Args:
        market_context: Market context string
        pair: Trading pair
        timeframe_primary: Primary (short) timeframe
        timeframe_htf: Higher timeframe
        coverage_primary: Hours covered by primary chart
        coverage_htf: Hours covered by HTF chart
        trendline_info_primary: Trendline info for primary timeframe
        trendline_info_htf: Trendline info for HTF

    Returns:
        Formatted prompt for multi-timeframe trend analysis
    """
    focus_text = TREND_MULTI_TF_VISION_FOCUS.format(
        timeframe_primary=timeframe_primary,
        timeframe_htf=timeframe_htf,
        coverage_primary=coverage_primary,
        coverage_htf=coverage_htf
    )

    # Build trendline context for both timeframes
    trendline_context = ""

    if trendline_info_primary:
        support_tl = trendline_info_primary.get("support_trendline")
        resist_tl = trendline_info_primary.get("resistance_trendline")
        if support_tl or resist_tl:
            trendline_context += f"\n## {timeframe_primary} 趋势线参数\n"
            if support_tl:
                trendline_context += f"- 支撑线: 斜率={support_tl.get('slope', 0):.6f}\n"
            if resist_tl:
                trendline_context += f"- 阻力线: 斜率={resist_tl.get('slope', 0):.6f}\n"

    if trendline_info_htf:
        support_tl = trendline_info_htf.get("support_trendline")
        resist_tl = trendline_info_htf.get("resistance_trendline")
        if support_tl or resist_tl:
            trendline_context += f"\n## {timeframe_htf} 趋势线参数\n"
            if support_tl:
                trendline_context += f"- 支撑线: 斜率={support_tl.get('slope', 0):.6f}\n"
            if resist_tl:
                trendline_context += f"- 阻力线: 斜率={resist_tl.get('slope', 0):.6f}\n"

    return f"""# {pair} 多时间周期趋势线视觉分析

{focus_text}
{trendline_context}

# 补充市场信息（供参考）

{market_context}

{OUTPUT_FORMAT}

请基于两张趋势线图进行多时间周期视觉分析，综合判断趋势方向和关键价位。"""
