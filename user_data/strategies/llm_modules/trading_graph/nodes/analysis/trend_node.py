"""
Trend Analysis Node for LangGraph.

Trend structure analysis: EMA structure, price structure, support/resistance.
Supports both text and vision analysis modes.
"""

import logging
import re
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from ...state import (
    AnalysisState,
    AgentReport,
    Signal,
    Direction,
    SignalStrength,
)
from ....prompts.analysis.trend_prompt import (
    TREND_SYSTEM_PROMPT,
    build_trend_prompt,
    build_trend_vision_prompt,
    build_trend_multi_tf_vision_prompt,
)
from ....lc_integration.llm_factory import LLMFactory, create_vision_model_with_retry

logger = logging.getLogger(__name__)


def trend_node(state: AnalysisState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Trend structure analysis node.

    Analyzes EMA structure, price structure, support/resistance levels.
    Supports vision mode with single or multi-timeframe charts.

    Args:
        state: Current analysis state
        config: Node configuration (contains llm_config)

    Returns:
        Dict with agent_reports and trend_report updates
    """
    pair = state.get("pair", "UNKNOWN")
    market_context = state.get("market_context", "")
    ohlcv_data = state.get("ohlcv_data")
    ohlcv_data_htf = state.get("ohlcv_data_htf")
    timeframe = state.get("timeframe", "15m")
    timeframe_htf = state.get("timeframe_htf", "1h")

    logger.debug(f"[TrendAgent] Starting analysis for {pair}")

    try:
        # Get LLM config from graph config
        llm_config = state.get("llm_config") or {}
        if not llm_config and config:
            llm_config = config.get("configurable", {}).get("llm_config", {})

        # Decide analysis mode
        use_vision = False
        use_multi_tf = False
        chart_images = []
        coverage_info = {}
        trendline_info_primary = {}
        trendline_info_htf = {}

        # Validate OHLCV data format
        def validate_ohlcv(data):
            if data is None:
                return None
            try:
                required_cols = ['open', 'high', 'low', 'close']
                cols_lower = [c.lower() for c in data.columns]
                if not all(col in cols_lower for col in required_cols):
                    return None
                return data
            except Exception:
                return None

        ohlcv_data = validate_ohlcv(ohlcv_data)
        ohlcv_data_htf = validate_ohlcv(ohlcv_data_htf)

        # Try to generate charts if OHLCV data available
        vision_enabled = llm_config.get("use_vision", False)
        multi_tf_enabled = llm_config.get("multi_timeframe_vision", True)

        if ohlcv_data is not None and vision_enabled:
            try:
                from ....utils.chart_generator import ChartGenerator
                chart_gen = ChartGenerator({"num_candles": 50})

                # Check if multi-timeframe is available and enabled
                if ohlcv_data_htf is not None and multi_tf_enabled:
                    # Generate multi-timeframe charts
                    mtf_result = chart_gen.generate_multi_timeframe_charts(
                        ohlcv_primary=ohlcv_data,
                        ohlcv_htf=ohlcv_data_htf,
                        pair=pair,
                        timeframe_primary=timeframe,
                        timeframe_htf=timeframe_htf,
                        chart_type="trend",
                        num_candles_primary=50,
                        num_candles_htf=50
                    )

                    if mtf_result.get("success"):
                        primary_chart = mtf_result.get("primary_chart")
                        htf_chart = mtf_result.get("htf_chart")

                        if primary_chart and primary_chart.get("image_base64"):
                            chart_images.append(primary_chart["image_base64"])
                            coverage_info["primary"] = primary_chart.get("coverage_hours", 12.5)
                            trendline_info_primary = {
                                "support_trendline": primary_chart.get("support_trendline"),
                                "resistance_trendline": primary_chart.get("resistance_trendline"),
                            }

                        if htf_chart and htf_chart.get("image_base64"):
                            chart_images.append(htf_chart["image_base64"])
                            coverage_info["htf"] = htf_chart.get("coverage_hours", 50.0)
                            trendline_info_htf = {
                                "support_trendline": htf_chart.get("support_trendline"),
                                "resistance_trendline": htf_chart.get("resistance_trendline"),
                            }
                            use_multi_tf = True

                        if chart_images:
                            use_vision = True
                            logger.debug(
                                f"[TrendAgent] Multi-TF Vision mode: "
                                f"{timeframe}({coverage_info.get('primary', 0):.1f}h) + "
                                f"{timeframe_htf}({coverage_info.get('htf', 0):.1f}h)"
                            )
                else:
                    # Single timeframe chart
                    chart_result = chart_gen.generate_trend_image(
                        ohlcv_data,
                        pair=pair,
                        num_candles=50,
                        use_gradient_descent=True
                    )
                    if chart_result and chart_result.get("success") and chart_result.get("image_base64"):
                        use_vision = True
                        chart_images.append(chart_result["image_base64"])
                        coverage_info["primary"] = chart_result.get("coverage_hours", 12.5)
                        trendline_info_primary = {
                            "support_trendline": chart_result.get("support_trendline"),
                            "resistance_trendline": chart_result.get("resistance_trendline"),
                        }
                        logger.debug(f"[TrendAgent] Single-TF Vision mode enabled")

            except Exception as e:
                logger.debug(f"[TrendAgent] Chart generation failed: {e}, using text mode")
                use_vision = False
                chart_images = []

        # Create LLM instance and invoke
        response = None

        # Try vision mode first if enabled
        if use_vision and chart_images:
            try:
                from langchain_core.messages import HumanMessage

                llm = create_vision_model_with_retry(llm_config)

                # Build prompt based on single or multi-timeframe
                if use_multi_tf and len(chart_images) >= 2:
                    prompt = build_trend_multi_tf_vision_prompt(
                        market_context=market_context,
                        pair=pair,
                        timeframe_primary=timeframe,
                        timeframe_htf=timeframe_htf,
                        coverage_primary=coverage_info.get("primary", 12.5),
                        coverage_htf=coverage_info.get("htf", 50.0),
                        trendline_info_primary=trendline_info_primary,
                        trendline_info_htf=trendline_info_htf
                    )
                else:
                    prompt = build_trend_vision_prompt(market_context, pair, trendline_info_primary)

                # Build message content with multiple images
                message_content = [{"type": "text", "text": prompt}]
                for i, img_base64 in enumerate(chart_images):
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    })

                response = llm.invoke([
                    {"role": "system", "content": TREND_SYSTEM_PROMPT},
                    HumanMessage(content=message_content)
                ])
            except Exception as e:
                logger.warning(f"[TrendAgent] Vision call failed: {e}, falling back to text mode")
                use_vision = False
                use_multi_tf = False
                response = None

        # Fall back to text mode if vision failed or was disabled
        if response is None:
            llm = LLMFactory.create_chat_model(llm_config, task_type="trend")
            prompt = build_trend_prompt(market_context)

            messages = [
                {"role": "system", "content": TREND_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            response = llm.invoke(messages)

        response_text = response.content if hasattr(response, 'content') else str(response)

        if not response_text:
            return _create_error_result("LLM returned empty response")

        # Parse response - use HTF trendline info if available (more important)
        trendline_info = trendline_info_htf if trendline_info_htf else trendline_info_primary
        parsed = _parse_trend_response(response_text, trendline_info)

        # Create report with mode tag
        if use_multi_tf:
            mode_tag = f"[Vision-MTF:{timeframe}+{timeframe_htf}]"
        elif use_vision:
            mode_tag = f"[Vision:{timeframe}]"
        else:
            mode_tag = "[Text]"

        report = AgentReport(
            agent_name="TrendAgent",
            analysis=f"{mode_tag}\n{response_text}",
            signals=parsed["signals"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            key_levels=parsed["key_levels"]
        )

        logger.info(
            f"[TrendAgent] {pair} {mode_tag} analysis complete: "
            f"direction={report.direction}, confidence={report.confidence:.0f}%"
        )

        return {
            "agent_reports": [report],
            "trend_report": report
        }

    except Exception as e:
        logger.error(f"[TrendAgent] Analysis failed: {e}")
        return _create_error_result(str(e))


def _create_error_result(error_msg: str) -> Dict[str, Any]:
    """Create error result with empty report."""
    error_report = AgentReport(
        agent_name="TrendAgent",
        analysis=f"Analysis failed: {error_msg}",
        signals=[],
        confidence=0.0,
        direction=None,
        error=error_msg
    )
    return {
        "agent_reports": [error_report],
        "trend_report": error_report,
        "errors": [f"TrendAgent: {error_msg}"]
    }


def _parse_trend_response(response: str, trendline_info: dict = None) -> Dict[str, Any]:
    """Parse trend analysis response."""
    result = {
        "signals": [],
        "direction": Direction.NEUTRAL,
        "confidence": 50.0,
        "key_levels": {"support": None, "resistance": None}
    }

    if not response:
        return result

    lines = response.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section markers
        if '[信号列表]' in line:
            current_section = 'signals'
            continue
        elif '[方向判断]' in line:
            current_section = 'direction'
            continue
        elif '[置信度]' in line:
            current_section = 'confidence'
            continue
        elif '[关键价位]' in line:
            current_section = 'key_levels'
            continue
        elif '[趋势线状态]' in line:
            current_section = 'trendline_status'
            continue
        elif '[分析摘要]' in line:
            current_section = 'summary'
            continue

        # Parse based on section
        if current_section == 'signals' and line.startswith('- '):
            signal = _parse_signal_line(line[2:])
            if signal:
                result['signals'].append(signal)

        elif current_section == 'direction':
            direction = _parse_direction(line)
            if direction:
                result['direction'] = direction

        elif current_section == 'confidence':
            confidence = _parse_confidence(line)
            if confidence is not None:
                result['confidence'] = confidence

        elif current_section == 'key_levels':
            level = _parse_key_level(line)
            if level:
                key, value = level
                result['key_levels'][key] = value

        elif current_section == 'trendline_status':
            # Parse trendline status and add as signals
            if '上升' in line.lower():
                if '支撑' in line:
                    result['signals'].append(Signal(
                        signal_type="支撑趋势线上升",
                        description="支撑线斜率为正",
                        strength=SignalStrength.MEDIUM,
                        direction=Direction.LONG
                    ))
                elif '阻力' in line:
                    result['signals'].append(Signal(
                        signal_type="阻力趋势线上升",
                        description="阻力线斜率为正",
                        strength=SignalStrength.WEAK,
                        direction=Direction.LONG
                    ))
            elif '下降' in line.lower():
                if '支撑' in line:
                    result['signals'].append(Signal(
                        signal_type="支撑趋势线下降",
                        description="支撑线斜率为负",
                        strength=SignalStrength.WEAK,
                        direction=Direction.SHORT
                    ))
                elif '阻力' in line:
                    result['signals'].append(Signal(
                        signal_type="阻力趋势线下降",
                        description="阻力线斜率为负",
                        strength=SignalStrength.MEDIUM,
                        direction=Direction.SHORT
                    ))

    # Supplement key levels from trendline info if not parsed
    if trendline_info:
        if not result['key_levels'].get('support'):
            support_tl = trendline_info.get('support_trendline')
            if support_tl:
                result['key_levels']['support'] = support_tl.get('end_price')
        if not result['key_levels'].get('resistance'):
            resist_tl = trendline_info.get('resistance_trendline')
            if resist_tl:
                result['key_levels']['resistance'] = resist_tl.get('end_price')

    return result


def _parse_signal_line(line: str) -> Optional[Signal]:
    """Parse a single signal line."""
    try:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            return None

        name = parts[0]
        direction = _parse_direction(parts[1]) or Direction.NEUTRAL
        strength = _parse_strength(parts[2])
        description = parts[4] if len(parts) > 4 else ""

        return Signal(
            signal_type=name,
            description=description,
            strength=strength,
            direction=direction
        )
    except Exception:
        return None


def _parse_direction(text: str) -> Optional[Direction]:
    """Parse direction from text."""
    text_lower = text.lower().strip()
    if 'long' in text_lower or '做多' in text_lower or '多' == text_lower:
        return Direction.LONG
    elif 'short' in text_lower or '做空' in text_lower or '空' == text_lower:
        return Direction.SHORT
    elif 'neutral' in text_lower or '中性' in text_lower:
        return Direction.NEUTRAL
    return None


def _parse_strength(text: str) -> SignalStrength:
    """Parse signal strength."""
    text_lower = text.lower().strip()
    if 'strong' in text_lower or '强' in text_lower:
        return SignalStrength.STRONG
    elif 'moderate' in text_lower or '中' in text_lower:
        return SignalStrength.MEDIUM
    elif 'weak' in text_lower or '弱' in text_lower:
        return SignalStrength.WEAK
    return SignalStrength.MEDIUM


def _parse_confidence(text: str) -> Optional[float]:
    """Parse confidence value."""
    try:
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            conf = float(numbers[0])
            return max(0, min(100, conf))
    except Exception:
        pass
    return None


def _parse_key_level(line: str) -> Optional[tuple]:
    """Parse key level line."""
    line_lower = line.lower()

    if '支撑' in line_lower or 'support' in line_lower:
        value = _parse_float(line)
        if value:
            return ('support', value)

    elif '阻力' in line_lower or 'resistance' in line_lower:
        value = _parse_float(line)
        if value:
            return ('resistance', value)

    return None


def _parse_float(text: str) -> Optional[float]:
    """Extract float from text."""
    if not text or 'N/A' in text.upper():
        return None
    try:
        numbers = re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))
        if numbers:
            return float(numbers[0])
    except Exception:
        pass
    return None
