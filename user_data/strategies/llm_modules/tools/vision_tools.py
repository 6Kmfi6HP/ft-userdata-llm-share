# -*- coding: utf-8 -*-
"""
Gemini视觉分析工具

使用 Google GenAI 官方 SDK 进行K线图视觉分析

官方文档参考：
- Python SDK: https://googleapis.github.io/python-genai/
- Vision API: https://ai.google.dev/gemini-api/docs/vision
- Image Understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
"""
import base64
import logging
from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field

# 使用官方 Google GenAI SDK
from google import genai
from google.genai import types

# 重试机制
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ==================== Pydantic 响应模型 ====================

class PatternInfo(BaseModel):
    """K线形态信息"""
    name: str = Field(description="形态名称")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 (0.0-1.0)")


class TrendJudgement(BaseModel):
    """趋势判断"""
    direction: Literal["up", "down", "sideways"] = Field(description="趋势方向")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 (0.0-1.0)")
    evidence: List[str] = Field(description="支撑证据列表")


class VisionAnalysisResult(BaseModel):
    """视觉分析结果（结构化输出）"""
    summary: str = Field(description="中文简短总结 (50-100字)")
    judgement: TrendJudgement = Field(description="趋势判断")
    patterns: List[PatternInfo] = Field(default_factory=list, description="识别的K线形态")
    risks: List[str] = Field(default_factory=list, description="风险提示")


class VisionTools:
    """
    Gemini 视觉分析工具
    
    使用官方 Google GenAI SDK（google-genai）
    API 参考：https://googleapis.github.io/python-genai/
    """
    
    def __init__(self, api_base: str, api_key: str, vision_model: str,
                 timeout: Optional[int] = None, skip_validation: bool = False):
        """
        初始化 Gemini 视觉工具

        Args:
            api_base: API 基础 URL（用于自定义端点，如 OneAPI 代理）
            api_key: Gemini API 密钥
            vision_model: 视觉模型名称（如 gemini-2.5-flash-lite）
            timeout: 超时时间（秒）
            skip_validation: 是否跳过 API 验证（使用代理时建议设为 True）
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("❌ Gemini API key is required for VisionTools")

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.vision_model = vision_model
        self.timeout = timeout or 60
        self.skip_validation = skip_validation

        # 初始化 Google GenAI 客户端（使用自定义 API base）
        # 参考：https://googleapis.github.io/python-genai/
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                api_version='v1beta',
                base_url=self.api_base  # 自定义 API 端点（支持 OneAPI 等代理）
            )
        )

        # 验证 API 连接（启动时检测问题）
        if not self.skip_validation:
            self._validate_api_connection()
        else:
            logger.info(f"⏭️  已跳过 Gemini API 连接验证（使用代理模式）")

    def _validate_api_connection(self):
        """
        验证 Gemini API 连接和模型可用性

        在初始化时调用，快速失败原则
        """
        try:
            # 尝试获取模型信息（轻量级验证调用）
            model_info = self.client.models.get(model=self.vision_model)

            if model_info:
                logger.info(f"✅ Gemini API 连接验证成功，模型: {self.vision_model}")
            else:
                logger.warning(f"⚠️  Gemini API 连接成功，但模型信息为空")

        except Exception as e:
            # API 验证失败时仅警告，不阻止启动（优雅降级）
            logger.warning(f"⚠️  Gemini API 连接验证失败: {e}")
            logger.warning(f"   API Base: {self.api_base}")
            logger.warning(f"   Model: {self.vision_model}")
            logger.warning(f"   视觉分析功能可能无法正常工作，但策略将继续运行")
            logger.info(f"   提示: 如果使用代理（如 OneAPI），API 验证可能失败但实际调用可正常工作")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def _call_gemini_with_retry(self, model: str, contents: list, config: types.GenerateContentConfig):
        """
        带重试机制的 Gemini API 调用

        Args:
            model: 模型名称
            contents: 请求内容
            config: 生成配置

        Returns:
            API 响应对象

        Raises:
            重试失败后抛出原始异常
        """
        try:
            logger.debug(f"调用 Gemini API: {model}")
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"⚠️  Gemini API 调用失败，准备重试: {e}")
            raise  # 触发 tenacity 重试
        except Exception as e:
            # 其他异常不重试（如参数错误、配额超限等）
            logger.error(f"❌ Gemini API 调用失败（不重试）: {e}")
            raise

    def _build_system_instruction(self, task: str) -> str:
        """
        构建系统指令（定义 AI 角色和行为）
        
        注意：图表已包含趋势线图和标准K线图两部分，无需区分 task 类型
        
        Args:
            task: 分析任务类型（trend 或 pattern）- 仅用于日志，不影响系统指令
            
        Returns:
            系统指令字符串
        """
        # 统一的系统指令（适用于合并图表）
        return """你是一位专业的加密货币技术分析专家，擅长综合分析趋势线、K线形态和价格行为。

**你的专长**：
1. **趋势线分析**：
   - 识别支撑线和阻力线的有效性
   - 分析价格与趋势线的互动关系（反弹、突破、假突破）
   - 基于趋势线斜率判断市场强度
   - 评估趋势线收敛/发散带来的市场含义

2. **K线形态识别**：
   - 识别经典反转形态（头肩顶/底、双顶/底、圆弧顶/底、V型反转等）
   - 识别持续形态（三角形、旗形、矩形等）
   - 识别楔形（上升/下降楔形）
   - 识别特殊形态（岛型反转、扩张三角等）

3. **价格行为分析**：
   - 分析成交量与价格的配合关系
   - 识别关键支撑/阻力位
   - 判断趋势强度和延续性

**已知的经典K线形态**：
- 反转形态：Inverse Head and Shoulders（头肩底）、Double Bottom（双底）、Double Top（双顶）、Rounded Bottom（圆弧底）、V-shaped Reversal（V型反转）
- 持续形态：Bullish/Bearish Flag（牛/熊旗）、Ascending/Descending Triangle（上升/下降三角）、Symmetrical Triangle（对称三角）、Rectangle（矩形）
- 楔形：Falling/Rising Wedge（下降/上升楔形）
- 其他：Island Reversal（岛型反转）、Expanding Triangle（扩张三角）

**分析原则**：
- 客观：基于图表客观证据，避免主观臆测
- 全面：综合趋势线、形态、价格行为等多个维度
- 精准：准确识别形态类型和完成度，给出合理的置信度评分
- 风险意识：明确指出潜在风险和不确定性
- 实用性：分析对交易决策的指导意义
- 简洁：总结控制在 50-100 字

**输出要求**：
- summary、evidence、risks 必须使用中文
- direction 只能是 up/down/sideways 之一
- confidence 必须是 0.0-1.0 之间的数值
- evidence 至少提供 2-3 条支撑证据
- **patterns 识别 0-3 个形态**，每个形态必须有 name 和 confidence（0.0-1.0）
- 若无明确的经典K线形态，可返回描述性市场状态（如 Consolidation、Range-bound、Choppy Market、Weak Momentum 等）或空数组"""
    
    def _build_user_prompt(self, task: str, time_frame: str, pair: str = None) -> str:
        """
        构建用户提示词（具体分析任务）
        
        注意：图表包含上下两部分，统一提示词涵盖所有分析维度
        
        Args:
            task: 分析任务类型（trend 或 pattern）- 仅用于日志
            time_frame: 时间框架
            pair: 交易对名称（如 "ETH/USDT:USDT"）
            
        Returns:
            用户提示词字符串
        """
        # 统一的用户提示（适用于合并图表）
        pair_info = f"**交易对**: {pair}\n" if pair else ""
        
        return f"""请综合分析这张 {time_frame} 时间框架的 K 线图（包含两部分）：

{pair_info}

**图表说明**：
- 上半部分：价格图（可能包含以下元素，需根据实际图表判断）
  * K线图（蜡烛图） - 必定存在
  * 蓝色线/红色线 = 支撑/阻力趋势线（如存在）
  * 橙色线/紫色线 = EMA20/EMA50均线（如存在）
  * 灰色虚线 = 布林带上中下轨（如存在）
  * 底部：成交量柱状图
- 下半部分：标准 K 线图 + 技术指标（可能包含以下元素，需根据实际图表判断）
  * K线图（用于形态识别） - 必定存在
  * 蓝色/红色线 = MACD 指标（如存在：蓝=MACD线，红=信号线，灰柱=柱状图）
  * 成交量柱状图
- ✅ Z2修复: 仅分析实际可见的指标,不要基于不存在的元素进行分析
- 注意观察：K线实体大小、上下影线、连续K线组合、价格与可见指标的位置关系

**请完成以下综合分析**：

1. **综合技术分析**（基于上半部分，仅分析可见的指标）：
   - **趋势线**（如存在）：价格与支撑/阻力线的互动（反弹、突破、假突破），斜率和收敛/发散
   - **均线系统**（如存在）：价格相对 EMA20/EMA50 的位置（上方/下方），均线多头/空头排列
   - **布林带**（如存在）：价格在布林带的位置（触及上轨/下轨，中轨支撑/压力），通道宽度变化
   - **成交量**：成交量与价格的配合关系（量价配合/背离），突破时的成交量确认
   - **关键位置**：价格是否在多个可见指标的共振位

2. **K线形态与动量分析**（基于下半部分，必须完成）：
   - **K线形态**：必须识别至少 1 个可能性最高的形态（经典形态或当前市场状态）
   - **K线组合**：观察最近 5-10 根 K 线的组合模式（如吞没、十字星、锤头线等）
   - **MACD 指标**（如存在）：
     * MACD 线与信号线的位置关系（金叉/死叉）
     * MACD 柱状图的变化趋势（扩张/收缩，正值/负值）
     * MACD 与价格的背离（顶背离/底背离）
   - **摆动结构**：识别价格区间的支撑/阻力位（摆动高点/低点）
   - **形态评估**：给出形态名称和置信度评分（0.0-1.0），说明完成度和有效性
   - **市场状态**：如果没有明显的经典形态，识别当前价格结构（如：Consolidation、Range-bound、Trending、Choppy Market、Sideways Channel、Weak Breakout Attempt 等）

3. **趋势判断**（综合所有视觉元素）：
   - 判断短期趋势方向：up（上涨）/ down（下跌）/ sideways（横盘）
   - 给出置信度评分（0.0-1.0）
   - 列举 2-3 条支撑证据，必须综合多个维度：
     * 价格行为（K线形态、实体大小）
     * 趋势线互动（突破/反弹）
     * 均线系统（多头/空头排列，价格与均线位置）
     * 布林带（位置和宽度）
     * MACD 动量（金叉/死叉，柱状图方向）
     * 成交量确认（量价关系）

4. **风险提示**：
   - 指出可能的反转风险
   - 标注关键支撑/阻力位
   - 形态失败的可能性（如果有形态）
   - 需要确认的信号

**分析要求**：
- **多维度综合**：必须综合考虑所有视觉元素（趋势线、均线、布林带、K线形态、MACD、成交量）
- **客观证据**：基于图表客观证据，避免过度解读
- **关键观察点**：
  * 最近几根 K 线的实体大小变化（动能强弱）
  * 影线长度（买卖压力）
  * K 线颜色连续性（趋势延续/反转迹象）
  * 价格波动范围的收敛/扩张
  * 价格与 EMA20/EMA50 的交叉和位置关系
  * 布林带通道的宽度变化（波动率）
  * MACD 柱状图的连续变化方向
  * 成交量的异常放大/缩小
- **信号冲突处理**：如果不同指标信号冲突（如趋势线看涨但 MACD 死叉），请明确说明并评估权重
- **总结简洁**：summary 控制在 50-100 字"""
    
    def _convert_pydantic_to_dict(self, task: str, result: VisionAnalysisResult) -> dict:
        """
        将 Pydantic 模型转换为字典（兼容旧格式）
        
        Args:
            task: 任务类型
            result: Pydantic 响应对象
            
        Returns:
            字典格式的结果
        """
        return {
            "vision_task": task,
            "summary": result.summary,
            "judgement": {
                "direction": result.judgement.direction,
                "confidence": result.judgement.confidence,
                "evidence": result.judgement.evidence
            },
            "patterns": [
                {"name": p.name, "confidence": p.confidence}
                for p in result.patterns
            ],
            "risks": result.risks
        }
    
    def analyze_image_with_gemini(self, 
                                   image_b64: str,
                                   task: Literal["trend", "pattern"] = "trend",
                                   time_frame: str = "15m",
                                   pair: str = None,
                                   return_format: Literal["json", "text"] = "json") -> dict:
        """
        使用 Gemini 视觉 API 分析 K 线图像
        
        使用官方 Google GenAI SDK（google-genai）和 Structured Output 功能
        API 参考：https://googleapis.github.io/python-genai/
        
        Args:
            image_b64: base64 编码的图像（PNG/JPEG）
            task: 分析任务类型（trend 或 pattern）
            time_frame: 时间框架
            pair: 交易对名称（如 "ETH/USDT:USDT"）
            return_format: 返回格式（json 或 text）
            
        Returns:
            分析结果字典
        """
        try:
            # 构建系统指令和用户提示词（分离角色定义和具体任务）
            system_instruction = self._build_system_instruction(task)
            user_prompt = self._build_user_prompt(task, time_frame, pair)
            
            # 解码 base64 图像数据
            image_bytes = base64.b64decode(image_b64)
            
            # 构建 Gemini API 请求内容（使用官方 SDK types）
            # 参考：https://ai.google.dev/gemini-api/docs/image-understanding
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=image_bytes
                            )
                        )
                    ]
                )
            ]
            
            # 配置生成参数
            if return_format == "json":
                # 使用 Structured Output（Pydantic 模型 + 系统指令）
                # 参考：https://ai.google.dev/gemini-api/docs/structured-output
                generate_config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                    top_p=0.9,        # ← 添加：核采样，进一步稳定
                    top_k=40,         # ← 添加：限制候选词数量
                    seed=42,          # ← 添加：固定种子（实验性）
                    response_mime_type='application/json',  # 强制 JSON 输出
                    response_schema=VisionAnalysisResult,    # Pydantic 模型
                    system_instruction=system_instruction,   # 系统指令（角色定义）
                )
                logger.info(f"调用 Gemini 视觉分析（JSON 模式 + 系统指令）: task={task}, model={self.vision_model}")
            else:
                # 文本模式（也使用系统指令）
                generate_config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    system_instruction=system_instruction,   # 系统指令（角色定义）
                )
                logger.info(f"调用 Gemini 视觉分析（文本模式 + 系统指令）: task={task}, model={self.vision_model}")
            
            logger.debug(f"使用 Google GenAI SDK，模型：{self.vision_model}")
            logger.debug(f"系统指令长度: {len(system_instruction)} 字符")
            logger.debug(f"用户提示词长度: {len(user_prompt)} 字符")

            # 调用 Gemini API（使用带重试的方法）
            response = self._call_gemini_with_retry(
                model=self.vision_model,
                contents=contents,
                config=generate_config
            )
            
            # 处理响应
            if return_format == "json":
                # JSON 模式：使用 response.parsed（自动解析的 Pydantic 对象）
                if hasattr(response, 'parsed') and response.parsed:
                    logger.info(f"✅ Gemini 返回结构化对象: {type(response.parsed).__name__}")
                    result = self._convert_pydantic_to_dict(task, response.parsed)
                    logger.debug(f"解析结果: direction={result['judgement']['direction']}, "
                               f"patterns={len(result['patterns'])}, risks={len(result['risks'])}")
                    return result
                else:
                    # 回退：尝试从 text 字段解析
                    logger.warning(f"⚠️ response.parsed 为空，尝试从 text 字段解析")
                    if hasattr(response, 'text') and response.text:
                        logger.debug(f"响应文本前200字符: {response.text[:200]}")
                        return {"vision_task": task, "summary": response.text}
                    else:
                        logger.error(f"❌ Gemini 返回空响应！")
                        return {
                            "vision_task": task,
                            "summary": "VISION_CALL_FAILED: Gemini returned empty content."
                        }
            else:
                # 文本模式
                content = response.text if hasattr(response, 'text') else ""
                if not content:
                    logger.error(f"Gemini 返回空内容！")
                    return {
                        "vision_task": task,
                        "summary": "VISION_CALL_FAILED: Gemini returned empty content."
                    }
                logger.debug(f"Gemini 返回内容长度: {len(content)} 字符")
                return {"vision_task": task, "summary": content}
            
        except Exception as e:
            logger.error(f"❌ Gemini 视觉分析失败: {e}")
            import traceback
            logger.debug(f"完整堆栈:\n{traceback.format_exc()}")
            return {
                "vision_task": task,
                "summary": f"VISION_CALL_FAILED: {str(e)}"
            }

