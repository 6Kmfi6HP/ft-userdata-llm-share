"""
提示词构建器模块
负责从外部 markdown 文件加载 LLM 系统提示词

重构历史：
- 2025-11-29: 重构为文件加载模式
  * 提示词从 user_data/prompts/{template}/ 目录加载
  * 支持多套模板切换（通过 template_name 参数）
  * 保留 StrategyConfig 用于模板变量替换
  * 移除所有硬编码的 _build_* 方法
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """可配置的策略阈值 - 集中管理所有魔数以便调优"""

    # ADX 趋势强度阈值
    ADX_NO_TREND: int = 20          # ADX < 20: 无趋势/震荡市场
    ADX_WEAK_TREND: int = 25        # ADX 20-25: 弱趋势形成中
    # ADX > 25: 趋势确立

    # RSI 极值区阈值
    RSI_OVERSOLD: int = 30          # RSI < 30: 超卖区
    RSI_OVERSOLD_WARN: int = 35     # RSI 30-35: 超卖警告区
    RSI_OVERBOUGHT_WARN: int = 65   # RSI 65-70: 超买警告区
    RSI_OVERBOUGHT: int = 70        # RSI > 70: 超买区

    # 风险收益比阈值
    MIN_RR_RATIO: float = 2.0       # 最低风险收益比 (R:R ≥ 2:1)

    # 置信度阈值
    MIN_CONFIDENCE: int = 80        # 最低置信度
    CONFIDENCE_HALF_POSITION: int = 85   # 置信度 80-85: 50% 仓位
    CONFIDENCE_NORMAL_POSITION: int = 90  # 置信度 85-90: 70% 仓位
    # 置信度 > 90: 100% 仓位

    # 持仓时间相关 (分钟)
    MIN_HOLD_MINUTES: int = 90      # 最小持仓时间 (3根30分钟K线)
    MIN_HOLD_KLINES: int = 3        # 对应的K线数量

    # 仓位调整阈值
    ADD_POSITION_MIN_PROFIT: float = 3.0   # 加仓最低盈利 (%)
    PARTIAL_TP_THRESHOLD_1: float = 5.0    # 部分止盈阈值1 (%)
    PARTIAL_TP_THRESHOLD_2: float = 8.0    # 部分止盈阈值2 (%)
    MFE_DRAWBACK_WARNING: float = 30.0     # MFE回撤警告阈值 (%)
    MFE_DRAWBACK_CRITICAL: float = 50.0    # MFE回撤严重阈值 (%)

    # 硬止损阈值
    HARD_STOPLOSS: float = -8.0     # 硬止损百分比

    # EMA 200 接近阈值 (ATR倍数)
    EMA200_PROXIMITY_ATR: float = 1.5

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "StrategyConfig":
        """从配置字典创建实例，支持部分覆盖"""
        if not config:
            return cls()

        strategy_config = config.get("strategy_config", {})
        return cls(**{k: v for k, v in strategy_config.items() if hasattr(cls, k)})


class PromptBuilder:
    """
    LLM提示词构建器 - 文件加载版本

    从 user_data/prompts/{template_name}/ 目录加载提示词：
    - entry_prompt.md: 开仓决策提示词
    - position_prompt.md: 持仓管理提示词

    支持模板变量替换（可选）：
    - 在 markdown 文件中使用 {{VAR_NAME}} 格式
    - 变量值从 StrategyConfig 中获取
    """

    def __init__(
        self,
        template_name: str = "default_old",
        include_timeframe_guidance: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化提示词构建器

        Args:
            template_name: 模板名称，对应 user_data/prompts/{template_name}/ 目录
            include_timeframe_guidance: 保留参数，用于向后兼容
            config: 配置字典，可包含 "strategy_config" 用于覆盖默认阈值
        """
        self.template_name = template_name
        self.include_timeframe_guidance = include_timeframe_guidance
        self.config = config or {}

        # 策略阈值配置
        self.sc = StrategyConfig.from_config(self.config)

        # 确定模板目录路径
        # 支持多种可能的路径（Docker 环境 vs 本地开发）
        self.template_dir = self._find_template_dir()

    def _find_template_dir(self) -> Path:
        """
        查找模板目录

        按以下顺序查找：
        1. user_data/prompts/{template_name}/
        2. ./user_data/prompts/{template_name}/
        3. 相对于当前文件的路径
        """
        possible_paths = [
            Path("user_data/prompts") / self.template_name,
            Path("./user_data/prompts") / self.template_name,
            Path(__file__).parent.parent.parent.parent / "prompts" / self.template_name,
        ]

        for path in possible_paths:
            if path.exists():
                logger.debug(f"Found template directory: {path}")
                return path

        # 如果找不到，使用默认路径（可能会在读取时报错）
        default_path = Path("user_data/prompts") / self.template_name
        logger.warning(f"Template directory not found, using default: {default_path}")
        return default_path

    def _substitute_variables(self, content: str) -> str:
        """
        替换模板变量 {{VAR_NAME}} 为实际值

        变量来源：StrategyConfig 的属性

        Args:
            content: 包含模板变量的内容

        Returns:
            替换后的内容
        """
        # 构建替换映射
        replacements = {
            "{{ADX_NO_TREND}}": str(self.sc.ADX_NO_TREND),
            "{{ADX_WEAK_TREND}}": str(self.sc.ADX_WEAK_TREND),
            "{{RSI_OVERSOLD}}": str(self.sc.RSI_OVERSOLD),
            "{{RSI_OVERSOLD_WARN}}": str(self.sc.RSI_OVERSOLD_WARN),
            "{{RSI_OVERBOUGHT_WARN}}": str(self.sc.RSI_OVERBOUGHT_WARN),
            "{{RSI_OVERBOUGHT}}": str(self.sc.RSI_OVERBOUGHT),
            "{{MIN_RR_RATIO}}": str(self.sc.MIN_RR_RATIO),
            "{{MIN_CONFIDENCE}}": str(self.sc.MIN_CONFIDENCE),
            "{{CONFIDENCE_HALF_POSITION}}": str(self.sc.CONFIDENCE_HALF_POSITION),
            "{{CONFIDENCE_NORMAL_POSITION}}": str(self.sc.CONFIDENCE_NORMAL_POSITION),
            "{{MIN_HOLD_MINUTES}}": str(self.sc.MIN_HOLD_MINUTES),
            "{{MIN_HOLD_KLINES}}": str(self.sc.MIN_HOLD_KLINES),
            "{{ADD_POSITION_MIN_PROFIT}}": str(self.sc.ADD_POSITION_MIN_PROFIT),
            "{{PARTIAL_TP_THRESHOLD_1}}": str(self.sc.PARTIAL_TP_THRESHOLD_1),
            "{{PARTIAL_TP_THRESHOLD_2}}": str(self.sc.PARTIAL_TP_THRESHOLD_2),
            "{{MFE_DRAWBACK_WARNING}}": str(self.sc.MFE_DRAWBACK_WARNING),
            "{{MFE_DRAWBACK_CRITICAL}}": str(self.sc.MFE_DRAWBACK_CRITICAL),
            "{{HARD_STOPLOSS}}": str(self.sc.HARD_STOPLOSS),
            "{{EMA200_PROXIMITY_ATR}}": str(self.sc.EMA200_PROXIMITY_ATR),
        }

        # 执行替换
        for key, value in replacements.items():
            content = content.replace(key, value)

        return content

    def _load_prompt_file(self, filename: str) -> str:
        """
        加载提示词文件

        Args:
            filename: 文件名（如 entry_prompt.md）

        Returns:
            文件内容（已替换变量）

        Raises:
            FileNotFoundError: 文件不存在
        """
        filepath = self.template_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {filepath}\n"
                f"Please run extract_prompts.py to generate prompt files, or check template_name."
            )

        content = filepath.read_text(encoding="utf-8")
        return self._substitute_variables(content)

    def build_entry_prompt(self) -> str:
        """
        构建开仓决策系统提示词

        从 {template_dir}/entry_prompt.md 加载

        Returns:
            开仓决策系统提示词
        """
        try:
            return self._load_prompt_file("entry_prompt.md")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

    def build_position_prompt(self) -> str:
        """
        构建持仓管理系统提示词

        从 {template_dir}/position_prompt.md 加载

        Returns:
            持仓管理系统提示词
        """
        try:
            return self._load_prompt_file("position_prompt.md")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise


# 保留向后兼容的导入
# 如果有代码依赖 few_shot_examples 模块，这里提供空实现
def get_format_examples_entry() -> str:
    """向后兼容：返回空字符串（示例已包含在 markdown 文件中）"""
    return ""


def get_format_examples_position() -> str:
    """向后兼容：返回空字符串（示例已包含在 markdown 文件中）"""
    return ""
