"""
Agent 编排器模块
协调多个专业 Agent 的执行顺序和结果聚合

职责:
1. 管理 Agent 的生命周期
2. 控制 Agent 的执行顺序（串行/并行）
3. 收集和聚合各 Agent 的分析结果
4. 生成最终的综合分析报告
"""

import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .agent_state import (
    AgentState,
    AgentReport,
    Signal,
    Direction,
    SignalStrength,
    create_initial_state,
    merge_state
)
from .base_agent import BaseAgent
from .indicator_agent import IndicatorAgent
from .trend_agent import TrendAgent
from .sentiment_agent import SentimentAgent
from .pattern_agent import PatternAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    多 Agent 系统编排器

    协调 IndicatorAgent, TrendAgent, SentimentAgent 的执行，
    聚合分析结果，生成综合报告供 ConsensusClient 使用
    """

    # Agent 权重配置（用于共识计算）
    DEFAULT_WEIGHTS = {
        "IndicatorAgent": 1.0,    # 技术指标
        "TrendAgent": 1.2,        # 趋势分析（权重略高，趋势为王）
        "SentimentAgent": 0.8,    # 情绪分析（权重略低，辅助参考）
        "PatternAgent": 1.1,      # 形态识别（视觉分析，权重中上）
    }

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化编排器

        Args:
            llm_client: LLM 客户端实例
            config: 配置选项
        """
        self.llm_client = llm_client
        self.config = config or {}

        # 配置选项
        self.parallel_execution = self.config.get("parallel_execution", True)
        self.timeout_per_agent = self.config.get("timeout_per_agent", 30)
        self.enabled_agents = self.config.get(
            "enabled_agents",
            ["indicator", "trend", "sentiment", "pattern"]  # 默认启用所有agent包括pattern
        )
        self.weights = self.config.get("agent_weights", self.DEFAULT_WEIGHTS)

        # OHLCV 数据缓存（用于 PatternAgent 的视觉分析）
        self._ohlcv_cache: Optional[Any] = None

        # 初始化各专业 Agent
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()

        logger.info(
            f"AgentOrchestrator 初始化完成: "
            f"agents={list(self.agents.keys())}, "
            f"parallel={self.parallel_execution}"
        )

    def _init_agents(self):
        """初始化启用的 Agent"""
        agent_config = self.config.get("agent_config", {})

        if "indicator" in self.enabled_agents:
            self.agents["IndicatorAgent"] = IndicatorAgent(
                self.llm_client,
                config=agent_config.get("indicator", {})
            )

        if "trend" in self.enabled_agents:
            self.agents["TrendAgent"] = TrendAgent(
                self.llm_client,
                config=agent_config.get("trend", {})
            )

        if "sentiment" in self.enabled_agents:
            self.agents["SentimentAgent"] = SentimentAgent(
                self.llm_client,
                config=agent_config.get("sentiment", {})
            )

        if "pattern" in self.enabled_agents:
            self.agents["PatternAgent"] = PatternAgent(
                self.llm_client,
                config=agent_config.get("pattern", {})
            )
            logger.info("✅ PatternAgent (K线形态视觉分析) 已启用")

        # 检查 TrendAgent 是否支持视觉分析
        if "trend" in self.enabled_agents:
            trend_config = agent_config.get("trend", {})
            if trend_config.get("prefer_vision", True):
                logger.info("✅ TrendAgent (趋势线视觉分析) 已启用")

    def set_ohlcv_data(self, ohlcv_data) -> None:
        """
        设置 OHLCV 数据缓存（供 PatternAgent 和 TrendAgent 视觉分析使用）

        Args:
            ohlcv_data: pandas DataFrame 包含 OHLCV 数据
        """
        self._ohlcv_cache = ohlcv_data
        logger.debug(f"OHLCV 数据已缓存: {len(ohlcv_data) if ohlcv_data is not None else 0} 条")

    def run_analysis(
        self,
        market_context: str,
        pair: str,
        current_price: Optional[float] = None,
        ohlcv_data=None,
        timeframe: str = "",
        **kwargs
    ) -> AgentState:
        """
        运行完整的多 Agent 分析流程

        Args:
            market_context: 市场上下文（来自 ContextBuilder）
            pair: 交易对
            current_price: 当前价格（可选）
            ohlcv_data: OHLCV 数据 DataFrame（可选，用于 PatternAgent 视觉分析）
            timeframe: 时间框架（可选，用于图表标题）
            **kwargs: 额外参数

        Returns:
            AgentState: 包含所有 Agent 分析结果的状态
        """
        start_time = time.time()

        # 缓存 OHLCV 数据
        if ohlcv_data is not None:
            self._ohlcv_cache = ohlcv_data

        logger.info("=" * 60)
        logger.info(f"🤖 多 Agent 分析开始: {pair}")
        logger.info(f"   启用 Agent: {list(self.agents.keys())}")
        logger.info(f"   执行模式: {'并行' if self.parallel_execution else '串行'}")

        # 检查视觉分析可用性
        vision_agents = []
        if "PatternAgent" in self.agents:
            vision_agents.append("PatternAgent(K线形态)")
        if "TrendAgent" in self.agents:
            vision_agents.append("TrendAgent(趋势线)")
        if vision_agents:
            vision_status = '可用' if self._ohlcv_cache is not None else '不可用 (无OHLCV数据)'
            logger.info(f"   视觉分析: {vision_status} - {', '.join(vision_agents)}")
        logger.info("=" * 60)

        # 初始化状态
        state = create_initial_state(
            pair=pair,
            current_price=current_price or 0.0,
            market_context=market_context
        )

        # 执行各 Agent 分析
        if self.parallel_execution and len(self.agents) > 1:
            reports = self._parallel_run(market_context, pair, timeframe)
        else:
            reports = self._sequential_run(market_context, pair, timeframe)

        # 收集报告到状态
        for report in reports:
            state['agent_sequence'].append(report.agent_name)

            if report.agent_name == "IndicatorAgent":
                state['indicator_report'] = report
            elif report.agent_name == "TrendAgent":
                state['trend_report'] = report
            elif report.agent_name == "SentimentAgent":
                state['sentiment_report'] = report
            elif report.agent_name == "PatternAgent":
                state['pattern_report'] = report

        # 聚合分析结果
        state = self._aggregate_results(state)

        # 记录执行时间
        state['execution_time_ms'] = (time.time() - start_time) * 1000

        # 输出汇总（格式化价格为2位小数）
        key_support = state.get('key_support')
        key_resistance = state.get('key_resistance')
        support_str = f"{key_support:.2f}" if isinstance(key_support, (int, float)) else 'N/A'
        resistance_str = f"{key_resistance:.2f}" if isinstance(key_resistance, (int, float)) else 'N/A'

        logger.info("-" * 60)
        logger.info(f"📊 多 Agent 分析汇总:")
        logger.info(f"   共识方向: {state.get('consensus_direction', 'N/A')}")
        logger.info(f"   共识置信度: {state.get('consensus_confidence', 0):.1f}%")
        logger.info(f"   关键支撑: {support_str}")
        logger.info(f"   关键阻力: {resistance_str}")
        logger.info(f"   总耗时: {state['execution_time_ms']:.0f}ms")
        logger.info("=" * 60)

        return state

    def _parallel_run(
        self,
        market_context: str,
        pair: str,
        timeframe: str = ""
    ) -> List[AgentReport]:
        """
        并行执行所有 Agent

        Args:
            market_context: 市场上下文
            pair: 交易对
            timeframe: 时间框架（用于 PatternAgent）

        Returns:
            AgentReport 列表
        """
        reports = []

        def run_agent(agent):
            """运行单个 agent，处理视觉分析 Agent 的特殊参数"""
            if agent.name == "PatternAgent":
                # PatternAgent 需要 OHLCV 数据进行 K线形态视觉分析
                return agent.analyze(
                    market_context,
                    pair,
                    ohlcv_data=self._ohlcv_cache,
                    timeframe=timeframe
                )
            elif agent.name == "TrendAgent":
                # TrendAgent 需要 OHLCV 数据进行趋势线视觉分析
                return agent.analyze(
                    market_context,
                    pair,
                    ohlcv_data=self._ohlcv_cache,
                    timeframe=timeframe
                )
            else:
                return agent.analyze(market_context, pair)

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(run_agent, agent): agent
                for agent in self.agents.values()
            }

            # 计算总超时时间：视觉分析 Agent 需要更长时间
            # 取最大单个超时 * 1.5 作为总超时，确保有足够时间
            total_timeout = self.timeout_per_agent * 3  # 默认 30 * 3 = 90秒
            completed_futures = set()

            try:
                for future in as_completed(futures, timeout=total_timeout):
                    completed_futures.add(future)
                    agent = futures[future]
                    try:
                        report = future.result(timeout=self.timeout_per_agent)
                        reports.append(report)

                        status = "✅" if report.is_valid else "⚠️"
                        vision_tag = " 📸" if agent.name in ["PatternAgent", "TrendAgent"] else ""
                        logger.info(
                            f"   {status} {agent.name}{vision_tag}: "
                            f"{report.direction or 'N/A'} ({report.confidence:.0f}%)"
                        )

                    except Exception as e:
                        logger.error(f"   ❌ {agent.name} 执行失败: {e}")
                        reports.append(AgentReport(
                            agent_name=agent.name,
                            analysis=f"执行失败: {e}",
                            signals=[],
                            confidence=0.0,
                            error=str(e)
                        ))

            except TimeoutError as e:
                # 超时时，为未完成的 Agent 创建错误报告
                logger.warning(f"⏱️ 多 Agent 分析超时: {e}")
                for future, agent in futures.items():
                    if future not in completed_futures:
                        logger.warning(f"   ⏱️ {agent.name} 超时未完成")
                        reports.append(AgentReport(
                            agent_name=agent.name,
                            analysis=f"执行超时 (>{total_timeout}s)",
                            signals=[],
                            confidence=0.0,
                            error=f"Timeout after {total_timeout}s"
                        ))
                        # 尝试取消未完成的任务
                        future.cancel()

        return reports

    def _sequential_run(
        self,
        market_context: str,
        pair: str,
        timeframe: str = ""
    ) -> List[AgentReport]:
        """
        串行执行所有 Agent

        Args:
            market_context: 市场上下文
            pair: 交易对
            timeframe: 时间框架（用于 PatternAgent）

        Returns:
            AgentReport 列表
        """
        reports = []

        for name, agent in self.agents.items():
            vision_tag = " 📸" if name in ["PatternAgent", "TrendAgent"] else ""
            logger.info(f"   执行 {name}{vision_tag}...")

            try:
                # 视觉分析 Agent 需要 OHLCV 数据
                if name == "PatternAgent":
                    # PatternAgent: K线形态视觉分析
                    report = agent.analyze(
                        market_context,
                        pair,
                        ohlcv_data=self._ohlcv_cache,
                        timeframe=timeframe
                    )
                elif name == "TrendAgent":
                    # TrendAgent: 趋势线视觉分析
                    report = agent.analyze(
                        market_context,
                        pair,
                        ohlcv_data=self._ohlcv_cache,
                        timeframe=timeframe
                    )
                else:
                    report = agent.analyze(market_context, pair)

                reports.append(report)

                status = "✅" if report.is_valid else "⚠️"
                vision_indicator = " 📸" if name in ["PatternAgent", "TrendAgent"] else ""
                logger.info(
                    f"   {status} {name}{vision_indicator}: "
                    f"{report.direction or 'N/A'} ({report.confidence:.0f}%)"
                )

            except Exception as e:
                logger.error(f"   ❌ {name} 执行失败: {e}")
                reports.append(AgentReport(
                    agent_name=name,
                    analysis=f"执行失败: {e}",
                    signals=[],
                    confidence=0.0,
                    error=str(e)
                ))

        return reports

    def _aggregate_results(self, state: AgentState) -> AgentState:
        """
        聚合各 Agent 的分析结果

        使用加权投票确定共识方向，合并关键价位和信号

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        reports = []

        # 收集有效报告（包括 PatternAgent）
        for report in [
            state.get('indicator_report'),
            state.get('trend_report'),
            state.get('sentiment_report'),
            state.get('pattern_report')  # 新增 PatternAgent 报告
        ]:
            if report and report.is_valid:
                reports.append(report)

        if not reports:
            state['consensus_direction'] = 'wait'
            state['consensus_confidence'] = 0.0
            state['combined_analysis'] = "所有 Agent 分析失败，建议观望"
            return state

        # 加权方向计算
        direction_scores = {
            Direction.LONG: 0.0,
            Direction.SHORT: 0.0,
            Direction.NEUTRAL: 0.0
        }

        total_weight = 0.0
        all_signals: List[Signal] = []

        for report in reports:
            weight = self.weights.get(report.agent_name, 1.0)
            confidence_weight = report.confidence / 100.0

            if report.direction:
                direction_scores[report.direction] += weight * confidence_weight

            total_weight += weight
            all_signals.extend(report.signals)

        # 确定共识方向
        consensus_direction = self._determine_consensus_direction(direction_scores)

        # 计算共识置信度
        consensus_confidence = self._calculate_consensus_confidence(
            reports, direction_scores, consensus_direction
        )

        # 聚合关键价位
        key_support, key_resistance = self._aggregate_key_levels(reports)

        # 合并分析文本
        combined_analysis = self._build_combined_analysis(reports, consensus_direction)

        # 筛选关键信号
        consensus_signals = self._filter_key_signals(all_signals, consensus_direction)

        # 更新状态
        state['consensus_direction'] = str(consensus_direction) if consensus_direction != Direction.NEUTRAL else 'neutral'
        state['consensus_confidence'] = consensus_confidence
        state['consensus_signals'] = consensus_signals
        state['combined_analysis'] = combined_analysis
        state['key_support'] = key_support
        state['key_resistance'] = key_resistance

        return state

    def _determine_consensus_direction(
        self,
        direction_scores: Dict[Direction, float]
    ) -> Direction:
        """
        确定共识方向

        规则：
        1. 如果某方向分数超过另一方向 20%，选择该方向
        2. 否则返回中性
        """
        long_score = direction_scores[Direction.LONG]
        short_score = direction_scores[Direction.SHORT]

        threshold = 1.2  # 20% 优势阈值

        if long_score > short_score * threshold:
            return Direction.LONG
        elif short_score > long_score * threshold:
            return Direction.SHORT
        else:
            return Direction.NEUTRAL

    def _calculate_consensus_confidence(
        self,
        reports: List[AgentReport],
        direction_scores: Dict[Direction, float],
        consensus_direction: Direction
    ) -> float:
        """
        计算共识置信度

        规则：
        1. 基础置信度 = 各 Agent 置信度的加权平均
        2. 如果方向不一致，降低置信度
        3. 如果共识方向是中性，置信度降低
        """
        if not reports:
            return 0.0

        # 加权平均置信度
        total_weight = 0.0
        weighted_confidence = 0.0

        for report in reports:
            weight = self.weights.get(report.agent_name, 1.0)
            weighted_confidence += report.confidence * weight
            total_weight += weight

        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # 方向一致性惩罚
        directions = [r.direction for r in reports if r.direction]
        unique_directions = set(directions)

        if len(unique_directions) > 1:
            # 存在分歧，降低置信度
            avg_confidence *= 0.7

        # 中性方向惩罚
        if consensus_direction == Direction.NEUTRAL:
            avg_confidence *= 0.6

        return min(100.0, max(0.0, avg_confidence))

    def _aggregate_key_levels(
        self,
        reports: List[AgentReport]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        聚合关键价位

        优先使用 TrendAgent 的价位，其次取平均值
        """
        supports = []
        resistances = []

        for report in reports:
            if report.key_levels:
                if report.key_levels.get('support'):
                    supports.append((
                        report.key_levels['support'],
                        self.weights.get(report.agent_name, 1.0)
                    ))
                if report.key_levels.get('resistance'):
                    resistances.append((
                        report.key_levels['resistance'],
                        self.weights.get(report.agent_name, 1.0)
                    ))

        # 加权平均
        support = None
        resistance = None

        if supports:
            total_weight = sum(w for _, w in supports)
            support = sum(v * w for v, w in supports) / total_weight

        if resistances:
            total_weight = sum(w for _, w in resistances)
            resistance = sum(v * w for v, w in resistances) / total_weight

        return support, resistance

    def _build_combined_analysis(
        self,
        reports: List[AgentReport],
        consensus_direction: Direction
    ) -> str:
        """
        构建合并分析文本（旧格式，用于简要摘要）
        """
        parts = [
            "## 多 Agent 分析报告",
            "",
            f"**共识方向**: {consensus_direction}",
            ""
        ]

        for report in reports:
            parts.append(f"### {report.agent_name}")
            parts.append(f"- 方向: {report.direction or 'N/A'}")
            parts.append(f"- 置信度: {report.confidence:.0f}%")

            if report.signals:
                parts.append("- 关键信号:")
                for signal in report.signals[:3]:  # 最多显示 3 个信号
                    parts.append(f"  - {signal.name} ({signal.direction})")

            parts.append("")

        # 添加综合建议
        parts.append("### 综合建议")
        if consensus_direction == Direction.LONG:
            parts.append("多数 Agent 看多，建议关注做多机会")
        elif consensus_direction == Direction.SHORT:
            parts.append("多数 Agent 看空，建议关注做空机会")
        else:
            parts.append("Agent 意见分歧或信号不明确，建议观望")

        return "\n".join(parts)

    def _build_quantagent_style_reports(self, state: AgentState) -> Dict[str, str]:
        """
        构建 QuantAgent 风格的完整分析报告

        返回四个独立的完整报告，供 Decision Agent 综合决策使用

        Args:
            state: Agent 分析状态

        Returns:
            包含四个报告的字典:
            - indicator_report: 技术指标分析报告
            - trend_report: 趋势结构分析报告
            - sentiment_report: 市场情绪分析报告
            - pattern_report: K线形态分析报告（视觉分析）
        """
        reports = {}

        # Indicator Report
        indicator = state.get('indicator_report')
        if indicator and indicator.is_valid:
            reports['indicator_report'] = indicator.analysis or self._format_agent_report(indicator, "技术指标")
        else:
            reports['indicator_report'] = "技术指标分析不可用"

        # Trend Report
        trend = state.get('trend_report')
        if trend and trend.is_valid:
            reports['trend_report'] = trend.analysis or self._format_agent_report(trend, "趋势结构")
        else:
            reports['trend_report'] = "趋势结构分析不可用"

        # Sentiment Report
        sentiment = state.get('sentiment_report')
        if sentiment and sentiment.is_valid:
            reports['sentiment_report'] = sentiment.analysis or self._format_agent_report(sentiment, "市场情绪")
        else:
            reports['sentiment_report'] = "市场情绪分析不可用"

        # Pattern Report (视觉分析)
        pattern = state.get('pattern_report')
        if pattern and pattern.is_valid:
            reports['pattern_report'] = pattern.analysis or self._format_agent_report(pattern, "K线形态")
        else:
            reports['pattern_report'] = "K线形态分析不可用"

        return reports

    def _format_agent_report(self, report: AgentReport, report_type: str) -> str:
        """
        格式化单个 Agent 报告为可读文本

        Args:
            report: Agent 报告
            report_type: 报告类型名称

        Returns:
            格式化的报告文本
        """
        parts = [f"### {report_type}分析报告"]
        parts.append(f"- **方向判断**: {report.direction or 'neutral'}")
        parts.append(f"- **置信度**: {report.confidence:.0f}%")

        if report.signals:
            parts.append("- **关键信号**:")
            for signal in report.signals:
                strength_map = {
                    SignalStrength.STRONG: "强",
                    SignalStrength.MODERATE: "中",
                    SignalStrength.WEAK: "弱"
                }
                strength = strength_map.get(signal.strength, "")
                parts.append(f"  - {signal.name}: {signal.description} ({strength}信号)")

        if report.key_levels:
            parts.append("- **关键价位**:")
            if report.key_levels.get('support'):
                parts.append(f"  - 支撑位: {report.key_levels['support']}")
            if report.key_levels.get('resistance'):
                parts.append(f"  - 阻力位: {report.key_levels['resistance']}")

        return "\n".join(parts)

    def _filter_key_signals(
        self,
        all_signals: List[Signal],
        consensus_direction: Direction
    ) -> List[Signal]:
        """
        筛选关键信号

        优先保留与共识方向一致的强信号
        """
        if not all_signals:
            return []

        # 按强度和方向排序
        def signal_score(s: Signal) -> float:
            score = 0.0

            # 强度分数
            strength_scores = {
                SignalStrength.STRONG: 3.0,
                SignalStrength.MODERATE: 2.0,
                SignalStrength.WEAK: 1.0,
                SignalStrength.NONE: 0.0
            }
            score += strength_scores.get(s.strength, 0.0)

            # 方向一致性加分
            if s.direction == consensus_direction:
                score += 2.0

            return score

        sorted_signals = sorted(all_signals, key=signal_score, reverse=True)

        # 返回前 5 个关键信号
        return sorted_signals[:5]

    def format_for_decision(self, state: AgentState) -> Dict[str, Any]:
        """
        将 AgentState 格式化为 Decision Agent 可用的数据

        返回 QuantAgent 风格的完整报告，供双 Decision Agent 综合决策

        Args:
            state: Agent 分析状态

        Returns:
            包含以下内容的字典:
            - indicator_report: 技术指标完整报告
            - trend_report: 趋势结构完整报告
            - sentiment_report: 市场情绪完整报告
            - pattern_report: K线形态完整报告（视觉分析）
            - consensus_direction: 预分析共识方向
            - consensus_confidence: 预分析共识置信度
        """
        reports = self._build_quantagent_style_reports(state)

        return {
            "indicator_report": reports.get('indicator_report', ''),
            "trend_report": reports.get('trend_report', ''),
            "sentiment_report": reports.get('sentiment_report', ''),
            "pattern_report": reports.get('pattern_report', ''),  # 新增视觉分析报告
            "consensus_direction": state.get('consensus_direction', 'neutral'),
            "consensus_confidence": state.get('consensus_confidence', 0),
            "key_support": state.get('key_support'),
            "key_resistance": state.get('key_resistance'),
            "pair": state.get('pair', ''),
            "combined_analysis": state.get('combined_analysis', '')  # 保留旧格式兼容
        }

    def format_for_logging(self, state: AgentState) -> Dict[str, Any]:
        """
        将 AgentState 格式化为日志记录格式

        Args:
            state: Agent 分析状态

        Returns:
            可 JSON 序列化的字典
        """
        result = {
            "pair": state.get('pair'),
            "consensus_direction": state.get('consensus_direction'),
            "consensus_confidence": state.get('consensus_confidence'),
            "key_support": state.get('key_support'),
            "key_resistance": state.get('key_resistance'),
            "execution_time_ms": state.get('execution_time_ms'),
            "agent_sequence": state.get('agent_sequence'),
            "created_at": state.get('created_at'),
            "reports": {}
        }

        # 添加各 Agent 报告摘要（包括 PatternAgent）
        for report_key in ['indicator_report', 'trend_report', 'sentiment_report', 'pattern_report']:
            report = state.get(report_key)
            if report:
                result["reports"][report.agent_name] = {
                    "direction": str(report.direction) if report.direction else None,
                    "confidence": report.confidence,
                    "signal_count": len(report.signals),
                    "execution_time_ms": report.execution_time_ms,
                    "error": report.error,
                    "is_vision_analysis": report.agent_name == "PatternAgent"  # 标记视觉分析
                }

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取编排器统计信息"""
        return {
            "enabled_agents": list(self.agents.keys()),
            "parallel_execution": self.parallel_execution,
            "timeout_per_agent": self.timeout_per_agent,
            "weights": self.weights
        }
