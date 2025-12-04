"""
QuantAgent 风格多 Agent 决策系统

架构设计（类似 QuantAgent）：
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: 专业 Agent 并行分析                  │
├─────────────────────────────────────────────────────────────────┤
│  IndicatorAgent → RSI, MACD, ADX, Stochastic 分析               │
│  TrendAgent → EMA 结构、支撑阻力、价格结构分析                    │
│  SentimentAgent → 资金费率、多空比、OI、恐惧贪婪分析              │
│           ↓                                                     │
│  AgentOrchestrator → 加权共识聚合                                │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: 双 Decision Agent 并行决策           │
├─────────────────────────────────────────────────────────────────┤
│  Decision Agent 1 (激进): 积极寻找交易机会                       │
│  Decision Agent 2 (保守): 严格风险评估                           │
│           ↓                                                     │
│  输入：三份完整的专业分析报告（QuantAgent 风格）                  │
│  输出：交易函数调用                                              │
│           ↓                                                     │
│  共识解决：置信度优先 / 保守策略                                 │
└─────────────────────────────────────────────────────────────────┘

核心改进（v3 - QuantAgent 风格）：
1. 三份完整的专业 Agent 报告（类似 QuantAgent 的 indicator_report, pattern_report, trend_report）
2. 双 Decision Agent 并行决策（替代原 OpportunityFinder + RiskAssessor）
3. 决策提示词采用 QuantAgent 的决策策略风格
4. 支持三报告一致性优先的共识机制
"""
import logging
import json
import copy
import re
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ConsensusClient:
    """
    双重决策共识客户端

    包装LLMClient，提供双重决策验证功能
    """

    # ========== QuantAgent 风格 Decision Agent 模式 ==========
    # Decision Agent 1: 激进决策者 - 积极寻找交易机会
    DECISION_AGENT_AGGRESSIVE_PREFIX = """# 决策角色：激进决策者 (Aggressive Decision Maker)

你是一位高频交易分析师，基于以下三份专业分析报告做出交易决策。

### 决策策略：
1. 只对**已确认**的信号采取行动 — 避免投机性信号
2. 优先考虑**三份报告方向一致**的情况
3. 给予以下信号更高权重：
   - 近期强动量信号（如 MACD 交叉、RSI 突破）
   - 明确的价格行为（如突破 K 线、拒绝影线）
   - 趋势线支撑/阻力位的确认
4. 如果报告存在分歧：
   - 选择有**更强、更近期确认**的方向
   - 优先选择**有动量支撑**的信号
5. 建议盈亏比在 **1.5 到 2.5** 之间

### 你的倾向：
- 积极寻找交易机会
- 在信号足够强时果断入场
- 相信动量和趋势的延续性

---

"""

    # Decision Agent 2: 保守决策者 - 严格风险评估
    DECISION_AGENT_CONSERVATIVE_PREFIX = """# 决策角色：保守决策者 (Conservative Decision Maker)

你是一位高频交易分析师，基于以下三份专业分析报告做出交易决策。

### 决策策略：
1. 只对**已确认**的信号采取行动 — 避免投机性信号
2. 优先考虑**三份报告方向一致**的情况
3. 给予以下信号更高权重：
   - 近期强动量信号（如 MACD 交叉、RSI 突破）
   - 明确的价格行为（如突破 K 线、拒绝影线）
   - 趋势线支撑/阻力位的确认
4. 如果报告存在分歧：
   - 选择**更防御性**的方向
   - 不确定时倾向于**观望**
5. 建议盈亏比在 **1.5 到 2.5** 之间

### 你的倾向：
- 严格评估风险和陷阱
- 只在信号非常明确时入场
- 宁可错过机会也不冒险

---

"""

    # Agent 报告注入模板（支持四个Agent，包括视觉分析）
    AGENT_REPORTS_TEMPLATE = """
## 专业分析报告

以下是四位专业分析师对当前市场的独立分析：

---
### 技术指标分析报告 (Technical Indicator Report)
{indicator_report}

---
### 趋势结构分析报告 (Trend Analysis Report)
{trend_report}

---
### 市场情绪分析报告 (Sentiment Report)
{sentiment_report}

---
### K线形态分析报告 (Pattern Recognition Report - 视觉分析)
{pattern_report}

---
### 预分析共识
- **共识方向**: {consensus_direction}
- **共识置信度**: {consensus_confidence:.1f}%
- **关键支撑位**: {key_support}
- **关键阻力位**: {key_resistance}

---

请基于以上四份报告的综合分析，结合市场数据做出最终交易决策。
注意：K线形态分析报告来自视觉分析Agent，可识别头肩顶/底、双顶/底、三角形等经典形态。
"""

    # 保守决策优先级（数字越小越保守）
    # 用于置信度相近时的决策参考，不直接决定结果
    ACTION_PRIORITY = {
        "wait": 1,        # 最保守 - 不开仓
        "hold": 2,        # 保守 - 不平仓
        "adjust": 2.5,    # 中等偏保守 - 调整仓位（可能是减仓保护）
        "exit": 3,        # 中等 - 平仓止损
        "enter_long": 4,  # 激进 - 开多仓
        "enter_short": 4, # 激进 - 开空仓
    }

    # 置信度差异阈值：差异超过此值时，置信度优先于保守性
    CONFIDENCE_DIFF_THRESHOLD = 15

    def __init__(
        self,
        llm_config: Dict[str, Any],
        function_executor,
        consensus_config: Optional[Dict[str, Any]] = None,
        trading_tools=None
    ):
        """
        初始化共识客户端

        Args:
            llm_config: LLM配置
            function_executor: 函数执行器
            consensus_config: 共识系统配置
            trading_tools: 交易工具实例（用于后置置信度验证）
        """
        self.llm_client = LLMClient(llm_config, function_executor)
        self.trading_tools = trading_tools

        # 共识配置
        config = consensus_config or {}
        self.enabled = config.get("enabled", True)
        self.parallel_requests = config.get("parallel_requests", True)
        self.conflict_strategy = config.get("conflict_strategy", "conservative")
        self.require_consensus = config.get("require_consensus", False)
        self.confidence_threshold = config.get("confidence_threshold", 80)

        # ===== 多 Agent 预分析系统配置 =====
        self.multi_agent_enabled = config.get("multi_agent_enabled", False)
        self.agent_orchestrator = None
        self._last_agent_state = None  # 缓存最近一次的 Agent 分析状态

        # ===== OHLCV 数据缓存（用于视觉分析 Agent）=====
        self._current_ohlcv = None  # 当前 K 线数据 (DataFrame)
        self._current_timeframe = None  # 当前时间框架 (如 "30m")
        self._current_pair = None  # 当前交易对

        if self.multi_agent_enabled:
            try:
                from ..agents.orchestrator import AgentOrchestrator
                agent_config = config.get("agent_config", {})
                self.agent_orchestrator = AgentOrchestrator(
                    self.llm_client,
                    config=agent_config
                )
                logger.info("✅ 多 Agent 预分析系统已启用")
            except ImportError as e:
                logger.warning(f"⚠️ 无法导入 AgentOrchestrator，多 Agent 模式已禁用: {e}")
                self.multi_agent_enabled = False
            except Exception as e:
                logger.error(f"❌ 初始化 AgentOrchestrator 失败: {e}")
                self.multi_agent_enabled = False

        logger.info(f"QuantAgent 风格决策系统已初始化: enabled={self.enabled}, "
                   f"parallel={self.parallel_requests}, strategy={self.conflict_strategy}, "
                   f"multi_agent={self.multi_agent_enabled}")

    def set_current_ohlcv(self, dataframe, timeframe: str, pair: str = None):
        """
        设置当前 K 线数据（供视觉分析 Agent 使用）

        在调用 call_with_functions 之前调用此方法，
        将 OHLCV 数据传递给多 Agent 预分析系统。

        Args:
            dataframe: pandas DataFrame 包含 OHLCV 数据
            timeframe: 时间框架字符串（如 "30m", "1h"）
            pair: 交易对（可选，用于日志记录）
        """
        self._current_ohlcv = dataframe
        self._current_timeframe = timeframe
        self._current_pair = pair
        logger.debug(f"已设置 OHLCV 数据: {pair}, timeframe={timeframe}, "
                    f"rows={len(dataframe) if dataframe is not None else 0}")

    def clear_current_ohlcv(self):
        """清除当前 OHLCV 数据缓存"""
        self._current_ohlcv = None
        self._current_timeframe = None
        self._current_pair = None

    def call_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        双重决策调用LLM

        与原LLMClient接口兼容，但内部执行两次决策并对比

        流程：
        1. 跳过置信度门槛检查（由共识后置验证）
        2. 执行两次LLM决策
        3. 计算平均置信度
        4. 用平均值验证是否满足门槛
        5. 更新或清除信号

        Args:
            messages: 消息列表
            functions: 可用的函数列表
            max_iterations: 最大迭代次数

        Returns:
            共识后的响应
        """
        if not self.enabled:
            # 禁用共识模式时直接使用原客户端
            return self.llm_client.call_with_functions(messages, functions, max_iterations)

        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("🔄 QuantAgent 风格多 Agent 决策开始")
        logger.info("=" * 60)

        # ===== 多 Agent 预分析（如果启用）=====
        if self.multi_agent_enabled and self.agent_orchestrator:
            messages = self._run_multi_agent_analysis(messages)

        # 在共识模式下，跳过置信度门槛检查（后置验证）
        if self.trading_tools:
            self.trading_tools.set_skip_confidence_check(True)

        try:
            # 执行两次决策（双角色：机会发现者 + 风险评估者）
            if self.parallel_requests:
                response_1, response_2 = self._parallel_call(
                    messages, functions, max_iterations
                )
            else:
                response_1, response_2 = self._sequential_call(
                    messages, functions, max_iterations
                )

            # 分析并合并结果
            consensus_result = self._analyze_consensus(response_1, response_2)

            # 后置置信度验证
            consensus_result = self._post_validate_confidence(consensus_result)

        finally:
            # 恢复置信度检查
            if self.trading_tools:
                self.trading_tools.set_skip_confidence_check(False)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️  QuantAgent 决策耗时: {elapsed:.2f}秒")
        logger.info("=" * 60)

        return consensus_result

    def _post_validate_confidence(self, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        后置置信度验证

        用平均置信度判断是否满足门槛：
        - 满足：更新信号中的置信度为平均值
        - 不满足：清除信号，返回失败
        """
        if not self.trading_tools:
            return consensus_result

        if not consensus_result.get("success"):
            return consensus_result

        # 获取共识置信度
        avg_confidence = consensus_result.get("consensus_confidence")
        if avg_confidence is None:
            # 非共识模式或单次响应
            return consensus_result

        # 从 function_calls 中提取 pair 和 action
        function_calls = consensus_result.get("function_calls", [])
        pair = None
        action = None
        merged_reason = consensus_result.get("merged_reason", "")

        for call in function_calls:
            func_name = call.get("function", "")
            args = call.get("arguments", {})
            if func_name in ("signal_entry_long", "signal_entry_short"):
                pair = args.get("pair")
                action = "enter_long" if "long" in func_name else "enter_short"
                break

        if not pair or not action:
            # 非开仓信号，不需要验证
            return consensus_result

        logger.info(f"📊 后置置信度验证: {pair}")
        logger.info(f"   平均置信度: {avg_confidence:.1f}, 门槛: {self.confidence_threshold}")

        if avg_confidence >= self.confidence_threshold:
            # 满足门槛，更新信号的置信度
            self.trading_tools.update_signal_confidence(pair, avg_confidence, merged_reason)
            logger.info(f"   ✅ 通过验证，信号有效")
        else:
            # 不满足门槛，清除信号
            self.trading_tools.clear_signal_for_pair(pair)
            logger.warning(f"   ❌ 平均置信度 {avg_confidence:.1f} < {self.confidence_threshold}，信号已清除")

            # 更新结果
            consensus_result["success"] = False
            consensus_result["confidence_rejected"] = True
            consensus_result["error"] = (
                f"共识平均置信度 {avg_confidence:.1f} 低于门槛 {self.confidence_threshold}，开仓信号已取消"
            )

        return consensus_result

    # ===== 多 Agent 预分析相关方法 =====

    def _run_multi_agent_analysis(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        运行多 Agent 预分析并将结果注入消息（QuantAgent 风格）

        流程：
        1. 提取市场上下文和交易对
        2. 并行执行三个专业 Agent 分析
        3. 生成 QuantAgent 风格的完整报告
        4. 注入到 Decision Agent 的消息中

        Args:
            messages: 原始消息列表

        Returns:
            注入了完整 Agent 分析报告的消息列表
        """
        try:
            # 从消息中提取市场上下文和交易对
            market_context = self._extract_market_context(messages)
            pair = self._extract_pair(messages)

            if not market_context:
                logger.warning("⚠️ 无法提取市场上下文，跳过多 Agent 分析")
                return messages

            logger.info(f"🤖 开始多 Agent 预分析 (QuantAgent 风格): {pair or 'UNKNOWN'}")

            # 检查是否有 OHLCV 数据可用于视觉分析
            has_ohlcv = self._current_ohlcv is not None and len(self._current_ohlcv) > 0
            if has_ohlcv:
                logger.info(f"   ✅ OHLCV 数据可用: {len(self._current_ohlcv)} 根 K 线, "
                           f"timeframe={self._current_timeframe}")
            else:
                logger.info("   ⚠️ 无 OHLCV 数据，视觉分析将不可用")

            # 运行 Agent 分析（并行执行专业 Agent，包括视觉分析）
            agent_state = self.agent_orchestrator.run_analysis(
                market_context=market_context,
                pair=pair or "UNKNOWN",
                ohlcv_data=self._current_ohlcv,  # 传递 OHLCV 数据用于图表生成
                timeframe=self._current_timeframe  # 传递时间框架
            )

            # 缓存分析状态
            self._last_agent_state = agent_state

            # 获取 QuantAgent 风格的完整报告（字典格式）
            agent_reports = self.agent_orchestrator.format_for_decision(agent_state)

            if agent_reports:
                # 注入完整的三份专业报告到消息
                messages = self._inject_agent_analysis(messages, agent_reports)

                logger.info("✅ QuantAgent 风格的多 Agent 报告已注入 Decision Agent")
                logger.info(f"   - 技术指标报告: {len(agent_reports.get('indicator_report', ''))} 字符")
                logger.info(f"   - 趋势结构报告: {len(agent_reports.get('trend_report', ''))} 字符")
                logger.info(f"   - 市场情绪报告: {len(agent_reports.get('sentiment_report', ''))} 字符")
                logger.info(f"   - 预分析共识: {agent_reports.get('consensus_direction')} "
                           f"({agent_reports.get('consensus_confidence', 0):.1f}%)")

            return messages

        except Exception as e:
            logger.error(f"❌ 多 Agent 分析失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return messages

    def _extract_market_context(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        从消息中提取市场上下文

        市场上下文通常在 user 消息中，包含 <market_data> 标签
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # 尝试提取 market_data 标签内容
                if "<market_data>" in content:
                    return content
                # 如果没有标签，但内容较长，可能就是市场上下文
                if len(content) > 500:
                    return content
        return None

    def _extract_pair(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        从消息中提取交易对

        交易对通常在 "交易对:" 或 "pair:" 后面
        """
        for msg in messages:
            content = msg.get("content", "")

            # 尝试匹配 "交易对: XXX/USDT:USDT" 格式
            match = re.search(r'交易对[:\s]+([A-Z]+/USDT(?::USDT)?)', content)
            if match:
                return match.group(1)

            # 尝试匹配 "pair: XXX/USDT" 格式
            match = re.search(r'pair[:\s]+([A-Z]+/USDT(?::USDT)?)', content, re.IGNORECASE)
            if match:
                return match.group(1)

            # 尝试匹配 "## 交易对: XXX" 格式
            match = re.search(r'##\s*交易对[:\s]+([A-Z]+/USDT(?::USDT)?)', content)
            if match:
                return match.group(1)

        return None

    def _inject_agent_analysis(
        self,
        messages: List[Dict[str, str]],
        agent_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        将 Agent 分析结果注入到消息中（QuantAgent 风格）

        注入位置：在 system message 末尾添加完整的三份专业报告

        Args:
            messages: 原始消息列表
            agent_analysis: Agent 分析数据字典，包含:
                - indicator_report: 技术指标报告
                - trend_report: 趋势结构报告
                - sentiment_report: 市场情绪报告
                - pattern_report: K线形态报告（视觉分析）
                - consensus_direction: 预分析共识方向
                - consensus_confidence: 预分析共识置信度
                - key_support: 关键支撑位
                - key_resistance: 关键阻力位

        Returns:
            注入后的消息列表
        """
        messages = copy.deepcopy(messages)

        # 格式化关键价位（保留2位小数）
        key_support = agent_analysis.get('key_support')
        key_resistance = agent_analysis.get('key_resistance')
        key_support_str = f"{key_support:.2f}" if isinstance(key_support, (int, float)) else 'N/A'
        key_resistance_str = f"{key_resistance:.2f}" if isinstance(key_resistance, (int, float)) else 'N/A'

        # 使用 QuantAgent 风格的报告模板（包括视觉分析报告）
        injection_text = self.AGENT_REPORTS_TEMPLATE.format(
            indicator_report=agent_analysis.get('indicator_report', '技术指标分析不可用'),
            trend_report=agent_analysis.get('trend_report', '趋势结构分析不可用'),
            sentiment_report=agent_analysis.get('sentiment_report', '市场情绪分析不可用'),
            pattern_report=agent_analysis.get('pattern_report', 'K线形态分析不可用'),
            consensus_direction=agent_analysis.get('consensus_direction', 'neutral'),
            consensus_confidence=agent_analysis.get('consensus_confidence', 0),
            key_support=key_support_str,
            key_resistance=key_resistance_str
        )

        # 在 system message 末尾注入
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + "\n" + injection_text
                break

        return messages

    def get_last_agent_state(self) -> Optional[Dict[str, Any]]:
        """
        获取最近一次的 Agent 分析状态

        用于日志记录和调试

        Returns:
            AgentState 字典或 None
        """
        if self._last_agent_state and self.agent_orchestrator:
            return self.agent_orchestrator.format_for_logging(self._last_agent_state)
        return None

    def _create_role_messages(
        self,
        messages: List[Dict[str, str]],
        role: str = "aggressive"
    ) -> List[Dict[str, str]]:
        """
        创建带有 Decision Agent 角色前缀的消息

        通过在 system message 开头注入角色定义，
        让两次 LLM 调用具有不同的决策倾向，实现双重 Decision Agent 验证。

        Args:
            messages: 原始消息列表
            role: 'aggressive'（激进决策者）或 'conservative'（保守决策者）

        Returns:
            带有角色前缀的消息列表
        """
        messages_modified = copy.deepcopy(messages)

        # 选择 Decision Agent 角色前缀
        if role == "aggressive":
            prefix = self.DECISION_AGENT_AGGRESSIVE_PREFIX
        else:
            prefix = self.DECISION_AGENT_CONSERVATIVE_PREFIX

        # 在 system message 开头注入角色前缀
        for msg in messages_modified:
            if msg.get("role") == "system":
                msg["content"] = prefix + msg["content"]
                break

        return messages_modified

    def _parallel_call(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]],
        max_iterations: int
    ) -> tuple:
        """
        并行执行两次决策（双 Decision Agent 模式）

        第1次：激进决策者 - 积极寻找交易机会
        第2次：保守决策者 - 严格风险评估
        """
        logger.info("📡 并行执行两次LLM决策（双 Decision Agent 模式）...")

        # 创建两个 Decision Agent 的消息
        messages_aggressive = self._create_role_messages(messages, role="aggressive")
        messages_conservative = self._create_role_messages(messages, role="conservative")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(
                self.llm_client.call_with_functions,
                messages_aggressive, functions, max_iterations
            )
            future_2 = executor.submit(
                self.llm_client.call_with_functions,
                messages_conservative, functions, max_iterations
            )

            response_1 = future_1.result()
            response_2 = future_2.result()

        logger.info("   ✅ 激进决策者 (Decision Agent 1) 完成")
        logger.info("   ✅ 保守决策者 (Decision Agent 2) 完成")

        return response_1, response_2

    def _sequential_call(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]],
        max_iterations: int
    ) -> tuple:
        """
        顺序执行两次决策（双 Decision Agent 模式）

        第1次：激进决策者 - 积极寻找交易机会
        第2次：保守决策者 - 严格风险评估
        """
        logger.info("📡 顺序执行两次LLM决策（双 Decision Agent 模式）...")

        # 创建两个 Decision Agent 的消息
        messages_aggressive = self._create_role_messages(messages, role="aggressive")
        messages_conservative = self._create_role_messages(messages, role="conservative")

        logger.info("   第1次决策（激进决策者）...")
        response_1 = self.llm_client.call_with_functions(
            messages_aggressive, functions, max_iterations
        )
        logger.info("   ✅ 激进决策者 (Decision Agent 1) 完成")

        logger.info("   第2次决策（保守决策者）...")
        response_2 = self.llm_client.call_with_functions(
            messages_conservative, functions, max_iterations
        )
        logger.info("   ✅ 保守决策者 (Decision Agent 2) 完成")

        return response_1, response_2

    def _analyze_consensus(
        self,
        response_1: Dict[str, Any],
        response_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析两次决策结果并生成共识

        Args:
            response_1: 第一次响应
            response_2: 第二次响应

        Returns:
            共识后的最终响应
        """
        # 检查两次请求是否都成功
        success_1 = response_1.get("success", False)
        success_2 = response_2.get("success", False)

        # 如果两次都失败，返回失败
        if not success_1 and not success_2:
            logger.error("❌ 两次决策都失败")
            return {
                "success": False,
                "error": "两次决策都失败",
                "response_1": response_1,
                "response_2": response_2,
                "consensus_type": "both_failed"
            }

        # 如果只有一次成功，使用成功的那次
        if not success_1:
            logger.warning("⚠️  第1次决策失败，使用第2次结果")
            return self._wrap_single_response(response_2, "fallback_to_second")

        if not success_2:
            logger.warning("⚠️  第2次决策失败，使用第1次结果")
            return self._wrap_single_response(response_1, "fallback_to_first")

        # 两次都成功，分析共识
        action_1, details_1 = self._extract_action(response_1)
        action_2, details_2 = self._extract_action(response_2)

        conf_1 = details_1.get("confidence_score", 50)
        conf_2 = details_2.get("confidence_score", 50)

        logger.info(f"📊 双 Decision Agent 决策对比:")
        logger.info(f"   【激进决策者】: {action_1} (置信度: {conf_1})")
        logger.info(f"   【保守决策者】: {action_2} (置信度: {conf_2})")

        # 判断是否达成共识
        if action_1 == action_2:
            # 动作一致 - 完全共识
            logger.info(f"✅ 完全共识: 双 Decision Agent 都同意 {action_1}")
            return self._merge_responses(
                response_1, response_2, details_1, details_2, "full_consensus"
            )
        else:
            # 动作不一致 - 需要决策
            logger.warning(f"⚠️  Decision Agent 分歧: 激进决策者主张 {action_1}, 保守决策者主张 {action_2}")
            return self._resolve_conflict(
                response_1, response_2,
                action_1, action_2,
                details_1, details_2
            )

    def _extract_action(
        self,
        response: Dict[str, Any]
    ) -> tuple:
        """
        从响应中提取决策动作和详细信息

        Returns:
            (action_type, details_dict)
        """
        function_calls = response.get("function_calls", [])

        if not function_calls:
            return "no_action", {}

        # 获取最后一个有效的交易函数调用
        for call in reversed(function_calls):
            func_name = call.get("function", "")
            result = call.get("result", {})
            args = call.get("arguments", {})

            if func_name.startswith("signal_"):
                # signal_entry_long, signal_entry_short, signal_exit, signal_hold, signal_wait
                action = result.get("action") or args.get("action")
                if not action:
                    # 从函数名推断
                    if "entry_long" in func_name:
                        action = "enter_long"
                    elif "entry_short" in func_name:
                        action = "enter_short"
                    elif "exit" in func_name:
                        action = "exit"
                    elif "hold" in func_name:
                        action = "hold"
                    elif "wait" in func_name:
                        action = "wait"

                return action, {
                    "function": func_name,
                    "confidence_score": args.get("confidence_score", 50),
                    "reason": args.get("reason", ""),
                    "leverage": args.get("leverage"),
                    "result": result
                }

            elif func_name == "adjust_position":
                return "adjust", {
                    "function": func_name,
                    "adjustment_pct": args.get("adjustment_pct", 0),
                    "confidence_score": args.get("confidence_score", 50),
                    "reason": args.get("reason", ""),
                    "result": result
                }

        return "no_action", {}

    def _merge_responses(
        self,
        response_1: Dict[str, Any],
        response_2: Dict[str, Any],
        details_1: Dict[str, Any],
        details_2: Dict[str, Any],
        consensus_type: str
    ) -> Dict[str, Any]:
        """
        合并两次响应

        - 置信度取平均值
        - reason合并两次结果
        """
        # 计算平均置信度
        conf_1 = details_1.get("confidence_score", 50)
        conf_2 = details_2.get("confidence_score", 50)
        avg_confidence = (conf_1 + conf_2) / 2

        # 合并reason
        reason_1 = details_1.get("reason", "")
        reason_2 = details_2.get("reason", "")
        merged_reason = self._merge_reasons(reason_1, reason_2)

        # 使用第一个响应作为基础，更新置信度和reason
        result = copy.deepcopy(response_1)
        result["consensus_type"] = consensus_type
        result["consensus_confidence"] = avg_confidence
        result["merged_reason"] = merged_reason
        result["original_confidences"] = [conf_1, conf_2]

        # 更新function_calls中的置信度和reason
        if result.get("function_calls"):
            for call in result["function_calls"]:
                args = call.get("arguments", {})
                if "confidence_score" in args:
                    args["confidence_score"] = avg_confidence
                    args["original_confidence_1"] = conf_1
                    args["original_confidence_2"] = conf_2
                if "reason" in args:
                    args["original_reason"] = args["reason"]
                    args["reason"] = merged_reason

        logger.info(f"📈 共识置信度: {avg_confidence:.1f} (来自 {conf_1} 和 {conf_2})")

        return result

    def _merge_reasons(self, reason_1: str, reason_2: str) -> str:
        """合并两次决策的理由"""
        if not reason_1 and not reason_2:
            return ""
        if not reason_1:
            return f"[验证决策] {reason_2}"
        if not reason_2:
            return f"[初始决策] {reason_1}"

        # 如果两个理由高度相似（>70%重叠），只保留较长的一个
        if self._text_similarity(reason_1, reason_2) > 0.7:
            return reason_1 if len(reason_1) >= len(reason_2) else reason_2

        return f"[初始决策] {reason_1}\n[验证决策] {reason_2}"

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _resolve_conflict(
        self,
        response_1: Dict[str, Any],
        response_2: Dict[str, Any],
        action_1: str,
        action_2: str,
        details_1: Dict[str, Any],
        details_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解决决策冲突

        策略：
        - conservative: 综合评估置信度和保守性
          - 置信度差异 > 阈值时，选择置信度更高的决策
          - 置信度相近时，选择更保守的决策
        - confidence: 选择置信度更高的决策
        - first: 始终使用第一次决策
        """
        conf_1 = details_1.get("confidence_score", 50)
        conf_2 = details_2.get("confidence_score", 50)
        conf_diff = abs(conf_1 - conf_2)

        priority_1 = self.ACTION_PRIORITY.get(action_1, 5)
        priority_2 = self.ACTION_PRIORITY.get(action_2, 5)

        logger.info(f"   置信度: {conf_1} vs {conf_2} (差异: {conf_diff})")
        logger.info(f"   保守性: {action_1}({priority_1}) vs {action_2}({priority_2})")

        if self.conflict_strategy == "confidence":
            # 纯置信度策略
            if conf_1 >= conf_2:
                logger.info(f"   采用【激进决策者】决策（置信度 {conf_1} >= {conf_2}）")
                chosen_response = response_1
                chosen_details = details_1
            else:
                logger.info(f"   采用【保守决策者】决策（置信度 {conf_2} > {conf_1}）")
                chosen_response = response_2
                chosen_details = details_2

        elif self.conflict_strategy == "first":
            # 始终使用激进决策者
            logger.info("   采用【激进决策者】决策（first策略）")
            chosen_response = response_1
            chosen_details = details_1

        else:  # conservative (默认) - 综合评估
            # 置信度差异显著时，优先选择高置信度决策
            if conf_diff > self.CONFIDENCE_DIFF_THRESHOLD:
                if conf_1 > conf_2:
                    logger.info(f"   采用【激进决策者】决策（置信度差异 {conf_diff} > {self.CONFIDENCE_DIFF_THRESHOLD}，{conf_1} > {conf_2}）")
                    chosen_response = response_1
                    chosen_details = details_1
                else:
                    logger.info(f"   采用【保守决策者】决策（置信度差异 {conf_diff} > {self.CONFIDENCE_DIFF_THRESHOLD}，{conf_2} > {conf_1}）")
                    chosen_response = response_2
                    chosen_details = details_2
            else:
                # 置信度相近，选择更保守的决策
                if priority_1 <= priority_2:
                    logger.info(f"   采用【激进决策者】决策（置信度相近，{action_1} 更保守）")
                    chosen_response = response_1
                    chosen_details = details_1
                else:
                    logger.info(f"   采用【保守决策者】决策（置信度相近，{action_2} 更保守）")
                    chosen_response = response_2
                    chosen_details = details_2

        # 如果require_consensus=True且存在冲突，降级为wait/hold
        if self.require_consensus:
            logger.warning("   require_consensus=True，决策分歧时降级为观望")
            return self._create_wait_response(response_1, response_2, details_1, details_2)

        # 合并信息
        result = copy.deepcopy(chosen_response)
        result["consensus_type"] = "conflict_resolved"
        result["conflict_resolution"] = {
            "strategy": self.conflict_strategy,
            "action_1": action_1,
            "action_2": action_2,
            "chosen_action": action_1 if chosen_response is response_1 else action_2
        }

        # 合并reason（记录分歧）
        merged_reason = self._merge_conflict_reasons(
            details_1.get("reason", ""),
            details_2.get("reason", ""),
            action_1, action_2
        )

        if result.get("function_calls"):
            for call in result["function_calls"]:
                args = call.get("arguments", {})
                if "reason" in args:
                    args["original_reason"] = args["reason"]
                    args["reason"] = merged_reason

        return result

    def _merge_conflict_reasons(
        self,
        reason_1: str,
        reason_2: str,
        action_1: str,
        action_2: str
    ) -> str:
        """合并冲突时的理由（记录分歧）"""
        return (
            f"[共识分歧 - {action_1} vs {action_2}]\n"
            f"[初始决策 ({action_1})] {reason_1}\n"
            f"[验证决策 ({action_2})] {reason_2}"
        )

    def _create_wait_response(
        self,
        response_1: Dict[str, Any],
        response_2: Dict[str, Any],
        details_1: Dict[str, Any],
        details_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """当require_consensus=True且存在冲突时，创建观望响应"""
        return {
            "success": True,
            "consensus_type": "conflict_wait",
            "message": "决策存在分歧，降级为观望",
            "function_calls": [{
                "function": "signal_wait",
                "arguments": {
                    "pair": details_1.get("result", {}).get("pair", ""),
                    "confidence_score": 50,  # 低置信度
                    "rsi_value": 50,
                    "reason": f"[共识分歧] 两次决策不一致，保守观望。"
                             f"初始: {self._extract_action(response_1)[0]}, "
                             f"验证: {self._extract_action(response_2)[0]}"
                },
                "result": {"success": True, "action": "wait"}
            }],
            "original_responses": {
                "response_1": response_1,
                "response_2": response_2
            }
        }

    def _wrap_single_response(
        self,
        response: Dict[str, Any],
        consensus_type: str
    ) -> Dict[str, Any]:
        """包装单个响应（当另一个失败时）"""
        result = copy.deepcopy(response)
        result["consensus_type"] = consensus_type
        return result

    # ========== 代理方法，保持与LLMClient接口兼容 ==========

    def simple_call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """代理到LLMClient的simple_call"""
        return self.llm_client.simple_call(messages, temperature, max_tokens, timeout)

    def manage_context_window(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 6000
    ) -> List[Dict[str, str]]:
        """代理到LLMClient的manage_context_window"""
        return self.llm_client.manage_context_window(messages, max_tokens)

    def add_to_history(self, role: str, content: str):
        """代理到LLMClient的add_to_history"""
        self.llm_client.add_to_history(role, content)

    def clear_history(self):
        """代理到LLMClient的clear_history"""
        self.llm_client.clear_history()

    def get_history(self, include_timestamp: bool = False) -> List[Dict[str, Any]]:
        """代理到LLMClient的get_history"""
        return self.llm_client.get_history(include_timestamp)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.llm_client.get_statistics()
        stats["consensus_enabled"] = self.enabled
        stats["conflict_strategy"] = self.conflict_strategy
        stats["parallel_requests"] = self.parallel_requests

        # 多 Agent 系统统计
        stats["multi_agent_enabled"] = self.multi_agent_enabled
        if self.multi_agent_enabled and self.agent_orchestrator:
            stats["agent_orchestrator"] = self.agent_orchestrator.get_statistics()

        return stats
