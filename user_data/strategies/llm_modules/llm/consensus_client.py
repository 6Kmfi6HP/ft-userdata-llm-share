"""
双重决策共识客户端模块
对同一模型使用相似提示词进行两次决策，通过对比结果提高决策可靠性

设计原则：
1. 使用相同模型进行两次独立决策
2. 第二次请求添加验证性提示词变体
3. 对比两次决策结果，采用共识或保守策略
4. 置信度取平均值，reason合并两次结果
"""
import logging
import json
import copy
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

    # ========== 双角色并行验证模式 ==========
    # 机会发现者角色前缀：积极识别高胜率交易机会
    OPPORTUNITY_FINDER_PREFIX = """# 决策角色：机会发现者 Opportunity Finder
你的主要职责是识别高胜率的交易机会。

在满足以下条件时积极建议入场：
- 至少2个独立信号确认
- 盈亏比 ≥ 2:1
- 趋势方向明确或反转信号充分

---

"""

    # 风险评估者角色前缀：识别潜在风险和交易陷阱
    RISK_ASSESSOR_PREFIX = """# 决策角色：风险评估者 Risk Assessor
你的主要职责是识别交易风险和潜在陷阱。

只在以下情况下才同意入场：
- 风险充分可控
- 盈亏比显著有利
- 无明显的陷阱迹象

如有重大风险疑虑，宁可错过机会也要保守观望。

---

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

        logger.info(f"双重决策共识客户端已初始化（双角色模式）: enabled={self.enabled}, "
                   f"parallel={self.parallel_requests}, strategy={self.conflict_strategy}")

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
        logger.info("🔄 双重决策共识验证开始（双角色模式）")
        logger.info("=" * 60)

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
        logger.info(f"⏱️  双重决策耗时: {elapsed:.2f}秒")
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

    def _create_role_messages(
        self,
        messages: List[Dict[str, str]],
        role: str = "opportunity"
    ) -> List[Dict[str, str]]:
        """
        创建带有角色前缀的消息

        通过在 system message 开头注入角色定义，
        让两次 LLM 调用具有不同的认知框架，实现真正独立的验证。

        Args:
            messages: 原始消息列表
            role: 'opportunity'（机会发现者）或 'risk'（风险评估者）

        Returns:
            带有角色前缀的消息列表
        """
        messages_modified = copy.deepcopy(messages)

        # 选择角色前缀
        if role == "opportunity":
            prefix = self.OPPORTUNITY_FINDER_PREFIX
        else:
            prefix = self.RISK_ASSESSOR_PREFIX

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
        并行执行两次决策（双角色模式）

        第1次：机会发现者 - 积极识别交易机会
        第2次：风险评估者 - 识别潜在风险陷阱
        """
        logger.info("📡 并行执行两次LLM决策（双角色模式）...")

        # 创建两个角色的消息
        messages_opportunity = self._create_role_messages(messages, role="opportunity")
        messages_risk = self._create_role_messages(messages, role="risk")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(
                self.llm_client.call_with_functions,
                messages_opportunity, functions, max_iterations
            )
            future_2 = executor.submit(
                self.llm_client.call_with_functions,
                messages_risk, functions, max_iterations
            )

            response_1 = future_1.result()
            response_2 = future_2.result()

        logger.info("   ✅ 机会发现者决策完成")
        logger.info("   ✅ 风险评估者决策完成")

        return response_1, response_2

    def _sequential_call(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]],
        max_iterations: int
    ) -> tuple:
        """
        顺序执行两次决策（双角色模式）

        第1次：机会发现者 - 积极识别交易机会
        第2次：风险评估者 - 识别潜在风险陷阱
        """
        logger.info("📡 顺序执行两次LLM决策（双角色模式）...")

        # 创建两个角色的消息
        messages_opportunity = self._create_role_messages(messages, role="opportunity")
        messages_risk = self._create_role_messages(messages, role="risk")

        logger.info("   第1次决策（机会发现者）...")
        response_1 = self.llm_client.call_with_functions(
            messages_opportunity, functions, max_iterations
        )
        logger.info("   ✅ 机会发现者决策完成")

        logger.info("   第2次决策（风险评估者）...")
        response_2 = self.llm_client.call_with_functions(
            messages_risk, functions, max_iterations
        )
        logger.info("   ✅ 风险评估者决策完成")

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

        logger.info(f"📊 双角色决策对比:")
        logger.info(f"   【机会发现者】: {action_1} (置信度: {conf_1})")
        logger.info(f"   【风险评估者】: {action_2} (置信度: {conf_2})")

        # 判断是否达成共识
        if action_1 == action_2:
            # 动作一致 - 完全共识
            logger.info(f"✅ 完全共识: 两个角色都同意 {action_1}")
            return self._merge_responses(
                response_1, response_2, details_1, details_2, "full_consensus"
            )
        else:
            # 动作不一致 - 需要决策
            logger.warning(f"⚠️  角色分歧: 机会发现者主张 {action_1}, 风险评估者主张 {action_2}")
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
                logger.info(f"   采用【机会发现者】决策（置信度 {conf_1} >= {conf_2}）")
                chosen_response = response_1
                chosen_details = details_1
            else:
                logger.info(f"   采用【风险评估者】决策（置信度 {conf_2} > {conf_1}）")
                chosen_response = response_2
                chosen_details = details_2

        elif self.conflict_strategy == "first":
            # 始终使用机会发现者
            logger.info("   采用【机会发现者】决策（first策略）")
            chosen_response = response_1
            chosen_details = details_1

        else:  # conservative (默认) - 综合评估
            # 置信度差异显著时，优先选择高置信度决策
            if conf_diff > self.CONFIDENCE_DIFF_THRESHOLD:
                if conf_1 > conf_2:
                    logger.info(f"   采用【机会发现者】决策（置信度差异 {conf_diff} > {self.CONFIDENCE_DIFF_THRESHOLD}，{conf_1} > {conf_2}）")
                    chosen_response = response_1
                    chosen_details = details_1
                else:
                    logger.info(f"   采用【风险评估者】决策（置信度差异 {conf_diff} > {self.CONFIDENCE_DIFF_THRESHOLD}，{conf_2} > {conf_1}）")
                    chosen_response = response_2
                    chosen_details = details_2
            else:
                # 置信度相近，选择更保守的决策
                if priority_1 <= priority_2:
                    logger.info(f"   采用【机会发现者】决策（置信度相近，{action_1} 更保守）")
                    chosen_response = response_1
                    chosen_details = details_1
                else:
                    logger.info(f"   采用【风险评估者】决策（置信度相近，{action_2} 更保守）")
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
        return stats
