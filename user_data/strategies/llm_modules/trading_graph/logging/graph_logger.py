"""
LangGraph 决策日志记录器

记录 LangGraph 交易决策系统的完整决策链到 JSONL 文件。
支持：
- 分析阶段日志（IndicatorAgent, TrendAgent, SentimentAgent, PatternAgent）
- 辩论阶段日志（BullAgent, BearAgent, JudgeAgent）
- Grounding 验证日志
- 持仓管理决策日志
- 最终执行决策日志
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class GraphDecisionLogger:
    """
    LangGraph 决策日志记录器
    
    将 LangGraph 的完整决策链记录到 JSONL 文件，
    便于事后分析和回溯决策逻辑。
    """
    
    def __init__(
        self,
        log_path: str = "./user_data/logs/graph_decisions.jsonl",
        enabled: bool = True
    ):
        """
        初始化 LangGraph 决策日志记录器
        
        Args:
            log_path: 日志文件路径
            enabled: 是否启用日志记录
        """
        self.log_path = Path(log_path)
        self.enabled = enabled
        
        # 确保目录存在
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()
                logger.info(f"已创建 LangGraph 决策日志文件: {self.log_path}")
        
        logger.info(f"GraphDecisionLogger 初始化完成: enabled={enabled}")
    
    def log_graph_decision(
        self,
        state: Dict[str, Any],
        execution_result: Dict[str, Any],
        execution_time_ms: float = 0.0
    ) -> bool:
        """
        记录完整的 LangGraph 决策链
        
        Args:
            state: LangGraph 最终状态
            execution_result: 执行结果
            execution_time_ms: 执行耗时（毫秒）
            
        Returns:
            是否记录成功
        """
        if not self.enabled:
            return True
            
        try:
            log_entry = self._build_log_entry(state, execution_result, execution_time_ms)
            self._write_jsonl(log_entry)
            
            pair = state.get("pair", "UNKNOWN")
            action = execution_result.get("action", "unknown")
            logger.debug(f"LangGraph 决策已记录: {pair} -> {action}")
            return True
            
        except Exception as e:
            logger.error(f"记录 LangGraph 决策失败: {e}")
            return False
    
    def _build_log_entry(
        self,
        state: Dict[str, Any],
        execution_result: Dict[str, Any],
        execution_time_ms: float
    ) -> Dict[str, Any]:
        """
        构建日志条目
        
        从 LangGraph state 提取关键信息，构建结构化日志条目。
        """
        pair = state.get("pair", "UNKNOWN")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "thread_id": state.get("thread_id", ""),
            "source": "langgraph",
            "execution_time_ms": execution_time_ms,
            
            # 输入信息
            "input": {
                "current_price": state.get("current_price", 0),
                "timeframe": state.get("timeframe", ""),
                "timeframe_htf": state.get("timeframe_htf"),
                "has_position": state.get("has_position", False),
                "position_side": state.get("position_side"),
                "position_profit_pct": state.get("position_profit_pct", 0),
            },
            
            # 分析阶段结果
            "analysis": self._extract_analysis_data(state),
            
            # 辩论阶段结果
            "debate": self._extract_debate_data(state),
            
            # Grounding 验证结果
            "grounding": self._extract_grounding_data(state),
            
            # 持仓管理（如果适用）
            "position_management": self._extract_position_data(state),
            
            # 最终决策
            "final_decision": {
                "action": execution_result.get("action", "signal_wait"),
                "confidence_score": execution_result.get("confidence_score", 0),
                "leverage": execution_result.get("leverage"),
                "reason": execution_result.get("reason", ""),
                "key_support": execution_result.get("key_support"),
                "key_resistance": execution_result.get("key_resistance"),
                "adjustment_pct": execution_result.get("adjustment_pct"),
            },
            
            # 错误信息
            "errors": state.get("errors", []),
        }
    
    def _extract_analysis_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取分析阶段数据"""
        analysis = {
            "consensus_direction": None,
            "consensus_confidence": state.get("consensus_confidence", 0),
            "key_support": state.get("key_support"),
            "key_resistance": state.get("key_resistance"),
            "weighted_scores": state.get("weighted_scores"),
            "agents": {}
        }
        
        # 提取共识方向
        consensus_dir = state.get("consensus_direction")
        if consensus_dir:
            if hasattr(consensus_dir, "value"):
                analysis["consensus_direction"] = consensus_dir.value
            else:
                analysis["consensus_direction"] = str(consensus_dir)
        
        # 提取各 Agent 报告
        agent_reports = {
            "indicator": state.get("indicator_report"),
            "trend": state.get("trend_report"),
            "sentiment": state.get("sentiment_report"),
            "pattern": state.get("pattern_report"),
        }
        
        for agent_name, report in agent_reports.items():
            if report:
                analysis["agents"][agent_name] = self._format_agent_report(report)
        
        return analysis
    
    def _format_agent_report(self, report: Any) -> Dict[str, Any]:
        """格式化 Agent 报告为可序列化字典"""
        if report is None:
            return {}
        
        # 如果是 AgentReport dataclass
        if hasattr(report, "to_dict"):
            return report.to_dict()
        
        # 如果是字典
        if isinstance(report, dict):
            return self._make_serializable(report)
        
        # 其他情况，尝试转换
        try:
            return {
                "agent_name": getattr(report, "agent_name", "unknown"),
                "analysis": getattr(report, "analysis", ""),
                "confidence": getattr(report, "confidence", 0),
                "direction": self._format_enum(getattr(report, "direction", None)),
                "error": getattr(report, "error", None),
            }
        except Exception:
            return {"raw": str(report)[:500]}
    
    def _extract_debate_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取辩论阶段数据"""
        debate = {
            "bull_argument": None,
            "bear_argument": None,
            "judge_verdict": None,
        }
        
        # Bull 论点
        bull = state.get("bull_argument")
        if bull:
            debate["bull_argument"] = self._format_debate_argument(bull)
        
        # Bear 论点
        bear = state.get("bear_argument")
        if bear:
            debate["bear_argument"] = self._format_debate_argument(bear)
        
        # Judge 裁决
        verdict = state.get("judge_verdict")
        if verdict:
            debate["judge_verdict"] = self._format_judge_verdict(verdict)
        
        return debate
    
    def _format_debate_argument(self, arg: Any) -> Dict[str, Any]:
        """格式化辩论论点"""
        if arg is None:
            return {}
        
        if hasattr(arg, "to_dict"):
            return arg.to_dict()
        
        if isinstance(arg, dict):
            return self._make_serializable(arg)
        
        try:
            return {
                "agent_role": getattr(arg, "agent_role", ""),
                "position": self._format_enum(getattr(arg, "position", None)),
                "confidence": getattr(arg, "confidence", 0),
                "key_points": getattr(arg, "key_points", []),
                "risk_factors": getattr(arg, "risk_factors", []),
                "recommended_action": getattr(arg, "recommended_action", ""),
                "reasoning": getattr(arg, "reasoning", "")[:500],  # 限制长度
            }
        except Exception:
            return {"raw": str(arg)[:500]}
    
    def _format_judge_verdict(self, verdict: Any) -> Dict[str, Any]:
        """格式化 Judge 裁决"""
        if verdict is None:
            return {}
        
        if hasattr(verdict, "to_dict"):
            return verdict.to_dict()
        
        if isinstance(verdict, dict):
            return self._make_serializable(verdict)
        
        try:
            return {
                "verdict": self._format_enum(getattr(verdict, "verdict", None)),
                "confidence": getattr(verdict, "confidence", 0),
                "winning_argument": getattr(verdict, "winning_argument", None),
                "key_reasoning": getattr(verdict, "key_reasoning", "")[:500],
                "recommended_action": getattr(verdict, "recommended_action", ""),
                "leverage": getattr(verdict, "leverage", None),
                "risk_assessment": getattr(verdict, "risk_assessment", ""),
            }
        except Exception:
            return {"raw": str(verdict)[:500]}
    
    def _extract_grounding_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取 Grounding 验证数据"""
        grounding = {
            "verified": state.get("grounding_verified", False),
            "result": None,
        }
        
        grounding_result = state.get("grounding_result")
        if grounding_result:
            grounding["result"] = self._format_grounding_result(grounding_result)
        
        return grounding
    
    def _format_grounding_result(self, result: Any) -> Dict[str, Any]:
        """格式化 Grounding 验证结果"""
        if result is None:
            return {}
        
        if hasattr(result, "to_dict"):
            return result.to_dict()
        
        if isinstance(result, dict):
            return self._make_serializable(result)
        
        try:
            return {
                "hallucination_score": getattr(result, "hallucination_score", 0),
                "confidence_penalty": getattr(result, "confidence_penalty", 0),
                "is_rejected": getattr(result, "is_rejected", False),
                "rejection_reason": getattr(result, "rejection_reason", ""),
                "total_claims": getattr(result, "total_claims", 0),
                "false_claims": getattr(result, "false_claims", 0),
                "verified_claims": getattr(result, "verified_claims", []),
            }
        except Exception:
            return {"raw": str(result)[:500]}
    
    def _extract_position_data(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取持仓管理数据（如果适用）"""
        if not state.get("has_position", False):
            return None
        
        position = {
            "metrics": state.get("position_metrics"),
            "bull_argument": None,
            "bear_argument": None,
            "judge_verdict": None,
            "grounding_result": None,
        }
        
        # 持仓辩论
        pos_bull = state.get("position_bull_argument")
        if pos_bull:
            position["bull_argument"] = self._format_debate_argument(pos_bull)
        
        pos_bear = state.get("position_bear_argument")
        if pos_bear:
            position["bear_argument"] = self._format_debate_argument(pos_bear)
        
        pos_verdict = state.get("position_judge_verdict")
        if pos_verdict:
            position["judge_verdict"] = self._format_position_verdict(pos_verdict)
        
        pos_grounding = state.get("position_grounding_result")
        if pos_grounding:
            position["grounding_result"] = self._format_grounding_result(pos_grounding)
        
        return position
    
    def _format_position_verdict(self, verdict: Any) -> Dict[str, Any]:
        """格式化持仓管理裁决"""
        if verdict is None:
            return {}
        
        if hasattr(verdict, "to_dict"):
            return verdict.to_dict()
        
        if isinstance(verdict, dict):
            return self._make_serializable(verdict)
        
        try:
            return {
                "verdict": self._format_enum(getattr(verdict, "verdict", None)),
                "confidence": getattr(verdict, "confidence", 0),
                "adjustment_pct": getattr(verdict, "adjustment_pct", None),
                "key_reasoning": getattr(verdict, "key_reasoning", "")[:500],
                "profit_protection_triggered": getattr(verdict, "profit_protection_triggered", False),
                "forced_rule_name": getattr(verdict, "forced_rule_name", None),
                "bull_score": getattr(verdict, "bull_score", 0),
                "bear_score": getattr(verdict, "bear_score", 0),
            }
        except Exception:
            return {"raw": str(verdict)[:500]}
    
    def _format_enum(self, enum_val: Any) -> Optional[str]:
        """格式化枚举值为字符串"""
        if enum_val is None:
            return None
        if hasattr(enum_val, "value"):
            return enum_val.value
        return str(enum_val)
    
    def _make_serializable(self, obj: Any) -> Any:
        """递归地将对象转换为 JSON 可序列化格式"""
        if obj is None:
            return None
        
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        
        if isinstance(obj, dict):
            return {
                str(k): self._make_serializable(v) 
                for k, v in obj.items()
            }
        
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        
        if hasattr(obj, "value"):  # Enum
            return obj.value
        
        if hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        
        # 其他类型转为字符串
        return str(obj)[:500]
    
    def _write_jsonl(self, log_entry: Dict[str, Any]) -> None:
        """写入 JSONL 文件"""
        serializable_entry = self._make_serializable(log_entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_entry, ensure_ascii=False) + "\n")
    
    def query_decisions(
        self,
        pair: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询决策日志
        
        Args:
            pair: 过滤交易对
            action: 过滤动作类型
            limit: 最大返回数量
            
        Returns:
            匹配的日志条目列表
        """
        if not self.log_path.exists():
            return []
        
        results = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        # 应用过滤条件
                        if pair and entry.get("pair") != pair:
                            continue
                        if action:
                            final_action = entry.get("final_decision", {}).get("action")
                            if final_action != action:
                                continue
                        
                        results.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            # 返回最新的 N 条
            return results[-limit:]
            
        except Exception as e:
            logger.error(f"查询决策日志失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取决策日志统计信息"""
        if not self.log_path.exists():
            return {"total_decisions": 0}
        
        try:
            action_counts = {}
            direction_counts = {}
            pair_counts = {}
            total = 0
            
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        total += 1
                        
                        # 统计动作
                        action = entry.get("final_decision", {}).get("action", "unknown")
                        action_counts[action] = action_counts.get(action, 0) + 1
                        
                        # 统计方向
                        direction = entry.get("analysis", {}).get("consensus_direction")
                        if direction:
                            direction_counts[direction] = direction_counts.get(direction, 0) + 1
                        
                        # 统计交易对
                        pair = entry.get("pair", "UNKNOWN")
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
                        
                    except json.JSONDecodeError:
                        continue
            
            return {
                "total_decisions": total,
                "action_counts": action_counts,
                "direction_counts": direction_counts,
                "pair_counts": pair_counts,
            }
            
        except Exception as e:
            logger.error(f"获取决策日志统计失败: {e}")
            return {"total_decisions": 0, "error": str(e)}
