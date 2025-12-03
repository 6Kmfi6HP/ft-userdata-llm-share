"""
决策查询引擎
从llm_decisions.jsonl查询上次分析决策，提供给LLM参考
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class DecisionQueryEngine:
    """决策查询引擎 - 查询历史LLM分析决策"""

    def __init__(self, decision_log_path: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化查询引擎

        Args:
            decision_log_path: llm_decisions.jsonl 文件路径
            config: 配置项
                - max_age_hours: 最大有效时间（小时），默认24
                - max_tokens: 最大token数（用于截断），默认800
        """
        self.decision_log_path = Path(decision_log_path)
        self.config = config or {}
        self.max_age_hours = self.config.get('previous_decision_max_age_hours', 24)
        self.max_chars = self.config.get('previous_decision_max_chars', 1500)  # 约等于500-800 tokens

        # 缓存：每个交易对的最新决策
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_load_time: Optional[datetime] = None
        self._file_mtime: Optional[float] = None

    def _should_reload(self) -> bool:
        """检查是否需要重新加载"""
        if not self.decision_log_path.exists():
            return False

        # 检查文件修改时间
        current_mtime = self.decision_log_path.stat().st_mtime
        if self._file_mtime is None or current_mtime > self._file_mtime:
            return True

        # 每60秒强制检查一次
        if self._last_load_time is None:
            return True
        if (datetime.now(timezone.utc) - self._last_load_time).total_seconds() > 60:
            return True

        return False

    def _load_latest_decisions(self):
        """
        从JSONL文件加载每个交易对的最新决策
        使用倒序读取优化性能
        """
        if not self.decision_log_path.exists():
            self._cache = {}
            return

        try:
            # 记录文件修改时间
            self._file_mtime = self.decision_log_path.stat().st_mtime

            # 倒序读取文件（最新的在最后）
            new_cache: Dict[str, Dict[str, Any]] = {}
            seen_pairs: set = set()

            with open(self.decision_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 从后往前遍历，每个交易对只保留最新的一条
            for line in reversed(lines):
                if not line.strip():
                    continue

                try:
                    decision = json.loads(line)
                    pair = decision.get('pair', '')

                    if pair and pair not in seen_pairs:
                        seen_pairs.add(pair)
                        new_cache[pair] = decision

                except json.JSONDecodeError:
                    continue

            self._cache = new_cache
            self._last_load_time = datetime.now(timezone.utc)
            logger.debug(f"已加载 {len(self._cache)} 个交易对的最新决策")

        except Exception as e:
            logger.error(f"加载决策日志失败: {e}")
            self._cache = {}

    def get_previous_decision(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        获取指定交易对的上次决策

        Args:
            pair: 交易对

        Returns:
            上次决策记录，如果不存在或已过期则返回None
        """
        # 检查是否需要重新加载
        if self._should_reload():
            self._load_latest_decisions()

        decision = self._cache.get(pair)
        if not decision:
            return None

        # 检查时效性
        timestamp_str = decision.get('timestamp', '')
        if timestamp_str:
            try:
                decision_time = self._parse_datetime(timestamp_str)
                age_hours = (datetime.now(timezone.utc) - decision_time).total_seconds() / 3600

                if age_hours > self.max_age_hours:
                    logger.debug(f"[{pair}] 上次决策已过期 ({age_hours:.1f}h > {self.max_age_hours}h)")
                    return None

            except Exception as e:
                logger.warning(f"解析决策时间失败: {e}")

        return decision

    def format_previous_decision_for_context(self, pair: str) -> str:
        """
        格式化上次决策为上下文文本（不包含结果和置信度）

        Args:
            pair: 交易对

        Returns:
            格式化的上次决策文本，如果不存在则返回空字符串
        """
        decision = self.get_previous_decision(pair)
        if not decision:
            return ""

        timestamp_str = decision.get('timestamp', '')
        decision_text = decision.get('decision', '')

        if not decision_text:
            return ""

        # 计算时间差
        time_ago_str = ""
        if timestamp_str:
            try:
                decision_time = self._parse_datetime(timestamp_str)
                time_diff = datetime.now(timezone.utc) - decision_time
                hours_ago = time_diff.total_seconds() / 3600

                if hours_ago < 1:
                    minutes_ago = int(time_diff.total_seconds() / 60)
                    time_ago_str = f"{minutes_ago}分钟前"
                elif hours_ago < 24:
                    time_ago_str = f"{hours_ago:.1f}小时前"
                else:
                    days_ago = hours_ago / 24
                    time_ago_str = f"{days_ago:.1f}天前"
            except Exception:
                time_ago_str = "未知时间"

        # 截断决策文本（保留完整性，在句子边界截断）
        truncated_text = self._truncate_text(decision_text, self.max_chars)

        # 构建格式化输出（不包含结果和置信度）
        lines = [
            f"### 上次分析 ({time_ago_str})",
            truncated_text
        ]

        return '\n'.join(lines)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """
        智能截断文本，尽量在句子边界截断

        Args:
            text: 原始文本
            max_chars: 最大字符数

        Returns:
            截断后的文本
        """
        if len(text) <= max_chars:
            return text

        # 在max_chars位置附近寻找句子边界
        truncated = text[:max_chars]

        # 寻找最后一个句子结束符
        for end_char in ['。', '！', '？', '\n\n', '.\n', '. ']:
            last_pos = truncated.rfind(end_char)
            if last_pos > max_chars * 0.6:  # 至少保留60%的内容
                return truncated[:last_pos + len(end_char)].rstrip() + "..."

        # 如果找不到句子边界，在最后一个换行符处截断
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.6:
            return truncated[:last_newline].rstrip() + "..."

        # 最后手段：直接截断
        return truncated.rstrip() + "..."

    def _parse_datetime(self, time_str: str) -> datetime:
        """
        解析 ISO 格式时间字符串

        Args:
            time_str: ISO 格式时间字符串

        Returns:
            UTC aware datetime 对象
        """
        if not time_str:
            return datetime.now(timezone.utc)

        try:
            # 处理多种ISO格式
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))

            # 如果是 naive datetime，假设是 UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            return dt
        except (ValueError, AttributeError) as e:
            logger.warning(f"解析时间字符串失败: {time_str}, 错误: {e}")
            return datetime.now(timezone.utc)

    def get_recent_decisions_for_pair(
        self,
        pair: str,
        limit: int = 5,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指定交易对的最近N条决策（可选功能，用于未来扩展）

        Args:
            pair: 交易对
            limit: 返回数量
            hours: 最近N小时内（可选）

        Returns:
            决策列表（按时间倒序）
        """
        if not self.decision_log_path.exists():
            return []

        results = []
        cutoff_time = None
        if hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            with open(self.decision_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in reversed(lines):
                if len(results) >= limit:
                    break

                if not line.strip():
                    continue

                try:
                    decision = json.loads(line)
                    if decision.get('pair') != pair:
                        continue

                    # 时间过滤
                    if cutoff_time:
                        timestamp_str = decision.get('timestamp', '')
                        if timestamp_str:
                            decision_time = self._parse_datetime(timestamp_str)
                            if decision_time < cutoff_time:
                                continue

                    results.append(decision)

                except json.JSONDecodeError:
                    continue

            return results

        except Exception as e:
            logger.error(f"查询决策失败: {e}")
            return []
