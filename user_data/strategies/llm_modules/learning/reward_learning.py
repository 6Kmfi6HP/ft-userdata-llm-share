"""
奖励惩罚学习机制
基于交易评价结果，构建奖励/惩罚系统指导模型学习
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RewardLearningSystem:
    """奖励学习系统"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化奖励学习系统

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 学习率
        self.learning_rate = self.config.get('learning_rate', 0.1)

        # 奖励衰减因子
        self.discount_factor = self.config.get('discount_factor', 0.95)

        # 历史奖励记录
        self.reward_history = deque(maxlen=1000)

        # 模式记忆 (什么样的决策导致好/坏的结果)
        self.pattern_memory: Dict[str, List[Dict[str, Any]]] = {
            'successful': [],    # 成功模式
            'failed': []         # 失败模式
        }

        # 持久化路径
        self.storage_path = Path(
            self.config.get('storage_path', './user_data/logs/reward_learning.json')
        )

        # 尝试加载历史数据
        self._load_history()

        logger.info("奖励学习系统已初始化")

    def record_reward(
        self,
        trade_id: int,
        pair: str,
        action_type: str,  # entry | exit | hold | adjust
        decision_context: Dict[str, Any],
        evaluation: Dict[str, Any]
    ):
        """
        记录奖励

        Args:
            trade_id: 交易ID
            pair: 交易对
            action_type: 决策类型
            decision_context: 决策上下文（市场状态、指标等）
            evaluation: 交易评价结果
        """
        reward = evaluation.get('reward', 0)
        score = evaluation.get('total_score', 50)

        # 创建奖励记录
        record = {
            'timestamp': datetime.now().isoformat(),
            'trade_id': trade_id,
            'pair': pair,
            'action_type': action_type,
            'reward': reward,
            'score': score,
            'grade': evaluation.get('grade', 'C'),
            'decision_context': decision_context,
            'evaluation': evaluation
        }

        self.reward_history.append(record)

        # 更新模式记忆
        if score >= 70:
            # 成功模式
            self.pattern_memory['successful'].append({
                'action_type': action_type,
                'decision_context': decision_context,
                'reward': reward,
                'score': score
            })
            # 限制大小
            if len(self.pattern_memory['successful']) > 100:
                self.pattern_memory['successful'] = self.pattern_memory['successful'][-100:]

        elif score < 40:
            # 失败模式
            self.pattern_memory['failed'].append({
                'action_type': action_type,
                'decision_context': decision_context,
                'reward': reward,
                'score': score
            })
            # 限制大小
            if len(self.pattern_memory['failed']) > 100:
                self.pattern_memory['failed'] = self.pattern_memory['failed'][-100:]

        logger.info(
            f"奖励记录: {pair} | {action_type} | "
            f"奖励={reward:+.4f} | 评分={score:.0f} | 评级={evaluation.get('grade')}"
        )

        # 定期持久化
        if len(self.reward_history) % 10 == 0:
            self._save_history()

    # TODO: 未集成到主流程 - 考虑实现集成或在未来版本中移除
    # 功能描述: 根据历史奖励记录，为当前决策提供学习指导
    def get_learning_guidance(
        self,
        current_context: Dict[str, Any],
        action_type: str
    ) -> Dict[str, Any]:
        """
        根据历史奖励，提供学习指导

        Args:
            current_context: 当前市场上下文
            action_type: 决策类型

        Returns:
            学习指导字典
        """
        # 查找相似的成功和失败案例
        similar_successes = self._find_similar_patterns(
            current_context,
            action_type,
            pattern_type='successful'
        )

        similar_failures = self._find_similar_patterns(
            current_context,
            action_type,
            pattern_type='failed'
        )

        # 构建指导信息
        guidance = {
            'has_guidance': False,
            'confidence_adjustment': 0,
            'warnings': [],
            'recommendations': [],
            'similar_successes': similar_successes[:3],
            'similar_failures': similar_failures[:3]
        }

        # 如果有相似的失败案例，发出警告
        if similar_failures:
            guidance['has_guidance'] = True
            guidance['warnings'].append(
                f"当前情况与{len(similar_failures)}个失败案例相似，建议谨慎"
            )
            guidance['confidence_adjustment'] = -0.2  # 降低信心

            # 提取失败原因
            for fail in similar_failures[:2]:
                eval_data = fail.get('evaluation', {})
                weaknesses = eval_data.get('comments', {}).get('weaknesses', [])
                if weaknesses:
                    guidance['warnings'].append(f"历史失败原因: {weaknesses[0]}")

        # 如果有相似的成功案例，提供建议
        if similar_successes:
            guidance['has_guidance'] = True
            avg_reward = sum(s.get('reward', 0) for s in similar_successes) / len(similar_successes)

            if avg_reward > 0.05:
                guidance['recommendations'].append(
                    f"当前情况与{len(similar_successes)}个成功案例相似，平均奖励{avg_reward:+.2%}"
                )
                guidance['confidence_adjustment'] = max(
                    guidance.get('confidence_adjustment', 0),
                    0.1
                )

            # 提取成功要素
            for success in similar_successes[:2]:
                eval_data = success.get('evaluation', {})
                strengths = eval_data.get('comments', {}).get('strengths', [])
                if strengths:
                    guidance['recommendations'].append(f"成功要素: {strengths[0]}")

        return guidance

    # TODO: 未集成到主流程 - 考虑实现集成或在未来版本中移除
    # 功能描述: 获取奖励趋势统计，分析交易表现变化
    def get_reward_trend(self, window_size: int = 20) -> Dict[str, Any]:
        """
        获取奖励趋势

        Args:
            window_size: 窗口大小

        Returns:
            趋势统计
        """
        if len(self.reward_history) < window_size:
            window_size = len(self.reward_history)

        if window_size == 0:
            return {'count': 0}

        recent_records = list(self.reward_history)[-window_size:]

        avg_reward = sum(r['reward'] for r in recent_records) / window_size
        avg_score = sum(r['score'] for r in recent_records) / window_size

        # 计算趋势（最近10个 vs 之前10个）
        if window_size >= 20:
            half = window_size // 2
            first_half_reward = sum(r['reward'] for r in recent_records[:half]) / half
            second_half_reward = sum(r['reward'] for r in recent_records[half:]) / half
            trend = 'improving' if second_half_reward > first_half_reward else 'declining'
        else:
            trend = 'insufficient_data'

        return {
            'count': window_size,
            'avg_reward': avg_reward,
            'avg_score': avg_score,
            'trend': trend,
            'recent_grades': [r['grade'] for r in recent_records[-5:]]
        }

    # TODO: 未集成到主流程 - 考虑实现集成或在未来版本中移除
    # 功能描述: 生成完整的学习报告，包含成功/失败模式分析
    def generate_learning_report(self) -> str:
        """
        生成学习报告

        Returns:
            格式化的学习报告文本
        """
        if not self.reward_history:
            return "尚无学习数据"

        lines = [
            "=" * 60,
            "【奖励学习系统 - 学习报告】",
            "=" * 60,
            ""
        ]

        # 1. 总体统计
        total_records = len(self.reward_history)
        avg_reward = sum(r['reward'] for r in self.reward_history) / total_records
        avg_score = sum(r['score'] for r in self.reward_history) / total_records

        lines.extend([
            "总体统计:",
            f"  记录数: {total_records}",
            f"  平均奖励: {avg_reward:+.4f}",
            f"  平均评分: {avg_score:.2f}",
            ""
        ])

        # 2. 成功模式
        success_count = len(self.pattern_memory['successful'])
        if success_count > 0:
            lines.append(f"成功模式 ({success_count} 个):")
            for pattern in self.pattern_memory['successful'][-3:]:
                lines.append(
                    f"  - {pattern['action_type']}: "
                    f"评分={pattern['score']:.0f}, 奖励={pattern['reward']:+.4f}"
                )
            lines.append("")

        # 3. 失败模式
        failure_count = len(self.pattern_memory['failed'])
        if failure_count > 0:
            lines.append(f"失败模式 ({failure_count} 个):")
            for pattern in self.pattern_memory['failed'][-3:]:
                lines.append(
                    f"  - {pattern['action_type']}: "
                    f"评分={pattern['score']:.0f}, 奖励={pattern['reward']:+.4f}"
                )
            lines.append("")

        # 4. 最近趋势
        trend = self.get_reward_trend(window_size=20)
        lines.extend([
            "最近趋势 (20笔):",
            f"  平均奖励: {trend['avg_reward']:+.4f}",
            f"  平均评分: {trend['avg_score']:.2f}",
            f"  趋势: {trend['trend']}",
            f"  最近评级: {' → '.join(trend['recent_grades'])}",
            ""
        ])

        lines.append("=" * 60)

        return "\n".join(lines)

    def _find_similar_patterns(
        self,
        current_context: Dict[str, Any],
        action_type: str,
        pattern_type: str,  # successful | failed
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        查找相似的模式

        Args:
            current_context: 当前上下文
            action_type: 决策类型
            pattern_type: 模式类型
            top_k: 返回数量

        Returns:
            相似模式列表
        """
        patterns = self.pattern_memory.get(pattern_type, [])

        # 过滤相同决策类型
        filtered = [p for p in patterns if p['action_type'] == action_type]

        if not filtered:
            return []

        # 简单相似度计算（基于上下文关键字匹配）
        # 实际应用中可以用向量相似度
        similarities = []
        for pattern in filtered:
            pattern_ctx = pattern.get('decision_context', {})
            similarity = self._calculate_context_similarity(current_context, pattern_ctx)
            similarities.append((similarity, pattern))

        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in similarities[:top_k]]

    def _calculate_context_similarity(
        self,
        ctx1: Dict[str, Any],
        ctx2: Dict[str, Any]
    ) -> float:
        """
        计算上下文相似度（简单版本）

        实际应用中应该使用向量相似度
        """
        # 提取关键指标
        keys = ['rsi', 'atr', 'trend', 'volatility', 'side']

        matches = 0
        total = 0

        for key in keys:
            if key in ctx1 and key in ctx2:
                total += 1
                val1 = ctx1[key]
                val2 = ctx2[key]

                # 数值类型：计算相对差异
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val2 != 0:
                        diff = abs(val1 - val2) / abs(val2)
                        if diff < 0.2:  # 差异小于20%
                            matches += 1
                # 字符串类型：完全匹配
                elif val1 == val2:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _save_history(self):
        """持久化奖励历史"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'reward_history': list(self.reward_history),
                'pattern_memory': self.pattern_memory,
                'config': self.config,
                'saved_at': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"奖励历史已保存: {self.storage_path}")

        except Exception as e:
            logger.error(f"保存奖励历史失败: {e}")

    def _load_history(self):
        """加载奖励历史"""
        if not self.storage_path.exists():
            logger.info("奖励历史文件不存在，从头开始")
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.reward_history = deque(data.get('reward_history', []), maxlen=1000)
            self.pattern_memory = data.get('pattern_memory', {'successful': [], 'failed': []})

            logger.info(f"奖励历史已加载: {len(self.reward_history)} 条记录")

        except Exception as e:
            logger.error(f"加载奖励历史失败: {e}")

    # TODO: 未集成到主流程 - 考虑实现集成或在未来版本中移除
    # 功能描述: 清空奖励历史数据
    def clear_history(self):
        """清空历史数据"""
        self.reward_history.clear()
        self.pattern_memory = {'successful': [], 'failed': []}
        logger.info("奖励历史已清空")
