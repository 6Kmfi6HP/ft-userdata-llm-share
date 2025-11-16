"""
持仓追踪模块 - 追踪MFE/MAE/回撤/hold次数
提供完整的持仓演变信息，让LLM自己判断而非硬编码限制
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class PositionTracker:
    """持仓状态追踪器 - 纯信息收集，无行为限制"""

    def __init__(self):
        # 存储结构: {trade_id: PositionMetrics}
        self.positions: Dict[int, Dict[str, Any]] = {}

    def update_position(
        self,
        trade_id: int,
        pair: str,
        current_price: float,
        open_price: float,
        is_short: bool,
        leverage: float,
        decision_type: str,  # 'hold', 'exit', 'adjust', 'check'
        decision_reason: str,
        open_date: datetime = None  # 真实开仓时间
    ) -> Dict[str, Any]:
        """
        更新持仓统计

        Args:
            trade_id: 交易ID
            pair: 交易对
            current_price: 当前价格
            open_price: 开仓价格
            is_short: 是否做空
            leverage: 杠杆倍数
            decision_type: 决策类型
            decision_reason: 决策理由

        Returns:
            持仓指标字典
        """

        # 计算当前盈亏百分比
        if is_short:
            profit_pct = (open_price - current_price) / open_price * leverage * 100
        else:
            profit_pct = (current_price - open_price) / open_price * leverage * 100

        # 初始化或获取现有记录
        if trade_id not in self.positions:
            # 使用传入的真实开仓时间,如果没有则使用UTC时间(向后兼容)
            actual_open_time = open_date if open_date is not None else datetime.now(timezone.utc)
            
            self.positions[trade_id] = {
                'pair': pair,
                'open_price': open_price,
                'open_time': actual_open_time,  # 使用真实开仓时间(UTC)
                'is_short': is_short,
                'leverage': leverage,
                'max_profit_pct': profit_pct,  # MFE (Maximum Favorable Excursion)
                'max_loss_pct': profit_pct,    # MAE (Maximum Adverse Excursion)
                'hold_count': 0,
                'hold_reasons': [],
                'decision_history': [],
                'profit_peak': profit_pct,
                'price_snapshots': [{'time': datetime.now(timezone.utc), 'price': current_price, 'profit_pct': profit_pct}]
            }
            logger.info(f"开始追踪持仓 #{trade_id}: {pair} (开仓时间: {actual_open_time})")

        pos = self.positions[trade_id]

        # 更新MFE/MAE
        pos['max_profit_pct'] = max(pos['max_profit_pct'], profit_pct)
        pos['max_loss_pct'] = min(pos['max_loss_pct'], profit_pct)

        # 更新盈利峰值
        if profit_pct > pos['profit_peak']:
            pos['profit_peak'] = profit_pct

        # 记录决策
        if decision_type == 'hold':
            pos['hold_count'] += 1
            pos['hold_reasons'].append({
                'time': datetime.now(timezone.utc),
                'reason': decision_reason,
                'profit_at_hold': profit_pct
            })

        pos['decision_history'].append({
            'time': datetime.now(timezone.utc),
            'type': decision_type,
            'price': current_price,
            'profit_pct': profit_pct,
            'reason': decision_reason[:200]  # 限制长度避免存储过大
        })

        # 添加价格快照
        pos['price_snapshots'].append({
            'time': datetime.now(timezone.utc),
            'price': current_price,
            'profit_pct': profit_pct
        })

        # 只保留最近50个快照
        if len(pos['price_snapshots']) > 50:
            pos['price_snapshots'] = pos['price_snapshots'][-50:]

        # 计算从峰值的回撤
        drawdown_from_peak = profit_pct - pos['profit_peak']

        # 分析hold决策模式
        hold_pattern = self._analyze_hold_pattern(pos['hold_reasons'])

        # 持仓时长(使用UTC时间,处理naive datetime)
        current_utc = datetime.now(timezone.utc)
        open_time = pos['open_time']
        # 如果open_time是naive datetime,假定为UTC
        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        time_in_position_hours = (current_utc - open_time).total_seconds() / 3600

        # 计算DD和dd_ratio (统一公式: DD = MFE - CP, dd_ratio = DD / MFE)
        dd = pos['max_profit_pct'] - profit_pct  # DD = MFE - CP
        dd_ratio = dd / max(pos['max_profit_pct'], 1e-6) if pos['max_profit_pct'] > 0 else 0.0
        
        # 持久化关键指标到positions（H5修复）
        pos['current_profit_pct'] = profit_pct
        pos['dd'] = dd
        pos['dd_ratio'] = dd_ratio
        pos['drawdown_from_peak_pct'] = drawdown_from_peak
        pos['time_in_position_hours'] = time_in_position_hours
        pos['recent_decisions'] = pos['decision_history'][-5:]  # ✅ Y2修复: 持久化最近5次决策,供get_position_metrics使用
        pos['hold_pattern'] = hold_pattern  # ✅ Z1修复: 持久化hold_pattern,供stuck_in_loop检测使用

        # 返回序列化后的数据（确保JSON兼容）
        result = {
            'trade_id': trade_id,
            'max_profit_pct': pos['max_profit_pct'],
            'max_loss_pct': pos['max_loss_pct'],
            'current_profit_pct': profit_pct,
            'drawdown_from_peak_pct': drawdown_from_peak,
            'dd': dd,  # DD = MFE - CP
            'dd_ratio': dd_ratio,  # dd_ratio = DD / MFE
            'hold_count': pos['hold_count'],
            'hold_pattern': hold_pattern,
            'time_in_position_hours': time_in_position_hours,
            'decision_count': len(pos['decision_history']),
            'recent_decisions': pos['recent_decisions'],  # 使用已持久化的数据
        }
        
        # 序列化整个结果（处理datetime等类型）
        return self._serialize_value(result)

    def _analyze_hold_pattern(self, hold_reasons: List[Dict]) -> Dict[str, Any]:
        """
        分析hold决策的模式

        检测:
        1. 是否重复使用相同理由
        2. 重复频率
        3. 是否陷入决策循环
        """
        if not hold_reasons:
            return {'pattern': 'no_holds', 'repeated_reason': None, 'repeat_count': 0}

        # 统计最常见的理由 (截取前50字符作为key)
        reason_counts = {}
        for h in hold_reasons:
            reason_key = h['reason'][:50]
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

        most_common_reason = max(reason_counts, key=reason_counts.get)
        repeat_count = reason_counts[most_common_reason]

        # 判断是否存在重复模式
        if repeat_count >= 3 and len(hold_reasons) >= 3:
            # 检查最近3次是否都是相同理由
            recent_3_reasons = [h['reason'][:50] for h in hold_reasons[-3:]]
            if all(r == most_common_reason for r in recent_3_reasons):
                pattern = 'stuck_in_loop'  # 陷入循环 - 连续3次相同理由
            else:
                pattern = 'repeated_reasoning'  # 重复推理 - 总体有重复但不连续
        else:
            pattern = 'normal'

        return {
            'pattern': pattern,
            'repeated_reason': most_common_reason,
            'repeat_count': repeat_count,
            'total_holds': len(hold_reasons),
            'diversity_score': len(reason_counts) / len(hold_reasons) if hold_reasons else 1.0
        }

    def _serialize_value(self, obj):
        """递归序列化对象为JSON兼容类型"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy数组
            return obj.tolist()
        elif hasattr(obj, 'item') and hasattr(obj, 'ndim') and obj.ndim == 0:  # numpy标量
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_value(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 未知类型，尝试转换为字符串
            return str(obj)
    
    def get_position_metrics(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """获取持仓指标（返回 JSON 可序列化的数据）"""
        pos = self.positions.get(trade_id)
        if not pos:
            return None
        
        # 使用递归序列化所有数据
        return self._serialize_value(pos)

    def get_all_positions(self) -> Dict[int, Dict[str, Any]]:
        """获取所有持仓"""
        return self.positions.copy()

    def remove_position(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        移除持仓记录并返回最终统计

        用于交易完成后的复盘
        """
        if trade_id in self.positions:
            final_metrics = self.positions[trade_id].copy()
            del self.positions[trade_id]
            logger.info(f"停止追踪持仓 #{trade_id}")
            return final_metrics
        return None

    def generate_position_summary(self, trade_id: int) -> str:
        """
        生成持仓摘要文本

        用于添加到LLM的context中
        """
        pos = self.positions.get(trade_id)
        if not pos:
            return "无持仓追踪数据"

        # 持仓时长(处理naive datetime)
        current_utc = datetime.now(timezone.utc)
        open_time = pos['open_time']
        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        time_hours = (current_utc - open_time).total_seconds() / 3600

        # 计算当前盈亏
        if pos['decision_history']:
            current_profit = pos['decision_history'][-1]['profit_pct']
        else:
            current_profit = 0

        drawdown = current_profit - pos['profit_peak']

        # 计算dd_ratio
        dd_ratio = (pos['max_profit_pct'] - current_profit) / max(pos['max_profit_pct'], 1e-6) if pos['max_profit_pct'] > 0 else 0.0

        hold_pattern = self._analyze_hold_pattern(pos['hold_reasons'])

        summary_parts = [
            "【持仓追踪数据】",
            f"  最大浮盈(MFE): {pos['max_profit_pct']:+.2f}%",
            f"  最大浮亏(MAE): {pos['max_loss_pct']:+.2f}%",
            f"  当前盈亏: {current_profit:+.2f}%",
            f"  盈利回撤: {drawdown:+.2f}% (从峰值{pos['profit_peak']:+.2f}%)",
            f"  dd_ratio(回撤比): {dd_ratio:.4f} = ({pos['max_profit_pct']:.2f}% - {current_profit:.2f}%) / {pos['max_profit_pct']:.2f}%",
            f"  持仓时长: {time_hours:.1f}小时",
            f"  hold决策次数: {pos['hold_count']}次"
        ]

        # hold模式分析
        if hold_pattern['pattern'] == 'stuck_in_loop':
            summary_parts.append(
                f"  [注意] 连续{hold_pattern['repeat_count']}次使用相似理由hold"
            )
            summary_parts.append(
                f"  重复理由: \"{hold_pattern['repeated_reason']}\""
            )
            summary_parts.append(
                f"  提示: 重复决策可能是确认偏差，考虑寻找反面证据"
            )
        elif hold_pattern['pattern'] == 'repeated_reasoning':
            summary_parts.append(
                f"  提示: 检测到理由重复({hold_pattern['repeat_count']}次)，多样性分数{hold_pattern['diversity_score']:.2f}"
            )

        # 最近决策历史（完整显示）
        if pos['decision_history']:
            summary_parts.append("  最近3次决策:")
            for d in pos['decision_history'][-3:]:
                time_str = d['time'].strftime("%H:%M")
                summary_parts.append(
                    f"    [{time_str}] {d['type']}: {d['reason']} (盈亏{d['profit_pct']:+.2f}%)"
                )

        return "\n".join(summary_parts)
