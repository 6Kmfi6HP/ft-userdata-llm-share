"""
RAG 检索管理器
整合向量存储、嵌入服务、交易评价和奖励学习
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .vector_store import VectorStore
from .embedding_service import EmbeddingService
from .trade_evaluator import TradeEvaluator
from .reward_learning import RewardLearningSystem

logger = logging.getLogger(__name__)


class RAGManager:
    """RAG 检索管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 RAG 管理器

        Args:
            config: 配置字典
                {
                    "embedding": {...},  # 嵌入服务配置
                    "vector_store": {...},  # 向量存储配置
                    "reward_learning": {...}  # 奖励学习配置
                }
        """
        self.config = config

        # 1. 初始化嵌入服务
        embedding_config = config.get('embedding', {})
        self.embedding_service = EmbeddingService(
            model_name=embedding_config.get('model_name', 'text-embedding-bge-m3'),
            api_url=embedding_config.get('api_url', 'http://localhost:11434'),
            api_type=embedding_config.get('api_type', 'ollama'),
            dimension=embedding_config.get('dimension', 1024)
        )

        # 2. 初始化向量存储
        vector_config = config.get('vector_store', {})
        self.vector_store = VectorStore(
            dimension=self.embedding_service.dimension,
            index_type=vector_config.get('index_type', 'flat'),
            storage_path=vector_config.get('storage_path', './user_data/rag/vector_store')
        )

        # 3. 初始化交易评价器
        self.trade_evaluator = TradeEvaluator()

        # 4. 初始化奖励学习系统
        reward_config = config.get('reward_learning', {})
        self.reward_learning = RewardLearningSystem(reward_config)

        logger.info("RAG 管理器已初始化")
        logger.info(f"  向量维度: {self.embedding_service.dimension}")
        logger.info(f"  存储记录: {self.vector_store.get_statistics()['total_vectors']}")

    def store_trade_experience(
        self,
        trade_id: int,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        profit_pct: float,
        leverage: float,
        stake_amount: float,
        entry_reason: str,
        exit_reason: str,
        position_metrics: Optional[Dict[str, Any]] = None,
        market_changes: Optional[Dict[str, Any]] = None,
        model_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        存储交易经验到 RAG 系统

        Args:
            (所有交易相关参数)

        Returns:
            评价结果
        """
        try:
            # 1. 评估交易质量（优先使用模型自评分）
            evaluation = self.trade_evaluator.evaluate_trade(
                trade_id=trade_id,
                pair=pair,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                profit_pct=profit_pct,
                leverage=leverage,
                stake_amount=stake_amount,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                position_metrics=position_metrics,
                market_changes=market_changes,
                model_score=model_score
            )

            # 2. 构建交易描述文本
            trade_text = self._build_trade_text(
                pair=pair,
                side=side,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                profit_pct=profit_pct,
                evaluation=evaluation,
                position_metrics=position_metrics
            )

            # 3. 生成嵌入向量
            embedding = self.embedding_service.embed(trade_text)

            # 4. 构建元数据
            metadata = {
                'trade_id': trade_id,
                'pair': pair,
                'side': side,
                'entry_time': entry_time.isoformat(),
                'exit_time': exit_time.isoformat(),
                'profit_pct': profit_pct,
                'leverage': leverage,
                'entry_reason': entry_reason,
                'exit_reason': exit_reason,
                'evaluation': evaluation,
                'position_metrics': position_metrics,
                'market_changes': market_changes,
                'trade_text': trade_text,
                'stored_at': datetime.now().isoformat()
            }

            # 5. 存储到向量库
            vector_id = self.vector_store.add(embedding, metadata)

            logger.info(
                f"交易经验已存储: {pair} | "
                f"盈亏={profit_pct:+.2f}% | 评分={evaluation['total_score']:.0f} | "
                f"向量ID={vector_id}"
            )

            # 6. 定期持久化
            if vector_id % 5 == 0:
                self.vector_store.save()

            return evaluation

        except Exception as e:
            logger.error(f"存储交易经验失败: {e}", exc_info=True)
            return {}

    def retrieve_similar_trades(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相似的交易经验

        Args:
            query_text: 查询文本（描述当前市场情况或决策意图）
            top_k: 返回最相似的K个结果
            filters: 过滤条件（如: {"pair": "BTC/USDT", "side": "long"}）

        Returns:
            相似交易列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.embed(query_text)

            # 检索相似向量
            results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            logger.debug(f"检索到 {len(results)} 个相似交易")

            return results

        except Exception as e:
            logger.error(f"检索相似交易失败: {e}")
            return []

    def get_contextual_advice(
        self,
        current_situation: str,
        action_type: str,  # entry | exit | hold
        pair: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        根据当前情况获取上下文建议

        Args:
            current_situation: 当前市场情况描述
            action_type: 决策类型
            pair: 交易对（可选）
            top_k: 检索数量

        Returns:
            建议字典
                {
                    "similar_trades": [...],
                    "advice": "...",
                    "warnings": [...],
                    "success_rate": 0.6
                }
        """
        # 构建查询文本
        query_text = f"{action_type}: {current_situation}"
        if pair:
            query_text = f"{pair} - {query_text}"

        # 检索相似交易
        similar_trades = self.retrieve_similar_trades(
            query_text=query_text,
            top_k=top_k * 2  # 多检索一些以便筛选
        )

        if not similar_trades:
            return {
                'similar_trades': [],
                'advice': '暂无历史经验可参考',
                'warnings': [],
                'success_rate': 0.5
            }

        # 分析相似交易
        successful = [t for t in similar_trades if t['metadata'].get('profit_pct', 0) > 0]
        failed = [t for t in similar_trades if t['metadata'].get('profit_pct', 0) <= 0]

        success_rate = len(successful) / len(similar_trades)

        # 生成建议
        advice_parts = []
        warnings = []

        if success_rate > 0.6:
            advice_parts.append(f"历史相似场景成功率较高 ({success_rate:.0%})")
            # 提取成功要素
            for trade in successful[:2]:
                eval_data = trade['metadata'].get('evaluation', {})
                comments = eval_data.get('comments', {})
                if comments.get('strengths'):
                    advice_parts.append(f"  成功要素: {comments['strengths'][0]}")

        elif success_rate < 0.4:
            warnings.append(f"警告: 历史相似场景成功率较低 ({success_rate:.0%})")
            # 提取失败教训
            for trade in failed[:2]:
                eval_data = trade['metadata'].get('evaluation', {})
                comments = eval_data.get('comments', {})
                if comments.get('weaknesses'):
                    warnings.append(f"  历史失败: {comments['weaknesses'][0]}")

        else:
            advice_parts.append(f"历史相似场景结果参半 ({success_rate:.0%})，需谨慎决策")

        return {
            'similar_trades': similar_trades[:top_k],
            'advice': '\n'.join(advice_parts) if advice_parts else '无明确建议',
            'warnings': warnings,
            'success_rate': success_rate
        }

    def format_similar_trades_context(
        self,
        similar_trades: List[Dict[str, Any]],
        max_length: int = 1000
    ) -> str:
        """
        格式化相似交易为上下文文本（用于LLM prompt）

        Args:
            similar_trades: 相似交易列表
            max_length: 最大文本长度

        Returns:
            格式化的文本
        """
        if not similar_trades:
            return ""

        lines = ["【相似历史交易】", ""]

        for i, trade_data in enumerate(similar_trades[:5], 1):
            meta = trade_data['metadata']
            score = trade_data.get('score', 0)

            profit = meta.get('profit_pct', 0)
            evaluation = meta.get('evaluation', {})
            grade = evaluation.get('grade', 'C')

            # 简洁格式
            lines.append(
                f"{i}. {meta.get('pair')} {meta.get('side')} | "
                f"盈亏={profit:+.2f}% | 评级={grade} | 相似度={score:.2f}"
            )

            # 添加关键经验
            comments = evaluation.get('comments', {})
            if comments.get('warnings'):
                lines.append(f"   警告: {comments['warnings'][0]}")
            if comments.get('suggestions'):
                lines.append(f"   建议: {comments['suggestions'][0]}")

            lines.append("")

        text = '\n'.join(lines)

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    def _build_trade_text(
        self,
        pair: str,
        side: str,
        entry_reason: str,
        exit_reason: str,
        profit_pct: float,
        evaluation: Dict[str, Any],
        position_metrics: Optional[Dict[str, Any]]
    ) -> str:
        """
        构建交易描述文本（用于嵌入）

        这个文本会被向量化，所以要包含所有关键信息
        """
        parts = [
            f"交易对: {pair}",
            f"方向: {side}",
            f"入场理由: {entry_reason}",
            f"出场理由: {exit_reason}",
            f"盈亏: {profit_pct:+.2f}%",
            f"评级: {evaluation.get('grade', 'C')}",
            f"评分: {evaluation.get('total_score', 50)}"
        ]

        # 添加评价评语
        comments = evaluation.get('comments', {})
        if comments.get('strengths'):
            parts.append(f"优势: {', '.join(comments['strengths'])}")
        if comments.get('weaknesses'):
            parts.append(f"弱点: {', '.join(comments['weaknesses'])}")
        if comments.get('suggestions'):
            parts.append(f"建议: {', '.join(comments['suggestions'][:2])}")

        # 添加持仓指标
        if position_metrics:
            mfe = position_metrics.get('max_profit_pct', 0)
            mae = position_metrics.get('max_loss_pct', 0)
            parts.append(f"最大浮盈: {mfe:+.2f}%")
            parts.append(f"最大浮亏: {mae:+.2f}%")

        return " | ".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """获取 RAG 系统统计信息"""
        vector_stats = self.vector_store.get_statistics()
        reward_trend = self.reward_learning.get_reward_trend(window_size=20)

        return {
            'vector_store': vector_stats,
            'reward_trend': reward_trend,
            'embedding_service': self.embedding_service.get_model_info()
        }

    def save_all(self):
        """持久化所有数据"""
        try:
            self.vector_store.save()
            self.reward_learning._save_history()
            logger.info("RAG 系统数据已全部保存")
        except Exception as e:
            logger.error(f"保存 RAG 数据失败: {e}")
