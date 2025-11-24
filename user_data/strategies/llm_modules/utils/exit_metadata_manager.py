"""
退出元数据管理器 - 缓存自动退出的技术参数

在 custom_stoploss 和 custom_exit 触发时记录技术参数，
供 confirm_trade_exit 中的 LLM 分析使用。
"""

import logging
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExitMetadataManager:
    """轻量级退出元数据缓存（仅记录技术参数）"""

    def __init__(self):
        """初始化退出元数据管理器"""
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.cache_expiry_minutes = 30  # 缓存过期时间

    def record_exit(
        self,
        pair: str,
        layer: str,
        **kwargs
    ) -> None:
        """
        记录退出元数据

        Args:
            pair: 交易对
            layer: 退出层 ("layer1" | "layer2" | "layer4")
            **kwargs: 各层的特定参数
        """
        with self._lock:
            self._cache[pair] = {
                "layer": layer,
                "timestamp": datetime.now(),
                "pair": pair,
                **kwargs
            }

            logger.debug(f"Exit metadata recorded for {pair}: {layer}")

    def get_and_clear(self, pair: str) -> Optional[Dict]:
        """
        原子操作：读取并清除缓存

        Args:
            pair: 交易对

        Returns:
            退出元数据字典，如果不存在或已过期则返回 None
        """
        with self._lock:
            if pair not in self._cache:
                return None

            metadata = self._cache.pop(pair)

            # 检查是否过期
            if self._is_expired(metadata):
                logger.warning(
                    f"Exit metadata for {pair} expired "
                    f"(age: {datetime.now() - metadata['timestamp']})"
                )
                return None

            return metadata

    def _is_expired(self, metadata: Dict) -> bool:
        """检查元数据是否过期"""
        age = datetime.now() - metadata['timestamp']
        return age > timedelta(minutes=self.cache_expiry_minutes)

    def clear_expired(self) -> int:
        """
        清理过期的缓存条目

        Returns:
            清理的条目数
        """
        with self._lock:
            expired_pairs = [
                pair for pair, metadata in self._cache.items()
                if self._is_expired(metadata)
            ]

            for pair in expired_pairs:
                del self._cache[pair]

            if expired_pairs:
                logger.info(f"Cleared {len(expired_pairs)} expired cache entries")

            return len(expired_pairs)

    def get_cache_size(self) -> int:
        """获取当前缓存大小"""
        with self._lock:
            return len(self._cache)

    def clear_all(self) -> None:
        """清空所有缓存（用于测试或重置）"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared all {count} cache entries")
