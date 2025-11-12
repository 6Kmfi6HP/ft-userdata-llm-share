"""
向量存储模块
使用 FAISS 存储和检索交易经验的向量表示
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，将使用简单的余弦相似度检索（性能较低）")


class VectorStore:
    """向量存储和检索系统"""

    def __init__(
        self,
        dimension: int = 1024,  # bge-m3 默认维度
        index_type: str = "flat",  # flat | hnsw | ivf
        storage_path: Optional[str] = None
    ):
        """
        初始化向量存储

        Args:
            dimension: 向量维度
            index_type: 索引类型 (flat=精确搜索, hnsw=快速近似, ivf=大规模)
            storage_path: 持久化存储路径
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None

        # 初始化索引
        if FAISS_AVAILABLE:
            self.index = self._create_faiss_index(index_type, dimension)
            self.use_faiss = True
            logger.info(f"使用 FAISS 索引 ({index_type}), 维度: {dimension}")
        else:
            self.index = None
            self.vectors = []  # 简单的列表存储
            self.use_faiss = False
            logger.info(f"使用简单向量存储, 维度: {dimension}")

        # 元数据存储 (trade info, timestamp, profit, etc.)
        self.metadata: List[Dict[str, Any]] = []
        self.id_counter = 0

        # 如果存储路径存在，尝试加载
        if self.storage_path and self.storage_path.exists():
            self.load()

    def _create_faiss_index(self, index_type: str, dimension: int):
        """创建 FAISS 索引"""
        if index_type == "flat":
            # 精确搜索 (适合小数据集 < 10k)
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "hnsw":
            # HNSW 近似搜索 (适合中等数据集 10k-1M)
            index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        elif index_type == "ivf":
            # IVF 索引 (适合大规模数据集 > 1M)
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # 需要训练，但先创建空索引
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

        return index

    def add(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> int:
        """
        添加向量和元数据

        Args:
            vector: 向量 (shape: [dimension])
            metadata: 元数据字典

        Returns:
            向量ID
        """
        # 验证维度
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimension}, 实际 {vector.shape[0]}"
            )

        # 标准化向量 (L2归一化)
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        # 添加到索引
        vector_2d = vector.reshape(1, -1).astype('float32')

        if self.use_faiss:
            self.index.add(vector_2d)
        else:
            self.vectors.append(vector)

        # 存储元数据
        vector_id = self.id_counter
        metadata['vector_id'] = vector_id
        self.metadata.append(metadata)
        self.id_counter += 1

        return vector_id

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        批量添加向量

        Args:
            vectors: 向量数组 (shape: [batch_size, dimension])
            metadata_list: 元数据列表

        Returns:
            向量ID列表
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("向量数量和元数据数量不匹配")

        # 标准化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors = (vectors / norms).astype('float32')

        # 添加到索引
        if self.use_faiss:
            self.index.add(vectors)
        else:
            for vec in vectors:
                self.vectors.append(vec)

        # 存储元数据
        ids = []
        for meta in metadata_list:
            vector_id = self.id_counter
            meta['vector_id'] = vector_id
            self.metadata.append(meta)
            ids.append(vector_id)
            self.id_counter += 1

        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的K个结果
            filters: 元数据过滤条件 (例如: {"pair": "BTC/USDT"})

        Returns:
            相似结果列表，每个包含 {metadata, score}
        """
        if len(self.metadata) == 0:
            return []

        # 标准化查询向量
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_vector = query_vector.reshape(1, -1).astype('float32')

        # 搜索
        if self.use_faiss:
            # FAISS 搜索
            distances, indices = self.index.search(query_vector, min(top_k * 2, len(self.metadata)))
            distances = distances[0]
            indices = indices[0]

            # 转换距离为相似度 (L2距离 -> 余弦相似度近似)
            similarities = 1 / (1 + distances)
        else:
            # 简单余弦相似度
            similarities = []
            for vec in self.vectors:
                sim = np.dot(query_vector[0], vec)
                similarities.append(sim)

            similarities = np.array(similarities)
            indices = np.argsort(similarities)[::-1][:top_k * 2]
            similarities = similarities[indices]

        # 构建结果
        results = []
        for idx, score in zip(indices, similarities):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # 应用过滤器
            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            results.append({
                'metadata': meta,
                'score': float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        仅根据元数据筛选（不使用向量相似度）

        Args:
            filters: 元数据过滤条件
            top_k: 最多返回数量

        Returns:
            匹配的元数据列表
        """
        results = []
        for meta in self.metadata:
            match = all(meta.get(k) == v for k, v in filters.items())
            if match:
                results.append(meta)

            if len(results) >= top_k:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            'total_vectors': len(self.metadata),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'use_faiss': self.use_faiss,
            'storage_path': str(self.storage_path) if self.storage_path else None
        }

    def save(self, path: Optional[str] = None):
        """
        持久化存储

        Args:
            path: 存储路径（如果未指定则使用初始化时的路径）
        """
        save_path = Path(path) if path else self.storage_path
        if not save_path:
            logger.warning("未指定存储路径，跳过持久化")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 保存 FAISS 索引
            if self.use_faiss and FAISS_AVAILABLE:
                index_path = save_path.with_suffix('.faiss')
                faiss.write_index(self.index, str(index_path))
            else:
                # 保存简单向量列表
                vectors_path = save_path.with_suffix('.vectors.pkl')
                with open(vectors_path, 'wb') as f:
                    pickle.dump(self.vectors, f)

            # 保存元数据
            meta_path = save_path.with_suffix('.meta.pkl')
            with open(meta_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_counter': self.id_counter,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)

            logger.info(f"向量存储已保存: {save_path}")

        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")

    def load(self, path: Optional[str] = None):
        """
        加载持久化数据

        Args:
            path: 存储路径（如果未指定则使用初始化时的路径）
        """
        load_path = Path(path) if path else self.storage_path
        if not load_path:
            logger.warning("未指定存储路径，跳过加载")
            return

        try:
            # 加载元数据
            meta_path = load_path.with_suffix('.meta.pkl')
            if not meta_path.exists():
                logger.info("元数据文件不存在，跳过加载")
                return

            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_counter = data['id_counter']
                self.dimension = data['dimension']
                self.index_type = data['index_type']

            # 加载索引/向量
            if self.use_faiss and FAISS_AVAILABLE:
                index_path = load_path.with_suffix('.faiss')
                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))
                    logger.info(f"FAISS 索引已加载: {index_path}")
            else:
                vectors_path = load_path.with_suffix('.vectors.pkl')
                if vectors_path.exists():
                    with open(vectors_path, 'rb') as f:
                        self.vectors = pickle.load(f)
                    logger.info(f"向量列表已加载: {vectors_path}")

            logger.info(f"向量存储已加载: {len(self.metadata)} 条记录")

        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")

    def clear(self):
        """清空所有数据"""
        if self.use_faiss:
            self.index = self._create_faiss_index(self.index_type, self.dimension)
        else:
            self.vectors = []

        self.metadata = []
        self.id_counter = 0
        logger.info("向量存储已清空")
