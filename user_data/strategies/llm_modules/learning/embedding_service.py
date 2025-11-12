"""
嵌入模型服务
支持 text-embedding-bge-m3 和其他嵌入模型
"""
import numpy as np
from typing import List, Union, Optional
import logging
import requests
import json

logger = logging.getLogger(__name__)


class EmbeddingService:
    """文本嵌入服务"""

    def __init__(
        self,
        model_name: str = "text-embedding-bge-m3",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_type: str = "openai",  # openai | ollama | local
        dimension: int = 1024,
        batch_size: int = 8
    ):
        """
        初始化嵌入服务

        Args:
            model_name: 模型名称
            api_url: API地址
            api_key: API密钥（OpenAI格式需要）
            api_type: API类型 (openai, ollama, local)
            dimension: 向量维度
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.api_url = api_url or "http://localhost:11434"  # 默认 Ollama 地址
        self.api_key = api_key
        self.api_type = api_type
        self.dimension = dimension
        self.batch_size = batch_size

        # 尝试初始化
        self._validate_connection()

        logger.info(f"嵌入服务已初始化: {model_name} ({api_type})")

    def _validate_connection(self):
        """验证连接"""
        try:
            if self.api_type == "ollama":
                # 测试 Ollama 连接
                response = requests.get(f"{self.api_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    logger.info("Ollama 连接成功")
                else:
                    logger.warning(f"Ollama 连接异常: {response.status_code}")
            elif self.api_type == "openai":
                logger.info("使用 OpenAI 兼容 API")
            else:
                logger.info("使用本地模型")

        except Exception as e:
            logger.warning(f"无法连接到嵌入服务: {e}")

    def embed(self, text: str) -> np.ndarray:
        """
        生成单个文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量 (shape: [dimension])
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量生成嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量数组 (shape: [batch_size, dimension])
        """
        if not texts:
            return np.array([])

        try:
            if self.api_type == "ollama":
                return self._embed_ollama(texts)
            elif self.api_type == "openai":
                return self._embed_openai(texts)
            else:
                # 本地模型（需要额外实现）
                return self._embed_local(texts)

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            # 返回零向量作为降级方案（避免污染向量库）
            logger.warning(f"使用零向量作为降级方案，共{len(texts)}个文本")
            return np.zeros((len(texts), self.dimension), dtype='float32')

    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """
        使用 Ollama API 生成嵌入

        Args:
            texts: 文本列表

        Returns:
            嵌入向量数组
        """
        embeddings = []

        for text in texts:
            try:
                response = requests.post(
                    f"{self.api_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = np.array(data['embedding'], dtype='float32')
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Ollama API 错误: {response.status_code}")
                    embeddings.append(np.zeros(self.dimension, dtype='float32'))

            except Exception as e:
                logger.error(f"Ollama 嵌入失败: {e}")
                embeddings.append(np.zeros(self.dimension, dtype='float32'))

        return np.array(embeddings)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """
        使用 OpenAI 兼容 API 生成嵌入

        Args:
            texts: 文本列表

        Returns:
            嵌入向量数组
        """
        try:
            # 使用标准 OpenAI API 格式（通过 requests）
            headers = {
                'Content-Type': 'application/json'
            }

            # 如果有 API Key，添加到 header
            if hasattr(self, 'api_key') and self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            embeddings = []

            # 批量处理（每次最多8个文本）
            batch_size = min(self.batch_size, 8)
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                payload = {
                    "model": self.model_name,
                    "input": batch_texts
                }

                # 尝试标准OpenAI路径 /v1/embeddings 和简化路径 /embeddings
                api_paths = ["/v1/embeddings", "/embeddings"]
                success = False

                for api_path in api_paths:
                    try:
                        # 确保api_url不以/结尾
                        base_url = self.api_url.rstrip('/')
                        response = requests.post(
                            f"{base_url}{api_path}",
                            json=payload,
                            headers=headers,
                            timeout=60
                        )

                        if response.status_code == 200:
                            try:
                                data = response.json()

                                # 标准 OpenAI 返回格式
                                for item in data.get('data', []):
                                    embedding = np.array(item['embedding'], dtype='float32')
                                    embeddings.append(embedding)
                                success = True
                                break
                            except json.JSONDecodeError as je:
                                logger.warning(f"OpenAI API 返回非JSON响应 ({api_path}): {response.text[:200]}")
                                continue
                        else:
                            logger.debug(f"尝试路径 {api_path} 失败: {response.status_code}")
                            continue
                    except Exception as e:
                        logger.debug(f"尝试路径 {api_path} 异常: {e}")
                        continue

                if not success:
                    logger.warning(f"OpenAI API 所有路径失败，使用零向量降级")
                    # 添加零向量作为降级
                    for _ in batch_texts:
                        embeddings.append(np.zeros(self.dimension, dtype='float32'))

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"OpenAI 嵌入失败: {e}")
            # 返回零向量作为降级方案（避免污染向量库）
            logger.warning(f"使用零向量作为降级方案，共{len(texts)}个文本")
            return np.zeros((len(texts), self.dimension), dtype='float32')

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """
        使用本地模型生成嵌入（需要安装 sentence-transformers）

        Args:
            texts: 文本列表

        Returns:
            嵌入向量数组
        """
        try:
            from sentence_transformers import SentenceTransformer

            # 懒加载模型
            if not hasattr(self, '_local_model'):
                self._local_model = SentenceTransformer(self.model_name)
                logger.info(f"本地模型已加载: {self.model_name}")

            embeddings = self._local_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            return embeddings.astype('float32')

        except ImportError:
            logger.error("sentence-transformers 未安装，无法使用本地模型")
            logger.error("请运行: pip install sentence-transformers")
            return np.random.randn(len(texts), self.dimension).astype('float32')
        except Exception as e:
            logger.error(f"本地嵌入失败: {e}")
            return np.random.randn(len(texts), self.dimension).astype('float32')

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        embeddings = self.embed_batch([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]

        # 余弦相似度
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
        )

        return float(similarity)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'api_type': self.api_type,
            'api_url': self.api_url,
            'dimension': self.dimension,
            'batch_size': self.batch_size
        }


# 工厂函数
def create_embedding_service(config: dict) -> EmbeddingService:
    """
    根据配置创建嵌入服务

    Args:
        config: 配置字典
            {
                "model_name": "text-embedding-bge-m3",
                "api_url": "http://localhost:11434",
                "api_key": "sk-xxx",
                "api_type": "ollama",
                "dimension": 1024
            }

    Returns:
        EmbeddingService 实例
    """
    return EmbeddingService(
        model_name=config.get("model_name", "text-embedding-bge-m3"),
        api_url=config.get("api_url", "http://localhost:11434"),
        api_key=config.get("api_key"),
        api_type=config.get("api_type", "ollama"),
        dimension=config.get("dimension", 1024),
        batch_size=config.get("batch_size", 8)
    )
