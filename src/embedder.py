"""
向量嵌入模块：使用sentence-transformers加载bge-small-zh模型
"""
import os
import logging
from typing import List

# 解决Windows环境下HuggingFace SSL证书问题
# 使用国内镜像站绕过CA证书验证
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用SSL验证（Windows开发环境证书链不完整）
os.environ.setdefault("CURL_CA_BUNDLE", "")

import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import httpx
# 全局禁用httpx SSL验证
original_init = httpx.Client.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_init

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 全局模型实例（避免重复加载）
_model = None


def get_model(model_name: str = "BAAI/bge-small-zh-v1.5") -> SentenceTransformer:
    """获取嵌入模型实例（单例模式）"""
    global _model
    if _model is None:
        logger.info(f"正在加载嵌入模型: {model_name}")
        _model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"模型加载完成，维度: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str], model_name: str = "BAAI/bge-small-zh-v1.5") -> List[List[float]]:
    """将文本列表转换为向量"""
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str = "BAAI/bge-small-zh-v1.5") -> List[float]:
    """将查询文本转换为向量"""
    model = get_model(model_name)
    embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return embedding[0].tolist()
