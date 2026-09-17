"""
向量存储与检索模块：使用ChromaDB
"""
import os
import logging
from typing import List, Dict

import chromadb

logger = logging.getLogger(__name__)

# 全局Chroma客户端
_client = None
_collection = None


def get_collection(db_path: str = "./chroma_db", collection_name: str = "mini_rag"):
    """获取ChromaDB集合（单例模式）"""
    global _client, _collection
    if _client is None:
        logger.info(f"初始化ChromaDB: {db_path}")
        _client = chromadb.PersistentClient(path=db_path)
        _collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB集合就绪: {collection_name}, 当前文档数: {_collection.count()}")
    return _collection


def add_documents(chunks: List[Dict], filename: str, 
                  db_path: str = "./chroma_db", 
                  collection_name: str = "mini_rag"):
    """
    将分块文本嵌入并存入向量库
    chunks: [{"chunk": text, "index": int}, ...]
    """
    from src.embedder import embed_texts
    
    collection = get_collection(db_path, collection_name)
    
    texts = [c["chunk"] for c in chunks]
    embeddings = embed_texts(texts)
    
    # 生成唯一ID：文件名_序号
    ids = [f"{filename}_{c['index']}" for c in chunks]
    
    # 元数据：来源文件名、块序号
    metadatas = [{"filename": filename, "chunk_index": c["index"]} for c in chunks]
    
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    logger.info(f"已入库: {filename}, {len(chunks)}个块")


def search(query: str, top_k: int = 5,
           db_path: str = "./chroma_db",
           collection_name: str = "mini_rag") -> List[Dict]:
    """
    语义检索：返回Top-K相关文本片段
    返回: [{"text": chunk, "filename": 来源文件, "distance": 相似度}, ...]
    """
    from src.embedder import embed_query
    
    collection = get_collection(db_path, collection_name)
    
    if collection.count() == 0:
        return []
    
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 整理结果
    search_results = []
    for i in range(len(results["documents"][0])):
        search_results.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i].get("filename", "未知"),
            "distance": results["distances"][0][i]
        })
    
    logger.info(f"检索完成: query='{query[:30]}...', 召回{len(search_results)}条结果")
    return search_results


def clear_collection(db_path: str = "./chroma_db", collection_name: str = "mini_rag"):
    """清空向量库"""
    global _client, _collection
    if _client is not None:
        _client.delete_collection(collection_name)
        _collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("向量库已清空")
    else:
        get_collection(db_path, collection_name)
        _client.delete_collection(collection_name)
        _collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("向量库已清空")


def list_documents(db_path: str = "./chroma_db", 
                   collection_name: str = "mini_rag") -> List[str]:
    """列出向量库中所有文档文件名"""
    collection = get_collection(db_path, collection_name)
    if collection.count() == 0:
        return []
    
    all_data = collection.get(include=["metadatas"])
    filenames = set()
    for meta in all_data["metadatas"]:
        filenames.add(meta.get("filename", "未知"))
    
    return sorted(filenames)


def delete_document(filename: str, db_path: str = "./chroma_db",
                    collection_name: str = "mini_rag"):
    """删除指定文档的所有向量"""
    collection = get_collection(db_path, collection_name)
    
    # 通过元数据过滤获取该文档的所有ID
    all_data = collection.get(
        where={"filename": filename},
        include=[]
    )
    
    if all_data["ids"]:
        collection.delete(ids=all_data["ids"])
        logger.info(f"已删除文档: {filename}, {len(all_data['ids'])}个块")
    else:
        logger.warning(f"未找到文档: {filename}")
