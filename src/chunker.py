"""
文本预处理与智能分块模块
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# 默认分块参数
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """文本清洗：去除多余换行、空格、特殊字符"""
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除连续空行（保留单个换行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除行首行尾空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 去除多余空格（连续空格变单个）
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 去除特殊控制字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    return text.strip()


def split_text_by_length(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    固定长度分块 + 重叠补全
    按字符数切分，相邻块之间有overlap个字符的重叠
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 下一个块的起始位置（向后移动chunk_size - overlap）
        start = start + chunk_size - overlap
        
        # 如果剩余文本不足以再分一个块，结束
        if end >= len(text):
            break
    
    return chunks


def split_text_by_paragraph(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    段落感知分块：优先按段落切分，段落过长时再按长度切
    这样能更好地保留语义完整性
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果当前段落本身就超过chunk_size，先保存当前累积的内容，再对长段落单独切分
        if len(para) > chunk_size:
            # 先保存之前累积的内容
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # 对长段落按长度切分
            sub_chunks = split_text_by_length(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
        else:
            # 累积段落，如果加上当前段落超过chunk_size，则切分
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
    
    # 最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, str]]:
    """
    文本分块主函数
    返回: [{"chunk": 分块文本, "index": 序号}, ...]
    """
    cleaned = clean_text(text)
    
    if not cleaned:
        return []
    
    # 使用段落感知分块
    chunks = split_text_by_paragraph(cleaned, chunk_size, overlap)
    
    # 过滤过短的块（少于10个字符的块丢弃）
    chunks = [c for c in chunks if len(c.strip()) >= 10]
    
    result = [{"chunk": c, "index": i} for i, c in enumerate(chunks)]
    
    logger.info(f"文本分块完成: {len(cleaned)}字符 -> {len(result)}个块 "
                f"(chunk_size={chunk_size}, overlap={overlap})")
    
    return result
