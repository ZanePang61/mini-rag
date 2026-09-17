"""
RAG问答模块：组装检索上下文 + 调用DeepSeek大模型生成回答
"""
import os
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 系统提示词：严格基于文档内容回答
SYSTEM_PROMPT = """你是一个专业的文档问答助手。请严格基于以下检索到的文档内容来回答用户的问题。

要求：
1. 只使用提供的文档内容作为回答依据，不要编造或使用外部知识
2. 如果文档内容无法回答该问题，请明确说明"根据现有文档内容，无法回答该问题"
3. 回答时请标注信息来源（文件名）
4. 表格/列表型问题需完整呈现关键字段（如工号、姓名、日期、数值等），不要只挑最显眼的列输出
5. 回答要准确、条理清晰

检索到的文档内容：
{context}
"""


def build_context(search_results: List[Dict]) -> str:
    """将检索结果组装为上下文文本"""
    if not search_results:
        return "（无相关文档内容）"
    
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"[片段{i}] 来源: {result['filename']}\n"
            f"内容: {result['text']}\n"
        )
    
    return "\n".join(context_parts)


def generate_answer(query: str, search_results: List[Dict],
                    model: str = "deepseek-chat",
                    temperature: float = 0.3,
                    max_tokens: int = 2000) -> Dict:
    """
    RAG问答：结合检索上下文调用大模型生成回答
    返回: {"answer": 回答文本, "sources": 来源文件列表, "context": 上下文}
    """
    context = build_context(search_results)
    sources = list(set(r["filename"] for r in search_results))
    
    system_message = SYSTEM_PROMPT.format(context=context)
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            temperature=temperature,  # 低温度保证回答稳定
            max_tokens=max_tokens,
        )
        
        answer = response.choices[0].message.content
        
        logger.info(f"问答完成: query='{query[:30]}...', 回答长度={len(answer)}")
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"大模型调用失败: {e}")
        return {
            "answer": f"问答生成失败: {str(e)}",
            "sources": sources,
            "context": context
        }
