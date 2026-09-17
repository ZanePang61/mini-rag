"""
Streamlit Web界面 - 小型轻量化RAG系统
ChatGPT风格：侧边栏(文档上传+管理) | 主区(聊天)
"""
import os
import sys
import time
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

from src.parser import parse_document, SUPPORTED_FORMATS
from src.chunker import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.retriever import add_documents, search, clear_collection, list_documents, delete_document
from src.rag import generate_answer
from src.history import (
    list_sessions, load_session, create_session,
    save_message, delete_session, rename_session,
)

# ===== 页面配置 =====
st.set_page_config(
    page_title="RAG 智能问答",
    page_icon="📚",
    layout="wide"
)

# ===== CSS =====
st.markdown("""
<style>
/* ===== 隐藏多余元素 ===== */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ===== 整体基础 ===== */
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

/* ===== 主区容器 ===== */
.block-container {
    max-width: min(1200px, 95vw) !important;
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}

/* ===== 聊天气泡：全宽自适应 ===== */
[data-testid="stChatMessage"] {
    max-width: 100% !important;
    width: 100% !important;
}
[data-testid="stChatMessageContent"] {
    max-width: 100% !important;
    width: 100% !important;
}

/* ===== 侧边栏 ===== */
[data-testid="stSidebar"] {
    background: #f7f7f8;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ===== Session State =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
# 参数配置默认值
if "cfg_chunk_size" not in st.session_state:
    st.session_state.cfg_chunk_size = DEFAULT_CHUNK_SIZE
if "cfg_chunk_overlap" not in st.session_state:
    st.session_state.cfg_chunk_overlap = DEFAULT_CHUNK_OVERLAP
if "cfg_top_k" not in st.session_state:
    st.session_state.cfg_top_k = 5
if "cfg_show_sources" not in st.session_state:
    st.session_state.cfg_show_sources = True
if "cfg_show_retrieval" not in st.session_state:
    st.session_state.cfg_show_retrieval = True
# 会话管理
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "sessions_cache" not in st.session_state:
    st.session_state.sessions_cache = []


def handle_upload(uf):
    if uf.name in st.session_state.uploaded_files:
        return
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    safe_name = f"{int(time.time()*1000)}_{uf.name}"
    tmp_path = os.path.join(data_dir, safe_name)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uf.getbuffer())
        with st.spinner(f"解析中..."):
            result = parse_document(tmp_path)
        chunks = chunk_text(result["text"], st.session_state.cfg_chunk_size, st.session_state.cfg_chunk_overlap)
        with st.spinner(f"入库中..."):
            add_documents(chunks, result["filename"])
        st.session_state.uploaded_files.append(uf.name)
        st.success(f"✅ {uf.name} 入库完成 ({len(chunks)}块)")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 上传失败: {e}")
        with st.expander("查看原始错误详情", expanded=False):
            st.exception(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    st.session_state.sessions_cache = list_sessions()

    # ===== 左侧边栏 =====
    with st.sidebar:
        st.title("📚 RAG 助手")

        if st.button("➕ 新建对话", use_container_width=True):
            new_id = create_session()
            st.session_state.current_session_id = new_id
            st.session_state.chat_history = []
            st.rerun()

        st.divider()

        st.subheader("📎 历史会话")
        sessions = st.session_state.sessions_cache
        if not sessions:
            st.caption("暂无历史对话")
        else:
            for s_meta in sessions[:20]:
                is_current = (s_meta["id"] == st.session_state.current_session_id)
                c1, c2 = st.columns([6, 1])
                with c1:
                    label = ("✅ " if is_current else "") + s_meta["title"]
                    if st.button(
                        label,
                        key=f"load_{s_meta['id']}",
                        use_container_width=True,
                        help=f"{s_meta['message_count']} 条消息 · {time.strftime('%m-%d %H:%M', time.localtime(s_meta['updated_at']))}"
                    ):
                        st.session_state.current_session_id = s_meta["id"]
                        data = load_session(s_meta["id"])
                        st.session_state.chat_history = data.get("messages", []) if data else []
                        st.rerun()
                with c2:
                    if st.button("✖", key=f"del_sess_{s_meta['id']}", help="删除该会话"):
                        delete_session(s_meta["id"])
                        if is_current:
                            st.session_state.current_session_id = None
                            st.session_state.chat_history = []
                        st.rerun()

        st.divider()

        st.subheader("📎 上传文档")
        uploaded = st.file_uploader(
            "支持 PDF / Word / Excel",
            type=["pdf", "docx", "xlsx", "xls"],
            accept_multiple_files=True,
            help="PDF(.pdf) Word(.docx) Excel(.xlsx/.xls)"
        )
        if uploaded:
            for uf in uploaded:
                handle_upload(uf)

        st.divider()

        st.subheader("🗂️ 已入库文档")
        try:
            docs = list_documents()
            if docs:
                for dn in docs:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.text(dn)
                    with c2:
                        if st.button("✕", key=f"del_{dn}"):
                            delete_document(dn)
                            st.rerun()
            else:
                st.info("暂无文档")
        except Exception as e:
            st.warning(f"加载失败: {e}")

        st.divider()

        st.subheader("⚙️ 参数配置")
        with st.expander("调整分块 / 检索参数", expanded=False):
            st.caption("参数变更对**新上传的文档**和**后续问题**生效")

            st.markdown("**📄 文档分块**")
            st.session_state.cfg_chunk_size = st.slider(
                "分块大小", 100, 2000, st.session_state.cfg_chunk_size, 50,
                help="每个文本块包含的字符数"
            )
            st.session_state.cfg_chunk_overlap = st.slider(
                "重叠大小", 0, 500, st.session_state.cfg_chunk_overlap, 10,
                help="相邻块重叠的字符数（避免边界信息丢失）"
            )

            st.markdown("**🔍 检索**")
            st.session_state.cfg_top_k = st.slider(
                "Top-K 片段数", 1, 20, st.session_state.cfg_top_k, 1,
                help="检索时返回的最相关片段数量"
            )

            st.markdown("**👁️ 展示**")
            st.session_state.cfg_show_retrieval = st.checkbox(
                "显示检索片段", value=st.session_state.cfg_show_retrieval,
                help="AI 回答上方是否展开检索到的原文片段"
            )
            st.session_state.cfg_show_sources = st.checkbox(
                "显示来源标签", value=st.session_state.cfg_show_sources,
                help="AI 回答末尾是否标注来源文档"
            )

        st.divider()

        if st.button("🗑️ 清空向量库", use_container_width=True):
            clear_collection()
            st.session_state.uploaded_files = []
            st.rerun()

    # ===== 右侧主区 - 聊天 =====
    # 当前会话标题
    cur = next((s for s in st.session_state.sessions_cache if s["id"] == st.session_state.current_session_id), None)
    if cur:
        st.markdown(
            f'''<div style="display:flex; align-items:center; gap:8px; padding:6px 0 12px; border-bottom:1px solid #e5e7eb; margin-bottom:16px; color:#374151; font-size:14px;"><span style="font-size:16px;">📝</span><span style="font-weight:600;">{cur["title"]}</span><span style="color:#9ca3af; font-size:12px;">&nbsp;· {cur["message_count"]} 条</span></div>''',
            unsafe_allow_html=True
        )
    try:
        doc_count = len(list_documents())
    except:
        doc_count = 0

    if doc_count == 0 and not st.session_state.chat_history:
        st.markdown("""
<div style="text-align:center; padding: 100px 20px 40px;">
  <div style="font-size: 56px; margin-bottom: 16px;">📚</div>
  <div style="font-size: 28px; font-weight: 600; margin-bottom: 10px;">RAG 智能问答助手</div>
  <div style="font-size: 15px; color: #6b7280; line-height: 1.6;">
    上传 PDF、Word 或 Excel 文档<br/>基于文档内容进行智能检索与问答
  </div>
</div>""", unsafe_allow_html=True)

    elif doc_count == 0:
        st.warning("⚠️ 请先在左侧边栏上传文档")

    # 历史消息
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant"):
                if st.session_state.cfg_show_retrieval and msg.get("retrieval"):
                    with st.expander(f"🔍 检索到 {len(msg['retrieval'])} 个相关片段"):
                        for i, r in enumerate(msg["retrieval"], 1):
                            st.write(f"**片段{i}** — `{r['filename']}` (相似度: {1-r['distance']:.4f})")
                            st.text(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
                            st.divider()
                st.write(msg["answer"])
                if msg.get("sources") and st.session_state.cfg_show_sources:
                    st.caption(f"📎 来源: {', '.join(msg['sources'])}")
    elif doc_count > 0:
        st.info("💬 在下方输入问题开始对话")

    # 底部输入框
    question = st.chat_input("输入你的问题...")

    # 阶段 1：用户刚提交 → 立即显示用户消息 + “正在检索”占位
    if question and question.strip():
        q = question.strip()
        st.session_state.chat_history.append({
            "question": q,
            "answer": "🔎 正在检索相关片段，请稍候…",
            "sources": [],
            "retrieval": [],
            "pending": True,
        })
        st.session_state.pending_question = q
        st.rerun()

    # 阶段 2：后台处理检索 + 生成
    pending = st.session_state.get("pending_question")
    if pending:
        q = pending
        # 找到刚才那个 pending 气泡的索引
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            if st.session_state.chat_history[i].get("pending"):
                pending_idx = i
                break
        else:
            pending_idx = None

        # 执行检索
        results = search(q, top_k=st.session_state.cfg_top_k)

        if not results:
            answer = "未能在文档中找到相关内容，请尝试换个问法或上传更多文档。"
            sources = []
        else:
            resp = generate_answer(q, results)
            answer = resp["answer"]
            sources = resp.get("sources", [])

        # 替换 pending 气泡为最终回答
        if pending_idx is not None:
            st.session_state.chat_history[pending_idx] = {
                "question": q,
                "answer": answer,
                "sources": sources,
                "retrieval": results if results else [],
            }
        else:
            st.session_state.chat_history.append({
                "question": q,
                "answer": answer,
                "sources": sources,
                "retrieval": results if results else [],
            })

        # 清理标记
        st.session_state.pop("pending_question", None)

        # 落盘
        if st.session_state.current_session_id is None:
            st.session_state.current_session_id = create_session(q)
        save_message(
            st.session_state.current_session_id,
            q, answer, sources,
            results if results else []
        )
        st.rerun()


if __name__ == "__main__":
    main()
