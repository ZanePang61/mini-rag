"""
问答历史记录管理
- 每个对话会话保存为单独 JSON 文件
- 目录: data/sessions/<session_id>.json
- 文件格式: {"id":..., "title":..., "created_at":..., "updated_at":..., "messages":[{q,a,sources,ts}]}
"""
import os
import json
import time
import uuid
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sessions"
)


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _now() -> float:
    return time.time()


def _format_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def list_sessions() -> List[Dict]:
    """列出所有会话（按 updated_at 倒序）"""
    _ensure_dir()
    sessions = []
    for fn in os.listdir(SESSIONS_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(_path(fn[:-5]), "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", fn[:-5]),
                "title": data.get("title", "(无标题)"),
                "created_at": data.get("created_at", 0),
                "updated_at": data.get("updated_at", 0),
                "message_count": len(data.get("messages", [])),
            })
        except Exception as e:
            logger.warning(f"读取会话失败 {fn}: {e}")
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> Optional[Dict]:
    """加载单个会话的完整数据"""
    p = _path(session_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载会话失败 {session_id}: {e}")
        return None


def create_session(first_question: str = "") -> str:
    """创建新会话，返回 session_id"""
    _ensure_dir()
    sid = uuid.uuid4().hex[:12]
    title = (first_question[:30] + "...") if len(first_question) > 30 else (first_question or "新对话")
    data = {
        "id": sid,
        "title": title,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": []
    }
    with open(_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return sid


def save_message(session_id: str, question: str, answer: str, sources: List[str], retrieval: List[Dict] = None) -> bool:
    """向指定会话追加一条问答"""
    p = _path(session_id)
    if not os.path.exists(p):
        # 会话不存在则创建
        create_session(question)
        return True
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        msg = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "retrieval": retrieval or [],
            "ts": _now(),
        }
        data["messages"].append(msg)
        # 首条问题时用 question 作标题
        if len(data["messages"]) == 1:
            data["title"] = (question[:30] + "...") if len(question) > 30 else question
        data["updated_at"] = _now()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存消息失败 {session_id}: {e}")
        return False


def delete_session(session_id: str) -> bool:
    p = _path(session_id)
    if os.path.exists(p):
        try:
            os.unlink(p)
            return True
        except Exception as e:
            logger.error(f"删除会话失败 {session_id}: {e}")
    return False


def rename_session(session_id: str, new_title: str) -> bool:
    p = _path(session_id)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = new_title[:50]
        data["updated_at"] = _now()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"重命名会话失败 {session_id}: {e}")
        return False
