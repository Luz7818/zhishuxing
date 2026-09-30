"""会话持久化:SQLite 单文件存储(零第三方依赖)。

把 TransferAssistant 的对话会话(历史 + 需求档案 + 失败记录)落到
`data/runs/sessions.db`,服务重启后按 session_id 恢复 —— 解决「重启即清」
的演示工程硬伤。设计取舍:

- **写穿读缓存**:内存 dict 仍是第一读写层(进程内零开销),落库只在
  handle 末尾与 reset 时发生一次;重启后首次访问经 `load_missing` 惰性回填;
- **单连接 + 锁**:SQLite 连接随 store 实例持有,写路径全部经
  `threading.Lock` 串行 —— 服务是单实例部署(AGENTS 边界),不追求多进程;
- **30 天过期**:`purge_expired()` 在服务启动时清理一次,过期会话静默删除;
- **profile 存 JSON**:PassengerProfile.to_dict()/from_dict 序列化,
  schema 演进时旧记录解析失败按"无档案"处理,不炸服务。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 天


class SessionStore:
    def __init__(self, db_path: Path, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                   session_id TEXT PRIMARY KEY,
                   updated_at REAL NOT NULL,
                   payload    TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    # ------------------------------------------------------------ 读写

    def save(self, session_id: str, session: Dict[str, Any]) -> None:
        payload = json.dumps(session, ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, updated_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, "
                "payload=excluded.payload",
                (session_id, time.time(), payload),
            )
            self._conn.commit()

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row[0])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def load_missing(self, sessions: Dict[str, Dict[str, Any]]) -> None:
        """把库里有、内存里没有的会话回填进内存(服务启动后首次使用时惰性执行)。"""
        with self._lock:
            rows = self._conn.execute("SELECT session_id, payload FROM sessions").fetchall()
        for session_id, payload in rows:
            if session_id in sessions:
                continue
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    sessions[session_id] = data
            except json.JSONDecodeError:
                continue

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def purge_expired(self) -> int:
        """删除超过 TTL 未活跃的会话;返回删除条数(服务启动时调用一次)。"""
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
