"""会话持久化:SQLite 存取、重启恢复、过期清理、持久化失败不阻断对话。"""


from zhishuxing.llm.session_store import SessionStore


def test_save_and_load_roundtrip(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    store.save("s-1", {"history": [{"role": "user", "content": "你好"}], "profile": None})

    assert store.load("s-1")["history"][0]["content"] == "你好"
    assert store.load("missing") is None
    store.close()


def test_overwrite_keeps_latest(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    store.save("s-1", {"history": [1]})
    store.save("s-1", {"history": [1, 2]})

    assert store.load("s-1")["history"] == [1, 2]
    store.close()


def test_delete_removes_row(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    store.save("s-1", {"history": []})
    store.delete("s-1")

    assert store.load("s-1") is None
    store.close()


def test_purge_expired_only_removes_old(tmp_path):
    import time

    store = SessionStore(db_path=tmp_path / "sessions.db", ttl_seconds=60)
    store.save("old", {"history": []})
    store.save("fresh", {"history": []})
    # 把 old 的 updated_at 拨回 2 小时前
    with store._lock:
        store._conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = 'old'",
                            (time.time() - 7200,))
        store._conn.commit()

    removed = store.purge_expired()

    assert removed == 1
    assert store.load("old") is None
    assert store.load("fresh") is not None
    store.close()


def test_load_missing_backfills_only_absent(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    store.save("from-db", {"history": ["库里的"]})
    sessions = {"in-mem": {"history": ["内存的"]}}

    store.load_missing(sessions)

    assert sessions["from-db"]["history"] == ["库里的"]
    assert sessions["in-mem"]["history"] == ["内存的"]   # 内存优先,不被覆盖
    store.close()


def test_corrupt_payload_tolerated(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    with store._lock:
        store._conn.execute("INSERT INTO sessions VALUES ('bad', 0, 'not-json{')")
        store._conn.commit()

    assert store.load("bad") is None
    sessions: dict = {}
    store.load_missing(sessions)
    assert "bad" not in sessions
    store.close()


def test_assistant_persists_and_recovers_across_instances(tmp_path, monkeypatch):
    """核心判据:同一 session_id 跨 assistant 实例(等价服务重启)可恢复。"""
    from zhishuxing.llm.assistant import TransferAssistant
    from zhishuxing.llm.kb import TransferKB

    class _Stub:
        pass

    db = tmp_path / "sessions.db"
    a1 = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=db)
    out = a1.handle("打开导航", session_id="persist-1")
    assert out["action"]["tab"] == "navigation"
    a1._store.close()

    a2 = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=db)
    assert "persist-1" not in a2._sessions           # 新实例内存为空
    a2.handle("再看看", session_id="persist-1")       # 访问触发回填
    assert "persist-1" in a2._sessions               # 会话已从库恢复
    a2._store.close()


def test_assistant_handles_db_failure_gracefully(tmp_path, monkeypatch):
    """落库失败只打印警告,对话正常返回(持久化是增强不是依赖)。"""
    from zhishuxing.llm.assistant import TransferAssistant
    from zhishuxing.llm.kb import TransferKB

    class _Stub:
        pass

    a = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=tmp_path / "s.db")

    class _Boom:
        def save(self, *args, **kwargs):
            raise RuntimeError("disk full")

        def delete(self, *args, **kwargs):
            raise RuntimeError("disk full")

        def load_missing(self, sessions):
            raise RuntimeError("disk full")

        def close(self):
            pass

    a._store = _Boom()
    out = a.handle("打开设置", session_id="boom-1")
    assert out["action"]["tab"] == "settings"        # 对话不受影响
    a.reset("boom-1")                                # reset 的删库失败同样被吞
    a._store.close()
