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


def test_assistant_restore_uses_primary_key_not_backfill(tmp_path, monkeypatch):
    """重启恢复走主键 load(PERF-01):只取该会话一行;整表回填 load_missing 不再出现在主链路,
    且恢复出的会话必须带历史(真正断言内容,而非"内存里存在"的空转断言)。"""
    from zhishuxing.llm.assistant import TransferAssistant
    from zhishuxing.llm.kb import TransferKB

    class _Stub:
        pass

    db = tmp_path / "sessions.db"
    a1 = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=db)
    a1.handle("我有点赶时间", session_id="restore-1")   # 非 intent 消息:完整链路,末尾写穿落库
    a1._store.close()

    a2 = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=db)

    def _boom_backfill(_sessions):
        raise AssertionError("重启恢复不应整表 load_missing 回填")

    monkeypatch.setattr(a2._store, "load_missing", _boom_backfill)
    a2.handle("打开导航", session_id="restore-1")        # intent 消息:主键恢复先行,随后短路返回
    assert "restore-1" in a2._sessions
    restored = a2._sessions["restore-1"]
    assert restored["history"], "历史应从库恢复,而非空会话"
    assert restored["history"][0]["role"] == "user"
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

        def load(self, session_id):
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


def test_save_opportunistic_purge_every_n_saves(tmp_path):
    """写入路径低频清理:每 PURGE_EVERY_N_SAVES 次落库机会式清一次过期会话,
    长驻进程不再只靠启动那一次 purge(PERF-06)。"""
    import time

    from zhishuxing.llm.session_store import PURGE_EVERY_N_SAVES

    store = SessionStore(db_path=tmp_path / "sessions.db", ttl_seconds=60)
    store.save("old", {"history": []})   # 计入落库计数
    with store._lock:
        store._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = 'old'", (time.time() - 7200,)
        )
        store._conn.commit()
        # 把计数拨到只差一次触发,隔离「前面落库次数」对断言的干扰
        store._saves_since_purge = PURGE_EVERY_N_SAVES - 1

    store.save("trigger", {"history": []})   # 这次落库应触发机会式清理

    assert store.load("old") is None          # 过期会话被清理
    assert store.load("trigger") is not None  # 本次落库不受影响
    assert store._saves_since_purge == 0      # 计数归零重新起算
    store.close()


def test_assistant_sessions_bounded_by_lru_eviction(tmp_path):
    """内存会话上限(LRU):超 SESSION_CAPACITY 按最近活跃淘汰,新 sid 不再无界增长;
    淘汰的是最久不活跃者,最新会话保留。"""
    from zhishuxing.llm.assistant import SESSION_CAPACITY, TransferAssistant
    from zhishuxing.llm.kb import TransferKB

    class _Stub:
        pass

    a = TransferAssistant(_Stub(), kb=TransferKB.empty(), db_path=tmp_path / "s.db")
    total = SESSION_CAPACITY + 8
    for i in range(total):
        a.handle("打开设置", session_id=f"sid-{i}")   # intent 短路消息:快速建会话

    assert len(a._sessions) <= SESSION_CAPACITY
    assert "sid-0" not in a._sessions                  # 最老被淘汰
    assert f"sid-{total - 1}" in a._sessions           # 最新保留
    a._store.close()


# -------------------------------------------------------- 上线三件套(WAL+busy_timeout+索引)

def test_wal_and_busy_timeout_applied_on_new_db(tmp_path):
    """上线三件套之一/之二(综合路线图 §3.2-1「上线硬前置」):SessionStore 打开的
    连接即处于 WAL 模式、busy_timeout=5000 —— 不做就带 SQLITE_BUSY 风险上公网。"""
    import sqlite3 as _sqlite3

    bare = _sqlite3.connect(str(tmp_path / "fresh.db"))    # 对照组:裸连接默认非 WAL,防恒真断言
    try:
        assert bare.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    finally:
        bare.close()

    store = SessionStore(db_path=tmp_path / "sessions.db")
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    store.close()


def test_wal_persists_across_reopen(tmp_path):
    """WAL 是库文件的持久属性:close 后用裸连接重开(等价服务重启/运维排查)仍是 WAL,
    不依赖「每次连接重设」这一兜底。"""
    import sqlite3

    db = tmp_path / "sessions.db"
    store = SessionStore(db_path=db)
    store.save("s-1", {"history": [1]})
    store.close()

    raw = sqlite3.connect(str(db))
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        raw.close()


def test_updated_at_index_serves_purge_query_plan(tmp_path):
    """上线三件套之三:idx_sessions_updated_at 在库,且 purge_expired 的
    `DELETE WHERE updated_at < ?` 查询计划走 SEARCH(索引)而非 SCAN(全表)——
    深评 §2 扣分点收口,TTL 清理成本随过期条数而非表大小增长。"""
    store = SessionStore(db_path=tmp_path / "sessions.db")
    names = {row[1] for row in store._conn.execute("PRAGMA index_list('sessions')").fetchall()}
    assert "idx_sessions_updated_at" in names

    plan = store._conn.execute(
        "EXPLAIN QUERY PLAN DELETE FROM sessions WHERE updated_at < ?", (0,)
    ).fetchall()
    detail = " ".join(str(row[-1]) for row in plan).lower()
    assert "idx_sessions_updated_at" in detail
    assert "scan" not in detail          # 有索引时计划为 SEARCH ... USING INDEX(本机实测)
    store.close()


def test_legacy_db_upgraded_in_place(tmp_path):
    """存量库升级:旧格式库(非 WAL、无索引、已有数据)被新代码打开时原位补齐
    WAL 与索引,旧数据完好可续写 —— 部署侧 git pull 重启即得,无需迁移脚本。"""
    import sqlite3

    db = tmp_path / "sessions.db"
    raw = sqlite3.connect(str(db))
    raw.execute(
        "CREATE TABLE sessions ("
        "session_id TEXT PRIMARY KEY, updated_at REAL NOT NULL, payload TEXT NOT NULL)"
    )
    raw.execute("INSERT INTO sessions VALUES ('legacy', 1, '{\"history\":[\"旧\"]}')")
    raw.commit()
    raw.close()

    store = SessionStore(db_path=db)
    assert store.load("legacy")["history"] == ["旧"]       # 旧数据原样可读
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    store.save("legacy", {"history": ["新"]})              # 原位续写不冲突
    assert store.load("legacy")["history"] == ["新"]
    store.close()
