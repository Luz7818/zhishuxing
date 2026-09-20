"""换乘经验知识库测试:入库幂等 / BM25 检索相关性 / hub 过滤。"""

from __future__ import annotations

import json

from zhishuxing.llm.kb import TransferKB, ingest_directory


def _make_source(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.md").write_text(
        "# 卫生间在哪里\n\n候车层两端各有一处卫生间,距离检票口约五十米。\n\n出站层A口内侧也有洗手间。\n",
        encoding="utf-8",
    )
    (src / "b.md").write_text(
        "# 直梯位置\n\n换乘通道东西两侧各一部无障碍直梯,行李多推轮椅优先走直梯。\n",
        encoding="utf-8",
    )
    (src / "c.html").write_text(
        "<html><body><h1>打车指引</h1><p>出租车蓄车区在东广场地下层,网约车在西广场。</p></body></html>",
        encoding="utf-8",
    )
    return src


def test_ingest_directory_builds_corpus(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    result = ingest_directory(src, out, hub="demo")
    assert result["files"] == 3
    assert result["docs"] >= 3

    docs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(d["hub"] == "demo" for d in docs)
    titles = " ".join(d["title"] for d in docs)
    assert "卫生间" in titles and "直梯" in titles and "打车" in titles
    # html 清洗后不应残留标签
    assert not any("<p>" in d["content"] for d in docs)


def test_ingest_is_idempotent(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    first = ingest_directory(src, out, hub="demo")
    second = ingest_directory(src, out, hub="demo")
    assert second["changed"] == 0
    assert second["docs"] == first["docs"]


def test_ingest_updates_changed_content(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    ingest_directory(src, out, hub="demo")
    (src / "a.md").write_text("# 卫生间在哪里\n\n位置调整:卫生间全部移到出站层西侧。\n", encoding="utf-8")
    result = ingest_directory(src, out, hub="demo")
    assert result["changed"] >= 1


def test_bm25_search_ranks_relevant_docs(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    ingest_directory(src, out, hub="demo")
    kb = TransferKB.load(out)
    assert len(kb) >= 3

    hits = kb.search("行李多想找直梯", hub="demo", top_k=2)
    assert hits
    assert "直梯" in hits[0]["doc"]["title"]

    hits_restroom = kb.search("卫生间", hub="demo", top_k=1)
    assert "卫生间" in hits_restroom[0]["doc"]["title"]


def test_search_filters_by_hub(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    ingest_directory(src, out, hub="demo")
    kb = TransferKB.load(out)
    assert kb.search("卫生间", hub="other_hub") == []
    assert kb.search("卫生间", hub="demo")


def test_load_default_kb_from_project_corpus():
    kb = TransferKB.load_default()
    # 项目自带深圳北站演示语料(由 kb-ingest 生成);缺失时视为空库不报错
    if len(kb) == 0:
        return
    hits = kb.search("优先直梯 行李", hub="shenzhen_north", top_k=3)
    assert hits
    assert all(hit["doc"]["hub"] == "shenzhen_north" for hit in hits)
