"""站内换乘经验知识库:JSONL 语料 + 纯 Python BM25 检索(零第三方依赖)。

文档结构:{id, hub, title, source, tags, content}。
分词:中文按 2-gram、ASCII 按小写单词,极端轻量,够支撑枢纽经验这类短文档检索。
语料来源:手工整理的公开攻略/站方指引(演示语料),或 zhishuxing kb-ingest 从任意
txt/md/html 目录抽取入库——后续可持续收录网页文本,无需改代码。
"""

from __future__ import annotations

import json
import math
import re
import html as html_mod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config as cfg

_ASCII_STOPWORDS = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "on"}


@dataclass
class _IndexedDoc:
    doc: Dict[str, Any]
    tf: Counter
    length: int


def tokenize(text: str) -> List[str]:
    """中文 2-gram + ASCII 小写单词(标题与正文一视同仁,标题权重由调用方重复拼接实现)。"""
    tokens: List[str] = []
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", (text or "").lower()):
        if run[0].isascii():
            if run not in _ASCII_STOPWORDS:
                tokens.append(run)
        elif len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class TransferKB:
    """内存 BM25 索引。文档量级(几十~几千条)下毫秒级检索,无需向量库。"""

    _K1 = 1.5
    _B = 0.75

    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None) -> None:
        self._docs: List[_IndexedDoc] = []
        self._df: Counter = Counter()
        self._avg_len: float = 0.0
        for doc in docs or []:
            self.add(doc)

    # ------------------------------------------------------------ 构建

    @classmethod
    def empty(cls) -> "TransferKB":
        return cls([])

    @classmethod
    def load(cls, corpus_path: Path) -> "TransferKB":
        docs: List[Dict[str, Any]] = []
        path = Path(corpus_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cls(docs)

    @classmethod
    def load_default(cls) -> "TransferKB":
        return cls.load(cfg.paths.kb_corpus)

    def add(self, doc: Dict[str, Any]) -> None:
        title = str(doc.get("title") or "")
        content = str(doc.get("content") or "")
        # 标题重复一次等效加权,让命中标题的文档排前
        tokens = tokenize(f"{title} {title} {content}")
        if not tokens:
            return
        tf = Counter(tokens)
        self._docs.append(_IndexedDoc(doc=doc, tf=tf, length=len(tokens)))
        self._df.update(tf.keys())
        self._avg_len = sum(item.length for item in self._docs) / len(self._docs)

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------ 检索

    def search(self, query: str, hub: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._docs:
            return []
        query_tokens = [t for t in tokenize(query) if t]
        if not query_tokens:
            return []
        total = len(self._docs)
        scored: List[Tuple[float, int]] = []
        for index, item in enumerate(self._docs):
            if hub and item.doc.get("hub") and item.doc["hub"] != hub:
                continue
            score = 0.0
            for token in query_tokens:
                freq = item.tf.get(token, 0)
                if not freq:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * (freq * (self._K1 + 1)) / (
                    freq + self._K1 * (1 - self._B + self._B * item.length / (self._avg_len or 1.0))
                )
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"doc": self._docs[index].doc, "score": score}
            for score, index in scored[: max(1, top_k)]
        ]


# ------------------------------------------------------------ 语料入库

_TAG_RULES: List[Tuple[str, List[str]]] = [
    ("地铁", ["地铁"]),
    ("高铁", ["高铁", "动车", "铁路"]),
    ("公交", ["公交"]),
    ("卫生间", ["卫生间", "洗手间", "厕所"]),
    ("直梯", ["直梯", "无障碍电梯", "垂直电梯"]),
    ("扶梯", ["扶梯"]),
    ("楼梯", ["楼梯"]),
    ("行李", ["行李", "托运"]),
    ("安检", ["安检"]),
    ("打车", ["打车", "出租", "网约车"]),
    ("出口", ["出口", "出站口"]),
    ("母婴", ["母婴"]),
    ("无障碍", ["无障碍", "轮椅"]),
    ("换乘", ["换乘"]),
]


def _strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html_mod.unescape(text)


def _split_chunks(text: str, max_len: int = 400) -> List[str]:
    """按空行分段再合并到 max_len,保证每条语料语义完整且长度可控。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 1 <= max_len:
            buffer = f"{buffer}\n{paragraph}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


def _auto_tags(text: str) -> List[str]:
    return [tag for tag, words in _TAG_RULES if any(word in text for word in words)]


def ingest_directory(
    src_dir: Path,
    out_path: Path,
    hub: str = "general",
    max_len: int = 400,
) -> Dict[str, Any]:
    """把目录下 .txt/.md/.html 清洗→切段→打标签,按 id 覆盖写入 JSONL(幂等可重跑)。

    id 规则 hub:相对路径:段序号 —— 同一文件重跑只更新不改增;换枢纽换 hub 前缀即可。
    """
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(f"源文档目录不存在: {src}")
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".htm"}
    )

    out = Path(out_path)
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if doc.get("id"):
                docs_by_id[doc["id"]] = doc

    changed = 0
    for file in files:
        raw = file.read_text(encoding="utf-8", errors="ignore")
        text = _strip_html(raw) if file.suffix.lower() in {".html", ".htm"} else raw
        text = re.sub(r"[ \t]+", " ", text)
        title = file.stem
        if file.suffix.lower() in {".html", ".htm"}:
            html_title = re.search(r"<title>(.*?)</title>|<h1[^>]*>(.*?)</h1>", raw, flags=re.S | re.I)
            if html_title:
                title = (html_title.group(1) or html_title.group(2)).strip()
        heading = re.search(r"^#\s+(.+)$", text, flags=re.M)
        if heading:
            title = heading.group(1).strip()
        relative = file.relative_to(src).as_posix()
        chunks = _split_chunks(text, max_len=max_len)
        for index, chunk in enumerate(chunks):
            doc_id = f"{hub}:{relative}:{index}"
            chunk_title = title if len(chunks) == 1 else f"{title} · {index + 1}"
            doc = {
                "id": doc_id,
                "hub": hub,
                "title": chunk_title,
                "source": relative,
                "tags": _auto_tags(f"{title} {chunk}"),
                "content": chunk,
            }
            if docs_by_id.get(doc_id, {}).get("content") == chunk:
                continue
            docs_by_id[doc_id] = doc
            changed += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = [docs_by_id[key] for key in sorted(docs_by_id)]
    out.write_text(
        "\n".join(json.dumps(doc, ensure_ascii=False) for doc in ordered) + ("\n" if ordered else ""),
        encoding="utf-8",
    )
    return {
        "files": len(files),
        "docs": len(ordered),
        "changed": changed,
        "out": str(out),
    }
