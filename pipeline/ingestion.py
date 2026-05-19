from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import tiktoken

import config

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Section:
    section_id: str
    doc_id: str
    title: str
    content: str
    position: int

    @property
    def token_count(self) -> int:
        return len(_enc.encode(self.content))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str
    text: str
    token_count: int
    position: int


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    source_path: str
    sections: List[Section] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(_enc.encode(self.content))


def _make_id(*parts: str) -> str:
    return hashlib.md5("||".join(parts).encode()).hexdigest()[:16]


def _parse_sections(doc_id: str, content: str) -> List[Section]:
    """Split document into sections based on markdown headings (## or #)."""
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    sections: List[Section] = []

    if not matches:
        # No headings: treat entire document as one section
        sid = _make_id(doc_id, "s0")
        sections.append(Section(sid, doc_id, "Main", content.strip(), 0))
        return sections

    # Text before first heading
    preamble = content[: matches[0].start()].strip()
    if preamble:
        sid = _make_id(doc_id, "preamble")
        sections.append(Section(sid, doc_id, "Preamble", preamble, 0))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if not body:
            continue
        sid = _make_id(doc_id, title, str(i))
        sections.append(Section(sid, doc_id, title, body, len(sections)))

    return sections


def load_documents(raw_dir: Path = config.RAW_DIR) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(raw_dir.glob("*.txt")) :
        content = path.read_text(encoding="utf-8")
        doc_id = _make_id(path.stem)
        title = path.stem.replace("_", " ").title()
        sections = _parse_sections(doc_id, content)
        docs.append(Document(doc_id, title, content, str(path), sections))
    return docs
