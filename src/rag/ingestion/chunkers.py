from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.schema import (
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)

from rag.ingestion.ids import make_chunk_id

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)


@dataclass
class Block:
    kind: str          
    text: str
    level: int = 0   
    lang: str = ""     


@dataclass
class Section:
    heading_path: str
    heading_path_list: list[str]
    section_title: str
    heading_level: int = 0
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks).rstrip()

    @property
    def has_code(self) -> bool:
        return any(b.kind == "code" for b in self.blocks)

    @property
    def code_languages(self) -> list[str]:
        return sorted(
            {b.lang for b in self.blocks if b.kind == "code" and b.lang}
        )


_HEADING_RE = re.compile(r"^[ ]{0,3}(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^[ ]{0,6}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")


def _fence_lang(info: str) -> str:
    if not info:
        return ""
    first = info.split()[0]
    if first.startswith("{"):
        return ""
    return first.lower()


def _is_close(line: str, fence_char: str, fence_len: int) -> bool:
    m = _FENCE_RE.match(line)
    if not m:
        return False
    if m.group(1)[0] != fence_char or len(m.group(1)) < fence_len:
        return False
    return m.group(2).strip() == ""




def parse_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        m = _FENCE_RE.match(line)
        if m:
            fence_str = m.group(1)
            fence_char = fence_str[0]
            fence_len = len(fence_str)
            lang = _fence_lang(m.group(2))
            body = [line]
            i += 1
            while i < n:
                body.append(lines[i])
                if _is_close(lines[i], fence_char, fence_len):
                    i += 1
                    break
                i += 1
            blocks.append(Block(kind="code", text="\n".join(body), lang=lang))
            continue

        m = _HEADING_RE.match(line)
        if m:
            blocks.append(Block(kind="heading", text=line, level=len(m.group(1))))
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s", line):
            body = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    break
                if _HEADING_RE.match(nxt) or _FENCE_RE.match(nxt):
                    break
                if re.match(r"^\s*([-*+]|\d+\.)\s", nxt) or nxt.startswith("  "):
                    body.append(nxt)
                    i += 1
                    continue
                break
            blocks.append(Block(kind="list", text="\n".join(body)))
            continue

        body = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt) or _FENCE_RE.match(nxt):
                break
            body.append(nxt)
            i += 1
        blocks.append(Block(kind="para", text="\n".join(body)))

    return blocks


def build_parents(
    blocks: list[Block],
    min_tokens: int = 50,
    max_tokens: int = 1200,
) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current: Optional[Section] = None

    for block in blocks:
        if block.kind == "heading":
            if current is not None:
                sections.append(current)
            level = block.level
            title = _HEADING_RE.match(block.text).group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path_list = [t for _, t in heading_stack]
            current = Section(
                heading_path=" > ".join(path_list),
                heading_path_list=path_list,
                section_title=title,
                heading_level=level,
                blocks=[block],
            )
        else:
            if current is None:
                current = Section(
                    heading_path="",
                    heading_path_list=[],
                    section_title="",
                    heading_level=0,
                )
            current.blocks.append(block)

    if current is not None:
        sections.append(current)

    merged = _merge_tiny(sections, min_tokens)
    out: list[Section] = []
    for s in merged:
        out.extend(_split_oversized(s, max_tokens))
    return out


def _merge_tiny(sections: list[Section], min_tokens: int) -> list[Section]:
    if not sections:
        return sections
    out: list[Section] = []
    for s in sections:
        if count_tokens(s.text) < min_tokens and out:
            out[-1].blocks.extend(s.blocks)
        else:
            out.append(s)
    return out


def _split_oversized(section: Section, max_tokens: int) -> list[Section]:
    if count_tokens(section.text) <= max_tokens:
        return [section]

    parts: list[Section] = []
    buf: list[Block] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            parts.append(_clone_section(section, buf))
            buf = []
            buf_tokens = 0

    for block in section.blocks:
        bt = count_tokens(block.text)
        if block.kind == "code" and bt > max_tokens:
            flush()
            parts.append(_clone_section(section, [block]))
            continue
        if buf_tokens + bt > max_tokens and buf:
            flush()
        buf.append(block)
        buf_tokens += bt

    flush()

    for idx, part in enumerate(parts):
        if idx == 0 or not section.heading_path:
            continue
        part.blocks.insert(0, Block(kind="para", text=f"[{section.heading_path}]"))
    return parts


def _clone_section(src: Section, blocks: list[Block]) -> Section:
    return Section(
        heading_path=src.heading_path,
        heading_path_list=list(src.heading_path_list),
        section_title=src.section_title,
        heading_level=src.heading_level,
        blocks=list(blocks),
    )



def build_leaves(parent: Section, max_leaf_tokens: int = 300) -> list[Section]:
    leaves: list[Section] = []
    buf: list[Block] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            leaves.append(_clone_section(parent, buf))
            buf = []
            buf_tokens = 0

    for block in parent.blocks:
        bt = count_tokens(block.text)
        if block.kind == "code":
            if bt > max_leaf_tokens:
                flush()
                leaves.append(_clone_section(parent, [block]))
            else:
                if buf_tokens + bt > max_leaf_tokens and buf:
                    flush()
                buf.append(block)
                buf_tokens += bt
            continue
        if buf_tokens + bt > max_leaf_tokens and buf:
            flush()
        buf.append(block)
        buf_tokens += bt

    flush()

    if leaves and parent.heading_path:
        first = leaves[0]
        first.blocks.insert(0, Block(kind="para", text=f"[{parent.heading_path}]"))
    return leaves



def _section_to_node(
    section: Section,
    doc: Document,
    strategy: str,
    level: int,
    chunk_index: int,
    parent_chunk_id: Optional[str] = None,
) -> TextNode:
    text = section.text
    meta = doc.metadata or {}

    node_id = make_chunk_id(
        library=str(meta.get("library", "")),
        version=str(meta.get("version", "")),
        source_path=str(meta.get("source_path", "")),
        strategy=strategy,
        level=level,
        heading_path=section.heading_path,
        ordinal_in_section=chunk_index,
        parent_chunk_id=parent_chunk_id, 
    )

    node = TextNode(
        id_=node_id,
        text=text,
        metadata={
            **meta,
            "heading_path": section.heading_path,
            "heading_path_list": list(section.heading_path_list),
            "section_title": section.section_title,
            "has_code": section.has_code,
            "code_languages": list(section.code_languages),
            "token_count": count_tokens(text),
            "level": level,
            "is_leaf": level > 0,
            "strategy": strategy,
        },
    )

    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc.doc_id)
    if parent_chunk_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_chunk_id)
    return node



def chunk_document(
    doc: Document,
    strategy: str = "md_codesafe_v1",
    min_parent_tokens: int = 50,
    max_parent_tokens: int = 1200,
    max_leaf_tokens: int = 300,
) -> tuple[list[TextNode], list[TextNode]]:
    blocks = parse_blocks(doc.text)
    if not blocks:
        return [], []

    sections = build_parents(blocks, min_parent_tokens, max_parent_tokens)

    parents: list[TextNode] = []
    leaves: list[TextNode] = []

    for i, section in enumerate(sections):
        parent_node = _section_to_node(section, doc, strategy, level=0, chunk_index=i)
        parents.append(parent_node)
        for j, leaf in enumerate(build_leaves(section, max_leaf_tokens)):
            leaf_node = _section_to_node(
                leaf, doc, strategy, level=1, chunk_index=j,
                parent_chunk_id=parent_node.id_,
            )
            leaves.append(leaf_node)

    for i, node in enumerate(leaves):
        if i > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=leaves[i - 1].id_
            )
        if i < len(leaves) - 1:
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=leaves[i + 1].id_
            )

    return parents, leaves