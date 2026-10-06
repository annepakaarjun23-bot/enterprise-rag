from __future__ import annotations

import hashlib


def make_chunk_id(
    library: str,
    version: str,
    source_path: str,
    strategy: str,
    level: int,
    heading_path: str,
    ordinal_in_section: int,
    parent_chunk_id: str | None = None,
) -> str:

    parts = [
        library,
        version,
        source_path,
        strategy,
        str(level),
        heading_path,
        str(ordinal_in_section),
    ]
    if parent_chunk_id:
        parts.append(parent_chunk_id)
    key = "|".join(parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:32]


def make_doc_id(library: str, version: str, source_path: str) -> str:
    key = f"{library}|{version}|{source_path}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]

def hex_to_uuid(h: str) -> str:
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"