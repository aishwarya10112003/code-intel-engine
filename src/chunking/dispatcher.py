"""
The dispatcher — picks the right chunking strategy for each file, with a safety net.

Rule:
  .py            → AST chunker  (falls back to line-windows if the file won't parse)
  .md/.markdown  → structural (heading) chunker
  everything else→ naive line-window chunker (our universal fallback)

The fallback matters: we should never silently drop a file just because we don't have
a fancy parser for its language yet. A rough chunk is better than no chunk. (In a later
phase we swap the fallback for tree-sitter to get real AST chunking across languages.)
"""
from __future__ import annotations

from pathlib import Path

from .base import Chunk
from .markdown_chunker import chunk_markdown
from .python_chunker import chunk_python

# Fallback chunker settings: overlapping windows so an idea split across a boundary
# still appears (partially) in both neighbours.
_WINDOW_LINES = 60
_OVERLAP_LINES = 10


def _chunk_fallback(path: Path, source: str, repo_root: Path) -> list[Chunk]:
    """Language-agnostic last resort: fixed-size overlapping line windows."""
    rel = str(path.relative_to(repo_root))
    lines = source.splitlines()
    chunks: list[Chunk] = []

    step = _WINDOW_LINES - _OVERLAP_LINES
    block_no = 0
    i = 0
    while i < len(lines):
        window = lines[i : i + _WINDOW_LINES]
        text = "\n".join(window).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{rel}::block{block_no}",
                    content=text,
                    metadata={
                        "file": rel,
                        "language": path.suffix.lstrip(".") or "text",
                        "kind": "line_window",
                        "start_line": i + 1,
                        "end_line": min(i + _WINDOW_LINES, len(lines)),
                    },
                )
            )
            block_no += 1
        i += step

    return chunks


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    """Read one file and turn it into chunks using the best available strategy."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Binary file or unreadable — skip it entirely.
        return []

    ext = path.suffix.lower()

    if ext == ".py":
        chunks = chunk_python(path, source, repo_root)
        return chunks if chunks else _chunk_fallback(path, source, repo_root)

    if ext in (".md", ".markdown"):
        return chunk_markdown(path, source, repo_root)

    return _chunk_fallback(path, source, repo_root)
