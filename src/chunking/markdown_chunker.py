"""
Structural chunking for Markdown / technical docs.

WHY:
  Same idea as the code chunker — split on meaning, not on character count. For docs,
  the natural boundary is the *heading*. We split the document into sections, one per
  heading, and we remember each section's "breadcrumb" (its chain of parent headings).

  That breadcrumb is gold for retrieval: a section titled "Configuration" is ambiguous
  on its own, but "Deployment > Kubernetes > Configuration" tells the LLM exactly where
  it sits. We store it in metadata and can prepend it to the text later.

Pure standard library — just a regex for headings.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import Chunk

# Matches ATX headings: one-to-six '#' characters, a space, then the title text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(path: Path, source: str, repo_root: Path) -> list[Chunk]:
    rel = str(path.relative_to(repo_root))
    lines = source.splitlines()

    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # ancestor headings: (level, title)
    buffer: list[str] = []             # lines of the section we're currently building
    current_title: str | None = None
    current_level = 0

    def breadcrumb() -> list[str]:
        """Titles of the current heading and all its ancestors, top-down."""
        return [title for (_level, title) in stack]

    def flush() -> None:
        """Emit the section we've buffered so far (if it has real content)."""
        text = "\n".join(buffer).strip()
        if not text:
            return
        crumbs = breadcrumb()
        name = " > ".join(crumbs) if crumbs else "(preamble)"
        chunks.append(
            Chunk(
                chunk_id=f"{rel}::{name}",
                content=text,
                metadata={
                    "file": rel,
                    "language": "markdown",
                    "kind": "section",
                    "heading": current_title,     # this section's own heading
                    "breadcrumb": crumbs,         # full path, e.g. ["Deploy", "K8s", "Config"]
                    "level": current_level,       # heading depth (1-6); 0 = preamble
                },
            )
        )

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            # A new heading starts a new section — first flush the previous one.
            flush()
            buffer = []

            level = len(match.group(1))
            title = match.group(2).strip()

            # Pop any headings at the same or deeper level so the stack holds only
            # this heading's true ancestors, then push this heading.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            current_level = level
            current_title = title
            buffer.append(line)  # keep the heading line as part of the section text
        else:
            buffer.append(line)

    flush()  # don't forget the final section
    return chunks
