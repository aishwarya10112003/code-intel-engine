"""
ingest.py — Phase 1 command-line tool.

Point it at a folder (a repo, a docs directory) and it walks every file, chunks each
one with the right strategy, writes all chunks to a JSON file, and prints a summary
so you can *see* what the chunker produced.

Usage:
    python ingest.py <path-to-folder> [output.json]

Example:
    python ingest.py sample_input
    python ingest.py ../Krishna_bakers_FINAL/backend/src backend_chunks.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from src.chunking import chunk_file

# Directories we never want to index (dependencies, build output, VCS internals).
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "coverage",
}

# File types we attempt to chunk. Anything here that isn't .py/.md goes through the
# fallback line-window chunker for now.
_ALLOWED_EXT = {
    ".py", ".md", ".markdown", ".txt", ".rst",
    ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb",
    ".json", ".yaml", ".yml", ".toml",
}


def iter_files(root: Path):
    """Yield every indexable file under `root`, skipping junk directories."""
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _ALLOWED_EXT:
            continue
        yield path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(1)

    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("chunks.json")

    all_chunks = []
    files_seen = 0
    for file_path in iter_files(root):
        files_seen += 1
        all_chunks.extend(chunk_file(file_path, root))

    # Write the chunks so we can inspect them by eye — the fastest way to sanity-check
    # a chunker is to actually read a few of its chunks.
    out_path.write_text(
        json.dumps([c.to_dict() for c in all_chunks], indent=2),
        encoding="utf-8",
    )

    # ---- Summary ----------------------------------------------------------------
    print(f"\nScanned: {root}")
    print(f"Files chunked: {files_seen}")
    print(f"Total chunks:  {len(all_chunks)}\n")

    by_lang = Counter(c.metadata.get("language", "?") for c in all_chunks)
    by_kind = Counter(c.metadata.get("kind", "?") for c in all_chunks)

    print("By language:")
    for lang, n in by_lang.most_common():
        print(f"  {lang:<10} {n}")
    print("By kind:")
    for kind, n in by_kind.most_common():
        print(f"  {kind:<15} {n}")

    sizes = [len(c.content) for c in all_chunks]
    if sizes:
        avg = sum(sizes) // len(sizes)
        print(f"\nChunk size (chars): min={min(sizes)}  avg={avg}  max={max(sizes)}")
        oversized = [c for c in all_chunks if len(c.content) > 4000]
        if oversized:
            # A talking point for later phases: very large chunks hurt retrieval,
            # so a future step is sub-splitting these.
            print(f"  NOTE: {len(oversized)} chunk(s) > 4000 chars may need sub-splitting later.")

    print(f"\nWrote {len(all_chunks)} chunks -> {out_path}\n")


if __name__ == "__main__":
    main()
