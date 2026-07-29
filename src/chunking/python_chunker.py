"""
AST-based chunking for Python files.

WHY not just split every 500 characters?
  Naive splitting cuts a function in half — the retriever then returns "half a
  function", which is useless to both the embedding model and the LLM. By parsing
  the file's Abstract Syntax Tree (AST) we split on *logical* boundaries: whole
  functions, whole methods, a class's signature+docstring. Each chunk is a
  self-contained, meaningful unit.

WHAT is an AST?
  Python's built-in `ast` module reads source code and returns a tree describing
  its structure ("this is a function named login, it spans lines 10-25, its body
  is..."). It's the same thing the Python interpreter uses. Zero external deps.

We produce, per file:
  * one chunk for the module-level docstring (if any),
  * one chunk per top-level function,
  * one "overview" chunk per class (its signature + docstring),
  * one chunk per method inside each class.
"""
from __future__ import annotations

import ast
from pathlib import Path

from .base import Chunk


def _class_overview(node: ast.ClassDef, source: str) -> str:
    """Build a compact 'header' for a class: `class Foo(Base):` + its docstring.

    We do NOT put the whole class body in one chunk — a big class would become one
    giant chunk that drowns out everything else in retrieval. Instead the class gets
    a small overview chunk, and each method becomes its own chunk (below).
    """
    # ast.unparse turns an AST node back into source text (Python 3.9+).
    bases = [ast.unparse(b) for b in node.bases]
    header = f"class {node.name}({', '.join(bases)}):" if bases else f"class {node.name}:"
    doc = ast.get_docstring(node)
    if doc:
        header += f'\n    """{doc}"""'
    return header


def chunk_python(path: Path, source: str, repo_root: Path) -> list[Chunk]:
    """Turn one Python file into a list of logical Chunks.

    Returns an empty list if the file can't be parsed (syntax error) — the caller
    (dispatcher) then falls back to naive line-window chunking so we never lose a file.
    """
    rel = str(path.relative_to(repo_root))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Broken/py2 file — let the caller fall back rather than crash the whole run.
        return []

    chunks: list[Chunk] = []

    def emit(node: ast.AST, kind: str, qualified_name: str, content: str) -> None:
        """Helper that records one chunk with consistent metadata."""
        chunks.append(
            Chunk(
                chunk_id=f"{rel}::{qualified_name}",
                content=content,
                metadata={
                    "file": rel,
                    "language": "python",
                    "kind": kind,  # module_doc | function | class_overview | method
                    "name": qualified_name,
                    "start_line": getattr(node, "lineno", 0),
                    "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                },
            )
        )

    # 1. Module-level docstring (the description at the very top of the file).
    module_doc = ast.get_docstring(tree)
    if module_doc:
        emit(tree, "module_doc", "<module docstring>", module_doc)

    # 2. Walk only the TOP-LEVEL statements of the file (tree.body).
    #    We intentionally don't recurse into nested functions here — keeping it
    #    simple and predictable for Phase 1. (Easy to extend later.)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A top-level function → one chunk with its full source.
            segment = ast.get_source_segment(source, node) or ""
            emit(node, "function", node.name, segment)

        elif isinstance(node, ast.ClassDef):
            # A class → one small overview chunk...
            emit(node, "class_overview", node.name, _class_overview(node, source))

            # ...plus one chunk per method inside it.
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    segment = ast.get_source_segment(source, item) or ""
                    emit(item, "method", f"{node.name}.{item.name}", segment)

    return chunks
