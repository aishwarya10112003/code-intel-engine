"""Chunking package: turns files into logical, self-contained Chunks."""
from .base import Chunk
from .dispatcher import chunk_file

__all__ = ["Chunk", "chunk_file"]
