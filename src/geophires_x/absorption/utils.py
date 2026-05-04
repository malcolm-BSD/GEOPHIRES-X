"""Utility helpers for absorption chiller package.

This file contains small helpers used by other modules. Keep utilities minimal
and well-tested.
"""
from typing import Iterable


def chunk_iterable(iterable: Iterable, size: int):
    """Yield successive chunks from iterable of given size.

    Parameters
    ----------
    iterable:
        Any iterable.
    size:
        Chunk size (>0).
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

