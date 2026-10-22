"""Keep freed memory for reuse instead of handing it back to the system.

Rendering a frame makes many large temporary NumPy arrays. By default glibc
gives each one fresh pages from the system (``mmap``) and returns them when
the array is freed, so every frame pays a page fault for each page it
touches. Raising the thresholds lets freed arrays be reused from the heap:
terrain rendering got about 40% faster this way (most of all under WSL). The
heap then holds on to a frame's worth of temporaries, some tens of MB.
"""

import ctypes
import ctypes.util

# mallopt parameters, from glibc's malloc.h.
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3

MMAP_THRESHOLD = 32 << 20  # glibc's largest allowed value on 64-bit systems
TRIM_THRESHOLD = 512 << 20


def keep_freed_memory() -> bool:
    """Tune glibc's allocator for large temporary arrays. Returns whether it
    could (only with glibc, so not on Windows or macOS)."""
    name = ctypes.util.find_library("c")
    if name is None:
        return False
    try:
        libc = ctypes.CDLL(name)
        mallopt = libc.mallopt
    except OSError, AttributeError:
        return False
    return bool(
        mallopt(_M_MMAP_THRESHOLD, MMAP_THRESHOLD)
        and mallopt(_M_TRIM_THRESHOLD, TRIM_THRESHOLD)
    )
