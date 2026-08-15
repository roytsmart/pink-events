import pathlib
import joblib

__all__ = [
    "path_cache",
    "memory",
]

#: The location on the filesystem where intermediate results are stored.
path_cache = pathlib.Path.home() / ".pinkevents/cache"

#: A representation of the cache which stores intermediate results.
memory = joblib.Memory(location=path_cache, mmap_mode="r", verbose=0)
