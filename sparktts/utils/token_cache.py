import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class CachedSegment:
    semantic_ids: List[int]
    global_ids: Optional[List[int]] = None


class TokenCache:
    """Simple disk-backed cache for semantic token segments."""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = cache_path
        self._store: Dict[str, CachedSegment] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            self.load()

    def get(self, key: str) -> Optional[CachedSegment]:
        return self._store.get(key)

    def set(
        self, key: str, semantic_ids: List[int], global_ids: Optional[List[int]] = None
    ) -> None:
        self._store[key] = CachedSegment(semantic_ids=semantic_ids, global_ids=global_ids)

    def load(self) -> None:
        with open(self.cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._store = {k: CachedSegment(**v) for k, v in raw.items()}

    def save(self) -> None:
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        serializable = {k: asdict(v) for k, v in self._store.items()}
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
