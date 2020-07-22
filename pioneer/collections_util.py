from typing import Dict, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


def set_optional_kv(d: Dict[K, V], k: K, v: Optional[V]):
    if v is not None:
        d[k] = v
