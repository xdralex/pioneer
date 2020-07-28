from typing import Dict, Optional, TypeVar, Any
import numpy as np

K = TypeVar('K')
V = TypeVar('V')


def set_optional_kv(d: Dict[K, V], k: K, v: Optional[V]):
    if v is not None:
        d[k] = v


def arr2str(arr: np.ndarray, fmt: str = '.3f') -> str:
    return '[' + ', '.join([f'{x:{fmt}}' for x in arr]) + ']'


def dict2str(d: Dict[str, Any], delim: str = ', ') -> str:
    return delim.join([f'{k}={v}' for k, v in d.items()])
