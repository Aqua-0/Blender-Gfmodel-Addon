"""Mathutils compatibility for running core code outside Blender.

Blender's Python ships `mathutils`, but plain CPython typically does not.
Core parsers/writers mostly need a lightweight vector container, so provide
an extremely small fallback implementation.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple, Union, overload

try:
                                                
    from mathutils import Vector as Vector                
except Exception:                    
    Number = Union[int, float]

    class Vector(tuple):
        @overload
        def __new__(cls, seq: Sequence[Number]) -> "Vector": ...

        @overload
        def __new__(cls, seq: Iterable[Number]) -> "Vector": ...

        def __new__(cls, seq) -> "Vector":
            return tuple.__new__(cls, tuple(float(x) for x in seq))

        def __iter__(self) -> Iterator[float]:
            return tuple.__iter__(self)                              

        @property
        def x(self) -> float:
            return float(self[0]) if len(self) > 0 else 0.0

        @property
        def y(self) -> float:
            return float(self[1]) if len(self) > 1 else 0.0

        @property
        def z(self) -> float:
            return float(self[2]) if len(self) > 2 else 0.0

        @property
        def w(self) -> float:
            return float(self[3]) if len(self) > 3 else 0.0

        def __mul__(self, other: Number) -> "Vector":
            return Vector((float(v) * float(other) for v in self))

        def __rmul__(self, other: Number) -> "Vector":
            return self.__mul__(other)

        def __add__(self, other: "Vector") -> "Vector":
            n = max(len(self), len(other))
            return Vector(
                (float(self[i]) if i < len(self) else 0.0)
                + (float(other[i]) if i < len(other) else 0.0)
                for i in range(n)
            )

        def __sub__(self, other: "Vector") -> "Vector":
            n = max(len(self), len(other))
            return Vector(
                (float(self[i]) if i < len(self) else 0.0)
                - (float(other[i]) if i < len(other) else 0.0)
                for i in range(n)
            )

        def to_tuple(self) -> Tuple[float, ...]:
            return tuple(float(v) for v in self)
